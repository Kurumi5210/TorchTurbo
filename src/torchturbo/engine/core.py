"""TorchTurboEngine — one-stop-shop for distributed training."""

from __future__ import annotations

from typing import Any, Optional

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ..logging import get_logger
from .backend_registry import get_available_backends, get_backend
from .backends.base import BaseBackend
from .config import EngineConfig, FreezePolicy
from .hooks import TrainHook, TrainState
from .utils import has_meta_tensor

from torchturbo.utils.device import is_npu_available

if is_npu_available:
    from torch_npu.contrib import transfer_to_npu as _


logger = get_logger(__name__)


class GatherAndInnerLoopSPDistributedSampler(DistributedSampler):
    """DistributedSampler subclass that gives each SP group contiguous index blocks.

    All ranks sharing the same ``dp_rank`` yield the identical sequence
    of ``sp_size`` indices per iteration block.

    Parameters
    ----------
    dataset:
        Dataset to sample from.
    sp_size:
        Sequence-parallel group size.  ``1`` = standard DistributedSampler.
    num_replicas:
        Total world size.  Defaults to ``dist.get_world_size()``.
    rank:
        Global rank.  Defaults to ``dist.get_rank()``.
    shuffle:
        Whether to shuffle indices each epoch.
    seed:
        Random seed used when *shuffle* is True.
    drop_last:
        Drop the tail to make the dataset evenly divisible.
    """

    def __init__(
        self,
        dataset: Dataset,
        sp_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        # Let DistributedSampler handle defaults for num_replicas / rank
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        if self.num_replicas % sp_size != 0:
            raise ValueError(
                f"world_size ({self.num_replicas}) must be divisible by sp_size ({sp_size})"
            )

        self.sp_size = sp_size
        self.dp_size = self.num_replicas // sp_size
        self.dp_rank = self.rank // sp_size

        # Recompute sizing: each DP group consumes sp_size samples per
        # iteration, so total iterations = dataset / (dp_size * sp_size).
        total = len(self.dataset)  # type: ignore[arg-type]
        if self.drop_last:
            self.num_iterations = total // (self.dp_size * self.sp_size)
        else:
            self.num_iterations = math.ceil(total / (self.dp_size * self.sp_size))
        self.total_size = self.num_iterations * self.dp_size * self.sp_size
        # Override parent's num_samples (used by __len__ in parent)
        self.num_samples = self.num_iterations * self.sp_size

        logger.info(
            "SPDistributedSampler: world=%d, rank=%d, sp_size=%d, "
            "dp_size=%d, dp_rank=%d, iterations=%d",
            self.num_replicas,
            self.rank,
            self.sp_size,
            self.dp_size,
            self.dp_rank,
            self.num_iterations,
        )

    def __iter__(self) -> Iterator[int]:
        # Build full index list (deterministic across all ranks)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Pad or truncate to total_size
        if len(indices) < self.total_size:
            # Cycle the full list until we reach total_size.
            indices = (indices * math.ceil(self.total_size / len(indices)))[: self.total_size]
        else:
            indices = indices[: self.total_size]

        # Each iteration block has dp_size * sp_size indices laid out as:
        #   [dp0_s0, dp0_s1, ..., dp0_sp-1, dp1_s0, ..., dpN_sp-1]
        #
        # DP rank d gets indices at offsets [d*sp_size .. d*sp_size+sp_size-1]
        # within each block.  All ranks in the same SP group (same dp_rank)
        # yield the identical list.
        rank_indices = []
        for i in range(self.num_iterations):
            base = i * self.dp_size * self.sp_size + self.dp_rank * self.sp_size
            for s in range(self.sp_size):
                rank_indices.append(indices[base + s])

        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class TorchTurboEngine:
    """One-stop-shop engine for distributed training.

    Owns the full lifecycle: config → model sharding → optimizer →
    training loop with hooks.

    Parameters
    ----------
    config:
        An :class:`EngineConfig`, a raw ``dict``, or ``None`` (defaults).

    Example::

        engine = TorchTurboEngine({"backend": "fsdp2", "mixed_precision": "bf16"})

        # Setup
        engine.prepare_optimizer(
            optimizer_cls=torch.optim.AdamW,
            lr=1e-4,
        )
        engine.prepare_lr_scheduler(
            scheduler_cls=CosineAnnealingLR,
            T_max=1000,
        )

        # Train — provide compute_loss as a callable
        engine.fit(
            train_dataloader=dataloader,
            compute_loss_fn=lambda model, batch, state: model(**batch).loss,
            num_epochs=3,
            hooks=[LoggingHook(), CheckpointHook()],
        )

        # Or subclass for more control
        class MyEngine(TorchTurboEngine):
            def compute_loss(self, model, batch, state):
                return model(**batch).loss

        engine = MyEngine(config)
        engine.prepare_optimizer(AdamW, lr=1e-4)
        engine.fit(dataloader, num_epochs=3)
    """

    def __init__(self, config: EngineConfig | dict[str, Any] | None = None) -> None:
        # Normalize to EngineConfig
        if config is None:
            self.config = EngineConfig()
        elif isinstance(config, dict):
            self.config = EngineConfig.from_dict(config)
        else:
            self.config = config

        # Select distributed backend
        backend_name = self.config.backend.value
        backend_cls = get_backend(backend_name)
        if backend_cls is None:
            raise ValueError(
                f"Unknown distributed backend {backend_name!r}. "
                f"Available: {get_available_backends()}"
            )

        self.backend: BaseBackend = backend_cls(self.config)

        # These are set by setup()
        self.model: nn.Module | None = None
        self.model_name_or_path: str | None = None
        self.optimizer: Any = None
        self.lr_scheduler: Any = None

        # Hooks (can be added before or after setup)
        self.hooks: list[TrainHook] = []

        logger.info("TorchTurboEngine initialized: backend=%s", backend_name)

    # ------------------------------------------------------------------
    # Model preparation (meta device → shard → load weights)
    # ------------------------------------------------------------------

    def build_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        trust_remote_code: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        peft_config: Any | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        """Build a distributed model instance without registering it on the engine.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model path (local or repo id).
        torch_dtype:
            Data type for model parameters.
        model_cls:
            Model class to use.  If ``None``, uses HuggingFace
            ``AutoModelForCausalLM`` (which will pick up registered
            torchturbo models).
        trust_remote_code:
            Passed to HuggingFace ``from_pretrained``.
        init_kwargs:
            Additional keyword arguments
        peft_config:
            Optional PEFT/LoRA configuration (``LoraConfig`` or ``dict``).
            When provided, LoRA adapters are applied **after** module
            replacement but **before** FSDP2 sharding.  Applying LoRA registers
            a custom handler for ``GroupedMLP`` (MoE experts with grouped
            matmul).
        load_weights:
            Whether to load checkpoint weights.  When ``False``, the model is
            still materialized onto real device storage but checkpoint I/O is
            skipped.

        Returns
        -------
        The sharded model instance.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if init_kwargs is None:
            init_kwargs = {}

        # Load config
        if "config" in init_kwargs:
            config = init_kwargs["config"]
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )

        if self.config.mtp.enabled:
            n_mtp = getattr(config, "num_nextn_predict_layers", 0)
            if n_mtp > 0:
                logger.info("MTP enabled: %d MTP layer(s) will be built", n_mtp)
            else:
                logger.warning(
                    "--tt_enable_mtp passed but model config has "
                    "num_nextn_predict_layers=0 — nothing to enable."
                )
        else:
            if hasattr(config, "num_nextn_predict_layers"):
                if config.num_nextn_predict_layers > 0:
                    logger.info(
                        "MTP off (default): stripping %d MTP layer(s) — "
                        "their checkpoint weights will be skipped. "
                        "Pass --tt_enable_mtp to enable.",
                        config.num_nextn_predict_layers,
                    )
                config.num_nextn_predict_layers = 0

        if model_cls is None:
            model_cls = AutoModelForCausalLM

        # 2. Build model on meta device (no memory)
        #
        # Try torchturbo's own model registry first (keyed by model_type).
        # This bypasses AutoModelForCausalLM.from_config entirely, avoiding
        # the type(config) mismatch that occurs when the config object was
        # created from a remote config class (e.g. via trust_remote_code).
        from ..models.base import TurboModel
        from ..models.registry import get_model
        from transformers import PreTrainedModel

        model_type = getattr(config, "model_type", None)
        turbo_cls = get_model(model_type) if model_type else None

        turbo_model_cls = turbo_cls if turbo_cls is not None else model_cls
        if not isinstance(turbo_model_cls, type) or not issubclass(
            turbo_model_cls, TurboModel
        ):
            raise TypeError(
                "TorchTurboEngine requires a TurboModel subclass. "
                f"Got {getattr(turbo_model_cls, '__name__', turbo_model_cls)!r} "
                f"for model_type={model_type!r}."
            )
        if not issubclass(turbo_model_cls, PreTrainedModel):
            raise TypeError(
                "TorchTurboModel classes must also inherit transformers.PreTrainedModel. "
                f"Got {turbo_model_cls.__name__}."
            )

        with turbo_model_cls.init_context(config):
            logger.info("Building model on meta device: %s", model_name_or_path)
            with torch.device("meta"):
                logger.info(
                    "Using torchturbo model class: %s", turbo_model_cls.__name__
                )
                model = turbo_model_cls._from_config(config)

                # Resolve parallel plan
                parallel_plan = self._resolve_parallel_plan(model)

                # Prepare model via backend (module replacement → LoRA → sharding).
                logger.info("Sharding model via %s backend", self.config.backend.value)
                model = self.backend.prepare_model(
                    model, parallel_plan=parallel_plan, peft_config=peft_config
                )

                parallel_context = self._get_parallel_context()
                model.on_parallelized(parallel_context)

        logger.info(
            "%s model via %s backend",
            "Loading" if load_weights else "Materializing",
            self.config.backend.value,
        )
        self.backend.load_model(
            model,
            model_name_or_path=model_name_or_path,
            parallel_context=parallel_context,
            load_weights=load_weights,
        )

        model.on_materialized()
        meta_tensor_names = has_meta_tensor(model)
        if meta_tensor_names:
            raise RuntimeError(
                "Model still has meta tensors after backend load/materialization "
                "and on_materialized(). Checkpoint loading must assign all "
                "parameters and persistent buffers; runtime buffers must be "
                "rebuilt after materialization or registered lazily with a real "
                "tensor on first use. Meta tensors: "
                f"{', '.join(meta_tensor_names)}"
            )

        logger.info("Model built: %s (%s)", type(model).__name__, torch_dtype)
        return model

    def _apply_freeze_policy(self, model: nn.Module, policy: FreezePolicy) -> None:
        if not policy.enabled:
            return

        get_parameter_groups = getattr(model, "get_parameter_groups", None)
        if not callable(get_parameter_groups):
            raise TypeError(
                "FreezePolicy requires models to expose get_parameter_groups(). "
                f"Got {type(model).__name__}."
            )

        parameter_groups = get_parameter_groups()
        available_groups = ", ".join(group.value for group in parameter_groups)

        for group in policy.parameter_groups:
            parameters = parameter_groups.get(group)
            if parameters is None:
                raise ValueError(
                    f"FreezePolicy requested unsupported parameter group "
                    f"{group.value!r} for {type(model).__name__}. "
                    f"Available groups: {available_groups or '<none>'}."
                )

            tensor_count = 0
            element_count = 0
            for _, param in parameters:
                param.requires_grad_(False)
                tensor_count += 1
                element_count += param.numel()

            if tensor_count == 0:
                raise ValueError(
                    f"FreezePolicy group {group.value!r} matched no parameters "
                    f"in {type(model).__name__}."
                )

            logger.info(
                "Froze %d parameter tensors (%d elements) in group %s",
                tensor_count,
                element_count,
                group.value,
            )

    def prepare_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        trust_remote_code: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        peft_config: Any | None = None,
    ) -> nn.Module:
        """Build and register the engine's primary training model."""
        model = self.build_model(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
            model_cls=model_cls,
            trust_remote_code=trust_remote_code,
            init_kwargs=init_kwargs,
            peft_config=peft_config,
            load_weights=True,
        )

        self._apply_freeze_policy(model, self.config.freeze_policy)

        self.model = model
        self.model_name_or_path = model_name_or_path

        # Synchronize printing
        if dist.is_initialized():
            dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Model prepared: %s (%s)", type(model).__name__, torch_dtype)
            self._log_model_info(model)
        
        return model

    def save_model(self, model: nn.Module, output_dir: str, **kwargs: Any) -> None:
        self.backend.save_model(
            model,
            self.model_name_or_path,
            output_dir,
            parallel_context=self._get_parallel_context(),
            **kwargs,
        )
        logger.info(
            "Model saved in %s via %s backend", output_dir, self.config.backend.value
        )

    def _get_parallel_context(self) -> dict[str, Any]:
        return {
            "dp": self.get_dp_info(),
            "sp": self.get_sp_info(),
            "ep": self.get_ep_info(),
        }

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------

    def prepare_optimizer(
        self,
        optimizer_cls: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an optimizer for the prepared model.

        Must be called **after** ``prepare_model()``.

        Example::

            model = engine.prepare_model("meta-llama/Llama-2-7b-hf")
            optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=1e-4)

        Returns the created optimizer (also stored as ``self.optimizer``).
        """
        if self.model is None:
            raise RuntimeError("No model prepared. Call engine.prepare_model() first.")

        if optimizer_cls is None:
            optimizer_cls = torch.optim.AdamW

        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("betas", (0.9, 0.999))
        kwargs.setdefault("weight_decay", 1e-2)

        self.optimizer = self.backend.prepare_optimizer(
            self.model, optimizer_cls, **kwargs
        )
        logger.info("Created optimizer: %s", type(self.optimizer).__name__)
        return self.optimizer

    def prepare_lr_scheduler(
        self,
        scheduler_cls: type,
        **kwargs: Any,
    ) -> Any:
        """Create an LR scheduler for the optimizer.

        Must be called **after** ``prepare_optimizer()``.

        Example::

            optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=1e-4)
            scheduler = engine.prepare_lr_scheduler(
                CosineAnnealingLR, T_max=1000,
            )

        Returns the created scheduler (also stored as ``self.lr_scheduler``).
        """
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer created. Call engine.prepare_optimizer() first."
            )

        self.lr_scheduler = scheduler_cls(self.optimizer, **kwargs)
        logger.info("Created LR scheduler: %s", type(self.lr_scheduler).__name__)
        return self.lr_scheduler

    def _resolve_parallel_plan(self, model: nn.Module):
        """Resolve the parallel plan for a model."""
        from ..models.base import TurboModel

        if not isinstance(model, TurboModel):
            raise TypeError(
                "TorchTurboEngine requires prepared models to inherit TurboModel. "
                f"Got {type(model).__name__}."
            )

        plan = type(model).get_default_parallel_plan()
        logger.info(
            "Auto-discovered default parallel plan from %s "
            "(plan_keys=%d, sp=%d, ep=%d)",
            type(model).__name__,
            len(getattr(plan, "plan", {})),
            getattr(plan, "sp_size", 1),
            getattr(plan, "ep_size", 1),
        )

        if self.config.parallel_plan is not None:
            logger.info("Overriding parallel plan with engine config")
            engine_plan = self.config.parallel_plan

            for attr in ["sp_size", "ep_size"]:
                if hasattr(engine_plan, attr):
                    val = getattr(engine_plan, attr)
                    setattr(plan, attr, val)

            if hasattr(engine_plan, "plan") and engine_plan.plan:
                if not hasattr(plan, "plan") or plan.plan is None:
                    plan.plan = {}
                plan.plan.update(engine_plan.plan)

        logger.info(
            "Parallel Plan for %s (sp=%d, ep=%d)",
            type(model).__name__,
            getattr(plan, "sp_size", 1),
            getattr(plan, "ep_size", 1),
        )
        return plan

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_hook(self, hook: TrainHook) -> None:
        """Register a training hook."""
        self.hooks.append(hook)

    def add_hooks(self, hooks: list[TrainHook]) -> None:
        """Register multiple training hooks."""
        self.hooks.extend(hooks)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def start_profiling(self, output_dir: str) -> None:
        """Start TorchTurbo profiling under ``output_dir``."""
        from .profiling import start_profiling

        start_profiling(output_dir)

    def step_profiling(self) -> None:
        """Advance TorchTurbo profiling by one optimizer step."""
        from .profiling import step_profiling

        step_profiling()

    def stop_profiling(self) -> None:
        """Stop TorchTurbo profiling and flush traces."""
        from .profiling import stop_profiling

        stop_profiling()

    # ------------------------------------------------------------------
    # Data loading (SP-aware)
    # ------------------------------------------------------------------

    def get_dp_info(self) -> dict[str, Any]:
        """Get data-parallel rank/size, accounting for sequence parallelism.

        When SP is active, ranks in the same SP group share the same data.
        The effective DP size = world_size / sp_size.

        Example::

            sp_size = 2
            info = engine.get_dp_info()
            # world_size=8 → dp_size=4, ranks [0,1] share dp_rank=0
        """
        return self.backend.get_dp_info()

    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Create a DataLoader with SP-aware distributed sampling.

        When ``sp_size > 1``, uses ``DistributedSampler`` with
        ``num_replicas=dp_size`` and ``rank=dp_rank``, so that all
        ranks in the same SP group receive the **same batch**.

        Example::

            dataloader = engine.create_dataloader(
                dataset, batch_size=4,
            )
            engine.fit(dataloader, ...)

        Parameters
        ----------
        dataset:
            A ``torch.utils.data.Dataset``.
        sp_size:
            Sequence parallel size.  ``1`` = standard DP.
        batch_size:
            Per-rank micro-batch size.
        """
        from torch.utils.data import DataLoader, DistributedSampler

        dp_info = self.get_dp_info()
        sp_info = self.get_sp_info()

        # sampler = DistributedSampler(
        #     dataset,
        #     num_replicas=dp_info["dp_size"],
        #     rank=dp_info["dp_rank"],
        #     shuffle=shuffle,
        #     drop_last=drop_last,
        # )
        # To align with version v1
        sampler = GatherAndInnerLoopSPDistributedSampler(
            dataset,
            sp_size=sp_info["sp_size"],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

        logger.info(
            "Created SP-aware DataLoader: dp_size=%d, dp_rank=%d, sp_size=%d",
            dp_info["dp_size"],
            dp_info["dp_rank"],
            sp_info["sp_size"],
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataloader: Iterable,
        num_epochs: int = 1,
        max_steps: int | None = None,
        compute_loss_fn: Any = None,
        return_outputs=False,
        hooks: list[TrainHook] | None = None,
        max_grad_norm: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> TrainState:
        """Run the training loop.

        **SP-aware**: Automatically detects ``sp_size`` from the resolved
        parallel plan and adjusts ``gradient_accumulation_steps`` to
        maintain the same global batch size.

        If ``train_dataloader`` is a ``Dataset``, wraps it with an
        SP-aware ``DistributedSampler``.  If it's already a ``DataLoader``,
        uses it as-is (caller is responsible for SP-aware sampling).

        Parameters
        ----------
        train_dataloader:
            A ``DataLoader``, ``Dataset``, or any iterable that yields batches.
        num_epochs:
            Number of training epochs.
        max_steps:
            If set, stop after this many optimizer steps.
        compute_loss_fn:
            A callable ``(model, batch, state, return_outputs) -> loss_tensor``.
        return_outputs:
            If set, return outputs.
        hooks:
            Additional hooks for this run.
        max_grad_norm:
            Gradient clipping norm.  ``None`` = no clipping.
        gradient_accumulation_steps:
            Number of micro-batches per optimizer step (before SP adjustment).
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call engine.prepare_optimizer() before engine.fit()")

        # --- Auto-detect SP from parallel plan ---
        sp_size = self.get_sp_info()["sp_size"]
        if sp_size > 1:
            original_grad_accum = gradient_accumulation_steps
            gradient_accumulation_steps *= sp_size
            logger.info(
                "SP auto-adapt: sp_size=%d, gradient_accumulation_steps %d → %d",
                sp_size,
                original_grad_accum,
                gradient_accumulation_steps,
            )

        # --- Wrap Dataset with SP-aware sampler if needed ---
        train_dataloader = self._ensure_sp_dataloader(train_dataloader, sp_size)

        # Merge per-run hooks with persistent hooks
        all_hooks = self.hooks + (hooks or [])

        # Resolve loss function
        if compute_loss_fn is not None:
            loss_fn = compute_loss_fn
        else:
            loss_fn = self.compute_loss

        # Build state
        state = TrainState(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )

        # --- Training loop ---
        self._fire(all_hooks, "on_train_begin", state)
        self.model.train()

        for epoch in range(num_epochs):
            state.epoch = epoch
            self._fire(all_hooks, "on_epoch_begin", state)

            # Update sampler epoch for proper shuffling
            if hasattr(train_dataloader, "sampler") and hasattr(
                train_dataloader.sampler, "set_epoch"
            ):
                train_dataloader.sampler.set_epoch(epoch)

            accumulation_count = 0
            is_step_start = True

            for batch in train_dataloader:
                state.batch = batch

                if is_step_start:
                    self._fire(all_hooks, "on_step_begin", state)
                    is_step_start = False

                self._fire(all_hooks, "on_microbatch_begin", state)

                # Forward
                self._fire(all_hooks, "before_forward", state)
                state.loss = loss_fn(self.model, batch, return_outputs, state)
                self._fire(all_hooks, "after_forward", state)

                # Scale for gradient accumulation
                if gradient_accumulation_steps > 1:
                    state.loss = state.loss / gradient_accumulation_steps

                # Backward
                self._fire(all_hooks, "before_backward", state)
                state.loss.backward()
                self._fire(all_hooks, "after_backward", state)
                self._fire(all_hooks, "on_microbatch_end", state)

                accumulation_count += 1
                if accumulation_count % gradient_accumulation_steps != 0:
                    continue

                # Optimizer step
                state.skip_optimizer_step = False
                self._fire(all_hooks, "before_optimizer_step", state)

                if max_grad_norm is not None:
                    state.grad_norm = self.backend.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    ).item()
                    if not torch.isfinite(torch.tensor(state.grad_norm)):
                        logger.warning("Non-finite grad norm — skipping step")
                        state.skip_optimizer_step = True

                if not state.skip_optimizer_step:
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()

                self._fire(all_hooks, "after_optimizer_step", state)

                state.global_step += 1
                state.metrics["loss"] = state.loss.item() * gradient_accumulation_steps

                self._fire(all_hooks, "on_step_end", state)
                is_step_start = True

                if state.should_stop:
                    break
                if max_steps is not None and state.global_step >= max_steps:
                    break

            self._fire(all_hooks, "on_epoch_end", state)

            if state.should_stop or (
                max_steps is not None and state.global_step >= max_steps
            ):
                break

        self._fire(all_hooks, "on_train_end", state)
        return state

    def get_sp_info(self) -> dict[str, Any]:
        """Get sequence-parallel rank, size and group."""
        return self.backend.get_sp_info()

    def get_ep_info(self) -> dict[str, Any]:
        """Get expert-parallel rank, size and group."""
        return self.backend.get_ep_info()

    def _ensure_sp_dataloader(self, dataloader_or_dataset: Any, sp_size: int) -> Any:
        """Wrap a Dataset with SP-aware sampling, or return DataLoader as-is."""
        from torch.utils.data import DataLoader, Dataset

        if not isinstance(dataloader_or_dataset, Dataset):
            # Already a DataLoader or custom iterable — use as-is
            return dataloader_or_dataset

        # Wrap Dataset with SP-aware DistributedSampler
        return self.create_dataloader(
            dataloader_or_dataset,
        )

    def compute_loss(
        self, model: nn.Module, inputs: Any, return_outputs: bool = False, **kwargs
    ):
        """Compute the scalar loss for a single micro-batch."""
        return self.backend.compute_loss(model, inputs, return_outputs, **kwargs)

    def compute_diffusion_loss(
        self,
        model: nn.Module,
        inputs: Any,
        ga: int = 1,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute block-diffusion loss for an ``is_diffusion_model`` model.

        ``ga`` is the effective gradient-accumulation steps (already SP-adjusted
        by the dataloader); the backend divides the loss by it to match HF's
        grad-accum normalization.
        """
        return self.backend.compute_diffusion_loss(
            model, inputs, ga=ga, return_outputs=return_outputs, **kwargs
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_model_info(self, model: nn.Module) -> None:
        """Log model architecture and parameter sharding map."""

        if dist.get_rank() != 0:
            return

        # --- Architecture ---
        logger.info("Model architecture:\n%s", model)

        # --- Sharding table ---
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            DTensor = None

        max_name_len = max(len(fqn) for fqn, _ in model.named_parameters())
        name_col = max(max_name_len, len("Parameter")) + 2  # +2 for padding

        header = f"{'Parameter':<{name_col}} {'Shape':<20} {'Dtype':<12} {'Device':<10} {'Mesh':<30} {'Placements'}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for fqn, param in model.named_parameters():
            shape = str(tuple(param.shape))
            dtype = str(param.dtype).replace("torch.", "")
            device = str(param.device)

            if DTensor is not None and isinstance(param, DTensor):
                mesh = param.device_mesh
                mesh_info = (
                    f"{mesh.mesh_dim_names}={list(mesh.mesh.shape)}"
                    if hasattr(mesh, "mesh_dim_names")
                    else str(mesh)
                )
                placements = str(param.placements)
            else:
                mesh_info = "-"
                placements = "local"

            lines.append(
                f"{fqn:<{name_col}} {shape:<20} {dtype:<12} {device:<10} {mesh_info:<30} {placements}"
            )

        lines.append(sep)
        logger.info("Parameter sharding map:\n%s", "\n".join(lines))

    @staticmethod
    def _fire(hooks: list[TrainHook], hook_name: str, state: TrainState) -> None:
        """Call the named method on all hooks."""
        for hook in hooks:
            method = getattr(hook, hook_name, None)
            if method is not None:
                method(state)

    def __repr__(self) -> str:
        setup_status = "ready" if self.model is not None else "not setup"
        return f"<TorchTurboEngine(backend={self.backend}, {setup_status})>"
