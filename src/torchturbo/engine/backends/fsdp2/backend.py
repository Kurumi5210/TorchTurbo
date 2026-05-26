"""FSDP2 distributed backend — ``torch.distributed.fsdp.fully_shard``."""

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm
import torch.distributed as dist

from torchturbo.logging import get_logger
from ..base import BaseBackend
from ...backend_registry import register_backend
from torchturbo.parallel import (
    FSDPPlanSpec,
    FSDPParallelPlan,
    FSDPPlacementSpec,
    ParallelPlan,
)
from .config import FSDPConfig
from .weight_loader import DistParamLoader
from .weight_saver import DistParamSaver
from torchturbo.utils.device import get_device_name
from ...config import EngineConfig
from torchturbo.models.modules.comm import all_gather_tensor
from torchturbo.models.modules.attention.ulysses import (
    slice_input_tensor,
    ulysses_pad_and_slice_inputs,
    ulysses_pad,
)
from .prefetch import set_moe_layer_forward_prefetch


logger = get_logger(__name__)


class FSDP2Backend(BaseBackend):
    """FSDP2 backend using ``torch.distributed.fsdp.fully_shard``.

    Reads FSDP-specific settings from ``config.fsdp`` and mixed precision
    from ``config.mixed_precision``.

    When ``config.parallel.sp_size > 1``, creates a sequence parallel
    process group for Ulysses SP.
    """

    name = "fsdp2"

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        fsdp: FSDPConfig = config.backend_config
        self.mixed_precision = config.mixed_precision
        self.sharding_strategy = fsdp.sharding_strategy
        self.reshard_after_forward = fsdp.reshard_after_forward
        self.forward_prefetch = fsdp.forward_prefetch
        self.cpu_offload = fsdp.cpu_offload
        self.pin_memory = fsdp.pin_memory

        # Sequence parallel process group and device mesh
        self.rank = dist.get_rank()
        self._mesh: Any = None  # torch.distributed.device_mesh.DeviceMesh
        self._world_mesh: Any = None
        self._sp_size: int = 1
        self._sp_group: dist.ProcessGroup = None
        self._sp_rank: int = 0

        self.device = get_device_name()
        self.use_transformer_engine = False
        self.gradient_scaling_factor = 1.0

    def _create_sp_group(self, sp_size: int) -> dist.ProcessGroup | None:
        """Create sequence parallel process group.

        Groups consecutive ranks into SP groups of size ``sp_size``.
        E.g. with 8 GPUs and sp_size=4:
          group 0: [0, 1, 2, 3]
          group 1: [4, 5, 6, 7]
        """
        if not dist.is_initialized():
            logger.warning(
                "torch.distributed not initialized — skipping SP group creation. "
                "SP groups will be created lazily when needed."
            )
            return None

        world_size = dist.get_world_size()
        rank = self.rank

        if world_size % sp_size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"sp_size ({sp_size}) for sequence parallelism."
            )

        # Create all SP groups (all ranks must call this)
        sp_group = None
        for start in range(0, world_size, sp_size):
            ranks = list(range(start, start + sp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                sp_group = group
        sp_rank = dist.get_rank(sp_group)

        logger.info(
            "Created SP process group: sp_size=%d, rank=%d, " "group_ranks=%s",
            sp_size,
            rank,
            list(range((rank // sp_size) * sp_size, (rank // sp_size + 1) * sp_size)),
        )

        self._sp_size = sp_size
        self._sp_group = sp_group
        self._sp_rank = sp_rank

    def _init_mesh(self, plan: Any) -> Any | None:
        """Create an N-dimensional DeviceMesh based on the parallel plan."""
        if self._mesh is not None:
            return self._mesh

        if not dist.is_initialized():
            return None

        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            raise ValueError("torch.distributed.device_mesh not available.")

        world_size = dist.get_world_size()

        ep_size = getattr(plan, "ep_size", 1)

        world_size = world_size
        device_name = get_device_name()
        self._world_mesh = init_device_mesh(device_name, (world_size,))
        self._mesh = init_device_mesh(
            device_name, (world_size // ep_size, ep_size), mesh_dim_names=("edp", "ep")
        )

        logger.info("Created DeviceMesh")

        self._create_sp_group(plan.sp_size)

    def prepare_model(self, model: nn.Module, parallel_plan: Any = None) -> nn.Module:
        if parallel_plan is None:
            plan = FSDPParallelPlan()
        elif not isinstance(parallel_plan, FSDPParallelPlan):
            raise ValueError(
                f"FSDP2Backend requires an FSDPParallelPlan, got {type(parallel_plan).__name__}"
            )
        else:
            plan = parallel_plan

        config = model.config

        # Initialize DeviceMesh and SP groups
        self._init_mesh(plan)

        # Module replacement or patching
        if plan.plan and isinstance(plan.plan, dict):
            for fqn, submodule in list(model.named_modules()):
                if fqn == "":
                    continue

                spec = plan.match(fqn)
                if isinstance(spec, FSDPPlanSpec) and spec.replace_module:
                    kwargs = spec.module_kwargs

                    kwargs["hf_config"] = config
                    kwargs["ep_info"] = self.get_ep_info()
                    kwargs["sp_info"] = self.get_sp_info()
                    new_module = spec.replace_module(submodule, **kwargs)

                    if new_module is not None and new_module is not submodule:
                        parent_fqn, _, name = fqn.rpartition(".")
                        parent = model.get_submodule(parent_fqn)
                        setattr(parent, name, new_module)
                        logger.info(
                            "FSDP2Backend repalced module %s with %s",
                            fqn,
                            type(new_module).__name__,
                        )
                    else:
                        logger.info("FSDP2Backend patched module %s in-place", fqn)

        model = self._shard_model(model, parallel_plan)

        ep_size = self.get_ep_info()["ep_size"]
        if ep_size > 1:
            self.gradient_scaling_factor = 1 / ep_size

        return model

    # TODO(zlj): Not used yet
    def _set_prefetch(self, model):
        outer_cls = model.model.layers[0].__class__
        if hasattr(model.config, "first_k_dense_replace"):
            inter_cls = model.model.layers[
                model.config.first_k_dense_replace
            ].mlp.experts.__class__
        else:
            inter_cls = model.model.layers[0].mlp.experts.__class__
        outermost_cls = model.__class__
        set_moe_layer_forward_prefetch(model, outer_cls, inter_cls, outermost_cls)

    def load_model(self, model: nn.Module, model_name_or_path: str):
        weight_loader = DistParamLoader(
            model=model,
            path=model_name_or_path,
            rank=self.rank,
            mesh=self._mesh,
            world_mesh=self._world_mesh,
            device=self.device,
            use_transformer_engine=self.use_transformer_engine,
        )
        weight_loader.load_weights()

    def save_model(
        self, model: nn.Module, model_name_or_path: str, output_dir: str, **kwargs: Any
    ):
        weight_saver = DistParamSaver(
            model=model,
            rank=self.rank,
            mesh=self._mesh,
            path=model_name_or_path,
            use_transformer_engine=self.use_transformer_engine,
            processing_class=None,
        )
        weight_saver.save_model(output_dir)

    def _shard_model(self, model: nn.Module, plan: FSDPParallelPlan) -> nn.Module:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        has_plan = plan.plan and isinstance(plan.plan, dict) and self._mesh is not None

        applied = 0
        cpu_offloaded = 0
        skip_fqns: set[str] = set()

        fsdp_kwargs = {}

        mp_policy = None
        if self.mixed_precision.value == "bf16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        elif self.mixed_precision.value == "fp16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                cast_forward_inputs=True,
            )
        if mp_policy:
            fsdp_kwargs["mp_policy"] = mp_policy

        if hasattr(self, "reshard_after_forward"):
            fsdp_kwargs["reshard_after_forward"] = self.reshard_after_forward

        def _wrap(mod: nn.Module, target_mesh: Any) -> None:
            kw = fsdp_kwargs.copy()
            if hasattr(mod, "_fsdp_cpu_offload"):
                kw["offload_policy"] = mod._fsdp_cpu_offload
            fully_shard(mod, mesh=target_mesh, **kw)

        if has_plan:
            # Module-level specifications (offload, no_shard)
            for fqn, submodule in model.named_modules():
                spec = plan.match(fqn)

                # Check for module-specific offload, OR global config offload
                should_offload = getattr(self, "cpu_offload", False)
                if isinstance(spec, FSDPPlanSpec):
                    if spec.cpu_offload:
                        should_offload = True

                if should_offload:
                    try:
                        from torch.distributed.fsdp.wrap import CPUOffloadPolicy

                        policy = CPUOffloadPolicy(
                            pin_memory=getattr(self, "pin_memory", True)
                        )
                        setattr(submodule, "_fsdp_cpu_offload", policy)
                        cpu_offloaded += 1
                    except ImportError:
                        pass

                    if isinstance(spec, FSDPPlanSpec) and spec.no_shard:
                        # Future-proofing for actual FSDP wrapping logic skipping
                        setattr(submodule, "_fsdp_no_shard", True)

            # Module-level Placements
            for fqn, submodule in reversed(list(model.named_modules())):
                spec = plan.match(fqn)
                placement_spec = None

                if isinstance(spec, FSDPPlanSpec) and spec.placement is not None:
                    placement_spec = spec.placement
                elif isinstance(spec, FSDPPlacementSpec):
                    placement_spec = spec

                if not placement_spec:
                    continue

                target_mesh = self._world_mesh
                if (
                    hasattr(placement_spec, "mesh_dim")
                    and placement_spec.mesh_dim is not None
                ):
                    if placement_spec.mesh_dim == "world":
                        target_mesh = self._world_mesh
                    else:
                        target_mesh = self._mesh[placement_spec.mesh_dim]

                fully_shard(submodule, mesh=target_mesh, **fsdp_kwargs)
                applied += 1

        logger.info(
            "FSDP2Backend: applied parallel plan placements to %d modules, %d modules offloaded",
            applied,
            cpu_offloaded,
        )

        for fqn, m in model.named_modules():
            if isinstance(m, nn.ModuleList):
                for child in m:
                    if not getattr(child, "_fsdp_no_shard", False):
                        _wrap(child, self._world_mesh)

        _wrap(model, self._world_mesh)

        # Apply prefetch policies if configured
        if hasattr(self, "forward_prefetch") and self.forward_prefetch:
            from torch.distributed.fsdp import register_fsdp_forward_method

            register_fsdp_forward_method(model, "forward")

        return model

    def get_dp_info(self) -> dict[str, Any]:
        """Compute data-parallel rank and size, accounting for SP.

        When SP is active, ranks in the same SP group process the same
        data. The effective DP size = world_size / sp_size.

        Returns ``{"dp_rank": ..., "dp_size": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        # If DeviceMesh is initialized, extract DP dimension directly
        if self._mesh is not None and "dp" in self._mesh.mesh_dim_names:
            dp_mesh = self._mesh["dp"]
            return {
                "dp_rank": dp_mesh.get_local_rank(),
                "dp_size": dp_mesh.size(),
            }

        world_size = dist.get_world_size()
        rank = self.rank

        sp_size = self._sp_size if self._sp_size else 1
        dp_size = world_size // sp_size
        dp_rank = rank // sp_size  # ranks in same SP group share dp_rank

        return {
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }

    def get_sp_info(self) -> dict[str, Any]:
        """Get sequence-parallel rank, size and group.

        Returns ``{"sp_size": ..., "sp_group": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        if self._sp_size and self._sp_group:
            return {
                "sp_size": self._sp_size,
                "sp_group": self._sp_group,
                "sp_rank": self._sp_rank,
            }
        else:
            return {"sp_size": 1, "sp_group": None, "sp_rank": 0}

    def get_ep_info(self) -> dict[str, Any]:
        """Get expert-parallel rank, size and group.

        Returns ``{"ep_rank": ..., "ep_size": ..., "ep_group": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        # If DeviceMesh is initialized, extract EP dimension directly
        if self._mesh is not None and "ep" in self._mesh.mesh_dim_names:
            ep_mesh = self._mesh["ep"]
            return {
                "ep_rank": ep_mesh.get_local_rank(),
                "ep_size": ep_mesh.size(),
                "ep_group": ep_mesh.get_group(),
            }
        else:
            return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

    def prepare_optimizer(
        self, model: nn.Module, optimizer_cls: type, **optimizer_kwargs
    ) -> Any:
        """For FSDP2, create the optimizer."""
        params = []
        moe_params = []
        edp = self._mesh["edp"]
        for param in model.parameters():
            if param.device_mesh != edp:
                params.append(param)
            else:
                moe_params.append(param)

        optimizer_params = [{"params": params}, {"params": moe_params}]

        return optimizer_cls(optimizer_params, **optimizer_kwargs)

    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm=1.0, norm_type=2.0):
        params = []
        grads = []
        moe_params = []
        moe_grads = []
        edp = self._mesh["edp"] if self._mesh is not None else None

        for param in parameters:
            # Skip parameters without gradients to ensure correctness.
            if param.grad is not None:
                if edp is not None and param.device_mesh == edp:
                    moe_params.append(param)
                    moe_grads.append(param.grad)
                else:
                    params.append(param)
                    grads.append(param.grad)

        edp_total_norm_list = []
        if moe_params:
            self._scale_moe_gradient(moe_params)
            edp_total_norm = _get_total_norm(moe_grads, norm_type=norm_type)
            edp_total_norm = edp_total_norm.full_tensor().unsqueeze(0)
            edp_total_norm = all_gather_tensor(
                edp_total_norm, self.get_ep_info()["ep_group"]
            )
            edp_total_norm_list.append(edp_total_norm)

        dp_total_norm = _get_total_norm(grads, norm_type=norm_type)
        dp_total_norm = dp_total_norm.full_tensor().unsqueeze(0)

        all_norms = edp_total_norm_list + [dp_total_norm]
        total_norm = torch.linalg.vector_norm(torch.cat(all_norms), norm_type)

        if moe_params:
            _clip_grads_with_norm_(moe_params, max_norm, total_norm)
        if params:
            _clip_grads_with_norm_(params, max_norm, total_norm)

        return total_norm

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if inputs.get("position_ids", None) is None:
            inputs["position_ids"] = torch.arange(
                inputs["input_ids"].size(1), device=inputs["input_ids"].device
            ).unsqueeze(0)
        inputs["use_cache"] = False

        labels = inputs.pop("labels", None)
        assert labels is not None, "'labels' should not be None in inputs"

        num_items_in_batch = kwargs.get("num_items_in_batch", None)

        sp_size = self.get_sp_info()["sp_size"]
        sp_group = self.get_sp_info()["sp_group"]
        sp_rank = self.get_sp_info()["sp_rank"]

        # assert num_items_in_batch is not None, "'num_items_in_batch' should not be None"

        # # Repeated calculation in SP group
        # num_items_in_batch = num_items_in_batch // sp_size

        is_vlm = getattr(model, "is_vlm", False)
        if is_vlm:
            # vlm model's inputs will be sliced after embedding
            inputs["input_ids"], inputs["position_ids"], _, pad_size = ulysses_pad(
                inputs["input_ids"],
                position_ids_rmpad=inputs["position_ids"],
                sp_size=sp_size,
            )
        else:
            inputs["input_ids"], inputs["position_ids"], _, pad_size = (
                ulysses_pad_and_slice_inputs(
                    inputs["input_ids"],
                    position_ids_rmpad=inputs["position_ids"],
                    sp_size=sp_size,
                    sp_group=sp_group,
                    sp_rank=sp_rank,
                )
            )
        assert pad_size == 0, "must be"

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        from torch.nn import CrossEntropyLoss

        loss_fct = CrossEntropyLoss(reduction="sum")

        # shift labels
        labels = nn.functional.pad(labels, (0, 1), value=loss_fct.ignore_index)
        labels = labels[..., 1:].contiguous()
        label_num = (labels != loss_fct.ignore_index).sum()
        if sp_size > 1:
            labels = slice_input_tensor(
                labels,
                dim=1,
                group=sp_group,
                rank=sp_rank,
                padding=False,
                sp_mode="ulysses",
            )

        labels = labels.view(-1).to(logits.device)
        logits = logits.view(-1, logits.size(-1))
        loss = loss_fct(logits, labels)

        if sp_size > 1 and sp_group is not None:
            # Keep SP loss reduction in the autograd graph.  A plain
            # dist.all_reduce updates the scalar value but drops remote loss
            # contributions during backward.
            loss = torch.distributed.nn.all_reduce(
                loss, op=dist.ReduceOp.SUM, group=sp_group
            )

        # Final loss — two paths matching HF Trainer convention:
        if num_items_in_batch:
            # New-style: use num_items_in_batch as denominator.
            # This is the total valid token count across all GA micro-batches,
            # passed down by HF Trainer for correct gradient accumulation.

            # Repeated calculation in SP group
            num_items_in_batch = num_items_in_batch // sp_size
            loss /= num_items_in_batch

            loss /= sp_size
            # Scale by world_size to match HF Trainer's
            # average_tokens_across_devices convention.
            world_size = dist.get_world_size()
            loss = loss * world_size
        else:
            # Old-style: local mean, no num_processes scaling.
            # HF Trainer will divide by gradient_accumulation_steps itself.
            loss /= label_num
            loss /= sp_size

        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    def _scale_moe_gradient(self, params):
        if len(params) < 1:
            return

        assert params[0].device_mesh == self._mesh["edp"]
        grads = [p.grad for p in params]
        torch._foreach_mul_(grads, self.gradient_scaling_factor)

    def __repr__(self) -> str:
        return (
            f"<FSDP2Backend(mp={self.mixed_precision.value}, "
            f"strategy={self.sharding_strategy.value}, "
            f"reshard={self.reshard_after_forward})>"
        )


# Auto-register on import
register_backend("fsdp2", FSDP2Backend)
