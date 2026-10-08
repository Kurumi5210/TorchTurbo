"""Distributed backend base class — the interface each backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from transformers import PretrainedConfig

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...logging import get_logger
from ...runtime import DeviceRuntime
from ..model_fqn import ModelFQNMap

if TYPE_CHECKING:
    from ..config import EngineConfig
    from ..optimizers.plan import OptimizerPlan


class BaseBackend(ABC):
    """Abstract base for a distributed backend.

    Subclass this and implement ``shard_model`` / ``prepare_optimizer``
    to add a new backend (e.g. Megatron).
    """

    name: str = ""

    def __init__(self, config: EngineConfig, device_runtime: DeviceRuntime) -> None:
        self.config = config
        self.logger = get_logger(type(self).__module__)
        if not isinstance(device_runtime, DeviceRuntime):
            raise TypeError(
                "device_runtime must be a DeviceRuntime. "
                "Create backends through TorchTurboEngine so the engine can "
                "select the process device runtime first."
            )
        self._device_runtime = device_runtime
        self._model_fqn_map: ModelFQNMap | None = None

    @property
    def device_runtime(self) -> DeviceRuntime:
        return self._device_runtime

    @property
    def device(self) -> torch.device:
        return self._device_runtime.device

    def get_model_fqn_map(self) -> ModelFQNMap:
        """Return the FQN map captured while preparing the current model."""
        if self._model_fqn_map is None:
            raise RuntimeError(
                "Model FQN map is unavailable. Call prepare_model() first."
            )
        return self._model_fqn_map

    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        parallel_plan: Any = None,
        peft_config: Any = None,
    ) -> nn.Module:
        """Wrap / shard the model according to this backend's strategy.

        Parameters
        ----------
        model:
            The raw model to shard.
        parallel_plan:
            Optional ``ParallelPlan`` with layer-name → placement mapping.
            If provided, applies per-layer sharding based on FQN patterns.
        peft_config:
            Optional PEFT/LoRA configuration (``LoraConfig`` or ``dict``).
            When provided, LoRA adapters are applied **after** module
            replacement but **before** sharding.  Backends that don't
            support PEFT should raise an error if this is non-None.
        """

    @abstractmethod
    def load_model(
        self,
        model: nn.Module,
        model_name_or_path: str,
        load_weights: bool = True,
    ) -> nn.Module:
        """Load weights or materialize the model without checkpoint I/O."""

    @abstractmethod
    def save_model(
        self,
        model: nn.Module,
        model_name_or_path: str,
        output_dir: str,
        format: str,
        **kwargs: Any,
    ) -> None:
        """Save weights."""

    @abstractmethod
    def load_from_checkpoint(
        self,
        model: nn.Module,
        checkpoint_dir: str,
        format: str,
        **kwargs: Any,
    ) -> None:
        """Re-load weights into an already-sharded model on resume (backend override).

        The loader is chosen by ``save_format`` and must match the format the
        checkpoint was written in.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support load_from_checkpoint.")

    @abstractmethod
    def save_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        output_dir: str,
        **kwargs: Any,
    ) -> None:
        """Save optimizer state for resume (backend override)."""
        raise NotImplementedError(f"{type(self).__name__} does not support save_optimizer.")

    @abstractmethod
    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        checkpoint_dir: str,
        **kwargs: Any,
    ) -> None:
        """Load optimizer state written by :meth:`save_optimizer` (backend override)."""
        raise NotImplementedError(f"{type(self).__name__} does not support load_optimizer.")

    @abstractmethod
    def prepare_optimizer(
        self,
        plan: OptimizerPlan,
    ) -> Optimizer:
        """Return a validated optimizer with all plan-defined step hooks installed."""

    def offload_optimizer(self) -> None:
        """Release prepared optimizer device resources at a cleared-gradient boundary."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support optimizer offload"
        )

    def restore_optimizer(self) -> None:
        """Restore the prepared optimizer's configured training state locations."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support optimizer restore"
        )

    @abstractmethod
    def get_dp_info(self) -> dict[str, Any]:
        """Compute data-parallel rank and size.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_ep_info(self) -> dict[str, Any]:
        """Compute expert-parallel rank, size and group.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_sp_info(self) -> dict[str, Any]:
        """Compute sequence-parallel rank, size and group.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_parallel_context(self) -> dict[str, Any]:
        """Return parallel-context dict of EP/SP/DP info.

        Override in subclass for distributed backends.
        """

    def prepare_inputs(self, model: nn.Module, inputs: Any) -> Any:
        """Apply backend-specific input layout requirements before a step."""
        return inputs

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, inputs: Any, return_outputs: bool = False, **kwargs
    ):
        """Compute Loss.

        Override in subclass for distributed backends.
        """

    def compute_diffusion_loss(
        self,
        model: nn.Module,
        inputs: Any,
        ga: int = 1,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute block-diffusion loss for ``is_diffusion_model`` models.

        Default is unsupported; a backend opts in by overriding this. The engine
        routes here only when ``model.is_diffusion_model`` is True, so a backend
        that never trains diffusion models can leave the default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support diffusion training "
            "(compute_diffusion_loss). Use a backend that implements it, e.g. FSDP2."
        )

    def clip_grad_norm_(self, parameters, max_norm=1.0, norm_type=2.0):
        """Clip the gradient norm of ``parameters`` in place; return the total norm.

        Default delegates to :func:`torch.nn.utils.clip_grad_norm_`. Distributed
        backends (e.g. FSDP2 with DTensor / expert-parallel grads) override this
        with a sharding-aware implementation.
        """
        return nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
