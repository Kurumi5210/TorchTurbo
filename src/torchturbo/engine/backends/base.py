"""Distributed backend base class — the interface each backend implements."""

from __future__ import annotations


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from transformers import PretrainedConfig

import torch.nn as nn

from ...logging import get_logger

if TYPE_CHECKING:
    from ..config import EngineConfig


class BaseBackend(ABC):
    """Abstract base for a distributed backend.

    Subclass this and implement ``shard_model`` / ``prepare_optimizer``
    to add a new backend (e.g. Megatron).
    """

    name: str = ""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.logger = get_logger(type(self).__module__)

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
        parallel_context: dict[str, Any],
        load_weights: bool = True,
    ) -> nn.Module:
        """Load weights or materialize the model without checkpoint I/O."""

    @abstractmethod
    def save_model(
        self,
        model: nn.Module,
        model_name_or_path: str,
        output_dir: str,
        parallel_context: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Save weights."""

    @abstractmethod
    def prepare_optimizer(
        self, model: nn.Module, optimizer_cls: type, **optimizer_kwargs
    ) -> Any:
        """Create an optimizer compatible with this backend.

        Some backends (e.g. Megatron) require engine-managed optimizers.
        The default can just instantiate the class normally.
        """

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
