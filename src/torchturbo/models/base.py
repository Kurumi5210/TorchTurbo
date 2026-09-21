from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

import torch
from torch import nn

from ..engine.config import EngineConfig, ParameterGroup
from ..logging import get_logger
from .weight_converter import BaseWeightConverter

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..engine.metrics.flops import BaseModelFlops
    from .modules.moe.moe_layer import MoERuntimeInputs


class SequenceParallelInputMode(str, Enum):
    PRE_EMBEDDING = "pre_embedding"
    POST_EMBEDDING = "post_embedding"


class TurboModel:
    """Base contract between TorchTurboEngine and TorchTurbo models."""

    sequence_parallel_input_mode = SequenceParallelInputMode.PRE_EMBEDDING
    is_diffusion_model = False  # Set to True for subclasses to indicate a diffusion model (uses block-diffusion loss).

    def get_norm(self) -> Optional[nn.Module]:
        """Final normalization layer of the model.

        The FSDP2 tail/root optimizer uses this to place the final norm in a
        child resident unit instead of the auto root unit (the root unit's lone
        managed norm breaks CPU offload). Default: ``<base_model>.norm`` (the
        standard causal-LM layout). Override when the norm is nested deeper
        (e.g. VLMs where it lives under ``model.language_model.norm``).
        """
        base_prefix = getattr(self, "base_model_prefix", "")
        base = getattr(self, base_prefix, None) if base_prefix else None
        norm = getattr(base, "norm", None) if isinstance(base, nn.Module) else None
        return norm if isinstance(norm, nn.Module) else None

    def get_lm_head(self) -> Optional[nn.Module]:
        """Output projection head. Default: ``self.lm_head``; override if the
        head is nested (e.g. under ``language_model``)."""
        lm_head = getattr(self, "lm_head", None)
        return lm_head if isinstance(lm_head, nn.Module) else None

    def get_output_embeddings(self) -> Optional[nn.Module]:
        """Output embedding module (drives the resident tail unit + tying check).

        Default: :meth:`get_lm_head` (the lm_head is the output embedding).
        Override if the output embedding differs from the lm_head.
        """
        return self.get_lm_head()

    @classmethod
    @contextmanager
    def init_context(
        cls, config: Any, engine_config: EngineConfig
    ) -> Iterator[None]:
        """Adapt model config and temporary state for meta-device construction."""
        n_mtp = getattr(config, "num_nextn_predict_layers", 0)
        if engine_config.mtp.enabled:
            if n_mtp > 0:
                logger.info_rank0("MTP enabled: %d MTP layer(s) will be built", n_mtp)
            else:
                logger.warning_rank0(
                    "MTP is enabled but model config has "
                    "num_nextn_predict_layers=0 - nothing to enable."
                )
        elif hasattr(config, "num_nextn_predict_layers"):
            if n_mtp > 0:
                logger.info_rank0(
                    "MTP off (default): stripping %d MTP layer(s) - "
                    "their checkpoint weights will be skipped. "
                    "Enable MTP in the engine config to retain them.",
                    n_mtp,
                )
            config.num_nextn_predict_layers = 0

        state = cls._force_config_dtype_float32(config)
        try:
            yield
        finally:
            cls._restore_config_dtype(config, state)

    @classmethod
    def get_default_parallel_plan(cls, **kwargs: Any):
        from torchturbo.parallel import FSDPParallelPlan

        return FSDPParallelPlan()

    def patch_ulysses_fa(
        self,
        sp_info: dict[str, Any],
    ) -> None:
        from torchturbo.models.modules.attention.ulysses import patch_flash_attention_forward_for_sp
        patch_flash_attention_forward_for_sp(sp_info)

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        """Hook called after backend module replacement/sharding."""

    def on_materialized(self) -> None:
        """Hook called after parameters and buffers are on a real device."""

    def get_moe_runtime_inputs(
        self,
        inputs: dict[str, Any],
        parallel_context: dict[str, Any],
    ) -> MoERuntimeInputs:
        from .modules.moe.moe_layer import MoERuntimeInputs

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or attention_mask.dim() != 2:
            return MoERuntimeInputs()

        padding_mask = attention_mask.eq(0)
        sp_info = parallel_context["sp"]
        if sp_info["sp_size"] > 1:
            from .modules.attention.ulysses import slice_input_tensor

            padding_mask = slice_input_tensor(
                padding_mask,
                dim=1,
                group=sp_info["sp_group"],
                rank=sp_info["sp_rank"],
                padding=False,
            )
        return MoERuntimeInputs(padding_mask=padding_mask)

    def get_sequence_parallel_input_mode(self) -> SequenceParallelInputMode:
        return self.sequence_parallel_input_mode

    def get_parameter_groups(
        self,
    ) -> dict[ParameterGroup, Iterable[tuple[str, nn.Parameter]]]:
        return {
            ParameterGroup.ALL: self.named_parameters(),
            ParameterGroup.LANGUAGE_MODEL: self.named_parameters(),
        }

    def get_weight_converter(self, parallel_context: dict[str, Any], model_path: str):
        return BaseWeightConverter(self, parallel_context, model_path)

    def get_flops_class(self) -> type[BaseModelFlops]:
        """Return the model's forward-FLOP recipe class (a :class:`BaseModelFlops` subclass)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an MFU flops recipe; "
            "implement get_flops_class(self) to enable MFU/HFU logging."
        )

    @staticmethod
    def _force_config_dtype_float32(config: Any) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if config is None:
            return state

        if hasattr(config, "dtype"):
            state["dtype"] = config.dtype
            config.dtype = torch.float32
        elif hasattr(config, "torch_dtype"):
            state["torch_dtype"] = config.torch_dtype
            config.torch_dtype = "float32"

        return state

    @staticmethod
    def _restore_config_dtype(config: Any, state: dict[str, Any]) -> None:
        if config is None:
            return

        if "dtype" in state:
            config.dtype = state["dtype"]
        elif "torch_dtype" in state:
            config.torch_dtype = state["torch_dtype"]


class TurboVLMModel(TurboModel):
    """Base class for multimodal Turbo models."""

    sequence_parallel_input_mode = SequenceParallelInputMode.POST_EMBEDDING
    nested_config_names = ("text_config", "vision_config")
    vision_tower_module_names = ("vision_tower",)
    multi_modal_projector_module_names = ("multi_modal_projector", "mm_projector")
    language_model_module_names = ("language_model",)

    @classmethod
    @contextmanager
    def init_context(
        cls, config: Any, engine_config: EngineConfig
    ) -> Iterator[None]:
        nested_state = {}
        with super().init_context(config, engine_config):
            for name in cls.nested_config_names:
                nested_config = getattr(config, name, None)
                child_state = cls._force_config_dtype_float32(nested_config)
                if child_state:
                    nested_state[name] = child_state

            try:
                yield
            finally:
                for name, child_state in nested_state.items():
                    cls._restore_config_dtype(getattr(config, name, None), child_state)

    def get_parameter_groups(
        self,
    ) -> dict[ParameterGroup, Iterable[tuple[str, nn.Parameter]]]:
        groups = super().get_parameter_groups()
        groups.update(
            {
                ParameterGroup.VISION_TOWER: self._named_parameters_from_modules(
                    self.vision_tower_module_names
                ),
                ParameterGroup.MULTI_MODAL_PROJECTOR: self._named_parameters_from_modules(
                    self.multi_modal_projector_module_names
                ),
                ParameterGroup.LANGUAGE_MODEL: self._named_parameters_from_modules(
                    self.language_model_module_names
                ),
            }
        )
        return groups

    def _named_parameters_from_modules(
        self, module_names: Iterable[str]
    ) -> Iterator[tuple[str, nn.Parameter]]:
        for module_name in module_names:
            try:
                module = self.get_submodule(module_name)
            except AttributeError:
                continue

            yield from module.named_parameters(prefix=module_name)
