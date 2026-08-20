"""Unit tests for ``build_config_from_swift_args`` and LoRA config mapping.

Mirrors ``tests/adapters/llamafactory_v0/test_config.py``: pure-function tests
on a lightweight args namespace, no swift import, no model build.

turbo ``--tt_*`` flags now live on the shared ``TorchTurboArgs``
(``set_torchturbo_args`` stashes them); swift-native fields (bf16/fp16, model
path, output_dir, tuner/rlhf type, LoRA) stay on the swift args namespace.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

# Disable auto-activation so importing torchturbo never touches swift during
# these unit tests (matches the llamafactory adapter test setup).
os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

# Some import chains inside torchturbo reference the optional `grouped_gemm`
# package; stub it so the adapter modules import cleanly without it installed.
if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg


@pytest.fixture(autouse=True)
def _reset_cli_turchturbo(monkeypatch):
    # Clean sys.argv AND the stashed TorchTurboArgs so each test starts from a
    # known state. Tests set turbo flags via _set_tt(...).
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from torchturbo.engine.args import set_torchturbo_args
    set_torchturbo_args(None)
    yield
    set_torchturbo_args(None)


from torchturbo.adapters.msswift.utils import (  # noqa: E402
    build_config_from_swift_args,
    build_lora_config_from_swift_args,
)
from torchturbo.engine.args import TorchTurboArgs  # noqa: E402
from torchturbo.engine.config import (  # noqa: E402
    DistributedBackend,
    FreezePolicy,
    MixedPrecisionPolicy,
    ParameterGroup,
)


def _set_tt(**overrides):
    """Stash a ``TorchTurboArgs`` for the config builder to read back."""
    from torchturbo.engine.args import set_torchturbo_args
    tt = TorchTurboArgs(**overrides)
    set_torchturbo_args(tt)
    return tt


def _base_args(**overrides):
    """A minimal TurboSftArguments-like namespace (swift-native fields only)."""
    defaults = dict(
        bf16=True,
        fp16=False,
        model="/path/to/qwen3.5-moe",
        output_dir="/tmp/out",
        tuner_type="full",
        rlhf_type=None,
        # lora fields (unused for full)
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        use_rslora=False,
        use_dora=False,
        target_modules=["all-linear"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_mixed_precision_from_bf16():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args(bf16=True))
    assert cfg.mixed_precision == MixedPrecisionPolicy.BF16


def test_mixed_precision_fp16():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args(bf16=False, fp16=True))
    assert cfg.mixed_precision == MixedPrecisionPolicy.FP16


def test_mixed_precision_override_by_tt():
    _set_tt(mixed_precision="fp32")
    cfg = build_config_from_swift_args(_base_args(bf16=True))
    assert cfg.mixed_precision == MixedPrecisionPolicy.FP32


def test_backend_default_fsdp2_and_override():
    _set_tt()
    assert build_config_from_swift_args(_base_args()).backend == DistributedBackend.FSDP2
    _set_tt(backend="fsdp2")
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend == DistributedBackend.FSDP2


def test_parallel_plan_sp_ep():
    _set_tt(sp_size=4, ep_size=8)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.parallel_plan.sp_size == 4
    assert cfg.parallel_plan.ep_size == 8


def test_default_sp_ep_is_one():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.parallel_plan.sp_size == 1
    assert cfg.parallel_plan.ep_size == 1


def test_freeze_policy_mapping():
    _set_tt(
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=True,
    )
    cfg = build_config_from_swift_args(_base_args())
    groups = set(cfg.freeze_policy.parameter_groups)
    assert groups == {
        ParameterGroup.VISION_TOWER,
        ParameterGroup.MULTI_MODAL_PROJECTOR,
        ParameterGroup.LANGUAGE_MODEL,
    }
    assert cfg.freeze_policy.enabled is True


def test_no_freeze_default():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.freeze_policy.enabled is False


def test_mtp_config_mapping():
    _set_tt(enable_mtp=True, mtp_loss_scaling_factor=0.3, mtp_log_loss=True)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == pytest.approx(0.3)
    assert cfg.mtp.log_loss is True


def test_mtp_off_default():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.mtp.enabled is False
    assert cfg.mtp.log_loss is False


def test_chunked_loss_config_mapping():
    _set_tt(enable_chunk_loss=True, loss_chunk_size=384)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.loss_chunk.enabled is True
    assert cfg.loss_chunk.chunk_size == 384


def test_activation_checkpointing_owned_by_engine():
    _set_tt(gradient_checkpointing=True)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.activation_checkpointing.enabled is True
    _set_tt(gradient_checkpointing=False)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.activation_checkpointing.enabled is False


def test_cpu_offload_to_fsdp_config():
    _set_tt(cpu_offload=True)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend_config.cpu_offload is True
    _set_tt(cpu_offload=False)
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.backend_config.cpu_offload is False


def test_checkpoint_paths():
    _set_tt()
    cfg = build_config_from_swift_args(_base_args())
    assert cfg.checkpoint.hf_model_path == "/path/to/qwen3.5-moe"
    assert cfg.checkpoint.dcp_path == "/tmp/out"


def test_unsupported_tuner_type_raises():
    _set_tt()
    with pytest.raises(NotImplementedError):
        build_config_from_swift_args(_base_args(tuner_type="adalora"))


def test_dpo_with_mtp_raises():
    _set_tt(enable_mtp=True)
    with pytest.raises(NotImplementedError):
        build_config_from_swift_args(
            _base_args(rlhf_type="dpo", tuner_type="full")
        )


def test_dpo_with_lora_raises():
    _set_tt()
    with pytest.raises(NotImplementedError):
        build_config_from_swift_args(_base_args(rlhf_type="dpo", tuner_type="lora"))


def test_lora_config_all_linear():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", lora_rank=16, lora_alpha=32, target_modules=["all-linear"])
    )
    assert lora is not None
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.target_modules == "all-linear"


def test_lora_config_explicit_targets():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", target_modules="q_proj,k_proj")
    )
    assert lora.target_modules == ["q_proj", "k_proj"]


def test_lora_config_dora_rslora_flags():
    pytest.importorskip("peft")
    _set_tt()
    lora = build_lora_config_from_swift_args(
        _base_args(tuner_type="lora", use_rslora=True, use_dora=True)
    )
    assert lora.use_rslora is True
    assert lora.use_dora is True
