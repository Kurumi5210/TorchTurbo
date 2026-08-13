"""Tests for unified EngineConfig loading (``--tt_config`` YAML/JSON + layering).

The config-loader feature assembles :class:`EngineConfig` from four layers:

1. ``TorchTurboArgs`` defaults / adapter-seeded values
2. ``TORCH_TURBO_*`` environment variables
3. the YAML/JSON config file passed via ``--tt_config``
4. explicitly supplied ``--tt_*`` CLI flags

Precedence is: CLI > file > env > TorchTurboArgs.
"""

import json

import pytest
import yaml

from torchturbo.engine.args import (
    TorchTurboArgs,
    get_arg_specs,
    parse_torchturbo_args,
)
from torchturbo.engine.config import (
    EngineConfig,
    MixedPrecisionPolicy,
    ParameterGroup,
    build_from_args,
)
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.base import TurboModel
from torchturbo.models.qwen3_5_moe.parallel_plan import (
    get_parallel_plan as get_qwen3_5_moe_plan,
)
from torchturbo.ops.base import OpPolicy
from torchturbo.parallel.fsdp import FSDPPlanSpec


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep every test hermetic against real ``TORCH_TURBO_*`` env vars."""
    for arg_spec in get_arg_specs():
        if arg_spec.env_name is not None:
            monkeypatch.delenv(arg_spec.env_name, raising=False)
    monkeypatch.delenv("TT_MODEL_NAME_OR_PATH", raising=False)
    monkeypatch.delenv("TT_OUTPUT_DIR", raising=False)


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _build_from_argv(argv: list[str]) -> EngineConfig:
    tt, _ = parse_torchturbo_args(argv)
    return build_from_args(tt)


class _FakeQwen35Moe(TurboModel):
    """Model whose default plan mirrors ``qwen3_5_moe.parallel_plan``."""

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        return get_qwen3_5_moe_plan()


def _resolve_engine_plan(cfg: EngineConfig):
    """Run the engine's model-default + engine-config plan merge."""
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.config = cfg
    return engine._resolve_parallel_plan(_FakeQwen35Moe())


# ---------------------------------------------------------------------------
# Explicit CLI override extraction
# ---------------------------------------------------------------------------


def test_parse_overrides_only_explicit_flags():
    tt, remaining = parse_torchturbo_args(
        ["--tt_sp_size", "2", "--model_name_or_path", "Qwen/Qwen2.5-7B"]
    )

    assert remaining == ["--model_name_or_path", "Qwen/Qwen2.5-7B"]
    assert tt.config_overrides == {"tt_sp_size": 2}
    # Default-valued flags must not leak into overrides.
    assert "tt_cpu_offload" not in tt.config_overrides
    assert "tt_gradient_checkpointing" not in tt.config_overrides
    assert "tt_prefetch_strategy" not in tt.config_overrides


def test_parse_explicit_false_is_an_override():
    tt, _ = parse_torchturbo_args(["--tt_cpu_offload", "false"])

    assert tt.config_overrides == {"tt_cpu_offload": False}


def test_parse_bool_flag_without_value_is_true_override():
    tt, _ = parse_torchturbo_args(["--tt_cpu_offload"])

    assert tt.config_overrides == {"tt_cpu_offload": True}


def test_parse_runtime_flags_excluded_from_overrides():
    tt, _ = parse_torchturbo_args(
        [
            "--tt_config",
            "cfg.yaml",
            "--tt_profiling",
            "--tt_log_level",
            "DEBUG",
            "--tt_memory_snapshot_steps",
            "2",
            "--tt_ep_size",
            "4",
        ]
    )

    assert tt.config_overrides == {"tt_ep_size": 4}


def test_parse_explicit_default_value_still_an_override():
    """Passing the default explicitly must still beat a config-file value."""
    tt, _ = parse_torchturbo_args(["--tt_prefetch_strategy", "auto"])

    assert tt.config_overrides == {"tt_prefetch_strategy": "auto"}


# ---------------------------------------------------------------------------
# YAML / JSON file loading
# ---------------------------------------------------------------------------

YAML_CFG = """\
engine_config:
  mixed_precision: fp16
  parallel_plan:
    sp_size: 2
    ep_size: 4
  activation_checkpointing:
    enabled: false
"""


def test_build_from_yaml_with_engineconfig_wrapper(tmp_path):
    cfg = _build_from_argv(["--tt_config", _write(tmp_path, "cfg.yaml", YAML_CFG)])

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP16
    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 4
    assert cfg.activation_checkpointing.enabled is False


def test_build_from_json_with_engineconfig_wrapper(tmp_path):
    data = {
        "engine_config": {
            "mixed_precision": "fp16",
            "parallel_plan": {"sp_size": 2, "ep_size": 4},
            "activation_checkpointing": {"enabled": False},
        }
    }
    cfg = _build_from_argv(
        ["--tt_config", _write(tmp_path, "cfg.json", json.dumps(data))]
    )

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP16
    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 4
    assert cfg.activation_checkpointing.enabled is False


def test_build_from_json_without_wrapper(tmp_path):
    data = {
        "mixed_precision": "fp32",
        "parallel_plan": {"sp_size": 3},
    }
    cfg = _build_from_argv(
        ["--tt_config", _write(tmp_path, "cfg.json", json.dumps(data))]
    )

    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32
    assert cfg.parallel_plan.sp_size == 3


def test_empty_config_file_is_noop(tmp_path):
    cfg = _build_from_argv(["--tt_config", _write(tmp_path, "empty.yaml", "")])

    assert cfg.parallel_plan.sp_size == 1
    assert cfg.activation_checkpointing.enabled is True
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP32


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_from_args(TorchTurboArgs(config_path=str(tmp_path / "nope.yaml")))


def test_malformed_yaml_raises(tmp_path):
    with pytest.raises(yaml.parser.ParserError):
        _build_from_argv(
            ["--tt_config", _write(tmp_path, "bad.yaml", "a: [\n")]
        )


def test_unknown_top_level_field_is_ignored(tmp_path):
    cfg = _build_from_argv(
        ["--tt_config", _write(tmp_path, "unknown.yaml", "bogus: 1\n")]
    )

    assert "bogus" not in cfg.extra


def test_oc_env_interpolation_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_MODEL_NAME_OR_PATH", "/mnt/models/qwen")
    path = _write(
        tmp_path,
        "cfg.yaml",
        "checkpoint:\n  hf_model_path: ${oc.env:TT_MODEL_NAME_OR_PATH}\n",
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.checkpoint.hf_model_path == "/mnt/models/qwen"


def test_oc_env_interpolation_without_env_raises(tmp_path):
    from omegaconf.errors import InterpolationResolutionError

    path = _write(
        tmp_path,
        "cfg.yaml",
        "checkpoint:\n  hf_model_path: ${oc.env:TT_MODEL_NAME_OR_PATH}\n",
    )

    with pytest.raises(InterpolationResolutionError):
        _build_from_argv(["--tt_config", path])


# ---------------------------------------------------------------------------
# Four-layer precedence: TT args < env < file < CLI
# ---------------------------------------------------------------------------


def test_env_overrides_tt_defaults(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")

    cfg = build_from_args()

    assert cfg.parallel_plan.sp_size == 2
    assert cfg.activation_checkpointing.enabled is False


def test_file_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_MIXED_PRECISION", "bf16")
    path = _write(
        tmp_path,
        "cfg.yaml",
        "parallel_plan:\n  sp_size: 4\nmixed_precision: fp16\n",
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP16


def test_cli_overrides_file(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "parallel_plan:\n  sp_size: 4\nactivation_checkpointing:\n  enabled: true\n",
    )

    cfg = _build_from_argv(
        [
            "--tt_config",
            path,
            "--tt_sp_size",
            "8",
            "--tt_gradient_checkpointing",
            "false",
        ]
    )

    assert cfg.parallel_plan.sp_size == 8
    assert cfg.activation_checkpointing.enabled is False


def test_full_four_layer_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")
    monkeypatch.setenv("TORCH_TURBO_MIXED_PRECISION", "bf16")
    monkeypatch.setenv("TORCH_TURBO_OP_POLICY", "native")
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  sp_size: 4
activation_checkpointing:
  enabled: true
mixed_precision: fp16
ops:
  policy: custom
""",
    )

    cfg = _build_from_argv(
        [
            "--tt_config",
            path,
            "--tt_sp_size",
            "8",
            "--tt_gradient_checkpointing",
            "false",
            "--tt_mixed_precision",
            "fp8",
            "--tt_op_policy",
            "auto",
        ]
    )

    assert cfg.parallel_plan.sp_size == 8
    assert cfg.activation_checkpointing.enabled is False
    assert cfg.mixed_precision is MixedPrecisionPolicy.FP8
    assert cfg.ops.policy is OpPolicy.AUTO


# ---------------------------------------------------------------------------
# Partial nested overrides keep unset values from lower layers
# ---------------------------------------------------------------------------


def test_partial_nested_overrides_keep_unset_base_values(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
backend_config:
  cpu_offload: true
diffusion:
  cap_lambda: 2.0
checkpoint:
  dcp_path: /file/out
mtp:
  enabled: true
""",
    )

    cfg = build_from_args(
        TorchTurboArgs(
            model_path="/tt/model",
            output_dir="/tt/out",
            config_path=path,
        )
    )

    assert cfg.backend_config.cpu_offload is True
    assert cfg.backend_config.prefetch_strategy.value == "auto"
    assert cfg.diffusion.cap_lambda == 2.0
    assert cfg.diffusion.bd_mask_low == 0.3
    assert cfg.checkpoint.dcp_path == "/file/out"
    assert cfg.checkpoint.hf_model_path == "/tt/model"
    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.1


def test_file_checkpoint_fully_overrides_adapter_seeded_paths(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
checkpoint:
  hf_model_path: /file/model
  dcp_path: /file/out
  save_format: hf
""",
    )

    cfg = build_from_args(
        TorchTurboArgs(
            model_path="/tt/model",
            output_dir="/tt/out",
            config_path=path,
        )
    )

    assert cfg.checkpoint.hf_model_path == "/file/model"
    assert cfg.checkpoint.dcp_path == "/file/out"
    assert cfg.checkpoint.save_format == "hf"


def test_tt_mtp_values_kept_when_file_only_touches_enabled(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "mtp:\n  enabled: false\n")

    cfg = build_from_args(
        TorchTurboArgs(
            enable_mtp=True,
            mtp_loss_scaling_factor=0.5,
            config_path=path,
        )
    )

    assert cfg.mtp.enabled is False
    assert cfg.mtp.loss_scaling_factor == 0.5


def test_cli_mtp_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "mtp:\n  enabled: true\n")

    cfg = _build_from_argv(["--tt_config", path, "--tt_enable_mtp", "false"])

    assert cfg.mtp.enabled is False


def test_cli_op_policy_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "ops:\n  policy: custom\n")

    cfg = _build_from_argv(["--tt_config", path, "--tt_op_policy", "auto"])

    assert cfg.ops.policy is OpPolicy.AUTO


def test_cli_use_deterministic_algorithms_overrides_file(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "use_deterministic_algorithms: true\n")

    from_file = _build_from_argv(["--tt_config", path])
    from_cli = _build_from_argv(
        ["--tt_config", path, "--tt_use_deterministic_algorithms", "false"]
    )

    assert from_file.use_deterministic_algorithms is True
    assert from_cli.use_deterministic_algorithms is False


# ---------------------------------------------------------------------------
# Freeze policy layering
# ---------------------------------------------------------------------------


def test_tt_freeze_flags_kept_without_file_or_cli():
    cfg = build_from_args(TorchTurboArgs(freeze_language_model=True))

    assert cfg.freeze_policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_legacy_file_freeze_groups_do_not_override_tt_freeze_flags(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "freeze_policy:\n  parameter_groups: [vision_tower]\n",
    )

    cfg = build_from_args(
        TorchTurboArgs(freeze_language_model=True, config_path=path)
    )

    assert cfg.freeze_policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_cli_freeze_replaces_file_freeze_groups(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "freeze_policy:\n  parameter_groups: [vision_tower]\n",
    )

    cfg = _build_from_argv(["--tt_config", path, "--tt_freeze_language_model"])

    assert cfg.freeze_policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_explicit_false_freeze_clears_all_groups(tmp_path):
    tt, _ = parse_torchturbo_args(["--tt_freeze_vision_tower", "false"])

    cfg = build_from_args(tt)

    assert cfg.freeze_policy.parameter_groups == ()


# ---------------------------------------------------------------------------
# parallel_plan.plan parsing from files
# ---------------------------------------------------------------------------


def test_file_parallel_plan_parsed_into_fsdp_plan_specs(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  sp_size: 4
  plan:
    "model.layers.*":
      activation_checkpoint: true
    "model.visual":
      standalone_world_shard: true
""",
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert isinstance(cfg.parallel_plan.plan["model.layers.*"], FSDPPlanSpec)
    assert cfg.parallel_plan.plan["model.layers.*"].activation_checkpoint is True
    assert cfg.parallel_plan.plan["model.visual"].standalone_world_shard is True


def test_file_plan_without_sp_ep_keeps_tt_values(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.layers.*":
      activation_checkpoint: true
""",
    )

    cfg = build_from_args(
        TorchTurboArgs(sp_size=2, ep_size=3, config_path=path)
    )

    assert cfg.parallel_plan.sp_size == 2
    assert cfg.parallel_plan.ep_size == 3
    assert "model.layers.*" in cfg.parallel_plan.plan


# ---------------------------------------------------------------------------
# Model default parallel plan vs config file conflicts
# ---------------------------------------------------------------------------


def test_config_sp_ep_override_model_default_plan(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "parallel_plan:\n  sp_size: 2\n  ep_size: 8\n",
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.sp_size == 2
    assert plan.ep_size == 8


def test_config_disables_model_default_activation_checkpoint(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.language_model.layers.*":
      activation_checkpoint: false
""",
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.language_model.layers.*"].activation_checkpoint is False
    # Untouched FQNs must keep their model defaults (MoE replacement, kwargs).
    mlp = plan.plan["model.language_model.layers.*.mlp"]
    assert mlp.replace_module is not None
    assert mlp.module_kwargs


def test_config_disables_model_default_standalone_world_shard(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.visual":
      standalone_world_shard: false
""",
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.visual"].standalone_world_shard is False


def test_config_adds_plan_entry_missing_from_model_default(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.new_module":
      activation_checkpoint: true
""",
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.plan["model.new_module"].activation_checkpoint is True


def test_config_no_shard_conflicts_with_model_placement(tmp_path):
    """A file-level no_shard on the MoE experts collides with the model's
    default expert placement and must surface as an error, not silently win."""
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.language_model.layers.*.mlp.experts":
      no_shard: true
""",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_engine_plan(_build_from_argv(["--tt_config", path]))


def test_sample_style_config_patches_model_default_without_losing_defaults(
    tmp_path,
):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  sp_size: 2
  ep_size: 8
  plan:
    "model.language_model.layers.*":
      activation_checkpoint: true
    "model.visual":
      standalone_world_shard: true
    "model.language_model.embed_tokens":
      standalone_world_shard: true
    "lm_head":
      standalone_world_shard: true
    "mtp.layers.*":
      activation_checkpoint: true
""",
    )

    plan = _resolve_engine_plan(_build_from_argv(["--tt_config", path]))

    assert plan.sp_size == 2
    assert plan.ep_size == 8
    # Model defaults that the file does not touch must survive the merge.
    assert plan.plan["model.language_model.layers.*.mlp"].replace_module is not None
    assert plan.plan["model.language_model.layers.*.mlp.experts"].placement is not None
    assert plan.plan["mtp.layers.*.mlp"].replace_module is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_env_sp_size_raises(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_SP_SIZE", "abc")

    with pytest.raises(ValueError):
        build_from_args()


def test_invalid_env_backend_raises(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_BACKEND", "bogus")

    with pytest.raises(ValueError):
        build_from_args()


def test_null_mixed_precision_in_file_raises(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "mixed_precision: null\n")

    with pytest.raises(ValueError):
        _build_from_argv(["--tt_config", path])


def test_flat_op_policy_in_file_is_ignored(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "op_policy: native\n")

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.ops.policy is OpPolicy.AUTO


# ---------------------------------------------------------------------------
# Known gaps (documented with strict xfail so fixes flip them to XPASS)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="from_dict supports flat sp_size/ep_size, but the file overlay only "
    "applies EngineConfig fields; flat keys are silently dropped",
)
def test_flat_parallel_keys_in_file_are_applied(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "sp_size: 4\nep_size: 2\n")

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4
    assert cfg.parallel_plan.ep_size == 2


def test_flat_mtp_keys_in_file_are_applied(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "enable_mtp: true\n")

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.mtp.enabled is True


@pytest.mark.xfail(
    strict=True,
    reason="from_dict supports flat dcp_path, but the file overlay only "
    "applies EngineConfig fields; flat keys are silently dropped",
)
def test_flat_checkpoint_keys_in_file_are_applied(tmp_path):
    path = _write(tmp_path, "cfg.yaml", "dcp_path: /file/out\n")

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.checkpoint.dcp_path == "/file/out"


@pytest.mark.xfail(
    strict=True,
    reason="legacy parallel key is accepted by from_dict but the file overlay "
    "only looks up EngineConfig.parallel_plan",
)
def test_legacy_parallel_key_in_file_is_applied(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "parallel:\n  sp_size: 4\n",
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.parallel_plan.sp_size == 4


@pytest.mark.xfail(
    strict=True,
    reason="legacy fsdp key is accepted by from_dict but the file overlay only "
    "looks up EngineConfig.backend_config",
)
def test_legacy_fsdp_key_in_file_is_applied(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        "fsdp:\n  cpu_offload: true\n",
    )

    cfg = _build_from_argv(["--tt_config", path])

    assert cfg.backend_config.cpu_offload is True


@pytest.mark.xfail(
    strict=True,
    reason="unknown plan spec keys are silently ignored and produce an "
    "all-None FSDPPlanSpec instead of raising like unknown top-level keys",
)
def test_unknown_plan_spec_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "cfg.yaml",
        """\
parallel_plan:
  plan:
    "model.layers.*":
      activation_checkpointing: true
""",
    )

    with pytest.raises(ValueError, match="Unknown"):
        _build_from_argv(["--tt_config", path])
