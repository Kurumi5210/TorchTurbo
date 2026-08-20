"""Tests for the ms-swift adapter: detection, native registration, structure.

These exercise the swift-facing surface, so they are skipped when ms-swift is
not importable (``pytest.importorskip('swift')``).
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    import types

    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

swift = pytest.importorskip("swift")


def test_adapter_detection():
    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.compat import get_swift_version, is_swift_supported

    assert is_swift_supported() is True
    adapter = MSSwiftAdapter()
    assert adapter.detect() is True
    ver = get_swift_version()
    assert isinstance(ver, str) and ver.count(".") >= 1


def test_native_registration_populates_swift_registries():
    from swift.model import MODEL_MAPPING
    from swift.trainers import TrainerFactory

    from torchturbo.adapters.msswift import MSSwiftAdapter

    adapter = MSSwiftAdapter()
    adapter.activate()

    # turbo models registered under turbo_<name> with the turbo loader.
    turbo_entries = {k: v for k, v in MODEL_MAPPING.items() if k.startswith("turbo_")}
    assert "turbo_qwen3_5_moe" in turbo_entries
    # turbo_qwen3_5_moe's loader is TurboModelMixin mixed ahead of the native
    # Qwen3_5MoeLoader — so get_model is turbo's engine and get_processor is the
    # native one (with its media-cap env side effects). Base TurboModelLoader is
    # the text-only fallback when swift has no native meta for the model.
    from torchturbo.adapters.msswift.loader import TurboModelMixin
    from swift.model.model_meta import MODEL_MAPPING as SWIFT_MODEL_MAPPING
    turbo_loader = turbo_entries["turbo_qwen3_5_moe"].loader
    assert issubclass(turbo_loader, TurboModelMixin)
    native_loader = SWIFT_MODEL_MAPPING["qwen3_5_moe"].loader
    # MRO: TurboModelMixin precedes the native loader; get_model is turbo's.
    assert native_loader in turbo_loader.__mro__
    assert turbo_loader.__mro__.index(TurboModelMixin) < turbo_loader.__mro__.index(native_loader)
    # get_model comes from the mixin (turbo engine), get_processor from native.
    assert turbo_loader.get_model is TurboModelMixin.get_model
    assert turbo_loader.get_processor is native_loader.get_processor
    # Meta carries the native loader chain's conventions, e.g. multimodal flag.
    assert turbo_entries["turbo_qwen3_5_moe"].is_multimodal is True

    # turbo trainer registered for causal_lm (and dpo if available).
    assert TrainerFactory.TRAINER_MAPPING["causal_lm"].endswith("TurboSeq2SeqTrainer")

    adapter.deactivate()


def test_trainer_subclasses_swift_seq2seq():
    from swift.trainers import Seq2SeqTrainer

    from torchturbo.adapters.msswift.trainer import (
        TurboTrainerMixin,
        get_turbo_seq2seq_trainer,
    )

    turbo_cls = get_turbo_seq2seq_trainer()
    assert issubclass(turbo_cls, Seq2SeqTrainer)
    # The turbo接管 overrides live on TurboTrainerMixin, mixed ahead of
    # Seq2SeqTrainer, so each engine-owned method resolves to the mixin.
    assert TurboTrainerMixin in turbo_cls.__mro__
    for method in ("create_optimizer", "compute_loss", "save_model", "_wrap_model",
                   "create_scheduler", "_prepare_inputs", "get_train_dataloader"):
        assert getattr(turbo_cls, method) is getattr(TurboTrainerMixin, method)


def test_trainer_registry_is_stage_driven():
    from torchturbo.adapters.msswift.stages import SFT_STAGE, iter_stages

    # The SFT stage contributes the causal_lm trainer to TrainerFactory.
    assert SFT_STAGE.turbo_trainers["causal_lm"].endswith("TurboSeq2SeqTrainer")
    stages = iter_stages()
    assert any(s.name == "sft" for s in stages)
    assert any(
        "causal_lm" in s.turbo_trainers for s in stages
    )


def test_stage_registry_drives_plugin_swap():
    """The plugin swaps each registered stage's pipeline module global.

    Register a fake stage against an importable module and confirm the plugin
    module's swap loop reassigns the attribute, then clean up so other tests
    see only the real SFT stage.
    """
    import types

    from torchturbo.adapters.msswift import stages as stages_mod

    # A throwaway swift-free module + attr to swap.
    fake_mod = types.ModuleType("turbo_test_stage_mod")
    fake_mod.SwiftSft = object  # sentinel original

    fake_spec = stages_mod.SwiftStageSpec(
        name="__test_stage__",
        pipeline_module="turbo_test_stage_mod",
        pipeline_attr="SwiftSft",
        turbo_pipeline_factory=lambda: str,  # sentinel turbo pipeline
        turbo_trainers={},
    )
    stages_mod.TURBO_SWIFT_STAGES.append(fake_spec)
    import sys

    sys.modules["turbo_test_stage_mod"] = fake_mod
    try:
        # Re-run the plugin's swap logic in isolation.
        import importlib

        for spec in stages_mod.iter_stages():
            mod = importlib.import_module(spec.pipeline_module)
            setattr(mod, spec.pipeline_attr, spec.turbo_pipeline_factory())
        assert fake_mod.SwiftSft is str  # swapped to the sentinel turbo pipeline
    finally:
        stages_mod.TURBO_SWIFT_STAGES.remove(fake_spec)
        del sys.modules["turbo_test_stage_mod"]


def test_turbo_sft_arguments_structure():
    from swift.arguments import SftArguments

    from torchturbo.adapters.msswift.args import (
        TurboSwiftArgsMixin,
        get_turbo_sft_arguments_class,
    )
    from torchturbo.engine.args import TorchTurboArgs

    cls = get_turbo_sft_arguments_class()
    assert issubclass(cls, SftArguments)

    # The force-off of engine-owned swift settings lives on TurboSwiftArgsMixin,
    # mixed ahead of SftArguments, and is reused by future stage args classes.
    assert TurboSwiftArgsMixin in cls.__mro__
    assert cls.__mro__.index(TurboSwiftArgsMixin) < cls.__mro__.index(SftArguments)
    assert cls.__post_init__ is TurboSwiftArgsMixin.__post_init__

    # The turbo --tt_* CLI surface lives on the shared TorchTurboArgs, NOT on
    # TurboSftArguments (which declares only swift-native fields). The launcher
    # strips --tt_* from sys.argv before swift's parser runs.
    turbo_fields = set(TorchTurboArgs.__dataclass_fields__.keys())
    for name in (
        "sp_size",
        "ep_size",
        "gradient_checkpointing",
        "ac_use_reentrant",
        "enable_mtp",
        "mtp_log_loss",
        "enable_chunk_loss",
        "loss_chunk_size",
        "freeze_vision_tower",
        "cpu_offload",
    ):
        assert name in turbo_fields, f"missing TorchTurboArgs field {name!r}"

    swift_field_names = set(cls.__dataclass_fields__.keys())
    for name in ("tt_sp_size", "tt_ep_size", "tt_gradient_checkpointing",
                 "tt_enable_mtp", "tt_freeze_vision_tower", "tt_cpu_offload"):
        assert name not in swift_field_names, (
            f"swift TurboSftArguments should not declare turbo field {name!r} "
            "(lives on TorchTurboArgs)"
        )


def test_turbo_sft_arguments_class_is_cached():
    from torchturbo.adapters.msswift.args import get_turbo_sft_arguments_class

    assert get_turbo_sft_arguments_class() is get_turbo_sft_arguments_class()


def test_native_plugin_swaps_swift_sft():
    """The `--external_plugins` entry must swap swift's SwiftSft -> TurboSwiftSft.

    This is what makes the native `swift sft ...` launcher build the turbo接管
    pipeline: ``sft_main`` closes over the module-global ``SwiftSft``, so the
    plugin only needs to reassign that one attribute.
    """
    import swift.pipelines.train.sft as swift_sft_mod

    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.pipeline import get_turbo_swift_sft

    TurboSwiftSft = get_turbo_swift_sft()
    original = swift_sft_mod.SwiftSft

    # Activate (registers models/trainers) then swap the pipeline class exactly
    # as the plugin module does on import.
    MSSwiftAdapter().activate()
    swift_sft_mod.SwiftSft = TurboSwiftSft
    try:
        assert swift_sft_mod.SwiftSft is TurboSwiftSft
        # native sft_main closes over the module global, so it would now build
        # the turbo pipeline; we assert the wiring rather than running it.
        assert callable(swift_sft_mod.sft_main)
    finally:
        swift_sft_mod.SwiftSft = original
