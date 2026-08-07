"""Unit tests for FSDP2 backend plan wrapping semantics."""

import os
import sys
import types
from contextlib import nullcontext

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import torch.nn as nn
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec


class TaggedModule(nn.Module):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def get_norm(self):
        base_prefix = getattr(self, "base_model_prefix", "")
        base = getattr(self, base_prefix, None) if base_prefix else None
        norm = getattr(base, "norm", None) if isinstance(base, nn.Module) else None
        return norm if isinstance(norm, nn.Module) else None


class FinalNorm(TaggedModule):
    """Class name contains 'Norm' so _find_final_norm detects it as the tail norm."""

    pass


def module_tags(module):
    modules = module if isinstance(module, list) else [module]
    tags = [m.tag for m in modules]
    return tags[0] if len(tags) == 1 else tags


def make_backend(*, mtp_enabled=False):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = types.SimpleNamespace(
        mtp=types.SimpleNamespace(enabled=mtp_enabled),
    )
    return backend


class CheckpointBlock(TaggedModule):
    def __init__(self, tag: str) -> None:
        super().__init__(tag)
        self.inner = TaggedModule(tag + ".inner")
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x, use_cache=False, past_key_values=None):
        if use_cache or past_key_values is not None:
            raise AssertionError("cache kwargs should be cleared")
        return x * self.weight


def test_standalone_world_shard_skips_default_descendant_wrapping(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), mesh))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.language_model = TaggedModule("language_model")
    model.language_model.layers = nn.ModuleList(
        [TaggedModule("L0"), TaggedModule("L1")]
    )

    plan = FSDPParallelPlan(
        plan={"language_model": FSDPPlanSpec(standalone_world_shard=True)}
    )

    backend._shard_model(model, plan)

    assert calls == [
        ("language_model", "world"),
        ("root", "world"),
    ]


def test_no_shard_matches_parameter_names(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.hc_head_fn = nn.Parameter(torch.ones(1))
    model.layer = TaggedModule("layer")
    model.layer.attn_sink = nn.Parameter(torch.ones(1))
    model.layer.weight = nn.Parameter(torch.ones(1))

    plan = FSDPParallelPlan(
        plan={
            "hc_head_fn": FSDPPlanSpec(no_shard=True),
            "**.attn_sink": FSDPPlanSpec(no_shard=True),
        }
    )

    backend._shard_model(model, plan)

    assert len(calls) == 1
    assert calls[0][0] == "root"
    assert calls[0][1]["ignored_params"] == {
        model.hc_head_fn,
        model.layer.attn_sink,
    }


def test_no_shard_module_skips_descendant_wrapping(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.visual = TaggedModule("visual")
    model.visual.proj = nn.Linear(2, 2)
    model.visual.blocks = nn.ModuleList(
        [TaggedModule("V0"), TaggedModule("V1")]
    )
    model.visual.blocks[0].weight = nn.Parameter(torch.ones(1))
    model.visual.blocks[1].weight = nn.Parameter(torch.ones(1))
    model.layers = nn.ModuleList([TaggedModule("L0")])

    plan = FSDPParallelPlan(plan={"visual": FSDPPlanSpec(no_shard=True)})

    backend._shard_model(model, plan)

    assert [call[0] for call in calls] == ["L0", "root"]
    assert calls[-1][1]["ignored_params"] == set(model.visual.parameters())


def test_default_policy_wraps_module_list_children(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), mesh))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.layers = nn.ModuleList([TaggedModule("L0"), TaggedModule("L1")])

    backend._shard_model(model, FSDPParallelPlan())

    assert calls == [
        ("L1", "world"),
        ("L0", "world"),
        ("root", "world"),
    ]


def test_global_cpu_offload_applies_to_all_fsdp_units(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=True, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.layers = nn.ModuleList([TaggedModule("L0"), TaggedModule("L1")])

    backend._shard_model(model, FSDPParallelPlan())

    assert [call[0] for call in calls] == ["L1", "L0", "root"]
    assert all("offload_policy" in kwargs for _, kwargs in calls)


def test_carves_tail_unit_with_reshard_false_and_root(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    embed = TaggedModule("embed")
    lm_head = TaggedModule("lm_head")
    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = lm_head
    model.tok = embed
    model.get_input_embeddings = lambda: embed
    model.get_output_embeddings = lambda: lm_head

    backend._shard_model(model, FSDPParallelPlan())

    # Untied input embedding is carved into its own unit with global reshard.
    assert ("embed", True) in calls
    # final_norm + lm_head are grouped into a single tail unit, reshard=False.
    assert (["norm", "lm_head"], False) in calls
    # root also uses reshard_after_forward=False.
    assert ("root", False) in calls


def test_tail_unit_uses_input_embedding_when_tied(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    shared = nn.Parameter(torch.ones(2, 2))
    embed = TaggedModule("embed")
    embed.weight = shared
    lm_head = TaggedModule("lm_head")
    lm_head.weight = shared  # tied to input embedding
    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = lm_head
    model.tok = embed
    model.get_input_embeddings = lambda: embed
    model.get_output_embeddings = lambda: lm_head

    backend._shard_model(model, FSDPParallelPlan())

    # Tied: use the input embedding in the tail unit and skip lm_head.
    assert (["norm", "embed"], False) in calls
    assert not any(
        isinstance(tags, list) and "lm_head" in tags for tags, _ in calls
    )


def test_tail_unit_does_not_guess_lm_head_attr(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = TaggedModule("lm_head")

    backend._shard_model(model, FSDPParallelPlan())

    assert ("norm", False) in calls
    assert not any(
        isinstance(tags, list) and "lm_head" in tags for tags, _ in calls
    )


class VLMOuter(TaggedModule):
    """Outer VLM wrapper exposing a nested final norm via ``get_norm``, mirroring
    the qwen3.5 / kimi turbo wrappers that delegate to the inner language model.
    The norm is invisible to the direct-child class-name scan (it lives under
    ``model.language_model``), so discovery must come from ``get_norm``;
    otherwise it falls into the auto root unit and breaks CPU offload.
    """

    base_model_prefix = "model"

    def get_norm(self):
        return self.model.language_model.norm

    def get_output_embeddings(self):
        return self.lm_head


def test_vlm_nested_norm_via_get_norm_joins_tail_unit(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = VLMOuter("root")
    inner = TaggedModule("model")
    language_model = TaggedModule("language_model")
    language_model.norm = FinalNorm("norm")
    language_model.embed_tokens = TaggedModule("embed")
    inner.language_model = language_model
    inner.visual = TaggedModule("visual")
    model.model = inner
    lm_head = TaggedModule("lm_head")
    model.lm_head = lm_head
    model.get_input_embeddings = lambda: language_model.embed_tokens

    backend._shard_model(model, FSDPParallelPlan())

    # The get_norm-declared nested norm is paired with lm_head in a resident
    # tail unit instead of falling through to the auto root unit.
    assert (["norm", "lm_head"], False) in calls
    # Root unit is still created (resident) but must not own the norm.
    assert ("root", False) in calls
    # norm must be covered by the tail unit, not carved as a standalone unit.
    assert not any(tags == "norm" for tags, _ in calls)


def test_mtp_keeps_vlm_norm_separate_from_reused_lm_head(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend(mtp_enabled=True)
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = VLMOuter("root")
    model.model = TaggedModule("model")
    model.model.language_model = TaggedModule("language_model")
    model.model.language_model.norm = FinalNorm("norm")
    model.model.language_model.embed_tokens = TaggedModule("embed")
    model.model.visual = TaggedModule("visual")
    model.lm_head = TaggedModule("lm_head")
    model.get_input_embeddings = lambda: model.model.language_model.embed_tokens

    plan = FSDPParallelPlan(
        plan={"lm_head": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, plan)

    assert ("norm", False) in calls
    assert ("lm_head", False) in calls
    assert (["norm", "lm_head"], False) not in calls


def test_tail_unit_skips_modules_inside_explicit_parent(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.language_model = TaggedModule("language_model")
    model.language_model.norm = FinalNorm("norm")
    model.language_model.lm_head = TaggedModule("lm_head")
    model.get_output_embeddings = lambda: model.language_model.lm_head

    plan = FSDPParallelPlan(
        plan={"language_model": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, plan)

    assert ("language_model", True) in calls
    assert not any(
        isinstance(tags, list) and ("norm" in tags or "lm_head" in tags)
        for tags, _ in calls
    )


def test_explicit_output_embedding_merges_with_tail_norm(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(cpu_offload=False, pin_memory=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = TaggedModule("lm_head")
    model.get_output_embeddings = lambda: model.lm_head

    plan = FSDPParallelPlan(
        plan={"lm_head": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, plan)

    assert (["norm", "lm_head"], False) in calls
    assert ("norm", False) not in calls
    assert ("lm_head", False) not in calls


def test_prepare_model_applies_activation_checkpointing_before_sharding():
    order = []
    model = TaggedModule("root")

    backend = make_backend()
    backend.prefetch_manager = types.SimpleNamespace(
        setup=lambda model: order.append("prefetch")
    )
    backend.get_ep_info = lambda: {"ep_size": 1}
    backend._patch_transformers = lambda: order.append("patch_transformers")
    backend._init_mesh = lambda plan: order.append("init_mesh")
    backend._replace_modules = lambda model, plan: order.append("replace") or model
    backend._apply_activation_checkpointing = (
        lambda model, plan: order.append("activation_checkpointing") or model
    )
    backend._shard_model = lambda model, plan: order.append("shard") or model

    assert backend.prepare_model(model, FSDPParallelPlan()) is model

    assert order == [
        "init_mesh",
        "replace",
        "activation_checkpointing",
        "shard",
        "prefetch",
        "patch_transformers",
    ]


def test_activation_checkpointing_patches_plan_targets(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda: nullcontext()
    )

    model = TaggedModule("root")
    model.config = types.SimpleNamespace(use_cache=True)
    model.layers = nn.ModuleList([CheckpointBlock("L0"), CheckpointBlock("L1")])
    model.layers[0].config = types.SimpleNamespace(use_cache=True)
    model.layers[0].gradient_checkpointing = True
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )
    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    backend._apply_activation_checkpointing(model, plan)

    assert hasattr(model.layers[0], "_tt_original_forward")
    assert hasattr(model.layers[1], "_tt_original_forward")
    assert not hasattr(model.layers[0], "_checkpoint_wrapped_module")
    assert model.config.use_cache is False
    assert model.layers[0].config.use_cache is False
    assert model.layers[0].gradient_checkpointing is False

    x = torch.ones(1, requires_grad=False)
    out = model.layers[0](x, use_cache=True, past_key_values=object())

    assert torch.equal(out, torch.ones(1))
    assert x.requires_grad is False
    assert checkpoint_calls[-1]["use_reentrant"] is False
    assert "context_fn" in checkpoint_calls[-1]


def test_activation_checkpointing_eval_uses_original_forward(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda: nullcontext()
    )

    model = TaggedModule("root")
    model.layers = nn.ModuleList([CheckpointBlock("L0")])
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )
    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    backend._apply_activation_checkpointing(model, plan)
    model.layers[0].eval()

    x = torch.ones(1, requires_grad=False)
    out = model.layers[0](x)

    assert torch.equal(out, torch.ones(1))
    assert checkpoint_calls == []
    assert x.requires_grad is False


def test_activation_checkpointing_disabled_warns_and_skips_plan_targets(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig
    from torchturbo.engine.backends.fsdp2 import backend as backend_module

    warnings = []
    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=False)
    )
    backend.rank = 0
    monkeypatch.setattr(
        backend_module,
        "logger",
        types.SimpleNamespace(
            warning_rank0=lambda *args, **kwargs: warnings.append(args),
        ),
    )

    model = TaggedModule("root")
    model.layers = nn.ModuleList([CheckpointBlock("L0")])
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )

    assert backend._apply_activation_checkpointing(model, plan) is model

    assert len(warnings) == 1
    assert "skipping activation checkpointing" in warnings[0][0]
    assert not hasattr(model.layers[0], "_tt_original_forward")
