"""Native TP/ETP on FSDP2: topology, plan admission, and single-device equivalence."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.parallel_context import (
    ParallelContext,
    validate_parallel_degrees,
)
from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.engine.optimizers import MuonPlan, OptimizerClassPlan
from torchturbo.engine.tensor_parallel import (
    GroupedMLPParallel,
    PrepareModuleMetadata,
    SharedHeadNormParallel,
    register_tensor_parallel_cache_guard,
    resolve_tensor_parallel_plan,
)
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
    TensorParallelPlanSpec,
)
from torchturbo.runtime.cpu import CpuRuntime


class DenseModel(nn.Module):
    """Column/row-parallel projections around a head-wise norm, plus a replicated head."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.config = SimpleNamespace()
        self.up = nn.Linear(8, 16, bias=False)
        self.norm = nn.RMSNorm(4)
        self.down = nn.Linear(16, 8, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.RMSNorm(8)
        self.head = nn.Linear(8, 5, bias=False)

    def forward(self, x):
        heads = self.norm(self.up(x).unflatten(-1, (-1, 4)))
        hidden = x + self.down(F.silu(heads.flatten(-2)))
        return self.head(self.final_norm(self.dropout(hidden)))

    def get_input_embeddings(self):
        raise NotImplementedError

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.final_norm


def _dense_tp_plan(**degrees) -> FSDPParallelPlan:
    return FSDPParallelPlan(
        tp_size=2,
        tensor_parallel_plan=TensorParallelPlanSpec(plan={
            "up": ColwiseParallel(),
            "norm": SharedHeadNormParallel(head_axis=2),
            "down": RowwiseParallel(input_layouts=Shard(-1)),
        }),
        **degrees,
    )


def test_parallel_degrees_must_divide_world():
    validate_parallel_degrees(8, sp_size=2, ep_size=2, tp_size=2, etp_size=2)
    with pytest.raises(ValueError, match=r"tp_size \(3\)"):
        validate_parallel_degrees(4, sp_size=1, ep_size=1, tp_size=3, etp_size=1)
    with pytest.raises(ValueError, match=r"sp_size \* tp_size \(8\)"):
        validate_parallel_degrees(4, sp_size=4, ep_size=1, tp_size=2, etp_size=1)


def test_plan_resolves_each_target_once_with_fresh_styles():
    model = nn.Module()
    model.layers = nn.ModuleList([DenseModel(), DenseModel()])
    fqn_map = build_model_fqn_map(model)
    style = ColwiseParallel()
    resolved = resolve_tensor_parallel_plan({"layers.*.up": style}, fqn_map)
    assert set(resolved) == {"layers.0.up", "layers.1.up"}
    assert resolved["layers.0.up"] is not style
    assert resolved["layers.0.up"] is not resolved["layers.1.up"]
    with pytest.raises(ValueError, match="matched no model modules"):
        resolve_tensor_parallel_plan({"layers.*.missing": ColwiseParallel()}, fqn_map)
    with pytest.raises(ValueError, match="matches multiple styles"):
        resolve_tensor_parallel_plan(
            {"layers.0.up": ColwiseParallel(), "layers.*.up": ColwiseParallel()}, fqn_map
        )
    with pytest.raises(TypeError, match="ParallelStyle"):
        resolve_tensor_parallel_plan({"layers.0.up": object()}, fqn_map)


def test_metadata_style_validates_before_dividing_counts():
    module = nn.Module()
    module.num_heads, module.head_dim = 4, 3
    style = PrepareModuleMetadata(divide=("num_heads",))
    style._apply(module, SimpleNamespace(size=lambda: 2))
    assert (module.num_heads, module.head_dim) == (2, 3)
    with pytest.raises(ValueError, match="divisible by mesh size"):
        style.validate(module, 4)
    with pytest.raises(ValueError, match="missing"):
        PrepareModuleMetadata(divide=("num_kv_heads",)).validate(module, 2)


def test_cache_guard_rejects_cached_calls():
    class Decoder(nn.Module):
        def forward(self, x, past_key_values=None, use_cache=None):
            return x

    decoder = Decoder()
    register_tensor_parallel_cache_guard(decoder)
    assert decoder(torch.ones(1)).item() == 1
    for kwargs in ({"use_cache": True}, {"past_key_values": object()}):
        with pytest.raises(ValueError, match="cache"):
            decoder(torch.ones(1), **kwargs)


@pytest.mark.parametrize("case,message", [
    ("peft", "PEFT"),
    ("muon", "Muon"),
    ("cpu_offload", "offload"),
    ("mtp", "MTP"),
    ("dense_with_ep", "Dense TP requires ep_size=etp_size=1"),
    ("unaligned_etp", "aligned tp_size=etp_size>1"),
    ("empty_plan", "nonempty tensor_parallel_plan"),
    ("dropout", "dropout=0"),
    ("no_shard", "no_shard"),
])
def test_admission_rejects_unsupported_features(monkeypatch, case, message):
    config = EngineConfig()
    backend = object.__new__(FSDP2Backend)
    backend.config, backend.fsdp_config = config, config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
    model, plan, peft_config = DenseModel(dropout=.1 if case == "dropout" else 0.), _dense_tp_plan(), None
    if case == "peft":
        peft_config = object()
    elif case == "muon":
        config.optimizer.name = MuonPlan.name
    elif case == "cpu_offload":
        config.backend_config.cpu_offload.enabled = True
    elif case == "mtp":
        config.mtp.enabled = True
    elif case == "dense_with_ep":
        plan.ep_size = 2
    elif case == "unaligned_etp":
        plan.etp_size = 1
        plan.tensor_parallel_plan.plan["up"] = GroupedMLPParallel()
    elif case == "empty_plan":
        plan.tensor_parallel_plan.plan.clear()
    elif case == "no_shard":
        plan.plan["head"] = FSDPPlanSpec(no_shard=True)
    with pytest.raises(ValueError, match=message):
        backend._validate_tensor_parallel_features(model, plan, peft_config)


def _init(rank, world, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))


def _ranks(mesh):
    return dist.get_process_group_ranks(mesh.get_group())


def _topology_worker(rank, path):
    _init(rank, 4, path)
    try:
        context = ParallelContext("cpu", sp_size=2, ep_size=2, tp_size=2, etp_size=2)
        # TP and ETP lanes are consecutive ranks; SP, EP and dense FSDP stride over them.
        tp_block = [rank - rank % 2, rank - rank % 2 + 1]
        lane = [rank % 2, rank % 2 + 2]
        assert _ranks(context.tp_mesh) == _ranks(context.etp_mesh) == tp_block
        assert _ranks(context.sp_mesh) == _ranks(context.ep_mesh) == lane
        assert _ranks(context.dense_fsdp_mesh) == lane
        assert context.expert_fsdp_mesh.size() == 1
        assert context.expert_checkpoint_owner_ranks == tp_block
        # SP and TP replicas consume the same samples.
        assert context.get_dp_info()["dp_size"] == 1
    finally:
        dist.destroy_process_group()


def test_parallel_context_groups(tmp_path):
    mp.spawn(_topology_worker, args=(str(tmp_path / "store"),), nprocs=4, join=True)


def _assert_grads_match(model, reference, slices=None):
    slices = slices or {}
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name).grad
        expected = slices.get(name, lambda value: value)(expected)
        actual = parameter.grad.full_tensor().reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


def _assert_params_match(model, reference, slices=None):
    slices = slices or {}
    for name, parameter in model.named_parameters():
        expected = slices.get(name, lambda value: value)(reference.get_parameter(name))
        actual = parameter.full_tensor().reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


def _dense_worker(rank, path, directory):
    from torchturbo.engine.checkpoint.metadata import validate_parallel_layout

    _init(rank, 4, path)
    try:
        torch.manual_seed(3)
        model = DenseModel()
        reference = copy.deepcopy(model)
        backend = FSDP2Backend(EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32), CpuRuntime())
        backend.prepare_model(model, _dense_tp_plan())
        context = backend._parallel_context
        assert context.tp_size == 2 and context.dense_fsdp_mesh.size() == 2
        optimizer = backend.prepare_optimizer(OptimizerClassPlan(model, torch.optim.AdamW, {"lr": .05, "weight_decay": 0.}))
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.05, weight_decay=0.)
        inputs = [torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(d)) for d in range(2)]
        data_rank = context.dense_fsdp_mesh.get_local_rank()

        def step(save=False):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            output = model(inputs[data_rank])
            torch.testing.assert_close(output, reference(inputs[data_rank]), rtol=1e-5, atol=1e-6)
            output.square().mean().backward()
            # FSDP averages gradients over the data ranks; TP lanes share their samples.
            (sum(reference(x).square().mean() for x in inputs) / len(inputs)).backward()
            _assert_grads_match(model, reference)
            norm = backend.clip_grad_norm_(list(model.parameters()), max_norm=.05)
            torch.testing.assert_close(norm, nn.utils.clip_grad_norm_(reference.parameters(), .05))
            if save:
                manager.save_native(model, str(directory / "model"))
                manager.save_optimizer(model, optimizer, str(directory / "optimizer"))
            optimizer.step()
            reference_optimizer.step()
            _assert_params_match(model, reference)

        manager = backend.create_checkpoint_manager()
        layout = manager.checkpoint_layout(model)
        assert layout["topology"] == dict(world_size=4, sp_size=1, tp_size=2, ep_size=1, etp_size=1)
        validate_parallel_layout({"schema_version": 2, "parallel_layout": layout}, layout)

        step()
        # Restore the pre-step state of a checkpoint taken mid-step, then repeat the step.
        snapshot = copy.deepcopy(reference)
        snapshot_optimizer = torch.optim.AdamW(snapshot.parameters(), lr=.05, weight_decay=0.)
        # load_state_dict keeps same-device state tensors, so copy them first.
        snapshot_optimizer.load_state_dict(copy.deepcopy(reference_optimizer.state_dict()))
        step(save=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        manager.load_native(model, str(directory / "model"))
        manager.load_optimizer(model, optimizer, str(directory / "optimizer"))
        reference, reference_optimizer = snapshot, snapshot_optimizer
        step()
    finally:
        dist.destroy_process_group()


def test_dense_tp_fsdp_matches_single_device_training(tmp_path):
    mp.spawn(_dense_worker, args=(str(tmp_path / "store"), tmp_path), nprocs=4, join=True)


class MoEModel(nn.Module):
    def __init__(self, ep_group, ep_size):
        from torchturbo.models.modules.moe.experts import GroupedMLP
        from torchturbo.models.modules.moe.moe_config import MoEConfig
        from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
        from torchturbo.models.modules.moe.module import ModuleSpec

        super().__init__()
        self.config = SimpleNamespace()
        config = MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=4, num_moe_experts=4, moe_router_topk=2,
            gated_linear_unit=True, activation_func=F.silu, expert_model_parallel_size=ep_size,
            moe_router_score_function="sigmoid", moe_token_dispatcher_type="alltoall",
        )
        self.moe = MoELayer(config, ep_group,
                            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
        self.norm = nn.RMSNorm(4)
        self.head = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.head(self.norm(self.moe(x)[0]))

    def get_input_embeddings(self):
        raise NotImplementedError

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm


class DenseMoE(nn.Module):
    """Literal top-2 sigmoid routing with routing weights applied inside experts."""

    def __init__(self, model, w1, w2):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = nn.Module()
        self.moe.router.weight = nn.Parameter(model.moe.router.weight.detach().clone())
        self.moe.experts = nn.Module()
        self.moe.experts.weight1 = nn.Parameter(w1.clone())
        self.moe.experts.weight2 = nn.Parameter(w2.clone())
        self.norm, self.head = copy.deepcopy(model.norm), copy.deepcopy(model.head)

    def forward(self, x):
        tokens = x.reshape(-1, 4)
        scores = (tokens @ self.moe.router.weight.T).sigmoid()
        ids = scores.topk(2, dim=-1).indices
        selected = scores.gather(1, ids)
        probs = selected / (selected.sum(-1, keepdim=True) + 1e-20)
        output = torch.zeros_like(tokens)
        for slot in range(2):
            expert = ids[:, slot]
            gate, up = torch.einsum("th,thf->tf", tokens, self.moe.experts.weight1[expert]).chunk(2, -1)
            weighted = F.silu(gate) * up * probs[:, slot:slot + 1]
            output = output + torch.einsum("tf,tfh->th", weighted, self.moe.experts.weight2[expert])
        return self.head(self.norm(output.reshape_as(x)))


def _routing_consensus(context):
    from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity

    lane = context.etp_mesh.get_local_rank()
    # Lane 1 alone would select experts {0, 2}; it must follow lane zero's {0, 1}.
    logits = torch.tensor([[.30, .29, .10, 0.]]) + lane * torch.tensor([0., -.25, 0., 0.])
    routing = topk_softmax_with_capacity(
        logits, topk=2, score_function="sigmoid", routing_group=context.etp_mesh.get_group()
    )
    assert routing.routing_map.tolist() == [[True, True, False, False]]
    selected = logits.sigmoid()[:, :2]
    torch.testing.assert_close(routing.probs[:, :2], selected / selected.sum(-1, keepdim=True))


def _etp_worker(rank, path):
    from torchturbo.ops import get_or_create_ops

    _init(rank, 4, path)
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        context = ParallelContext("cpu", sp_size=1, ep_size=2, tp_size=2, etp_size=2)
        _routing_consensus(context)
        torch.manual_seed(5)
        model = MoEModel(context.ep_mesh.get_group(), ep_size=2)
        w1 = torch.randn(4, 4, 8) * .4
        w2 = torch.randn(4, 4, 4) * .4
        owner = context.ep_mesh.get_local_rank()
        with torch.no_grad():
            model.moe.router.weight.copy_(torch.randn(4, 4))
            model.moe.experts.weight1.copy_(w1[owner * 2:owner * 2 + 2].reshape(8, 8))
            model.moe.experts.weight2.copy_(w2[owner * 2:owner * 2 + 2].reshape(8, 4))
        reference = DenseMoE(model, w1, w2)
        backend = FSDP2Backend(EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32), CpuRuntime())
        backend._parallel_context = context
        backend._world_mesh, backend._mesh = context.dense_fsdp_mesh, context.expert_mesh
        backend.prepare_model(model, FSDPParallelPlan(
            ep_size=2, tp_size=2, etp_size=2,
            plan={"moe.experts": FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), "edp"))},
            tensor_parallel_plan=TensorParallelPlanSpec(plan={"moe.experts": GroupedMLPParallel()}),
        ))
        assert model.moe.router.routing_group is context.etp_mesh.get_group()
        assert backend._parameter_ownership.is_expert(model.moe.experts.weight1)
        optimizer = backend.prepare_optimizer(OptimizerClassPlan(model, torch.optim.AdamW, {"lr": .05, "weight_decay": 0.}))
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.05, weight_decay=0.)
        # This EP rank owns two consecutive experts in the logical gate/up layout.
        slices = {
            "moe.experts.weight1": lambda value: value[owner * 2:owner * 2 + 2].reshape(8, 2, 4),
            "moe.experts.weight2": lambda value: value[owner * 2:owner * 2 + 2],
        }
        inputs = [torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(d)) for d in range(2)]
        data_rank = context.dense_fsdp_mesh.get_local_rank()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            output = model(inputs[data_rank])
            torch.testing.assert_close(output, reference(inputs[data_rank]), rtol=1e-5, atol=1e-6)
            output.square().mean().backward()
            (sum(reference(x).square().mean() for x in inputs) / len(inputs)).backward()
            # Router gradients flow through in-expert routing weights on every ETP lane.
            _assert_grads_match(model, reference, slices)
            norm = backend.clip_grad_norm_(list(model.parameters()), max_norm=.05)
            torch.testing.assert_close(norm, nn.utils.clip_grad_norm_(reference.parameters(), .05))
            optimizer.step()
            reference_optimizer.step()
            _assert_params_match(model, reference, slices)
    finally:
        dist.destroy_process_group()


def test_etp_moe_matches_single_device_training(tmp_path):
    mp.spawn(_etp_worker, args=(str(tmp_path / "store"),), nprocs=4, join=True)
