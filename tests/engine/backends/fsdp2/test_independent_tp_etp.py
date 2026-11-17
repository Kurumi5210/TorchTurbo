"""Independent dense and expert TP against literal FP32 MoE training."""

import copy
import json
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from transformers import PretrainedConfig

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from torchturbo.engine.config import (
    ActivationCheckpointingConfig,
    EngineConfig,
    MixedPrecisionPolicy,
)
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.engine.tensor_parallel import GroupedMLPParallel
from torchturbo.models.modules.attention.ulysses.ulysses import SequenceParallelInputSlicingSpec
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
    TensorParallelPlanSpec,
)
from torchturbo.runtime.cpu import CpuRuntime

from .test_tensor_parallel import _init


class _SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4, 12, bias=False)
        self.down = nn.Linear(12, 4, bias=False)

    def forward(self, hidden):
        return self.down(F.silu(self.up(hidden)))


class _MoEModel(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=4, moe_intermediate_size=12)
        self.all_tied_weights_keys = {}
        config = MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=12, num_moe_experts=4,
            moe_router_topk=2, gated_linear_unit=True, activation_func=F.silu,
            expert_model_parallel_size=context.ep_size,
            moe_router_score_function="sigmoid", moe_router_dtype="fp32",
            moe_aux_loss_coeff=0.03, moe_z_loss_coeff=0.02,
            moe_router_enable_expert_bias=True,
        )
        self.moe = MoELayer(
            config, context.ep_mesh.get_group(),
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            sp_group=context.sp_mesh.get_group() if context.sp_size > 1 else None,
        )
        self.moe.shared_expert = _SharedMLP()
        self.moe.shared_expert_gate = nn.Linear(4, 1, bias=False)
        self.moe.shared_experts = nn.Linear(4, 4, bias=False)
        self.norm = nn.RMSNorm(4)
        self.head = nn.Linear(4, 3, bias=False)

    def forward(self, hidden):
        output, router_logits = self.moe(hidden)
        return self.head(self.norm(output)), router_logits

    def get_input_embeddings(self):
        raise NotImplementedError

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def get_sequence_parallel_input_slicing_spec(self):
        return SequenceParallelInputSlicingSpec(
            self, _MoEModel.forward, sliced_inputs=(("hidden", 1),), padded_inputs=(),
        )


class _DenseReference(nn.Module):
    """Literal top-2 experts with no distributed or Turbo routing operators."""

    def __init__(self, model, weight1, weight2):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = nn.Module()
        self.moe.router.weight = nn.Parameter(model.moe.router.weight.detach().clone())
        self.moe.experts = nn.Module()
        self.moe.experts.weight1 = nn.Parameter(weight1.clone())
        self.moe.experts.weight2 = nn.Parameter(weight2.clone())
        self.moe.shared_expert = copy.deepcopy(model.moe.shared_expert)
        self.moe.shared_expert_gate = copy.deepcopy(model.moe.shared_expert_gate)
        self.moe.shared_experts = copy.deepcopy(model.moe.shared_experts)
        self.norm = copy.deepcopy(model.norm)
        self.head = copy.deepcopy(model.head)

    def forward(self, hidden, padding):
        tokens = hidden.reshape(-1, 4)
        logits = tokens @ self.moe.router.weight.T
        scores = logits.sigmoid()
        indices = scores.topk(2, dim=-1).indices
        selected = scores.gather(1, indices)
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
        probs = torch.zeros_like(scores).scatter(1, indices, selected)
        probs.retain_grad()
        output = torch.zeros_like(tokens)
        for slot in range(2):
            experts = indices[:, slot]
            gate, up = torch.einsum(
                "th,thf->tf", tokens, self.moe.experts.weight1[experts]
            ).chunk(2, dim=-1)
            weighted = F.silu(gate) * up * probs.gather(1, indices[:, slot:slot + 1])
            output = output + torch.einsum(
                "tf,tfh->th", weighted, self.moe.experts.weight2[experts]
            )
        shared = self.moe.shared_expert(hidden)
        shared = shared * self.moe.shared_expert_gate(hidden).sigmoid()
        output = output.reshape_as(hidden) + shared + self.moe.shared_experts(hidden)

        valid = ~padding.reshape(-1)
        count = valid.sum().clamp_min(1)
        assignments = torch.zeros_like(scores).scatter(1, indices, 1)
        counts = (assignments * valid[:, None]).sum(0)
        normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        probability_sum = (normalized * valid[:, None]).sum(0)
        aux = (probability_sum * counts).sum() * 4 / (2 * count.square())
        z = (logits.logsumexp(-1).square() * valid).sum() / count
        return self.head(self.norm(output)), logits, probs, aux, z, counts


def _plan(tp_size, etp_size, ep_size, checkpoint, sp_size):
    return FSDPParallelPlan(
        tp_size=tp_size, etp_size=etp_size, ep_size=ep_size, sp_size=sp_size,
        plan={
            "moe": FSDPPlanSpec(activation_checkpoint=checkpoint is not None),
            "moe.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
        },
        tensor_parallel_plan=TensorParallelPlanSpec(plan={
            "moe.experts": GroupedMLPParallel(),
            "moe.shared_expert.up": ColwiseParallel(),
            "moe.shared_expert.down": RowwiseParallel(input_layouts=Shard(-1)),
        }),
    )


def _full_tensor(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _expert_slice(name, value, context):
    if name.startswith("moe.experts."):
        num_local_experts = 4 // context.ep_size
        start = context.ep_mesh.get_local_rank() * num_local_experts
        return value[start:start + num_local_experts]
    return value


def _assert_parameters(model, reference, context, *, gradients=False):
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name)
        if gradients:
            parameter, expected = parameter.grad, expected.grad
            assert parameter is not None and expected is not None, name
        expected = _expert_slice(name, expected, context)
        actual = _full_tensor(parameter).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, msg=name)


def _assert_optimizer(optimizer, model, reference_optimizer, reference, context):
    for name, parameter in model.named_parameters():
        actual = optimizer.state[parameter]
        expected = reference_optimizer.state[reference.get_parameter(name)]
        assert actual.keys() == expected.keys(), name
        for field, value in actual.items():
            ref_value = expected[field]
            if field != "step":
                ref_value = _expert_slice(name, ref_value, context)
            value = _full_tensor(value).reshape_as(ref_value)
            torch.testing.assert_close(
                value, ref_value, rtol=2e-5, atol=2e-6, msg=f"{name}.{field}",
            )


def _optimizer(backend, model):
    return backend.prepare_optimizer(OptimizerClassPlan(
        model, torch.optim.AdamW, {"lr": 0.015, "weight_decay": 0.0},
    ))


def _hf_state(reference):
    state = {}
    for name, parameter in reference.named_parameters():
        if name == "moe.experts.weight1":
            for index, weight in enumerate(parameter):
                gate, up = weight.chunk(2, dim=-1)
                state[f"moe.experts.{index}.gate_proj.weight"] = gate.T.detach().contiguous()
                state[f"moe.experts.{index}.up_proj.weight"] = up.T.detach().contiguous()
        elif name == "moe.experts.weight2":
            for index, weight in enumerate(parameter):
                state[f"moe.experts.{index}.down_proj.weight"] = weight.T.detach().contiguous()
        else:
            state[name.replace("moe.router", "moe.gate")] = parameter.detach().clone()
    state["moe.gate.e_score_correction_bias"] = torch.zeros(4)
    return state


def _load_hf_source(model, reference, manager, context, directory):
    source = directory / "source"
    if dist.get_rank() == 0:
        source.mkdir(parents=True)
        state = _hf_state(reference)
        save_file(state, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(value.numel() * value.element_size() for value in state.values())},
            "weight_map": dict.fromkeys(state, "model.safetensors"),
        }))
        model.config.save_pretrained(source)
    dist.barrier()
    converter_context = {"ep": context.get_ep_info(), "etp": context.get_etp_info()}
    converter = BaseWeightConverter(model, converter_context, source)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    manager.load_hf(model, str(source), weight_converter=converter)
    _assert_parameters(model, reference, context)
    return source, converter_context, converter


def _export_hf(model, reference, manager, context, directory, source, converter_context, converter):
    export = directory / "export"
    manager.save_hf(model, str(export), source_path=str(source), weight_converter=converter)
    index = json.loads((export / "model.safetensors.index.json").read_text())
    actual = {}
    for filename in set(index["weight_map"].values()):
        actual.update(load_file(export / filename))
    expected = _hf_state(reference)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=2e-5, atol=2e-6, msg=name)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    exported_converter = BaseWeightConverter(model, converter_context, export)
    manager.load_hf(model, str(export), weight_converter=exported_converter)
    _assert_parameters(model, reference, context)


def _batches(dense_size, step, sp_size):
    batches = []
    for microbatch in range(2):
        replicas = []
        for source in range(dense_size):
            # One token gives empty ownership shards for TP>1. Other sources
            # have unequal odd lengths, so ETP must remove communication padding.
            length = 1 if source == 0 and microbatch == 0 else 3 + 2 * source
            generator = torch.Generator().manual_seed(1100 + 100 * step + 10 * microbatch + source)
            hidden = torch.randn(1, length, 4, generator=generator)
            padding = torch.zeros(1, length, dtype=torch.bool)
            if microbatch == 1:
                padding[:, -1] = True
            if microbatch == 1 and source == dense_size - 1 and dense_size > 1:
                padding.fill_(True)
            replicas.append((hidden, padding))
        for start in range(0, dense_size, sp_size):
            length = max(value.shape[1] for value, _ in replicas[start:start + sp_size])
            for index in range(start, start + sp_size):
                hidden, padding = replicas[index]
                tail = length - hidden.shape[1]
                replicas[index] = (F.pad(hidden, (0, 0, 0, tail)), F.pad(padding, (0, tail), value=True))
        batches.append(replicas)
    return batches


def _sp_auxiliary_losses(results, batch, sp_size):
    for start in range(0, len(results), sp_size):
        counts = sum(result[5] for result in results[start:start + sp_size])
        total = (counts.sum() / 2).clamp_min(1)
        for index in range(start, start + sp_size):
            output, logits, probs, _, z, local_counts = results[index]
            valid = ~batch[index][1].reshape(-1)
            scores = logits.sigmoid()
            scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
            probability_sum = (scores * valid[:, None]).sum(0)
            aux = (probability_sum * counts).sum() * 4 * sp_size / (2 * total.square())
            z = z * valid.sum() * sp_size / total
            results[index] = output, logits, probs, aux, z, local_counts


def _train_step(model, reference, optimizer, reference_optimizer, backend, context, step, checkpoint):
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    router = model.moe.router
    if "local_tokens_per_expert" in router._buffers:
        router.local_tokens_per_expert.zero_()
    dense_size = context.dense_fsdp_mesh.size()
    data_rank = context.dense_fsdp_mesh.get_local_rank()
    batches = _batches(dense_size, step, context.sp_size)
    total = sum((~padding).sum().item() for batch in batches for _, padding in batch)
    expected_counts = torch.zeros(4)
    observed_probs = []

    def capture_probs(module, inputs, result):
        probs = result[0].probs
        if probs.requires_grad:
            probs.retain_grad()
            observed_probs.append(probs)

    handle = router.register_forward_hook(capture_probs)
    try:
        for batch in batches:
            observed_probs.clear()
            source, padding = batch[data_rank]
            start = data_rank - data_rank % context.sp_size
            if context.sp_size > 1:
                source = torch.cat([value for value, _ in batch[start:start + context.sp_size]], dim=1)
            hidden = source.clone().requires_grad_()
            reference_inputs = [value.clone().requires_grad_() for value, _ in batch]
            model.moe.set_runtime_inputs(MoERuntimeInputs(padding_mask=padding))
            # AUX/z are per-microbatch means, independent of the main loss's
            # valid-token denominator; FSDP supplies their dense replica average.
            MoEAuxLossAutoScaler.set_loss_scale(1 / len(batches))
            output, logits = model(hidden=hidden)
            expected = [
                reference(value, source_padding)
                for value, (_, source_padding) in zip(reference_inputs, batch, strict=True)
            ]
            if context.sp_size > 1:
                _sp_auxiliary_losses(expected, batch, context.sp_size)
            local = expected[data_rank]
            torch.testing.assert_close(output, local[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(logits, local[1], rtol=1e-5, atol=1e-6)
            loss = (output.square() * (~padding)[..., None]).sum() * dense_size / total
            reference_loss = sum(
                (result[0].square() * (~source_padding)[..., None]).sum() / total
                + (0.03 * result[3] + 0.02 * result[4]) / (dense_size * len(batches))
                for result, (_, source_padding) in zip(expected, batch, strict=True)
            )
            loss.backward()
            reference_loss.backward()
            expected_hidden_grad = torch.zeros_like(hidden)
            offset = (data_rank % context.sp_size) * padding.shape[1]
            expected_hidden_grad[:, offset:offset + padding.shape[1]] = reference_inputs[data_rank].grad * dense_size
            torch.testing.assert_close(
                hidden.grad, expected_hidden_grad,
                rtol=2e-5, atol=2e-6, msg="hidden gradient",
            )
            if checkpoint is None:
                assert len(observed_probs) == 1
                torch.testing.assert_close(
                    observed_probs[0].grad, local[2].grad * dense_size,
                    rtol=2e-5, atol=2e-6, msg="routing probability gradient",
                )
            metrics = router.pop_metrics()
            torch.testing.assert_close(metrics["aux_loss"], local[3].detach())
            torch.testing.assert_close(metrics["z_loss"], local[4].detach())
            expected_counts += local[5].detach()
    finally:
        handle.remove()
        MoEAuxLossAutoScaler.set_loss_scale(1.0)
    torch.testing.assert_close(router.local_tokens_per_expert, expected_counts)
    _assert_parameters(model, reference, context, gradients=True)
    norm = backend.clip_grad_norm_(list(model.parameters()), max_norm=0.08)
    ref_norm = nn.utils.clip_grad_norm_(reference.parameters(), max_norm=0.08)
    torch.testing.assert_close(norm, ref_norm, rtol=2e-5, atol=2e-6)
    optimizer.step()
    reference_optimizer.step()
    _assert_parameters(model, reference, context)
    _assert_optimizer(optimizer, model, reference_optimizer, reference, context)


def _worker(rank, world_size, init_path, directory, tp_size, etp_size, ep_size, checkpoint, restore, sp_size=1):
    _init(rank, world_size, init_path)
    try:
        context = ParallelContext(
            "cpu", sp_size=sp_size, ep_size=ep_size, tp_size=tp_size, etp_size=etp_size,
        )
        assert context.get_dp_info()["dp_size"] == world_size // (tp_size * sp_size)
        torch.manual_seed(27)
        model = _MoEModel(context)
        weight1 = torch.randn(4, 4, 24) * 0.2
        weight2 = torch.randn(4, 12, 4) * 0.2
        with torch.no_grad():
            model.moe.router.weight.copy_(torch.randn(4, 4) * 0.4)
            for name, value in (("weight1", weight1), ("weight2", weight2)):
                parameter = model.moe.experts.get_parameter(name)
                parameter.copy_(_expert_slice(f"moe.experts.{name}", value, context).reshape_as(parameter))
        reference = _DenseReference(model, weight1, weight2)
        backend = FSDP2Backend(EngineConfig(
            mixed_precision=MixedPrecisionPolicy.FP32,
            activation_checkpointing=ActivationCheckpointingConfig(
                enabled=checkpoint is not None, use_reentrant=checkpoint == "reentrant",
            ),
        ), CpuRuntime())
        backend._parallel_context = context
        backend._world_mesh, backend._mesh = context.dense_fsdp_mesh, context.expert_mesh
        backend.prepare_model(model, _plan(tp_size, etp_size, ep_size, checkpoint, sp_size))
        expected_group = None if context.tp_mesh is None else context.tp_mesh.get_group()
        assert model.moe.router.routing_group is expected_group
        assert backend.gradient_scaling_factor == 1.0
        assert model.moe.token_dispatcher.sort_input_by_local_experts.device.type == "cpu"
        assert model.moe.experts.weight1.ndim == (2 if etp_size == 1 else 3)
        if restore:
            manager = backend.create_checkpoint_manager()
            source, converter_context, converter = _load_hf_source(
                model, reference, manager, context, directory,
            )
        optimizer = _optimizer(backend, model)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.015, weight_decay=0.0)
        _train_step(model, reference, optimizer, reference_optimizer, backend, context, 0, checkpoint)

        if restore:
            _export_hf(
                model, reference, manager, context, directory,
                source, converter_context, converter,
            )
            layout = manager.checkpoint_layout(model)
            assert layout["topology"] == dict(
                world_size=world_size, sp_size=sp_size, tp_size=tp_size,
                ep_size=ep_size, etp_size=etp_size,
            )
            optimizer.zero_grad(set_to_none=True)
            manager.save_native(model, str(directory / "model"))
            manager.save_optimizer(model, optimizer, str(directory / "optimizer"))
            snapshot = copy.deepcopy(reference)
            snapshot_optimizer = copy.deepcopy(reference_optimizer.state_dict())

        _train_step(model, reference, optimizer, reference_optimizer, backend, context, 1, checkpoint)
        if restore:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            optimizer = _optimizer(backend, model)
            manager.load_native(model, str(directory / "model"))
            manager.load_optimizer(model, optimizer, str(directory / "optimizer"))
            reference = snapshot
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.015, weight_decay=0.0)
            reference_optimizer.load_state_dict(snapshot_optimizer)
            _assert_parameters(model, reference, context)
            _assert_optimizer(optimizer, model, reference_optimizer, reference, context)
            _train_step(model, reference, optimizer, reference_optimizer, backend, context, 1, checkpoint)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("tp_size,etp_size,ep_size", [
    (1, 2, 2), (2, 1, 2), (2, 2, 2), (2, 4, 2), (4, 2, 2),
    # K=W/T is smaller than expert EDP when there is a single EP rank.
    (4, 2, 1),
    # Non-nested dense and expert TP groups share six ranks.
    (2, 3, 1), (3, 2, 1),
])
def test_independent_tp_etp_matches_dense_training(tmp_path, tp_size, etp_size, ep_size):
    world_size = math.lcm(tp_size, ep_size * etp_size)
    world_size *= max(1, math.ceil(4 / world_size))
    mp.spawn(
        _worker,
        args=(world_size, str(tmp_path / "store"), tmp_path, tp_size, etp_size, ep_size, None, False),
        nprocs=world_size, join=True,
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("tp_size,etp_size", [(1, 2), (2, 1)])
@pytest.mark.parametrize("checkpoint", ["reentrant", "nonreentrant"])
def test_independent_tp_etp_recompute_and_checkpoint(tmp_path, tp_size, etp_size, checkpoint):
    mp.spawn(
        _worker,
        args=(4, str(tmp_path / "store"), tmp_path, tp_size, etp_size, 2, checkpoint, True),
        nprocs=4, join=True,
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("tp_size,etp_size", [(1, 2), (2, 1)])
def test_independent_tp_etp_with_sequence_parallel(tmp_path, tp_size, etp_size):
    mp.spawn(
        _worker,
        args=(4, str(tmp_path / "store"), tmp_path, tp_size, etp_size, 2, None, False, 2),
        nprocs=4, join=True,
    )
