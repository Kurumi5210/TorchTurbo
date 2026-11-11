"""Vocabulary TP with native FSDP2, language-model losses, and checkpoints."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.checkpoint.metadata import validate_parallel_layout
from torchturbo.engine.config import EngineConfig, LossChunkConfig, MixedPrecisionPolicy
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.engine.tensor_parallel import apply_tensor_parallel_plan
from torchturbo.models.base import TurboModel
from torchturbo.models.modules.loss import chunk_loss, logits_loss
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.ops import get_or_create_ops
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec, TensorParallelPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.utils.torch_utils import torch_version


class _VocabModel(TurboModel, nn.Module):
    """Exercise both embedding and hidden-state gradients with optional tying."""

    def __init__(self, tied: bool, vocab_size: int = 12) -> None:
        super().__init__()
        self.config = PretrainedConfig(
            vocab_size=vocab_size, hidden_size=8, tie_word_embeddings=tied,
        )
        self.embed_tokens = nn.Embedding(vocab_size, 8, padding_idx=0)
        self.projection = nn.Linear(8, 8, bias=False)
        self.final_norm = nn.RMSNorm(8)
        self.lm_head = nn.Linear(8, vocab_size, bias=False)
        self.all_tied_weights_keys = (
            {"lm_head.weight": "embed_tokens.weight"} if tied else {}
        )
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_norm(self) -> nn.Module:
        return self.final_norm

    def forward(self, input_ids: torch.Tensor, **kwargs) -> CausalLMOutput:
        hidden = torch.tanh(self.projection(self.embed_tokens(input_ids)))
        return CausalLMOutput(logits=self.lm_head(self.final_norm(hidden)))


def _parallel_plan() -> FSDPParallelPlan:
    return FSDPParallelPlan(
        tp_size=2,
        plan={
            "embed_tokens": FSDPPlanSpec(standalone_world_shard=True),
            "projection": FSDPPlanSpec(standalone_world_shard=True),
            "lm_head": FSDPPlanSpec(standalone_world_shard=True),
        },
        tensor_parallel_plan=TensorParallelPlanSpec(plan={
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Replicate(),
            ),
            "lm_head": ColwiseParallel(use_local_output=False),
        }),
    )


def _batches() -> list[dict[str, torch.Tensor]]:
    # Every data rank uses tokens on both TP lanes, including the padding row.
    batches = [
        {"input_ids": torch.tensor([[0, 1, 7, 11, 2, 9, 3]])},
        {"input_ids": torch.tensor([[6, 2, 0, 10, 5, 8, 1]])},
    ]
    for batch in batches:
        batch["labels"] = batch["input_ids"].clone()
    # Unequal valid counts catch normalization by local token count or TP size.
    # With chunk_size=2, the first rank also has a fully ignored first chunk.
    batches[0]["labels"][0, 1:3] = -100
    batches[1]["labels"][0, 5] = -100
    return batches


def _targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.pad(batch["labels"][..., 1:], (0, 1), value=-100)


def _reference_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(batch["input_ids"]).logits
    return F.cross_entropy(
        logits.flatten(0, 1), _targets(batch).flatten(),
        reduction="sum", ignore_index=-100,
    )


def _assert_vocab_storage(model: _VocabModel, tied: bool) -> None:
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    for module in (model.embed_tokens, model.lm_head):
        parameter = module.weight
        assert isinstance(parameter, DTensor)
        assert parameter.device_mesh.mesh_dim_names == ("dp", "tp")
        assert parameter.placements[-1] == Shard(0)
        fsdp_placement = parameter.placements[0]
        assert isinstance(fsdp_placement, (Shard, _StridedShard)) and fsdp_placement.dim == 0
        assert parameter.shape == (12, 8)
        assert parameter.to_local().shape == (3, 8)
        assert parameter.to_local().numel() == parameter.numel() // 4


def _assert_parameters(model: nn.Module, reference: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.full_tensor(), reference.get_parameter(name),
            rtol=1e-5, atol=1e-6, msg=name,
        )


def _assert_gradients(model: nn.Module, reference: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name).grad
        assert parameter.grad is not None and expected is not None, name
        torch.testing.assert_close(
            parameter.grad.full_tensor(), expected,
            rtol=2e-5, atol=2e-6, msg=name,
        )


def _assert_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    reference_optimizer: torch.optim.Optimizer,
    reference: nn.Module,
) -> None:
    for name, parameter in model.named_parameters():
        actual = optimizer.state[parameter]
        expected = reference_optimizer.state[reference.get_parameter(name)]
        assert actual.keys() == expected.keys()
        for field, value in actual.items():
            if isinstance(value, DTensor):
                value = value.full_tensor()
            torch.testing.assert_close(
                value, expected[field], rtol=2e-5, atol=2e-6,
                msg=f"{name}.{field}",
            )


def _write_hf(reference: _VocabModel, directory: Path) -> None:
    if dist.get_rank() == 0:
        directory.mkdir(parents=True)
        # HF tied checkpoints omit the output alias, preserving one source.
        state = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        filename = "model.safetensors"
        save_file(state, directory / filename)
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {
                "total_size": sum(t.numel() * t.element_size() for t in state.values()),
            },
            "weight_map": {name: filename for name in state},
        }))
        reference.config.save_pretrained(directory)
    dist.barrier()


def _assert_hf(reference: _VocabModel, directory: Path) -> None:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    saved = {}
    for filename in set(index["weight_map"].values()):
        saved.update(load_file(directory / filename))
    expected = dict(reference.named_parameters())
    assert saved.keys() == expected.keys()
    for name, parameter in expected.items():
        torch.testing.assert_close(saved[name], parameter, rtol=1e-5, atol=1e-6)


def _optimizer(backend: FSDP2Backend, model: nn.Module) -> torch.optim.Optimizer:
    return backend.prepare_optimizer(OptimizerClassPlan(
        model, torch.optim.AdamW, {"lr": 0.02, "weight_decay": 0.01},
    ))


def _reference_optimizer(model: _VocabModel) -> torch.optim.Optimizer:
    # OptimizerClassPlan excludes the one-dimensional RMSNorm weight from decay.
    return torch.optim.AdamW([
        {
            "params": [
                parameter for parameter in model.parameters()
                if parameter is not model.final_norm.weight
            ],
            "weight_decay": 0.01,
        },
        {"params": [model.final_norm.weight], "weight_decay": 0.0},
    ], lr=0.02)


def _training_case(directory: Path, *, tied: bool, chunked: bool) -> None:
    torch.manual_seed(17)
    model = _VocabModel(tied)
    reference = copy.deepcopy(model)
    backend = FSDP2Backend(EngineConfig(
        mixed_precision=MixedPrecisionPolicy.FP32,
        loss_chunk=LossChunkConfig(enabled=chunked, chunk_size=2),
    ), CpuRuntime())
    # prepare_model invokes the shared plan applier before native fully_shard.
    backend.prepare_model(model, _parallel_plan())
    context = backend._parallel_context
    assert context.tp_size == 2 and context.dense_fsdp_mesh.size() == 2
    _assert_vocab_storage(model, tied)
    _assert_parameters(model, reference)

    manager = backend.create_checkpoint_manager()
    layout = manager.checkpoint_layout(model)
    validate_parallel_layout({"schema_version": 2, "parallel_layout": layout}, layout)
    embedding_layout = next(
        entry for entry in layout["parameters"] if entry["fqn"] == "embed_tokens.weight"
    )
    assert embedding_layout["aliases"] == (
        ["embed_tokens.weight", "lm_head.weight"] if tied else ["embed_tokens.weight"]
    )

    source = directory / "source"
    _write_hf(reference, source)
    converter_context = {"ep": context.get_ep_info(), "etp": context.get_etp_info()}
    converter = BaseWeightConverter(model, converter_context, source)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    manager.load_hf(model, str(source), weight_converter=converter)
    _assert_parameters(model, reference)
    _assert_vocab_storage(model, tied)

    optimizer = _optimizer(backend, model)
    reference_optimizer = _reference_optimizer(reference)
    batches = _batches()
    data_rank = context.dense_fsdp_mesh.get_local_rank()
    target_count = sum(int((_targets(batch) != -100).sum()) for batch in batches)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        loss, outputs = backend.compute_loss(
            model, batches[data_rank], return_outputs=True,
            num_items_in_batch=target_count,
        )
        reference_losses = [_reference_loss(reference, batch) for batch in batches]
        reference_loss = sum(reference_losses) / target_count
        torch.testing.assert_close(
            loss.detach(), reference_losses[data_rank].detach() * 2 / target_count,
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(outputs.main_loss, reference_loss.detach(), rtol=1e-5, atol=1e-6)
        if chunked:
            assert outputs.logits is None
        else:
            assert isinstance(outputs.logits, DTensor)
            placement, = outputs.logits.placements
            assert placement.is_shard() and placement.dim % outputs.logits.ndim == 2
            assert outputs.logits.to_local().shape == (1, 7, 6)
            torch.testing.assert_close(
                outputs.logits.detach().full_tensor(),
                reference(batches[data_rank]["input_ids"]).logits.detach(),
                rtol=1e-5, atol=1e-6,
            )
        loss.backward()
        reference_loss.backward()
        _assert_gradients(model, reference)
        norm = backend.clip_grad_norm_(list(model.parameters()), max_norm=0.1)
        torch.testing.assert_close(
            norm, nn.utils.clip_grad_norm_(reference.parameters(), max_norm=0.1),
            rtol=2e-5, atol=2e-6,
        )
        optimizer.step()
        reference_optimizer.step()
        _assert_parameters(model, reference)
        _assert_optimizer(optimizer, model, reference_optimizer, reference)
        _assert_vocab_storage(model, tied)

    step()
    if not chunked:
        export = directory / "export"
        manager.save_hf(model, str(export), source_path=str(source), weight_converter=converter)
        _assert_hf(reference, export)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        exported_converter = BaseWeightConverter(model, converter_context, export)
        manager.load_hf(model, str(export), weight_converter=exported_converter)
        _assert_parameters(model, reference)

    optimizer.zero_grad(set_to_none=True)
    manager.save_native(model, str(directory / "model_distcp"))
    manager.save_optimizer(model, optimizer, str(directory / "optimizer_distcp"))
    snapshot = copy.deepcopy(reference)
    snapshot_optimizer = copy.deepcopy(reference_optimizer.state_dict())
    step()

    # Restore into a fresh optimizer and repeat the continuation step.
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer = _optimizer(backend, model)
    manager.load_native(model, str(directory / "model_distcp"))
    manager.load_optimizer(model, optimizer, str(directory / "optimizer_distcp"))
    reference = snapshot
    reference_optimizer = _reference_optimizer(reference)
    reference_optimizer.load_state_dict(snapshot_optimizer)
    _assert_parameters(model, reference)
    _assert_optimizer(optimizer, model, reference_optimizer, reference)
    assert manager.checkpoint_layout(model) == layout
    step()


def _vocab_worker(rank: int, init_path: str, directory: Path, tied: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        for chunked in (False, True):
            _training_case(
                directory / ("chunked" if chunked else "regular"),
                tied=tied, chunked=chunked,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch_version() < (2, 10, 0), reason="Native TP/FSDP2 requires PyTorch >= 2.10")
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_vocab_tp_fsdp_loss_training_and_checkpoint(tmp_path: Path, tied: bool) -> None:
    mp.spawn(
        _vocab_worker, args=(str(tmp_path / "store"), tmp_path, tied),
        nprocs=4, join=True,
    )


def _tp_loss_case(mesh, *, tied: bool, frozen: str | None, chunked: bool) -> None:
    torch.manual_seed(29)
    model = _VocabModel(tied, vocab_size=13)
    if frozen == "head":
        model.lm_head.requires_grad_(False)
    elif frozen == "backbone":
        model.embed_tokens.requires_grad_(False)
        model.projection.requires_grad_(False)
        model.final_norm.requires_grad_(False)
    reference = copy.deepcopy(model)
    apply_tensor_parallel_plan(
        model, _parallel_plan().tensor_parallel_plan.plan, tp_mesh=mesh,
    )
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    for module in (model.embed_tokens, model.lm_head):
        assert module.weight.placements == (Shard(0),)
        assert module.weight.to_local().shape == (7 if mesh.get_local_rank() == 0 else 6, 8)

    ids = [torch.tensor([[0, 1, 12, 6, 7]]), torch.tensor([[12, 2, 10]])]
    hidden = [
        model.final_norm(torch.tanh(model.projection(model.embed_tokens(value))))
        for value in ids
    ]
    reference_hidden = [
        reference.final_norm(torch.tanh(reference.projection(reference.embed_tokens(value))))
        for value in ids
    ]
    diagnostic = torch.randn(1, 4, 8, requires_grad=True)
    hidden.append(diagnostic)
    reference_hidden.append(diagnostic.detach().clone().requires_grad_())
    for value in (*hidden, *reference_hidden):
        if value.requires_grad:
            value.retain_grad()
    labels = [
        torch.tensor([[-100, -100, 12, 6, 0]]),
        torch.tensor([[2, 12, -100]]),
        torch.tensor([[3, 12, -100, 6]]),
    ]
    normalizers, scales = [7.0, 5.0, 3.0], [1.0, 0.3, 0.0]
    if chunked:
        loss, components = chunk_loss(
            hidden, labels, lm_head=model.lm_head, normalizers=normalizers,
            loss_scales=scales, chunk_size=2,
        )
    else:
        predictions = [model.lm_head(value) for value in hidden]
        loss, components = logits_loss(
            predictions, labels, normalizers=normalizers, loss_scales=scales,
        )
    reference_components = [
        F.cross_entropy(
            reference.lm_head(value).flatten(0, 1), label.flatten(),
            reduction="sum", ignore_index=-100,
        ) / normalizer
        for value, label, normalizer in zip(reference_hidden, labels, normalizers)
    ]
    reference_loss = reference_components[0] + 0.3 * reference_components[1]
    assert not isinstance(loss, DTensor) and not isinstance(components, DTensor)
    assert not components.requires_grad
    torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        components, torch.stack(reference_components).detach(), rtol=1e-5, atol=1e-6,
    )
    # The custom chunk backward must honor a caller's upstream loss scaling.
    (loss * 2.5).backward()
    (reference_loss * 2.5).backward()
    assert diagnostic.grad is None
    for value, expected in zip(hidden, reference_hidden):
        if expected.grad is None:
            assert value.grad is None
        else:
            torch.testing.assert_close(value.grad, expected.grad, rtol=2e-5, atol=2e-6)
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name)
        assert parameter.requires_grad == expected.requires_grad
        if expected.grad is None:
            assert parameter.grad is None, name
            continue
        actual = parameter.grad
        if isinstance(actual, DTensor):
            actual = actual.full_tensor()
        torch.testing.assert_close(actual, expected.grad, rtol=2e-5, atol=2e-6, msg=name)


def _tp_loss_worker(rank: int, init_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        empty_model = _VocabModel(tied=True, vocab_size=1)
        original = dict(empty_model.named_parameters(remove_duplicate=False))
        with pytest.raises(ValueError, match="nonempty embedding shards"):
            apply_tensor_parallel_plan(
                empty_model, _parallel_plan().tensor_parallel_plan.plan, tp_mesh=mesh,
            )
        for name, parameter in empty_model.named_parameters(remove_duplicate=False):
            assert parameter is original[name] and not isinstance(parameter, DTensor)
        for tied, frozen in ((False, None), (True, None), (False, "head"), (False, "backbone")):
            for chunked in (False, True):
                _tp_loss_case(mesh, tied=tied, frozen=frozen, chunked=chunked)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch_version() < (2, 10, 0), reason="Native TP requires PyTorch >= 2.10")
def test_vocab_tp_uneven_loss_streams_and_freezing(tmp_path: Path) -> None:
    mp.spawn(_tp_loss_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
