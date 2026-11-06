"""Swift DPO with MTP: chosen-only targets, stage shifts and gradient routing."""

import importlib
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from torchturbo.adapters.msswift import engine_registry  # noqa: E402
from torchturbo.adapters.msswift.trainer import TurboDPOTrainerMixin  # noqa: E402
from torchturbo.models.base import TurboModel  # noqa: E402
from torchturbo.ops import get_or_create_ops  # noqa: E402
from torchturbo.runtime.cpu import CpuRuntime  # noqa: E402

from .test_trainer_packed import _Engine, _PAIRS, _packed_batch  # noqa: E402

_HIDDEN = 6
_VOCAB = 11
_PAD = -100


@pytest.fixture
def cpu_ops(monkeypatch):
    """Configure the dispatcher the chunked cross-entropy resolves through."""
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=runtime)


class _MTPModel(TurboModel, torch.nn.Module):
    """Minimal model carrying the MTP output contract of the Qwen3.5 family."""

    def __init__(self, num_stages: int = 2):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
        # A real lm_head so the trainer can shadow it; fixed stage multiples
        # keep the expected loss writable in closed form.
        self.lm_head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(_VOCAB, _HIDDEN, generator=generator))
            self.lm_head.weight.copy_(
                torch.randn(_VOCAB, _HIDDEN, generator=generator)
            )
        self.stage_scales = [0.5, 0.25][:num_stages]
        self.config = SimpleNamespace(num_nextn_predict_layers=num_stages)

    def backbone_hidden(self, input_ids, position_ids):
        return self.embed(input_ids) + 0.1 * position_ids.unsqueeze(-1)

    def forward(self, input_ids, position_ids, logits_to_keep=0, **kwargs):
        hidden = self.backbone_hidden(input_ids, position_ids)
        mtp_hidden = [hidden * scale for scale in self.stage_scales]
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            mtp_logits=[self.lm_head(state) for state in mtp_hidden],
            mtp_hidden_states=mtp_hidden,
        )


def _prepare_chunk_loss_lm_head(model, reference):
    """Stand in for the backend helper on the plain (non-FSDP) path."""
    weight = model.get_lm_head().weight
    return weight, reference, weight.new_zeros(())


def _trainer(monkeypatch, mtp_enabled=True, mtp_loss_scale=0.1, ga=4):
    engine = _Engine(mtp_enabled=mtp_enabled, mtp_loss_scale=mtp_loss_scale)
    engine.backend._prepare_chunk_loss_lm_head = _prepare_chunk_loss_lm_head
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer = object.__new__(TurboDPOTrainerMixin)
    trainer.label_pad_token_id = _PAD
    trainer.loss_type = ["sigmoid"]
    trainer.args = SimpleNamespace(ld_alpha=None)
    trainer.aux_loss_enabled = False
    trainer.current_gradient_accumulation_steps = ga
    trainer.get_use_logits_to_keep = lambda: True
    return trainer, engine


def _expected_stage_labels(stage):
    """Targets for one MTP stage: ``stage + 2`` ahead, chosen segments only."""
    chosen = [(prompt, response) for prompt, response, _ in _PAIRS]
    rejected = [(prompt, response) for prompt, _, response in _PAIRS]
    shift = stage + 2
    expected = []
    for index, (prompt, response) in enumerate(chosen + rejected):
        row = [_PAD] * len(prompt) + list(response)
        is_chosen = index < len(_PAIRS)
        for position in range(len(row)):
            target = position + shift
            inside_row = target < len(row)
            expected.append(row[target] if is_chosen and inside_row else _PAD)
    return expected


def _reference_mtp_loss(model, batch, loss_scale):
    """Per-stage chunk-free cross-entropy, matching the kernel's normalizer."""
    hidden = model.backbone_hidden(batch["input_ids"], batch["position_ids"])
    total = hidden.new_zeros(())
    for stage, scale in enumerate(model.stage_scales):
        logits = model.lm_head(hidden * scale)
        labels = torch.tensor([_expected_stage_labels(stage)])
        supervised = (labels != _PAD).sum().clamp_min(1)
        stage_loss = F.cross_entropy(
            logits[0].float(), labels[0], ignore_index=_PAD, reduction="sum"
        )
        total = total + loss_scale * stage_loss / supervised
    return total


def test_stage_labels_shift_per_stage_and_drop_rejected(monkeypatch):
    """Stage ``i`` targets ``i + 2`` ahead and never supervises rejected rows."""
    from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

    trainer, _ = _trainer(monkeypatch)
    batch = _packed_batch()
    cu_seqlens = batch["cu_seq_lens_q"]
    num_segments = cu_seqlens.numel() - 1
    segment_ids = cu_seqlens_to_segment_ids(cu_seqlens, batch["labels"].size(-1))
    stage_labels, normalizers = trainer._mtp_stage_labels(
        batch["labels"], segment_ids, num_segments // 2, 2
    )

    for stage, labels in enumerate(stage_labels):
        expected = torch.tensor([_expected_stage_labels(stage)])
        torch.testing.assert_close(labels, expected)
        assert normalizers[stage] == (expected != _PAD).sum().clamp_min(1)


@pytest.mark.parametrize("ga", [1, 4])
def test_mtp_enters_the_gradient_but_not_the_reported_outputs(monkeypatch, cpu_ops, ga):
    """MTP contributes ``scaled_loss / ga`` to the gradient and nothing else."""
    loss_scale = 0.1
    batch = _packed_batch()

    trainer_off, _ = _trainer(monkeypatch, mtp_enabled=False, ga=ga)
    model_off = _MTPModel()
    outputs_off = trainer_off.concatenated_forward(model_off, batch)
    outputs_off["chosen_logps"].sum().backward()

    trainer_on, _ = _trainer(monkeypatch, mtp_loss_scale=loss_scale, ga=ga)
    model_on = _MTPModel()
    outputs_on = trainer_on.concatenated_forward(model_on, batch)
    # Shadowing the LM head must leave the DPO-side outputs bit-for-bit equal.
    for key in ("chosen_logps", "rejected_logps", "nll_loss"):
        torch.testing.assert_close(outputs_on[key], outputs_off[key])
    outputs_on["chosen_logps"].sum().backward()

    model_ref = _MTPModel()
    mtp_grads = torch.autograd.grad(
        _reference_mtp_loss(model_ref, batch, loss_scale),
        [model_ref.embed.weight, model_ref.lm_head.weight],
    )
    for actual, base, mtp_grad in zip(
        (model_on.embed.weight, model_on.lm_head.weight),
        (model_off.embed.weight, model_off.lm_head.weight),
        mtp_grads,
    ):
        torch.testing.assert_close(
            actual.grad, base.grad + mtp_grad / ga, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("loss_scale", [0.0, 0.1])
def test_mtp_metrics_report_unscaled_stages_and_a_scaled_total(
    monkeypatch, cpu_ops, loss_scale
):
    """Per-stage values stay unscaled, so a zero factor still reports them."""
    trainer, engine = _trainer(monkeypatch, mtp_loss_scale=loss_scale)
    trainer.concatenated_forward(_MTPModel(num_stages=2), _packed_batch())

    stages = [engine.metrics["mtp_loss/0"], engine.metrics["mtp_loss/1"]]
    assert all(stage > 0 for stage in stages)
    assert engine.metrics["mtp_loss/avg"] == pytest.approx(sum(stages).item() / 2)
    assert engine.metrics["mtp_loss/scaled"] == pytest.approx(
        sum(stages).item() * loss_scale
    )


def test_reference_pass_reports_no_mtp(monkeypatch):
    """The reference model never turns MTP into a loss or a metric."""
    trainer, engine = _trainer(monkeypatch)
    with torch.no_grad():
        outputs = trainer.concatenated_forward(
            _MTPModel(), _packed_batch(), is_ref_model=True
        )
    assert "nll_loss" not in outputs
    assert engine.metrics == {}


def test_missing_mtp_outputs_points_at_the_template(monkeypatch, cpu_ops):
    """A template that drops ``input_ids`` fails with an actionable message."""
    trainer, _ = _trainer(monkeypatch)

    class _NoMTPOutputs(_MTPModel):
        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs.mtp_logits = None
            return outputs

    with pytest.raises(RuntimeError, match="turbo-registered"):
        trainer.concatenated_forward(_NoMTPOutputs(), _packed_batch())


def test_sp_mtp_loss_matches_single_rank(monkeypatch, cpu_ops):
    """Per-rank partial sums over a globally counted normalizer rebuild the whole."""
    from torchturbo.models.modules.attention.ulysses import (
        patch_forward_for_sp_input_slicing,
        ulysses_pad_inputs,
    )

    loss_scale = 0.1
    trainer, engine = _trainer(monkeypatch, mtp_loss_scale=loss_scale)
    trainer.concatenated_forward(_MTPModel(), _packed_batch())
    expected_stages = [engine.metrics["mtp_loss/0"], engine.metrics["mtp_loss/1"]]

    sp_size = 4
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: sp_size)
    model = _MTPModel()
    spec = model.get_sequence_parallel_input_slicing_spec()
    engine.backend.prepare_inputs = lambda model, inputs: ulysses_pad_inputs(
        inputs, spec, sp_size
    )
    # The MTP reduction carries the total plus one entry per stage.
    mtp_width = 1 + len(model.stage_scales)

    def forward_on_every_rank(reduce_mtp, reduce_stats):
        def all_reduce(values, group):
            reduce = reduce_mtp if values.numel() == mtp_width else reduce_stats
            return reduce(values)

        monkeypatch.setattr("torch.distributed.nn.functional.all_reduce", all_reduce)
        for sp_rank in range(sp_size):
            engine.sp_info = {"sp_size": sp_size, "sp_group": None, "sp_rank": sp_rank}
            patch_forward_for_sp_input_slicing(spec, None, sp_size, sp_rank)
            trainer.concatenated_forward(model, _packed_batch())

    # Collect each rank's partial sums, then replay every rank with the totals.
    mtp_partials, stats_partials = [], []

    def record(store):
        def collect(values):
            store.append(values)
            return values

        return collect

    forward_on_every_rank(record(mtp_partials), record(stats_partials))
    mtp_total = torch.stack(mtp_partials).sum(0)
    stats_total = torch.stack(stats_partials).sum(0)

    forward_on_every_rank(lambda values: mtp_total, lambda values: stats_total)
    for stage, expected in enumerate(expected_stages):
        torch.testing.assert_close(
            engine.metrics[f"mtp_loss/{stage}"], expected, rtol=2e-5, atol=2e-6
        )
