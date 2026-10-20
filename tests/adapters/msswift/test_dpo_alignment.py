"""Reference ordering, padding-independent traces and complete-step comparison."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from torchturbo.adapters.msswift.dpo_alignment import (
    DPOTraceCollator,
    DPOTraceWriter,
    IndexedDPOItem,
    MegatronDPOSampler,
    TRACE_KEY,
)

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "compare_dpo_traces", _ROOT / "internal/scripts/diagnostics/compare_dpo_traces.py",
)
_COMPARE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPARE)


# Captured from Swift acb9f4cf's MegatronPretrainingRandomSampler: total=31,
# DP=2, micro=1, epoch=0. Three complete optimizer steps use GA=4.
_MEGATRON_ORDER = (
    [18, 19, 29, 8, 30, 12, 15, 5, 7, 6, 13, 21],
    [10, 2, 22, 14, 28, 17, 16, 23, 3, 9, 0, 4],
)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("skip", [0, 4, 8, 12])
def test_matches_megatron_complete_steps_and_resume(rank, skip):
    sampler = MegatronDPOSampler(
        31, dp_rank=rank, dp_size=2, micro_batch_size=1,
        gradient_accumulation_steps=4, skip_batches=skip,
    )
    assert list(sampler) == _MEGATRON_ORDER[rank][skip:]
    assert len(sampler) == 12 - skip
    assert sampler.num_steps == 3


def test_sampler_uses_epoch_seed_independent_of_training_rng():
    sampler = MegatronDPOSampler(
        31, dp_rank=0, dp_size=2, micro_batch_size=1, gradient_accumulation_steps=4,
    )
    torch.manual_seed(42)
    assert list(sampler) == _MEGATRON_ORDER[0]
    sampler.set_epoch(1)
    assert list(sampler) == [10, 14, 29, 22, 0, 21, 18, 15, 28, 8, 7, 13]


def test_original_983_pack_run_has_40_complete_steps():
    ranks = [list(MegatronDPOSampler(
        983, dp_rank=rank, dp_size=2, micro_batch_size=1, gradient_accumulation_steps=12,
    )) for rank in range(2)]
    assert [len(rank) for rank in ranks] == [480, 480]
    assert set(ranks[0]).isdisjoint(ranks[1])
    assert len(set(ranks[0] + ranks[1])) == 960


def _samples():
    return [
        {"chosen_input_ids": [1, 2, 3], "chosen_labels": [-100, -100, 3],
         "rejected_input_ids": [1, 5], "rejected_labels": [-100, 5]},
        {"chosen_input_ids": [6, 7, 8, 9], "chosen_labels": [-100, -100, 8, 9],
         "rejected_input_ids": [6, 7, 10], "rejected_labels": [-100, -100, 10]},
    ]


def _collator(pad_each=1, pad_tail=0, corrupt=False):
    def collate(batch):
        samples = [sample for pack in batch for sample in pack]
        ids, labels, positions = [], [], []
        for prefix in ("chosen_", "rejected_"):
            for sample in samples:
                row = sample[prefix + "input_ids"]
                padding = -len(row) % pad_each
                ids.extend(row + [0] * padding)
                labels.extend(sample[prefix + "labels"] + [-100] * padding)
                positions.extend(range(len(row) + padding))
        if corrupt:
            labels[2] = -100
        return {
            "input_ids": torch.tensor([ids + [0] * pad_tail]),
            "labels": torch.tensor([labels + [-100] * pad_tail]),
            "position_ids": torch.tensor([positions + list(range(pad_tail))]),
        }
    return collate


def _record(collator=None):
    batch = [IndexedDPOItem(9, _samples())]
    return DPOTraceCollator(collator or _collator())(batch)[TRACE_KEY]


def test_trace_matches_across_cp_and_sp_padding():
    gpu = _record(_collator(pad_each=8, pad_tail=8))
    npu = _record(_collator(pad_tail=3))
    assert gpu == npu
    assert gpu["encoded"] == gpu["collated"]
    assert gpu["pairs_per_pack"] == [2]
    assert [part["length"] for part in gpu["collated"]] == [3, 4, 2, 3]


def _write_trace(directory, *, max_steps=2, written_steps=2):
    directory.mkdir()
    (directory / "dataset.json").write_text(json.dumps({"schema": 1, "train": {"num_packs": 31}}))
    for rank in range(2):
        writer = DPOTraceWriter(
            str(directory), dp_rank=rank, dp_size=2, gradient_accumulation_steps=2,
            micro_batch_size=1, start_step=0, max_steps=max_steps,
        )
        for step in range(written_steps):
            for _ in range(2):
                writer.write(step, _record())


def test_compare_all_steps_and_explicit_complete_prefix(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_trace(a, max_steps=40)
    _write_trace(b, max_steps=40)
    assert _COMPARE.compare_traces(a, b, steps=2) == 8
    with pytest.raises(ValueError, match="Missing batch step=3"):
        _COMPARE.compare_traces(a, b)


@pytest.mark.parametrize("change", ["labels", "missing_rank", "collator", "duplicate", "packing"])
def test_compare_rejects_mismatch_or_incomplete_traces(tmp_path, change):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_trace(a)
    _write_trace(b)
    file = next(b.glob("dp-00001-*.jsonl"))
    if change == "missing_rank":
        file.unlink()
    elif change == "packing":
        (b / "dataset.json").write_text(json.dumps({"schema": 1, "train": {"num_packs": 30}}))
    else:
        lines = [json.loads(line) for line in file.read_text().splitlines()]
        if change == "duplicate":
            lines.append(deepcopy(lines[-1]))
        else:
            lines[1]["collated"][0]["labels"] = "changed-labels"
            if change == "labels":
                lines[1]["encoded"][0]["labels"] = "changed-labels"
        file.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError):
        _COMPARE.compare_traces(a, b)


def test_empty_traces_are_not_a_match(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_trace(a, written_steps=0)
    _write_trace(b, written_steps=0)
    with pytest.raises(ValueError, match="Missing batch"):
        _COMPARE.compare_traces(a, b)


def test_trace_counts_consumed_batches_and_resume_steps(tmp_path):
    writer = DPOTraceWriter(
        str(tmp_path), dp_rank=0, dp_size=2, gradient_accumulation_steps=2,
        micro_batch_size=1, start_step=17, max_steps=40,
    )
    # Worker collation can run ahead; only the trainer's write counts a batch.
    prefetched = [_record() for _ in range(6)]
    assert len(writer.path.read_text().splitlines()) == 1
    writer.write(17, prefetched[0])
    writer.write(17, prefetched[1])
    writer.write(18, prefetched[2])
    records = [json.loads(line) for line in writer.path.read_text().splitlines()][1:]
    assert [(r["step"], r["micro_step"]) for r in records] == [(18, 0), (18, 1), (19, 0)]
    with pytest.raises(ValueError, match="discontinuous"):
        writer.write(19, prefetched[3])


def test_turbo_trainer_records_before_loss_and_strips_metadata(monkeypatch, tmp_path):
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.trainer import TurboDPOTrainerMixin

    class NativeTrainer:
        def _prepare_inputs(self, inputs):
            assert TRACE_KEY not in inputs
            return inputs

        def compute_loss(self, model, inputs, **kwargs):
            assert TRACE_KEY not in inputs
            return torch.tensor(0.)

    class Trainer(TurboDPOTrainerMixin, NativeTrainer):
        pass

    engine = SimpleNamespace(
        get_dp_info=lambda: {"dp_rank": 0, "dp_size": 2, "dp_group": None},
        get_sp_info=lambda: {"sp_rank": 0, "sp_size": 16, "sp_group": None},
    )
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: engine)
    trainer = object.__new__(Trainer)
    trainer._dpo_megatron_data = True
    trainer._dpo_data_trace_dir = str(tmp_path)
    trainer._dpo_trace_writer = None
    trainer._train_batch_size = 1
    trainer.train_dataset = [_samples() for _ in range(8)]
    trainer.data_collator = _collator()
    trainer.args = SimpleNamespace(
        gradient_accumulation_steps=2, train_dataloader_shuffle=True,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        dataloader_persistent_workers=False, dataloader_prefetch_factor=None,
    )
    trainer.state = SimpleNamespace(global_step=0, max_steps=2)
    dataloader = trainer.get_train_dataloader()
    assert len(dataloader) == 4
    for i, batch in enumerate(dataloader):
        trainer.state.global_step = i // 2
        inputs = trainer._prepare_inputs(batch)
        trainer.compute_loss(None, inputs)
    records = [json.loads(line) for line in trainer._dpo_trace_writer.path.read_text().splitlines()][1:]
    assert [(r["step"], r["micro_step"]) for r in records] == [(1, 0), (1, 1), (2, 0), (2, 1)]


def test_gpu_plugin_uses_training_consumption_and_excludes_validation(monkeypatch, tmp_path):
    pytest.importorskip("swift")
    from swift.dataset import PackingDataset
    from torch.utils.data import DataLoader

    class NativeMegatronTrainer:
        def _prepare_dataloader(self, train, val):
            return (DataLoader(train, batch_size=1, collate_fn=self.data_collator),
                    DataLoader(val, batch_size=1, collate_fn=self.data_collator))

        def get_batch(self, data_iterator, vp_stage=None):
            batch = next(data_iterator)
            assert TRACE_KEY not in batch
            return batch

    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0, get_context_parallel_rank=lambda: 0,
        is_pipeline_first_stage=lambda **kwargs: True,
        get_data_parallel_rank=lambda: 0, get_data_parallel_world_size=lambda: 2,
    )
    for name in ("megatron", "megatron.core", "swift.megatron", "swift.megatron.trainers",
                 "swift.megatron.trainers.dpo_trainer"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["megatron.core"].mpu = mpu
    sys.modules["swift.megatron.trainers.dpo_trainer"].MegatronDPOTrainer = NativeMegatronTrainer
    monkeypatch.setenv("DPO_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "megatron_dpo_trace_plugin", _ROOT / "internal/scripts/diagnostics/megatron_dpo_trace.py",
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    dataset = object.__new__(PackingDataset)
    dataset.dataset = _samples()
    dataset.packing_length = 128000
    dataset.packed_idx = [[0, 1]] * 4
    dataset.packed_length = [12] * 4
    trainer = NativeMegatronTrainer()
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path), packing=True, streaming=False, group_by_length=False,
        micro_batch_size=1, global_batch_size=4, train_iters=2,
    )
    trainer.state = SimpleNamespace(iteration=0)
    trainer.unwrapped_models = [SimpleNamespace(training=True)]
    trainer.data_collator = _collator(pad_each=8, pad_tail=8)
    train, val = trainer._prepare_dataloader(dataset, dataset)
    iterator = iter(train)
    for i in range(4):
        trainer.state.iteration = i // 2
        trainer.get_batch(iterator)
    writer = trainer._turbo_dpo_trace_writer
    assert len(writer.path.read_text().splitlines()) == 5
    trainer.unwrapped_models[0].training = False
    trainer.get_batch(iter(val))
    assert len(writer.path.read_text().splitlines()) == 5
