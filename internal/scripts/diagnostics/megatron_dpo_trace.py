"""Swift Megatron external plugin for comparing DPO inputs with TorchTurbo.

Usage: DPO_TRACE_DIR=/path/to/gpu-trace megatron rlhf ... --external_plugins /path/to/this/file.py
Targets the Swift acb9f4cf / Megatron Core 0.19.0 trainer interface.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


def _load_trace_module() -> Any:
    # Import the shared diagnostic without importing TorchTurbo's engine or
    # registering its models into the GPU process. Keep the module importable
    # by name so spawned dataloader workers can unpickle the collator.
    path = Path(__file__).resolve().parents[3] / "src/torchturbo/adapters/msswift"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("dpo_alignment")


def install() -> None:
    """Trace only TP0/CP0/first-PP-stage for each data-parallel rank."""
    from megatron.core import mpu
    from swift.megatron.trainers.dpo_trainer import MegatronDPOTrainer

    trace = _load_trace_module()
    trainer_cls = MegatronDPOTrainer
    if "_turbo_dpo_trace_installed" in trainer_cls.__dict__:
        return
    prepare_dataloader = trainer_cls._prepare_dataloader
    get_batch = trainer_cls.get_batch

    def prepare(self: Any, train_dataset: Any, val_dataset: Any) -> Any:
        self._turbo_dpo_trace_writer = None
        self._turbo_dpo_trace_dir = os.environ.get("DPO_TRACE_DIR", str(Path(self.args.output_dir) / "data-trace"))
        enabled = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
            and mpu.is_pipeline_first_stage(ignore_virtual=True)
        )
        if enabled:
            if not self.args.packing or self.args.streaming or self.args.group_by_length:
                raise ValueError("DPO trace plugin requires packed, non-streaming DPO without group_by_length.")
            if mpu.get_data_parallel_rank() == 0:
                trace.save_dataset_manifest(self._turbo_dpo_trace_dir, train_dataset, val_dataset)
            self.data_collator = trace.DPOTraceCollator(self.data_collator)
            train_dataset = trace.IndexedDPODataset(train_dataset)
            if val_dataset is not None:
                val_dataset = trace.IndexedDPODataset(val_dataset)
        return prepare_dataloader(self, train_dataset, val_dataset)

    def traced_batches(self: Any, data_iterator: Any) -> Any:
        for data in data_iterator:
            record = data.pop(trace.TRACE_KEY, None)
            if record is not None and self.unwrapped_models[0].training:
                if self._turbo_dpo_trace_writer is None:
                    dp_size = mpu.get_data_parallel_world_size()
                    micro = self.args.micro_batch_size
                    if self.args.global_batch_size % (dp_size * micro):
                        raise ValueError("Global batch size is not divisible by DP * micro batch size.")
                    self._turbo_dpo_trace_writer = trace.DPOTraceWriter(
                        self._turbo_dpo_trace_dir,
                        dp_rank=mpu.get_data_parallel_rank(), dp_size=dp_size,
                        gradient_accumulation_steps=self.args.global_batch_size // (dp_size * micro),
                        micro_batch_size=micro, start_step=self.state.iteration,
                        max_steps=self.args.train_iters,
                    )
                self._turbo_dpo_trace_writer.write(self.state.iteration, record)
            yield data

    def traced_get_batch(self: Any, data_iterator: Any, vp_stage: Any = None) -> Any:
        return get_batch(self, traced_batches(self, data_iterator), vp_stage)

    trainer_cls._prepare_dataloader = prepare
    trainer_cls.get_batch = traced_get_batch
    trainer_cls._turbo_dpo_trace_installed = True


install()
