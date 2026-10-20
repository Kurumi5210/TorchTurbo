"""Validate the debug launcher against Swift 4.4.2's real argument parser."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "internal/scripts/runs/qwen3_8_dense/swift_text_dpo_debug.sh"


def _dry_run(**overrides):
    env = {**os.environ, "MODEL": "/model", "DATASET": "/dataset", "OUTPUT_DIR": "/output",
           "DRY_RUN": "1", "NNODES": "2", "NPROC_PER_NODE": "16", "SP_SIZE": "16", **overrides}
    return subprocess.run(["bash", str(_SCRIPT)], env=env, text=True, capture_output=True)


def test_debug_launcher_rejects_a_different_effective_dp_size():
    result = _dry_run(NNODES="1")
    assert result.returncode != 0
    assert "2 * SP_SIZE" in result.stderr


def test_debug_launcher_arguments_parse_without_unknown_flags(monkeypatch):
    swift = pytest.importorskip("swift")
    if swift.__version__ != "4.4.2":
        pytest.skip("The debug launcher targets Swift 4.4.2")
    from transformers import HfArgumentParser
    from torchturbo.adapters.msswift.args import TurboRLHFArguments
    from torchturbo.engine.args import parse_torchturbo_args

    result = _dry_run()
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    argv = command[command.index("dpo") + 1:]
    turbo, argv = parse_torchturbo_args(argv)
    # Validate names, types and values without initializing distributed training.
    monkeypatch.setattr(TurboRLHFArguments, "__post_init__", lambda self: None)
    args, = HfArgumentParser(TurboRLHFArguments).parse_args_into_dataclasses(argv)
    assert args.template == "qwen3_8" and args.agent_template == "qwen3_5"
    assert args.max_length == args.packing_length == 128000
    assert args.split_dataset_ratio == 0.01 and args.load_from_cache_file is False
    assert args.dpo_megatron_data is True and args.num_train_epochs == 1
    assert args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 12
    assert turbo.gbs == 24 and turbo.sp_size == 16
    assert args.adam_beta2 == 0.95 and args.weight_decay == 0.1
    assert args.lr_scheduler_type == "cosine_with_min_lr"
    assert args.lr_scheduler_kwargs == '{"min_lr":1e-7}'
    assert args.loss_scale == "last_round+ignore_empty_think"
    assert args.eval_strategy == "no" and args.save_only_model is True
