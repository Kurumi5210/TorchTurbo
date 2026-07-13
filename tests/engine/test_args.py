"""Tests for TorchTurbo CLI argument parsing."""

import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


def test_string_to_bool_matches_transformers_semantics():
    from torchturbo.engine.args import string_to_bool

    assert string_to_bool("y") is True
    assert string_to_bool("f") is False


def test_parse_torchturbo_args_prefetch_depth():
    from torchturbo.engine.args import parse_torchturbo_args

    args, _ = parse_torchturbo_args([])
    assert args.forward_prefetch_depth == 1
    assert args.backward_prefetch_depth == 1

    args, _ = parse_torchturbo_args(
        ["--tt_forward_prefetch_depth", "3", "--tt_backward_prefetch_depth", "2"]
    )
    assert args.forward_prefetch_depth == 3
    assert args.backward_prefetch_depth == 2


def test_parse_torchturbo_args_prefetch_defaults():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert remaining == []
    assert args.prefetch_strategy == "auto"
    assert args.prefetch_direction == "forward_backward"


def test_parse_torchturbo_args_gradient_checkpointing():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert args.gradient_checkpointing is True
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_gradient_checkpointing", "true", "--foo", "bar"]
    )
    assert args.gradient_checkpointing is True
    assert remaining == ["--foo", "bar"]

    args, _ = parse_torchturbo_args(["--tt_gradient_checkpointing=false"])
    assert args.gradient_checkpointing is False

    args, _ = parse_torchturbo_args(["--tt_gradient_checkpointing"])
    assert args.gradient_checkpointing is True


def test_parse_torchturbo_args_cpu_offload():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args([])
    assert args.cpu_offload is False
    assert remaining == []

    args, remaining = parse_torchturbo_args(
        ["--tt_cpu_offload", "false", "--foo", "bar"]
    )
    assert args.cpu_offload is False
    assert remaining == ["--foo", "bar"]

    args, _ = parse_torchturbo_args(["--tt_cpu_offload=true"])
    assert args.cpu_offload is True

    args, _ = parse_torchturbo_args(["--tt_cpu_offload"])
    assert args.cpu_offload is True


def test_parse_torchturbo_args_freeze_flags():
    from torchturbo.engine.args import parse_torchturbo_args

    args, remaining = parse_torchturbo_args(
        [
            "--tt_freeze_vision_tower",
            "--tt_freeze_multi_modal_projector",
            "--foo",
            "bar",
        ]
    )

    assert remaining == ["--foo", "bar"]
    assert args.freeze_vision_tower is True
    assert args.freeze_multi_modal_projector is True
    assert args.freeze_language_model is False
