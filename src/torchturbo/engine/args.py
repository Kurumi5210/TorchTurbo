"""CLI argument definitions for torchturbo, replacing TORCH_TURBO_* env vars.

Usage::

    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    argv = sys.argv[1:]
    tt_args, remaining_argv = parse_torchturbo_args(argv)
    set_torchturbo_args(tt_args)
    # pass remaining_argv to the training framework

The parsed args can later be retrieved::

    from torchturbo.engine.args import get_torchturbo_args

    tt_args = get_torchturbo_args()
    overrides = tt_args.to_overrides_dict()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level context for stashing parsed args (single-process training)
# ---------------------------------------------------------------------------

_current_torchturbo_args: TorchTurboArgs | None = None


def set_torchturbo_args(args: TorchTurboArgs) -> None:
    """Store parsed torchturbo args in module-level context."""
    global _current_torchturbo_args
    _current_torchturbo_args = args


def get_torchturbo_args() -> TorchTurboArgs | None:
    """Retrieve previously stashed torchturbo args."""
    return _current_torchturbo_args


# ---------------------------------------------------------------------------
# TorchTurboArgs — mirrors TORCH_TURBO_* env vars
# ---------------------------------------------------------------------------


@dataclass
class TorchTurboArgs:
    """CLI arguments for torchturbo.

    All fields default to ``None`` meaning "not specified" — only explicitly
    provided values take effect.  When converted to overrides via
    :meth:`to_overrides_dict`, only non-None fields are included.

    Attributes
    ----------
    backend:
        Distributed backend (``"fsdp2"``).
    mixed_precision:
        Mixed precision policy (``"bf16"``, ``"fp16"``, ``"fp32"``, ``"fp8"``).
    sp_size:
        Sequence parallelism group size.
    ep_size:
        Expert parallelism group size.
    """

    backend: str | None = None
    mixed_precision: str | None = None
    sp_size: int | None = None
    ep_size: int | None = None
    mtp_loss_scaling_factor: float | None = None

    def to_overrides_dict(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for ``EngineConfig.from_dict()``.

        Only fields that were explicitly set (non-None) are included, so
        unspecified arguments do not override values from other sources.
        """
        d: dict[str, Any] = {}
        if self.backend is not None:
            d["backend"] = self.backend
        if self.mixed_precision is not None:
            d["mixed_precision"] = self.mixed_precision
        if self.sp_size is not None:
            d["sp_size"] = self.sp_size
        if self.ep_size is not None:
            d["ep_size"] = self.ep_size
        if self.mtp_loss_scaling_factor is not None:
            d["mtp_loss_scaling_factor"] = self.mtp_loss_scaling_factor
        return d


# ---------------------------------------------------------------------------
# Argparse-based CLI parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``--tt_*`` flags."""
    parser = argparse.ArgumentParser(
        description="torchturbo arguments",
        add_help=False,
    )

    parser.add_argument(
        "--tt_backend",
        type=str,
        default=None,
        metavar="BACKEND",
        help="Distributed backend (e.g. fsdp2). Replaces TORCH_TURBO_BACKEND.",
    )
    parser.add_argument(
        "--tt_mixed_precision",
        type=str,
        default=None,
        metavar="POLICY",
        help="Mixed precision policy (bf16, fp16, fp32, fp8). "
        "Replaces TORCH_TURBO_MIXED_PRECISION.",
    )
    parser.add_argument(
        "--tt_sp_size",
        type=int,
        default=None,
        metavar="SIZE",
        help="Sequence parallelism group size. Replaces TORCH_TURBO_SP_SIZE.",
    )
    parser.add_argument(
        "--tt_ep_size",
        type=int,
        default=None,
        metavar="SIZE",
        help="Expert parallelism group size. Replaces TORCH_TURBO_EP_SIZE.",
    )
    parser.add_argument(
        "--tt_mtp_loss_scaling_factor",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Multi-Token-Prediction loss weight. Overrides "
        "config.json's mtp_loss_scaling_factor. Set 0 to keep MTP forward "
        "(for logging) but exclude it from gradients. Only effective when "
        "the model config has num_nextn_predict_layers > 0.",
    )

    return parser


def parse_torchturbo_args(argv: list[str]) -> tuple[TorchTurboArgs, list[str]]:
    """Parse ``--tt_*`` arguments from *argv*, returning parsed args and
    the remaining (unconsumed) arguments.

    This uses ``parse_known_args`` so that non-torchturbo arguments are
    left untouched and can be forwarded to the training framework.

    Parameters
    ----------
    argv:
        Argument list (e.g. ``sys.argv[1:]``).

    Returns
    -------
    A tuple ``(TorchTurboArgs, remaining_argv)`` where *remaining_argv*
    contains all arguments that are not ``--tt_*`` flags.

    Example::

        tt_args, remaining = parse_torchturbo_args(sys.argv[1:])
        # remaining can be passed to LlamaFactory's run_exp()
    """
    parser = _build_parser()
    parsed, remaining = parser.parse_known_args(argv)

    args = TorchTurboArgs(
        backend=parsed.tt_backend,
        mixed_precision=parsed.tt_mixed_precision,
        sp_size=parsed.tt_sp_size,
        ep_size=parsed.tt_ep_size,
        mtp_loss_scaling_factor=parsed.tt_mtp_loss_scaling_factor,
    )

    if any(
        v is not None
        for v in [
            args.backend,
            args.mixed_precision,
            args.sp_size,
            args.ep_size,
            args.mtp_loss_scaling_factor,
        ]
    ):
        logger.info("torchturbo CLI args: %s", args)

    return args, remaining