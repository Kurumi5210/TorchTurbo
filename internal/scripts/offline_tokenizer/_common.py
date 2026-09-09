"""Shared helpers for the offline tokenization tools."""

from __future__ import annotations

import os
import sys


def ensure_repo_src() -> None:
    """Make ``torchturbo`` importable when running without ``pip install -e``."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_src = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


DISTRIBUTED_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
    "RDZV_ID",
)


def strip_distributed_env() -> list[str]:
    """Remove launcher-injected distributed env vars; return what was removed."""
    removed = [v for v in DISTRIBUTED_ENV_VARS if os.environ.pop(v, None) is not None]
    return removed


def pop_flag(argv: list[str], name: str) -> tuple[list[str], bool]:
    """Remove ``--name`` (a boolean flag) from argv; return (argv, present)."""
    present = name in argv
    return [a for a in argv if a != name], present


def pop_value(argv: list[str], name: str) -> tuple[list[str], str | None]:
    """Remove ``--name <value>`` / ``--name=<value>`` from argv."""
    value, out, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == name and i + 1 < len(argv):
            value, i = argv[i + 1], i + 2
            continue
        if a.startswith(name + "="):
            value, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    return out, value


#: ms-swift args that change the encoded/tokenized output and therefore must
#: match between export and training. Checked by ``verify_tokenized.py``.
#: NOTE: packing is deliberately absent — swift's export rejects ``--packing``
#: ("handled during training"), and ``PackingDataset`` wraps the cached rows
#: at train time anyway.
SWIFT_CONTRACT_ARGS = (
    "template",
    "max_length",
    "truncation_strategy",
    "loss_scale",
    "padding_free",
    "response_prefix",
    "template_backend",
    "dataset_shuffle",
)

#: Name of the contract file written next to swift cached-dataset artifacts.
SWIFT_CONTRACT_FILENAME = "torchturbo_tokenize_args.json"
