#!/usr/bin/env python
"""Launch LlamaFactory training with torchturbo from the torchturbo repo.

All arguments are passed directly to LlamaFactory's run_exp.
torchturbo is always imported; use TORCH_TURBO_DISABLE=1 to run vanilla LF.

Usage:
    torchrun --nproc_per_node=8 scripts/runs/run_lf.py \
        --stage sft \
        --model_name_or_path /path/to/model \
        --dataset your_dataset \
        --template qwen3 \
        --output_dir ./output/my_run
"""

from __future__ import annotations

import os
import sys


# Ensure torchturbo is importable when running without pip install -e
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def auto_detect_dataset_dir() -> str | None:
    """Auto-detect LlamaFactory's data/ directory from the package location."""
    try:
        import llamafactory
    except ImportError:
        return None
    pkg_dir = os.path.dirname(llamafactory.__file__)
    for _ in range(4):
        pkg_dir = os.path.dirname(pkg_dir)
        candidate = os.path.join(pkg_dir, "data")
        if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
            return candidate
    return None


def main():
    import torchturbo  # noqa: F401

    # Auto-inject dataset_dir if not explicitly provided
    argv = sys.argv[1:]
    if "--dataset_dir" not in argv:
        dataset_dir = auto_detect_dataset_dir()
        if dataset_dir is not None:
            argv.extend(["--dataset_dir", dataset_dir])

    from llamafactory.train.tuner import run_exp
    run_exp(args=argv)


if __name__ == "__main__":
    main()
