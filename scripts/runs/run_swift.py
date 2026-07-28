#!/usr/bin/env python3
"""turbo launch entry for ``swift sft`` — import the msswift plugin FIRST.

Why this wrapper exists (and why ``--external_plugins`` alone is NOT enough):
    swift loads ``--external_plugins`` inside ``BaseArguments.__post_init__`` —
    i.e. *during* the ``SwiftSft`` arg parse that ``sft_main`` already started.
    By then ``sft_main`` has already resolved the module-global ``SwiftSft``
    and locked its ``args_class``, so the plugin's ``SwiftSft -> TurboSwiftSft``
    swap is too late for THIS run: the turbo engine never gets built.
    Importing the plugin HERE swaps that module global *before* ``sft_main()``
    reads it, so ``sft_main`` constructs ``TurboSwiftSft``
    (``args_class = TurboSftArguments``) and the turbo engine is built up front.
    Registering turbo models/trainers into swift is a side effect of the same
    import (``MSSwiftAdapter.activate`` is idempotent), so the script does NOT
    pass ``--external_plugins``.

Launch — mirrors swift's own ``cli_main`` re-exec, just swapping the worker file
from ``swift/cli/sft.py`` to this module::

    torchrun --nproc_per_node ${NPROC_PER_NODE} --nnodes ${NNODES} \\
             --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} \\
             --master_port ${MASTER_PORT} run_swift.py \\
             --model_type turbo_qwen3_5_moe --model <path> --template qwen3_5 \\
             --packing true --attn_impl flash_attn \\
             --tt_sp_size 2 --tt_ep_size 4 --tt_freeze_vision_tower true \\
             --tuner_type full --dataset <swift_dataset> --output_dir ./out \\
             --bf16 true
"""
from __future__ import annotations

import os

if __name__ == '__main__':
    # Match swift/cli/sft.py's worker preamble for device/dist-init parity.
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()

    # Activate turbo AND swap SwiftSft -> TurboSwiftSft, *before* sft_main reads
    # the module global.
    import torchturbo.adapters.msswift.plugin  # noqa: F401

    # Strip the turbo ``--tt_*`` CLI flags from sys.argv BEFORE swift's parser
    # runs (sft_main reads sys.argv[1:] directly). ``TurboSftArguments`` declares
    # NO ``tt_*`` fields — the single source of truth is ``TorchTurboArgs`` in
    # engine/args.py, shared with the llamafactory adapter — so any ``--tt_*``
    # left in argv would hit swift's ``remaining_argv`` rejection. Parse them
    # here, stash via set_torchturbo_args, and leave swift a clean argv.
    import sys
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args
    _tt_args, _remaining = parse_torchturbo_args(sys.argv[1:])
    set_torchturbo_args(_tt_args)
    sys.argv = [sys.argv[0]] + _remaining

    from swift.pipelines import sft_main
    sft_main()
