#!/usr/bin/env python3
"""turbo launch entry for native swift stages: ``run_swift.py [sft|pt] ...``.
The leading positional mirrors swift's subcommand (omitted = sft). The
plugin is imported BEFORE the stage's ``main()`` (``--external_plugins``
fires too late), then the stage's native main() dispatches.
"""
from __future__ import annotations

import sys

if __name__ == '__main__':
    # Match swift's cli worker preamble for device/dist-init parity.
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()

    # Optional leading positional stage token (swift's subcommand form):
    # ``run_swift.py pt --model ...``. Anything else (a flag) means sft.
    from torchturbo.adapters.msswift.stages import iter_stages, stage_wiring

    argv = sys.argv[1:]
    stages = {spec.name: spec for spec in iter_stages()}
    stage = argv[0] if argv and argv[0] in stages else "sft"
    if argv and argv[0] == stage:
        argv = argv[1:]

    # Shared tt parser strips all --tt_*; swift sees only its own args.
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    tt_args, swift_argv = parse_torchturbo_args(argv)
    set_torchturbo_args(tt_args)
    sys.argv = [sys.argv[0]] + swift_argv

    main = stage_wiring(stage)[1]

    # Import = activation + pipeline-global swap; must run before the main
    # reads its pipeline global (external_plugins fires too late; see docstring).
    import torchturbo.adapters.msswift.plugin  # noqa: F401  (import = activation)

    main()
