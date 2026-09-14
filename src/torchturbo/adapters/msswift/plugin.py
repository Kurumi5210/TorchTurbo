"""swift ``--external_plugins`` entry: activates the msswift adapter (turbo
models/trainers into swift's registries) and swaps each registered stage's
swift pipeline global for its turbo pipeline (via ``stages.stage_wiring``),
so the native ``<stage>_main`` builds the turbo pipeline. Must run before
the main resolves its pipeline global.
"""

from __future__ import annotations

import importlib

import torchturbo  # noqa: F401  (ensure the package + adapter are importable)

from torchturbo.adapters.msswift.stages import iter_stages, stage_wiring

# Explicit activation — ``import torchturbo`` no longer activates anything.
# This plugin is the swift opt-in: importing it from the launcher (or loading
# it via ``--external_plugins``) activates the msswift adapter deterministically,
# mirroring how ``run_lf.py`` calls ``torchturbo.activate("llamafactory_v0")``.
torchturbo.activate("msswift")  # idempotent

# Swap every registered stage's pipeline module global to the turbo pipeline,
# so swift's native ``<stage>_main`` (which closes over the module global)
# builds the turbo takeover pipeline. A stage whose swift submodule is missing
# (e.g. an RLHF module not present in this swift build) is skipped, never
# breaking the stages that ARE present.
for _spec in iter_stages():
    try:
        _mod = importlib.import_module(_spec.pipeline_module)
    except Exception:
        continue
    setattr(_mod, _spec.pipeline_attr, stage_wiring(_spec.name)[0])