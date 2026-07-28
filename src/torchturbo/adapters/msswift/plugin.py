"""swift ``--external_plugins`` entry — activate turbo takeover for a native run.

Imported by swift during arg parsing (``BaseArguments._import_external_plugins``),
before ``sft_main`` runs and before any model loads. It:

1. Activates :class:`MSSwiftAdapter` — registers the turbo models (via
   :class:`TurboModelLoader`) into ``MODEL_MAPPING`` and the turbo trainers
   (per stage spec) into ``TrainerFactory.TRAINER_MAPPING``.
2. Swaps each registered stage's swift pipeline global for its turbo pipeline
   (SFT: ``swift.pipelines.train.sft.SwiftSft`` → :class:`TurboSwiftSft`), so
   the native ``sft_main`` (closed over that module global) builds the turbo
   pipeline — engine up front, ``--tt_*`` parsed natively via
   ``TurboSftArguments``.

Adding a stage is local to :mod:`stages`; this loop picks it up unchanged.

Run with native swift CLI (swift owns the distributed launch)::

    NPROC_PER_NODE=8 swift sft \
        --external_plugins /path/to/this/plugin.py \
        --model_type turbo_qwen3_5_moe --model <path> --template qwen3_5 \
        --tt_sp_size 4 --tt_ep_size 8 \
        --tuner_type full --packing true --max_length 32768 \
        --dataset <swift_dataset> --output_dir ./output --bf16 true

Only ``--tt_*`` is turbo-owned; everything else is plain swift.
"""

from __future__ import annotations

import importlib

import torchturbo  # noqa: F401  (ensure the package + adapter are importable)

from torchturbo.adapters.msswift.stages import iter_stages

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
    setattr(_mod, _spec.pipeline_attr, _spec.turbo_pipeline_factory())