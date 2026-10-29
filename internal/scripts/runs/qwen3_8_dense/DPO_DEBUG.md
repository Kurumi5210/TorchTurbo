# Qwen3.8 DPO alignment on Swift 4.4.2

This debug path backports the Qwen3.8 template behavior from Swift
`acb9f4cf055ce9d23ce191647c0985c9a3801939` and the last-tool loss boundary
from `cf6316ca2b827d849b0ab054e25aa10551413dcc`. The first commit is an
upstream match for the GPU log's source lines, not a confirmed hash of the
private GPU checkout. TorchTurbo and the installed Swift version remain
unchanged outside the opt-in `qwen3_8` template and `--dpo_megatron_data true`.

The original patch was based on TorchTurbo `qwen38-swift-dpo-rc1`, commit
`c9c1ad2cf9b0ad2f59fbbc566c390c13775fc4cf`. The template/data adaptation is
`6fa824b6`; the batch/split initialization fix is `91bc5c97`.
The reusable handoff branch is `feat/qwen38-dpo-data-alignment`.
This fixed experiment branch is `feat/qwen38-dpo-global176-experiment`.

## Verified scope as of 2026-10-29

This branch provides template/sampler alignment and a verifier for every
consumed microbatch. Complete GPU/NPU input equivalence has **not** passed
for the historical runs.

- The original full CPU preprocessing replay matched all 1171 retained
  training pairs, 12 validation pairs, and 983/11 train/validation packs.
  The sampler replay matched all 40 update steps.
- Actual NPU consumption matched that CPU reference for the first 5 steps:
  120 microbatches, 145 pairs, and 290 logical sequences, including input,
  unshifted label and text-position hashes. The original GPU runtime trace
  was unavailable for this check. The temporary CPU replay cache is no
  longer available; a new verification requires a new reference replay.
- A later audit of real GPU v7 and NPU training traces found different pack
  length compositions in 244 of 960 consumed packs, affecting all 40
  updates. NPU consumed 1144 pairs and GPU consumed 1143. Equal lengths in
  the remaining packs do not prove equal tokens.
- The old 10-shard RL export and the newer 18-shard `global_step_176` export
  are different: 17 of 18 selected shared tensors differed when their full
  contents were compared. This is a selected-tensor check, not a
  whole-model difference percentage. Directory names alone do not identify
  the same initialization.

The verifier detects mismatches; it does not repair pack membership or
force two different preprocessing pipelines to consume identical data.
To certify a new pair of runs, first match dataset, policy/reference
weights, tokenizer files, template settings, split/filtering, all pack
memberships and sampler settings. Capture actual consumption on both
backends and run the full comparison below without `--steps`. For the
40-step, DP2, GA12 setup, success must cover all 960 microbatches. A short
prefix pass does not establish full-run alignment. Matching inputs also
does not establish matching floating-point arithmetic or trained weights.

The template preserves historical thinking, inserts empty thinking prefixes
in historical assistant turns, and applies `last_round` after the last user
**or tool** message. Explicit per-message `loss` and `loss_scale` still take
precedence. `qwen3_5` keeps its Swift 4.4.2 behavior.

It also backports JSON tool argument literals from
`ec8d1b010e64e2d880709c58e0e2c81c76985c4c`: booleans and null are rendered
as `true`, `false`, `null`, instead of `True`, `False`, `None`. Full-data
comparison found 9 training pairs where this changed prompt tokens even
though sequence lengths and label hashes already matched. This override
is scoped to `qwen3_8`; the global `qwen3_5` agent registration is unchanged.

## Launch on NPU

Run this experiment with the repository-root `launch_experiment.sh`. It fixes the
policy and reference to the transferred 18-shard `global_step_176`, the
business DPO dataset, and the output directory. These paths, NNODES=2,
NPROC_PER_NODE=16, SP_SIZE=16 and HASH_MODEL_WEIGHTS=1 are defined in the
entry and do not depend on external parameter assignments. Change the
entry's output directory before starting a separate repeat.

The platform must provide `MASTER_ADDR` and either `NODE_RANK` or `RANK`
as the node index (0 or 1). The script delegates to the existing debug
launcher: BF16, MTP disabled, GBS24 and actual-consumption data traces.

```bash
bash launch_experiment.sh
```

Use the same command on both nodes. For a command-only check, use
`DRY_RUN=1 bash launch_experiment.sh`. Both nodes use 16 processes,
SP16 and effective DP2. Swift 4.4.2 and binpacking 1.5.2 are required.

The underlying historical debug launcher defaults to the RL checkpoint recorded in the NPU logs for
both policy and reference, and to the uploaded dataset at
`/mnt/sfs_turbo/cx02308786/business_user_flag_answer.json`. `MODEL`,
`REF_MODEL` and `DATASET` can override those defaults when invoking that
historical launcher directly. The experiment entry above sets its own paths.
`OUTPUT_DIR` defaults to `/mnt/sfs_turbo/cx02308786/ckpt/qwen38-dpo-debug`.
The GPU run used RL step 176; matching directory names or model config
does not establish matching weight shards. Set `HASH_MODEL_WEIGHTS=1` to
include every safetensors shard in the launch manifest; this reads the full
checkpoint once on node 0.

```bash
OUTPUT_DIR=/path/to/new-npu-debug-run \
NNODES=2 NPROC_PER_NODE=16 NODE_RANK=0 MASTER_ADDR=<master-host> \
bash internal/scripts/runs/qwen3_8_dense/swift_text_dpo_debug.sh
```

Run the same command on the other node with `NODE_RANK=1`. The default is
32 NPUs, SP16, effective DP2, micro batch 1, GA12, global batch 24. The
launcher reads `NODE_RANK` or the platform's `RANK`, along with `NNODES`,
`NPUS_PER_NODE`/`NPROC_PER_NODE`, `MASTER_ADDR` and `MASTER_PORT`.
Pass framework `gradient_accumulation_steps=1` together with `tt_gbs=24`;
the engine derives the actual GA12. Passing both `tt_gbs=24` and an explicit
framework GA12 fails the engine's mutual-exclusion check.
The launcher rejects a topology with a different DP size. `DRY_RUN=1` prints
the command without loading the runtime. `SWIFT_SRC_DIR`, if set, must
resolve to Swift 4.4.2. Use the existing model dependency installer.
The engine exports the final weights as Hugging Face safetensors under the
checkpoint's `huggingface/` directory; Megatron's `--save_safetensors` flag
is not accepted by this Swift 4.4.2 launcher.

The downloaded dataset has SHA256
`d9910f21d7695d28f4310ef11baae0682f8fbdcdf14e90f9cef463c2b25d87e0`.
With the GPU tokenizer and preprocessing, the earlier log showed 1203 raw
pairs, 1171 training pairs after the 1% split and filtering, 983 packs,
and 40 optimizer steps. These are comparison targets, not hard-coded
filter counts. The packing limit is 128000, not 131072. Swift packs by
chosen-length + rejected-length, but filters each sequence by max_length;
an individual pair can exceed the packing limit.

The dataset has 20 rows where a tool response is followed directly by a
user message, with no intervening assistant response. Native GPU Swift
and the backport both reject them with `response_role: "user"`. Filtering
these rows is expected; do not repair or pre-delete them on only one side.
The debug argument override preserves the 1% held-out split when
`eval_strategy=no`; native Swift 4.4.2 would otherwise silently reset it to 0.

The sampler matches Megatron's `data_sharding=false` permutation and DP
striding. Its seed is **epoch 0**, while dataset split/shuffle use seed 42.
It uses the full pack count for `randperm`, then stops at complete global
batches. For 983 packs, both backends consume 960 packs in 40 steps. This
mode deliberately supports one epoch; multi-epoch rollover is different
between the trainers. HF resume skipping is retained. A weights-only
checkpoint cannot reproduce optimizer/RNG continuation; use
`--save_only_model false` in a modified launcher if full-state resume is
needed. `MAX_STEPS` changes the LR schedule, so a shortened smoke run is
only a data-path check.

## Compare actual GPU and NPU batches

Make this checkout available on the GPU machines and add the following to
the original GPU command, retaining its native Swift and Megatron versions:

```bash
export DPO_TRACE_DIR=/path/to/new-gpu-debug-run/data-trace
# Append to megatron rlhf:
--external_plugins /path/to/TorchTurbo/internal/scripts/diagnostics/megatron_dpo_trace.py
```

The plugin loads only the shared diagnostic module, without activating
TorchTurbo on GPU. It records TP0/CP0/first PP stage for each DP rank and
keeps native sampling and preprocessing. Traces are emitted when training
consumes a batch, so dataloader worker prefetch and validation do not count
as training steps. NPU tracing likewise happens at `compute_loss`, once per
microbatch, before policy/reference forwards.

```bash
python3 internal/scripts/diagnostics/compare_dpo_traces.py \
    /path/to/gpu/data-trace /path/to/npu/data-trace
# Compare only the first three complete optimizer steps:
python3 internal/scripts/diagnostics/compare_dpo_traces.py \
    /path/to/gpu/data-trace /path/to/npu/data-trace --steps 3
```

`dataset.json` records all pack membership and lengths, including validation.
The JSONL files record `(step, DP rank, micro step)`, pack indices, pair
counts, SHA256 of encoded and collated input IDs/labels/text positions, and
supervised token counts. Steps start at 1; ranks and micro steps start at 0.
Chosen sequences precede rejected sequences. Hashes compare logical text
tokens before the one-token label shift, excluding GPU CP padding and NPU
SP tail padding. Traces do not dump the source text.

The comparison fails on missing DP ranks, incomplete steps, duplicate
resume records, changed pack membership, or the first differing token/label
hash. It also checks that collation preserves each logical sequence. Both
traces stopping early is not a full-run pass; select a complete prefix with
`--steps` if intentional. Resume files start at the completed optimizer step;
use a new trace directory when restarting at the same step. These traces
verify data alignment, not attention kernels, gradients or model quality.

## Local verification

CPU replay used the downloaded dataset and the tokenizer from the GPU model,
with PyTorch 2.10.0, Transformers 5.9.0, datasets 5.0.1 and binpacking 1.5.2.
The reference was native Swift `acb9f4cf`; the comparison used patched Swift
4.4.2. Neither replay loaded model weights or performed a training forward.

- All 1171 retained training pairs and 12 validation pairs matched exactly
  in chosen/rejected token hashes, unshifted label hashes, lengths and
  supervised token counts. All 983 training and 11 validation pack
  memberships and packed lengths also matched.
- The real GPU log sample (raw row 558, zero-based) matched token for token:
  chosen length 77127 with 1877 supervised tokens; rejected length 77111
  with 1861 supervised tokens. Its labels matched the logged arrays too.
- Both DP ranks matched the native Megatron sampler for all 40 optimizer
  steps and for resumes after steps 1, 17 and 39.
- Native CPU collators were compared on the logged long pair and a
  two-pair batch. After removing GPU CP/FP8 padding, all logical inputs,
  labels, text positions and three mRoPE axes matched. This used the actual
  Hugging Face `Qwen3_5Model.get_rope_index`, without a model forward.

Run the adapter and launcher checks in a Swift 4.4.2 environment:

```bash
PYTHONPATH=src python -m pytest -q tests/adapters/msswift \
    tests/scripts/test_dpo_scripts.py tests/scripts/test_qwen38_dpo_debug.py
bash -n internal/scripts/runs/qwen3_8_dense/swift_text_dpo_debug.sh
```

These checks include the actual Swift argument parser, scoped template
behavior, resume/step trace validation, and trainer/plugin integration.
The regression checks also exercise real argument initialization, dataset
splitting with evaluation disabled, and the engine's GBS/GA contract.
The original private GPU checkout hash remains unverified. The later
weight comparison and runtime data audit are summarized in the verified
scope above. Distributed numerical equivalence and business score recovery
still require matched hardware runs and the same evaluation set.

## Parameter mapping and remaining differences

| GPU Megatron | NPU debug path |
| --- | --- |
| model_type qwen3_5, template qwen3_8, agent qwen3_5 | turbo_qwen3_5, backported qwen3_8, agent qwen3_5 |
| sigmoid, beta 0.1, RPO 0.1, smoothing 0.05 | Same |
| split 0.01, delete overlength, length 128000, binpack | Same; cache disabled, packing_num_proc=1 |
| DP2, micro1, global24 | DP2, micro1, GA12, global24 |
| Adam beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.1 | Explicitly set; these are GPU upstream defaults |
| cosine LR 1e-6 to 1e-7, warmup 0.05, clip 1.0 | cosine_with_min_lr, same values |
| Full layer recompute, uniform one layer | TorchTurbo layer activation checkpointing |
| Freeze ViT/aligner, train LLM | Same; both Swift and engine freeze flags supplied |
| Epoch save, no optimizer/RNG save | Epoch save, save_only_model=true |
| End-of-epoch validation | Disabled: this TorchTurbo adapter rejects evaluation; split retained |
| FP8 blockwise e4m3, precision-aware optimizer | BF16 compute; FP8 arithmetic cannot be reproduced |
| TP4 / PP4 / CP4 on 128 GPUs | FSDP2 / SP16 on 32 NPUs |
| MTP layer 1, MTP loss 0, save_missing_weights | MTP disabled; exported checkpoint contents may differ |
| linear_decoupled_in_proj, dp_balanced packing scheduler | No direct engine flag; trace verifies resulting logical batches |

Changing the template and supervised spans can change both context and RPO
loss. Matching these hashes does not establish GPU/NPU numerical equality
or prove recovery of the business evaluation score. Confirm weight hashes
and run the same evaluation after the data comparison succeeds.
