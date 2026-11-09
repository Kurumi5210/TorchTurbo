# Ling tau2 full and LoRA SFT MFU runs

These entrypoints run the same tau2 recipe in full and attention-LoRA modes.
The recipe was exercised on 32 Ascend NPU processes with TorchTurbo
`d3e214e170ca92dee5a0a694a811eed31053e525`; both modes completed 78 steps.
The scripts use this checkout's `src` and `scripts/runs/run_swift.py` directly.

## Launch

Run from the checkout root on every node. Use the existing dependency installer
in the same Python environment as training. It includes `binpacking==1.5.2`
and the validated NumPy / PyArrow / datasets versions `1.26.4 / 21.0.0 / 4.8.4`.
The base image supplies PyTorch, torch_npu, CANN, FLA and Triton Ascend.

For full SFT:

```python
command += (
    'bash scripts/runs/bailing_v3/install_swift_deps.sh && '
    'bash scripts/runs/bailing_v3/tau2_mfu/run_full.sh '
)
```

For a separate LoRA job:

```python
command += (
    'bash scripts/runs/bailing_v3/install_swift_deps.sh && '
    'bash scripts/runs/bailing_v3/tau2_mfu/run_lora.sh '
)
```

The existing job configuration supplies these environment variables:

- `MODEL`: required local Ling checkpoint directory, including tokenizer files.
- `DATA_DIR`: directory containing `train.jsonl` and `valid.jsonl`; defaults to
  `data/tau2` under the checkout. `TRAIN_FILE` and `VALID_FILE` can override the
  two input files independently.
- `NNODES`, `NPROC_PER_NODE` (or `NPUS_PER_NODE`), `NODE_RANK` (or `RANK`),
  and `MASTER_ADDR` (or `MASTER_IP`). The measured recipe uses 2 nodes with
  16 processes each. `MASTER_PORT` is also inherited when supplied.
- `MAX_STEPS`, `NUM_TRAIN_EPOCHS` (or `EPOCHS`), `LEARNING_RATE`, `SEQ_LEN`,
  `SEED`, `TT_CONFIG`, `OUTPUT_DIR`, `SAVE_STRATEGY`, and `DRY_RUN` are optional.
  Defaults are 3 epochs, no step cap, and no checkpoint saving.

`TT_REPO` optionally overrides the checkout root. `PYTHON` defaults to `python3`;
the existing install chain uses `pip` and `python3`, so these must refer to the
training environment. No training packages are installed by `run_sft.sh` itself.

## Shared recipe

`shared.yaml` sets FSDP2, BF16, SP8, EP8, GBS4, optimizer CPU offload, reentrant
activation checkpointing, and chunked loss with chunk size 1024. The launcher
uses micro batch size 1, gradient accumulation 1, AdamW at constant LR 1e-4,
zero warmup and weight decay, betas 0.9/0.999, epsilon 1e-8, and gradient clip 1.
Sequence length is 32768 and seed is 42. MTP and MoE auxiliary/z losses are off.

`ops.overrides.kda: fla_kda` selects FLA/Triton KDA through the existing operator
dispatcher. MFU is enabled for `910c` with the repository's original formula.

Full SFT sets `--tuner_type full`. LoRA sets `--tuner_type lora`, rank 8,
alpha 16, dropout 0, with RSLoRA and DoRA disabled. Its target modules are
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `kv_a_proj_with_mqa`, and `kv_b_proj`.

## Data contract

Provide 421 training and 50 validation cases as JSONL. Each case has a unique
string `id`, a nonempty `prompt` message list ending in a user or tool message,
and a `completion` list containing exactly one assistant message. `tools` is
optional. The completion must contain text or tool calls.

`prepare_hxb_data.py` renders the checkpoint's chat template with thinking
disabled, masks the complete prompt, and supervises only the final completion.
It validates the Ling architecture, attention adapter inventory, sequence
lengths and disjoint training/validation IDs before loading model weights.

The pack plan consumes each training case once. It splits the final packs at
case boundaries to fill the last global batch without repeating examples.
The reference data produces 104 packs and 26 steps per epoch. The plugin reads
the prepared tokens and labels directly, resets positions at each case boundary,
and supplies isolated packed-sequence metadata without Swift re-tokenization.
This fixes the data semantics for comparison; it is not a requirement of the
underlying full or LoRA training engine.

## Outputs and MFU interpretation

Default outputs are `outputs/tau2_mfu_full` and `outputs/tau2_mfu_lora`.
Each node writes its bootstrap log, training log, command and prepared manifest.
Swift creates a versioned run directory with `logging.jsonl` inside it.
Each worker uses a separate temporary Triton/Inductor/extension cache, so initial
compilation contributes to startup and the first step.

Compare the same step window and exclude the initialization step. Logged MFU
uses the existing `forward FLOPs * 3` estimate for both modes; LoRA FLOPs are
not corrected. Keep that distinction when reporting the comparison.

Training data, model files, generated manifests, logs and outputs are supplied
or generated at runtime and are not part of this recipe.
