# antllm pre-pack + read (v3 plugin API)

One `--tt_plugin` entry point, `antllm_pack_plugin.py` (`antllm_prepack`), does the
whole flow in a single run: read an mdata dataset, run antllm
`SFTPrePackProcesserV2` into `save_path` (re-packed every run, no cache),
then **delegate reading** to `antllm_prepacked_reader.py` — a library module
(not a `--tt_plugin` entry of its own) that memmaps the pack and emits the engine's
packed-input contract.

`pack_compare.py` is an offline pack-comparison utility; not loaded as a plugin.

## Usage

```
--tt_plugin <repo>/internal/dataset/antllm_pack/antllm_pack_plugin.py:<pack.yaml>
--dataset <save_path>          # must equal dataset.args.save_path in pack.yaml
```

Run script: `internal/scripts/runs/bailing_v3/swift_tiny_sft_muon_mdata.sh`. Each
run re-packs into `save_path` (no cache; multi-node: each node's `LOCAL_RANK==0`
packs its share), then reads.

If no `--tt_plugin` is passed, the run falls back to swift's native dataset path
(the plugin bridge returns `None` and the trainer uses `SwiftInputCollator`).

## Pack config (`pack.yaml`)
```yaml
dataset:
  type: antllm.data.sft_pre_pack.sft_packed_dataset.SFTPackedDataset
  args:
    data_path: <raw or mdata://...>
    save_path: <antllm pre-pack output dir: holds meta.json>
  pack_args:
    antllm_path: <parent dir of the antllm package you cloned/installed>
    tokenizer_path: /workspace/bin/antllm-resource/bailing_tokenizer_v3/config_ling_adaptive
    max_length: 131072
    rotary_type: full-1d
    use_compatible_mode: true
    use_true_multiturn: true
    skip_abnormal_data: true
    max_pack_sample_num: 300
    think_enable_datasets: [ ... ]   # you maintain this; the plugin reads it verbatim
```

`antllm_path` is added to `sys.path` so `import antllm.*` resolves; set it
manually (like `tokenizer_path`), no runtime git clone.

## What the reader module does
`antllm_prepacked_reader.py` (imported by the pack plugin's `build_dataset`):
- Reads the pack directory (`meta.json` + `pack_data/pack_meta.json` + per-shard
  `meta.pkl` slice index over memmapped `data.<dtype>.npy` files). Pure memmap, no
  antllm import at read time; shard open/load is retried (240s) to tolerate
  cross-node shared-FS propagation lag (mirrors antllm's `retry_func_until_timeout`).
- Recovers per-dataset sampling ratios from the `data_path` the pack recorded in
  `meta.json` (mounted via the internal `mdatasets` package), re-keyed the way
  antllm's `SFTConcatSampledDataset` reads them. Raises if the mdata snapshot does
  not match the packed directories (catching a wrong-mixture run antllm would only
  log).
- Expands each dataset directory through antllm's index sampler (whole repeats
  plus a head/tail/random fraction) and concatenates them into one global index
  space.
- Rewrites each packed row into the engine's packed-input contract: `input_ids`,
  `labels` (HF-aligned), `position_ids` (zero-reset per sub-sequence),
  `cu_seq_lens_q`/`cu_seq_lens_k` (int32), no `attention_mask`. `--per_device_train_batch_size`
  must be 1 (one packed row per step); `--tt_gbs` for a larger global batch.
- `collate_prepacked` is served by the pack plugin's `get_collate_fn()` override;
  it builds the single-row packed batch and runs `assert_packed_inputs`.

A standalone self-check (no training):
```
python internal/dataset/antllm_pack/antllm_prepacked_reader.py <save_path>
```

## Collator hook
The pack plugin overrides `get_collate_fn()` to return `collate_prepacked`; the
pipeline (`TurboEnginePipelineMixin._get_trainer_kwargs`) resolves it via the
bridge's `collator_for(args)` and hands it to the trainer as the `plugin_collator`
kwarg, which `TurboTrainerMixin._get_data_collator` returns. Pack-grouped and
no-plugin runs fall back to `SwiftInputCollator`.

## Label convention (v3)
The FSDP2 backend applies the left-shift to `labels` itself, once, globally, in
`FSDP2Backend._prepare_loss_labels`, and does **not** mask per `cu_seq_lens`
boundaries (attention is block-diagonal via `cu_seq_lens`). The reader therefore
emits **HF-aligned labels** (`labels.shape == input_ids.shape`, undoing antllm's
Megatron pre-shift) with **-100 at the first position of every sub-sequence**, so
the single cross-boundary pair the global shift creates is masked.
