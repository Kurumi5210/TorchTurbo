# Tensor parallelism

TorchTurbo applies model-authored PyTorch tensor-parallel styles before FSDP2
sharding. TP requires PyTorch 2.10 or newer and an initialized distributed process
group. Use the existing `parallel_plan.tp_size` setting, `--tt_tp_size`, or
`TORCH_TURBO_TP_SIZE`; vocabulary sharding has no separate switch.

The following default plans shard both the input embedding and LM head when
`tp_size > 1`:

| Model plan | Input embedding |
| --- | --- |
| `qwen3_moe` | `model.embed_tokens` |
| `qwen3_5_moe` | `model.language_model.embed_tokens` |
| `qwen3_5_moe_text` | `model.embed_tokens` |
| `bailing_hybrid` | `model.word_embeddings` |
| `bailing_moe_v3_vl` | `model.word_embeddings` |

These routed-expert plans require aligned `tp_size = etp_size > 1`. For example,
on four ranks:

```yaml
engine_config:
  parallel_plan:
    sp_size: 1
    ep_size: 1
    tp_size: 2
    etp_size: 2
```

The existing model validators still check attention heads, expert widths, and SP
compatibility. TP currently excludes PEFT, CPU/optimizer offload, Muon, nonzero
dropout, MTP/DSpark, diffusion models, and cached generation.

Input embedding weights use native `RowwiseParallel`: each TP rank owns vocabulary
rows, and embedding outputs contain complete hidden states on every TP rank. The
LM head uses `ColwiseParallel`, retaining logits as a DTensor sharded on the
vocabulary dimension. Token IDs and labels remain global vocabulary indices.
Uneven vocabulary partitions are supported when every TP rank has at least one
row. Tied embedding/head weights retain one shared parameter, and frozen weights
remain frozen. FSDP further shards the TP-local parameters over its dense domain.

Regular and chunked LM losses use PyTorch `loss_parallel` for vocabulary-sharded
cross entropy. Both return ordinary scalar losses; callers can use
`loss.backward()` without an additional context manager. The existing exact
optimizer-step target count and loss normalization still apply. Set
`loss_chunk.enabled: true` or `--tt_enable_chunk_loss true` to compute local logits
in token chunks; `loss_chunk.chunk_size` counts flattened tokens. Chunked loss
supports tied weights under TP. With `tp_size = 1`, the existing loss behavior,
including the tied-weight chunked-loss restriction, is unchanged.

Native checkpoints record parameter layouts. Checkpoints from the earlier TP
plans, which kept vocabulary weights complete on each TP rank, have incompatible
embedding/head layouts even when the TP degree is unchanged. To migrate weights,
load the checkpoint with its original code and layout, export Hugging Face
weights, and initialize the new vocabulary-TP run from that export. Hugging Face
weights preserve the logical vocabulary tensors; this migration starts a new
optimizer and does not transfer the previous optimizer state.

Vocabulary TP partitions stored parameters and their optimizer state. PyTorch
2.10's native row-sharded embedding backward can still materialize a full
`[vocabulary, hidden]` gradient before redistributing it to the parameter layout.
Embedding backward peak memory therefore does not necessarily decrease in
proportion to `tp_size`.
