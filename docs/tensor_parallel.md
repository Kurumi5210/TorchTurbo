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

Dense TP and expert TP can be configured independently with the existing
`parallel_plan.tp_size` and `parallel_plan.etp_size` settings. For example, on
four ranks, this configuration partitions dense computation over two ranks and
keeps each routed expert's features complete:

```yaml
engine_config:
  parallel_plan:
    sp_size: 1
    ep_size: 2
    tp_size: 2
    etp_size: 1
```

Setting `tp_size: 1` and `etp_size: 2` instead partitions routed expert features
without sharding dense layers, embeddings, or the LM head. The world size must
be divisible by both `sp_size * tp_size` and `ep_size * etp_size`. Attention and
shared-expert widths follow TP; routed-expert feature widths follow ETP. The
existing model validators still check attention heads, widths, and SP
compatibility. TP or ETP currently excludes PEFT, CPU/optimizer offload, Muon, nonzero
dropout, MTP/DSpark, diffusion models, and cached generation.

When the two degrees differ, the routed branch transfers unique token pieces
from the dense TP layout to the expert TP layout before EP dispatch, then
restores the original token order after EP combine. Hidden states, routing
probabilities, and expert selections move together. The router, its auxiliary
losses and statistics, and shared experts retain the original dense token
layout. Unequal token counts and empty ownership pieces are supported;
communication padding is removed before expert dispatch. Equal TP/ETP degrees
retain the existing path without this transfer.

Data sampling has `world_size / (sp_size * tp_size)` independent replicas,
regardless of ETP. Swift and LLaMA-Factory SFT/PT require an explicit `tt_gbs`
when either degree exceeds one and count the exact valid targets across each
optimizer step. Dense and expert gradients use the same dense-domain divisor
`world_size / tp_size`; callers must not add an ETP or EP compensation factor.

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
supports tied weights under TP. With `tp_size = 1`, the tied-weight chunked-loss
restriction still applies, including ETP-only training. When both TP and ETP are
one, the existing loss normalization is unchanged.

Native checkpoints record parameter layouts. Checkpoints from the earlier TP
plans, which kept vocabulary weights complete on each TP rank, have incompatible
embedding/head layouts even when the TP degree is unchanged. To migrate weights,
load the checkpoint with its original code and layout, export Hugging Face
weights, and initialize the new vocabulary-TP run from that export. Hugging Face
weights preserve the logical vocabulary tensors; this migration starts a new
optimizer and does not transfer the previous optimizer state.

Independent TP/ETP retains packed 2D expert weights when ETP is one and logical
feature-sharded expert weights when ETP exceeds one. Native model and optimizer
restore requires the same saved topology and layouts; changing the degrees
requires exporting and reinitializing from Hugging Face weights.

Vocabulary TP partitions stored parameters and their optimizer state. PyTorch
2.10's native row-sharded embedding backward can still materialize a full
`[vocabulary, hidden]` gradient before redistributing it to the parameter layout.
Embedding backward peak memory therefore does not necessarily decrease in
proportion to `tp_size`.
