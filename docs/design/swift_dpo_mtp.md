# Multi-Token Prediction in Swift DPO

How MTP was wired into the ms-swift DPO path for Qwen3.5 Dense and Qwen3.8
Dense 27B. Both checkpoints share the `qwen3_5` architecture, so one
implementation covers them; they differ only in the conversation template
(`qwen3_5` vs `qwen3_8`) and the weights.

## The problem

MTP and Swift DPO both already worked, but could not be combined. The
configuration layer rejected the pair outright:

```python
if rlhf_type == "dpo":
    ...
    tt = get_torchturbo_args()
    if tt is not None and getattr(tt, "enable_mtp", False):
        unsupported.append("DPO with MTP")
```

That check was not the obstacle itself, only a marker for one. The real
obstacle is where the two code paths live.

The engine computes the MTP loss inside the FSDP2 backend's `compute_loss`.
The SFT trainer reaches it; the DPO trainer does not, because it resolves
straight past the engine to Swift's own loss assembly:

```python
# adapters/msswift/trainer.py:490
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Resolve directly to Swift's DPO loss implementation.
    return super(TurboTrainerMixin, self).compute_loss(...)
```

`super(TurboTrainerMixin, self)` skips the Turbo mixin in the MRO, so the
backend's loss code never runs under DPO. The model still *computes* its MTP
hidden states on every training forward, but nothing turns them into a loss.
So MTP had to be re-attached inside the DPO forward.

ms-swift offers nothing to copy here. Its HF training paths, both SFT and DPO,
have no MTP support at all; MTP exists only on its Megatron path, where the
arguments are passed through to mcore and `MegatronDPOTrainer.loss_func`
ignores them. The implementation is TorchTurbo's own.

## Design decisions

**MTP supervises the chosen completions only.** It is an auxiliary
language-modelling objective, so supervising the rejected segments would raise
their likelihood and pull against the preference objective. This matches the
policy NLL term (`rpo_alpha`), which is also chosen-only.

**MTP is an auxiliary objective added on top of DPO,** not a term inside the
sigmoid preference loss. The reference model does not run it.

**The MTP streams go through the chunked kernel,** always. The vocabulary is
roughly 248k and a DPO step already carries `2N` sequences, so a plain
projection is not affordable.

## Memory: shadow the LM head

The model projects every MTP stage over the whole sequence, unconditionally:

```python
# models/qwen3_5/modeling_qwen3_5.py:2076
mtp_logits = [self.lm_head(h) for h in mtp_hidden_states]
```

The backend's chunked SFT path already solves this: it temporarily replaces
`lm_head` with `nn.Identity()`, so the model hands back hidden states where
logits would be (`backend.py:1115`). The shadow is written onto the parent
module's attribute with `object.__setattr__`, which leaves the head's FSDP
module registration intact.

The DPO forward does the same (`trainer.py:746`), and the effect on each
output is:

| Output | With the head shadowed |
| --- | --- |
| `outputs.logits` | backbone hidden states, already cut down by the `logits_to_keep` mask |
| `outputs.mtp_logits` | full-length MTP hidden states, one per stage |

The caller then projects the backbone's kept positions itself, with the
all-gathered weight from `engine.backend._prepare_chunk_loss_lm_head`
(`backend.py:1134`):

```python
outputs = self._forward_with_shadowed_lm_head(model, inputs)
# The head was bypassed, so project the kept positions here instead.
lm_head_weight, main_hidden_states, loss_anchor = (
    engine.backend._prepare_chunk_loss_lm_head(model, outputs.logits)
)
main_logits = F.linear(main_hidden_states.to(lm_head_weight.dtype), lm_head_weight)
```

Because `logits_to_keep` already narrowed the hidden states inside the model,
this projection is exactly the work the in-model one would have done: the
backbone's cost is unchanged. The MTP streams go to `chunk_loss`, which never
materialises a `[tokens, vocab]` tensor.

`modeling_qwen3_5.py` needed no changes.

The head is shadowed on the reference pass too. This trainer keeps the
reference model out of accelerate's hands, so it stays in training mode and
reaches the model's MTP block; shadowing bounds that pass's memory as well,
while only the policy pass turns the result into a loss.

## Labels: generalise the shift

Packed DPO arrives as one flattened row holding `2 * N` segments delimited by
`cu_seq_lens_q`: the `N` chosen samples, then their rejected counterparts. The
existing helper shifted labels one position, inside each segment:

```python
def _shift_packed_labels(labels, segment_ids, pad_token_id, shift=1):
    """Shift labels ``shift`` positions ahead, only inside each segment."""
    next_labels = F.pad(labels, (0, shift), value=pad_token_id)[..., shift:]
    segment = segment_ids.to(torch.long)
    same_segment = segment == F.pad(segment, (0, shift), value=0)[..., shift:]
    valid = segment.ne(0) & same_segment
    return torch.where(valid, next_labels, torch.full_like(labels, pad_token_id))
```

A `shift` parameter was enough: the backbone predicts the next token
(`shift=1`), and MTP stage `i` predicts `i + 2` positions ahead. Targets that
would cross a segment boundary are masked, matching the backend's
`_prepare_loss_labels`.

`_mtp_stage_labels` (`trainer.py:722`) builds one target row and one
normalizer per stage on top of that, and masks every rejected segment via
`segment_ids > num_examples`.

Two ordering constraints, both easy to get wrong:

1. Stage labels are built from the **unshifted** row, before the backbone's own
   shift overwrites it.
2. Normalizers are counted over the **whole** row, before sequence-parallel
   slicing, so every rank divides by the same number. The labels are sharded
   afterwards, alongside the other loss-side tensors.

## Loss hookup: an autograd fake node

The repository's convention is that an auxiliary loss must not change the
reported loss value. `tests/adapters/msswift/test_dpo_moe_loss.py:155` pins
this down for the MoE router losses with `actual_loss == main_loss`: the
gradient is injected through an autograd "fake node" and the value is reported
separately as a metric.

MTP follows the same pattern, which also avoids overriding Swift's ~90-line
`get_batch_loss_metrics`:

```python
class _MTPAuxLossAutoScaler(torch.autograd.Function):
    """Add the MTP gradient to the graph, as ``MoEAuxLossAutoScaler`` does."""

    @staticmethod
    def forward(ctx, output, mtp_loss, loss_scale):
        ctx.save_for_backward(mtp_loss)
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mtp_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(mtp_loss) * ctx.loss_scale, None
```

`forward` is the identity, so the DPO outputs are bit-for-bit what they were
without MTP. `backward` seeds the MTP loss's gradient, which pulls its whole
subgraph into the same backward pass.

The node is applied to `per_token_logps`, a tensor the DPO loss already
depends on:

```python
per_token_logps = _MTPAuxLossAutoScaler.apply(
    per_token_logps, mtp_loss, 1.0 / gradient_accumulation_steps
)
```

Unlike `MoEAuxLossAutoScaler`, the scale is passed per call rather than held
on the class, because it folds in the gradient-accumulation divisor that the
engine sets globally for the router losses.

### Where the two scale factors are applied

`mtp.loss_scaling_factor` is handed to `chunk_loss` rather than multiplied in
afterwards. At zero the kernel detaches the streams, so the diagnostics mode
(`loss_scaling_factor: 0`, metrics only) does not unshard the MTP modules for a
backward that contributes nothing. The returned per-stage losses stay
unscaled, which is what the backend reports on the SFT path.

The gradient-accumulation divisor is the only factor left for the autoscaler.

## Reporting

The DPO path never runs `engine.compute_loss`, so the engine's own MTP
reporting never fires. `_record_mtp_metrics` (`trainer.py:810`) produces the
same keys through `engine.metric_tracker`:

| Metric | Value |
| --- | --- |
| `mtp_loss/{i}` | stage `i` cross-entropy, unscaled |
| `mtp_loss/avg` | mean over stages, unscaled |
| `mtp_loss/scaled` | sum over stages, times `loss_scaling_factor` |

Per-stage values stay unscaled so they remain readable at
`loss_scaling_factor: 0`.

## Sequence parallelism

Each rank runs its own token shard. The MTP partial sums are reduced over the
SP group, merged into the same collective as the per-stage breakdown:

```python
if sp_size > 1:
    reduced = all_reduce(
        torch.cat((total_loss.reshape(1), stage_losses)), group=sp_group
    )
    total_loss = reduced[0]
    stage_losses = reduced[1:]
```

Because the normalizers were counted globally, summing the per-rank partials
reproduces the single-rank value exactly.
`test_sp_mtp_loss_matches_single_rank` asserts this by replaying every rank.

## Template requirement

MTP needs `input_ids` in the forward kwargs. Swift's templates drop them
unless the template preserves them explicitly, which the turbo-registered ones
do for exactly this reason:

```python
# models/qwen3_5/swift_template.py:17
PRESERVE_KEYS = ("input_ids",)
```

A template without this silently produces no MTP outputs. Rather than let that
turn into a confusing downstream failure, the forward checks the contract and
names the fix:

```python
if not outputs.mtp_logits:
    raise RuntimeError(
        "MTP is enabled but the model returned no MTP outputs. MTP needs "
        "'input_ids' in the forward kwargs, which Swift drops unless the "
        "template preserves them; use a turbo-registered template such as "
        "qwen3_5 or qwen3_8."
    )
```

## Configuration

No new fields. `mtp.enabled` and `mtp.loss_scaling_factor` start taking effect
on the DPO path, and `loss_chunk.chunk_size` is borrowed as the chunk size.
`loss_chunk.enabled` is deliberately **not** consulted: it gates the SFT path's
choice between chunked and plain loss, while DPO has no plain option.

The default `loss_scaling_factor` of `0.1` sits at the same magnitude as the
`rpo_alpha: 0.1` used in production DPO configs, which is also a chosen-only
auxiliary CE, so it needs no separate calibration.

`scripts/runs/qwen3_5_dense/swift_text_dpo.{sh,yaml}` is the launch pair. Set
`TEMPLATE=qwen3_8` for a Qwen3.8 checkpoint.

## A deliberate difference from Megatron

Every MTP stage carries the full `loss_scaling_factor`, matching the SFT path
(`backend.py:1950`), so one configured value means the same thing in both
stages. Megatron-LM divides the factor by the stage count
(`gpt_model.py:653`). Comparing runs across the two frameworks therefore needs
the factor scaled by hand: `loss_scaling_factor = 0.1 / num_stages`.

The two formulas are identical whenever there is a single MTP stage, which is
the common case.

## Tests

`tests/adapters/msswift/test_dpo_mtp.py`, on CPU, no NPU required:

| Test | Covers |
| --- | --- |
| `test_stage_labels_shift_per_stage_and_drop_rejected` | stage `i` targets `i + 2` ahead; rejected rows are never supervised |
| `test_mtp_enters_the_gradient_but_not_the_reported_outputs` | DPO outputs unchanged by shadowing; gradient equals a hand-computed reference over `loss_scaling_factor / GA`, for `GA` in `{1, 4}` |
| `test_mtp_metrics_report_unscaled_stages_and_a_scaled_total` | metric keys and scaling, for `loss_scaling_factor` in `{0.0, 0.1}` |
| `test_reference_pass_reports_no_mtp` | the reference pass produces no MTP loss and no metrics |
| `test_missing_mtp_outputs_points_at_the_template` | the template contract guard fires |
| `test_sp_mtp_loss_matches_single_rank` | per-rank partials over a global normalizer rebuild the whole |

`tests/adapters/msswift/test_config.py::test_dpo_accepts_mtp` replaces the old
`test_dpo_rejects_mtp`.

Not yet done: a real NPU run. The implementation is verified on CPU only.
`mtp_loss/0`, `mtp_loss/avg` and `mtp_loss/scaled` appearing in the logs, with
`rewards/chosen` and `rewards/rejected` behaving as they do with MTP off, is
the check to make on hardware.

## Out of scope

- **DSpark + DPO.** DeepSeek V4's draft-training objective is rejected
  explicitly; it is a different mechanism (deferred closures, hand-written
  gradients).
- **MoE variants.** Only dense is enabled. MoE MTP layers carry a router, and
  the dense MTP input shift has no segment-boundary check, so cross-sample
  bleed would pollute the router statistics. Dense has no router, and the
  label-side masking covers the targets, so the same bleed is harmless here.
  Fixing it for MoE needs the `padding_mask` machinery from `deepseek_v4`.
- **LoRA + DPO** remains rejected, as before.
