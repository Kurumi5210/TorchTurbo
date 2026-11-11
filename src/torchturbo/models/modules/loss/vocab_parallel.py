"""Native vocabulary-parallel cross entropy with a local scalar result."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Keep native CE forward and backward inside their required context."""

    @staticmethod
    def forward(ctx, logits, labels, ignore_index):
        with torch.enable_grad(), loss_parallel():
            detached_logits = logits.detach().requires_grad_(True)
            loss = F.cross_entropy(
                detached_logits.float(), labels,
                reduction="sum", ignore_index=ignore_index,
            )
        ctx.save_for_backward(detached_logits, loss)
        return loss.to_local()

    @staticmethod
    def backward(ctx, grad_output):
        logits, loss = ctx.saved_tensors
        with loss_parallel():
            grad_loss = DTensor.from_local(
                grad_output, loss.device_mesh, [Replicate()], run_check=False,
            )
            (grad_logits,) = torch.autograd.grad(loss, logits, grad_loss)
        return grad_logits, None, None


def vocab_parallel_cross_entropy(
    logits: DTensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return summed FP32 CE for logits sharded over the vocabulary.

    Args:
        logits: A two-dimensional DTensor sharded on the class dimension
            over a one-dimensional TP mesh.
        labels: Complete class indices replicated over the TP mesh.
        ignore_index: Target value excluded from the loss and gradients.

    Returns:
        A plain scalar tensor whose backward needs no external TP context.
    """
    if logits.ndim != 2 or logits.device_mesh.ndim != 1:
        raise ValueError("Vocabulary CE requires 2D logits on a 1D TP mesh.")
    placement, = logits.placements
    if not isinstance(placement, Shard) or placement.dim % logits.ndim != 1:
        raise ValueError("Vocabulary CE requires logits sharded on the class dimension.")
    if torch.is_grad_enabled() and logits.requires_grad:
        return _VocabParallelCrossEntropy.apply(logits, labels, ignore_index)
    with loss_parallel():
        return F.cross_entropy(
            logits.float(), labels, reduction="sum", ignore_index=ignore_index,
        ).to_local()
