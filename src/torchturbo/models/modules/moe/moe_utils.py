import torch


def mask_padding_tokens(
    values: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Clear padding rows from a flattened token-by-expert tensor."""
    if padding_mask is None:
        return values
    valid_mask = (~padding_mask.reshape(-1).bool()).unsqueeze(-1)
    return values * valid_mask


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the bias-free AUX routing map; True in padding_mask is excluded."""
    logits = logits.reshape(-1, logits.size(-1))
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    top_indices = torch.topk(scores, k=topk, dim=-1).indices
    routing_map = torch.zeros_like(logits, dtype=torch.int32).scatter_(
        1, top_indices, 1
    ).bool()

    if padding_mask is not None:
        scores = mask_padding_tokens(scores, padding_mask)
        routing_map = mask_padding_tokens(routing_map, padding_mask).bool()

    return routing_map, scores


class MoEAuxLossAutoScaler(torch.autograd.Function):
    loss_scale = 1.0

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss) * MoEAuxLossAutoScaler.loss_scale

    @classmethod
    def set_loss_scale(cls, scale: float) -> None:
        cls.loss_scale = scale
