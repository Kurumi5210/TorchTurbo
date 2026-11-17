"""Transfer unique routed tokens between dense and expert TP layouts."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from .utils import RoutingResult


@dataclass(frozen=True)
class _TokenPartition:
    mesh: DeviceMesh | None
    lengths: tuple[int, ...]

    @property
    def rank(self) -> int:
        return self.mesh.get_local_rank() if self.mesh is not None else 0

    @property
    def slot_size(self) -> int:
        # Empty owners still participate in collectives, without routing padding.
        return max(1, max(self.lengths))

    def _pad(self, tokens: torch.Tensor) -> torch.Tensor:
        padding = tokens.new_zeros((self.slot_size - tokens.size(0), *tokens.shape[1:]))
        return torch.cat((tokens, padding), dim=0)

    def split(self, tokens: torch.Tensor) -> torch.Tensor:
        """Select owned tokens; backward gathers complete token gradients."""
        if self.mesh is None:
            return tokens
        slots = torch.cat([self._pad(piece) for piece in tokens.split(self.lengths)], dim=0)
        owned = DTensor.from_local(
            slots, self.mesh, [Replicate()], run_check=False,
        ).redistribute(placements=[Shard(0)]).to_local(grad_placements=[Shard(0)])
        return owned[:self.lengths[self.rank]]

    def gather(self, tokens: torch.Tensor) -> torch.Tensor:
        """Gather tokens; backward selects the owner's already complete gradient.

        GroupedMLPParallel sums feature-shard input gradients separately. Using
        a partial gradient here would reduce them a second time across ETP.
        """
        if self.mesh is None:
            return tokens
        slots = DTensor.from_local(
            self._pad(tokens), self.mesh, [Shard(0)], run_check=False,
        ).redistribute(placements=[Replicate()]).to_local(grad_placements=[Replicate()])
        return torch.cat([
            slot[:length]
            for slot, length in zip(slots.split(self.slot_size), self.lengths)
        ], dim=0)


class MoETokenLayout:
    """Per-forward ownership metadata for a routed dense-TP/expert-TP bridge.

    Router statistics and shared experts retain the original dense token domain.
    Communication slots are padded uniformly, then stripped before EP dispatch.
    """

    def __init__(
        self, hidden_states: torch.Tensor,
        tp_mesh: DeviceMesh | None, etp_mesh: DeviceMesh | None,
    ) -> None:
        self.hidden_shape = hidden_states.shape
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        tp_size = tp_mesh.size() if tp_mesh is not None else 1
        quotient, remainder = divmod(num_tokens, tp_size)
        self.dense = _TokenPartition(
            tp_mesh, tuple(quotient + (rank < remainder) for rank in range(tp_size)),
        )
        owned_length = self.dense.lengths[self.dense.rank]
        if etp_mesh is None:
            expert_lengths = (owned_length,)
        else:
            length = torch.tensor([owned_length], dtype=torch.int64, device=hidden_states.device)
            lengths = [torch.empty_like(length) for _ in range(etp_mesh.size())]
            dist.all_gather(lengths, length, group=etp_mesh.get_group())
            expert_lengths = tuple(int(value.item()) for value in lengths)
        self.expert = _TokenPartition(etp_mesh, expert_lengths)

    def dispatch(
        self, hidden_states: torch.Tensor, routing: RoutingResult,
    ) -> tuple[torch.Tensor, RoutingResult]:
        """Move hidden states and all routing tensors to the expert token domain."""
        def transfer(tensor: torch.Tensor) -> torch.Tensor:
            return self.expert.gather(self.dense.split(tensor))

        return transfer(hidden_states.reshape(-1, hidden_states.shape[-1])), RoutingResult(
            probs=transfer(routing.probs),
            routing_map=transfer(routing.routing_map),
            expert_indices=transfer(routing.expert_indices),
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore the original dense token order, shape, and gradient ownership."""
        return self.dense.gather(self.expert.split(hidden_states)).reshape(self.hidden_shape)
