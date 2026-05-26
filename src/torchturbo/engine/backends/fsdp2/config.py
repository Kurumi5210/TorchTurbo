"""FSDP2-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShardingStrategy(str, Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "full_shard"


@dataclass
class FSDPConfig:
    """FSDP2-specific configuration.

    Attributes
    ----------
    sharding_strategy:
        How to shard model parameters.
    reshard_after_forward:
        Whether to reshard params after forward (trade memory for speed).
    forward_prefetch:
        Prefetch the next FSDP unit during forward.
    cpu_offload:
        Offload params / gradients to CPU.
    pin_memory:
        Pin CPU offloaded memory for faster transfers.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    cpu_offload: bool = False
    pin_memory: bool = True
