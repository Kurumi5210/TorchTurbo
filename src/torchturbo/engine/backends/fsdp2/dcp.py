"""DCP (``torch.distributed.checkpoint``) checkpointing for the FSDP2 backend.

Turbo MoE experts are split across EP ranks at module construction: every EP
rank holds a disjoint expert subset under the SAME model FQN (e.g.
``mlp.experts.weight1``), with the DTensor placement carrying only ``Shard(0)``
on the edp mesh dim. DCP keys shards by ``(fqn, global_offset)``; the EP
dimension is invisible to it, so every EP rank computes the same offset for the
same expert FQN and ``dedup_save_plans`` would cross-wire experts (and expert
optimizer state) across EP ranks.

Stamp the EP rank onto expert FQNs as a suffix before handing the
state_dict to DCP, and strip it before writing back into the model/optimizer.
Dense/router params are left untouched.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

import torch
import torch.distributed as dist

from ....logging import get_logger
from ....models.weight_converter import BaseWeightConverter
from torchturbo.engine.utils import strip_peft_prefix

logger = get_logger(__name__)

MODEL_DISTCP_DIR = "model_distcp"
OPTIMIZER_DISTCP_DIR = "optimizer_distcp"

# DCP writes the metadata file (``.metadata`` globally, ``__<rank>.metadata``
# per-rank) *last*, atomically via a ``.tmp`` rename, after all data shards
# (``*.distcp``). Its presence is therefore the canonical "save finished
# cleanly" signal — a directory that exists without it was interrupted and is
# treated as incomplete by the guards below, not as a usable checkpoint.
_DCP_METADATA_GLOBAL = ".metadata"

class DcpCheckpoint:
    """Per-rank DCP save/load of model weights and optimizer state.

    Expert params are EP-suffixed before DCP so ``(fqn, offset)`` keys stay
    unique across EP ranks (see module docstring); the suffix is stripped
    before writing back. Each shard is saved at its HF source-checkpoint dtype
    (mapped via the weight converter), not a single mixed-precision dtype.

    Optimizer state follows the model-aware DTensor layout exposed by each
    optimizer's ``state_dict()``; loading lets DCP fill local tensors in place
    before the state dict is written back through the optimizer.
    """

    _EP_SUFFIX = "::ep"

    def __init__(
        self,
        rank: int,
        weight_converter: BaseWeightConverter,
    ) -> None:
        self.rank = rank
        self.weight_converter = weight_converter

    def _ep_rank(self, parallel_context: dict[str, Any]) -> int:
        return parallel_context["ep"]["ep_rank"]

    def _rank0_reset_dir(self, ckpt_dir: str) -> None:
        """Recreate a checkpoint dir so DCP never overwrites a stale save."""
        if self.rank == 0:
            if os.path.isdir(ckpt_dir):
                logger.info_rank0(
                    "[save] replacing existing checkpoint dir %s", ckpt_dir
                )
                shutil.rmtree(ckpt_dir)
            os.makedirs(ckpt_dir, exist_ok=True)
        dist.barrier()

    def _dcp_metadata_present(self, dcp_dir: str) -> bool:
        """True iff a finished DCP metadata file sits in ``dcp_dir``."""
        if not os.path.isdir(dcp_dir):
            return False
        if os.path.isfile(os.path.join(dcp_dir, _DCP_METADATA_GLOBAL)):
            return True
        # Per-rank metadata layout: ``__<rank>.metadata``.
        return any(name.endswith(_DCP_METADATA_GLOBAL) and name.startswith("__") for name in os.listdir(dcp_dir))

    # ------------------------------------------------------------------ #
    # EP expert-FQN suffixing (private).
    # ------------------------------------------------------------------ #
    def _stamp_expert_fqn(self, fqn: str, ep_rank: int, is_expert_fqn) -> str:
        if not is_expert_fqn(fqn):
            return fqn
        suffix = self._EP_SUFFIX + str(ep_rank)
        return fqn if fqn.endswith(suffix) else f"{fqn}{suffix}"

    def _strip_expert_fqn(self, fqn: str) -> str:
        idx = fqn.find(self._EP_SUFFIX)
        return fqn[:idx] if idx != -1 else fqn

    def _stamp_model_state(self, state_dict: dict, ep_rank: int, is_expert_fqn) -> dict:
        return {
            self._stamp_expert_fqn(k, ep_rank, is_expert_fqn): v for k, v in state_dict.items()
        }

    def _strip_model_state(self, state_dict: dict) -> dict:
        return {self._strip_expert_fqn(k): v for k, v in state_dict.items()}

    def _stamp_optim_state(self, optim_state: dict, ep_rank: int, is_expert_fqn) -> dict:
        new_state = {
            self._stamp_expert_fqn(fqn, ep_rank, is_expert_fqn): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [
                self._stamp_expert_fqn(p, ep_rank, is_expert_fqn)
                for p in g.get("params", [])
            ]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    def _strip_optim_state(self, optim_state: dict) -> dict:
        new_state = {
            self._strip_expert_fqn(fqn): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [self._strip_expert_fqn(p) for p in g.get("params", [])]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    # ------------------------------------------------------------------ #
    # Model weights.
    # ------------------------------------------------------------------ #
    def has_model_weights(self, checkpoint_dir: str) -> bool:
        """True iff a *complete* DCP model save is present.

        Requires the ``model_distcp`` dir AND its finished ``.metadata`` file —
        a dir without metadata is an interrupted save and must not be loaded.
        """
        return self._dcp_metadata_present(os.path.join(checkpoint_dir, MODEL_DISTCP_DIR))

    def model_weights_dir_exists(self, checkpoint_dir: str) -> bool:
        """Shallow existence (dir present at all), for ''missing vs incomplete'' messaging."""
        return os.path.isdir(os.path.join(checkpoint_dir, MODEL_DISTCP_DIR))

    def save_model(
        self,
        model: torch.nn.Module,
        output_dir: str,
        parallel_context: dict[str, Any],
    ) -> None:
        import torch.distributed.checkpoint as dcp

        ckpt_dir = os.path.join(output_dir, MODEL_DISTCP_DIR)
        self._rank0_reset_dir(ckpt_dir)

        # Store the natural state dict; expert weight1/weight2 get an EP-rank
        # suffix so DCP's (fqn, offset) keys stay unique across EP ranks.
        stamped = self._stamp_model_state(
            model.state_dict(), self._ep_rank(parallel_context), self.weight_converter.is_expert_weight
        )
        start = time.perf_counter()
        dcp.save({"model": stamped}, checkpoint_id=ckpt_dir)
        logger.info_rank0("[save] model DCP saved to %s in %.2fs", ckpt_dir, time.perf_counter() - start)

    def load_model(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        parallel_context: dict[str, Any],
    ) -> None:
        import torch.distributed.checkpoint as dcp

        ckpt_dir = os.path.join(checkpoint_dir, MODEL_DISTCP_DIR)
        # Stamp the same EP suffix onto the live state dict so DCP can fill it,
        # then strip back to natural names for load_state_dict.
        state_dict = self._stamp_model_state(
            model.state_dict(), self._ep_rank(parallel_context), self.weight_converter.is_expert_weight
        )
        start = time.perf_counter()
        dcp.load({"model": state_dict}, checkpoint_id=ckpt_dir)
        loaded = self._strip_model_state(state_dict)
        model.load_state_dict(loaded)
        logger.info_rank0(
            "[load] model DCP loaded from %s in %.2fs (%d params)",
            ckpt_dir, time.perf_counter() - start, len(loaded),
        )

    # ------------------------------------------------------------------ #
    # Optimizer state.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap_optimizer(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer | tuple[torch.optim.Optimizer, ...]:
        """Return the optimizer instances DCP should traverse.

        ``OptimizersContainer`` uses a named-child state format for generic
        serialization. DCP needs the standard state owned by each child so it
        can translate parameter identities to FQNs and reshard them
        independently.
        """
        from ...optimizers import OptimizersContainer

        if isinstance(optimizer, OptimizersContainer):
            return tuple(optimizer)
        return optimizer

    def has_optimizer(self, checkpoint_dir: str) -> bool:
        """True iff a *complete* DCP optimizer save is present (dir + ``.metadata``)."""
        return self._dcp_metadata_present(os.path.join(checkpoint_dir, OPTIMIZER_DISTCP_DIR))

    def optimizer_dir_exists(self, checkpoint_dir: str) -> bool:
        """Shallow existence (dir present at all), for ''missing vs incomplete'' messaging."""
        return os.path.isdir(os.path.join(checkpoint_dir, OPTIMIZER_DISTCP_DIR))

    def save_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        output_dir: str,
        parallel_context: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
        )

        ckpt_dir = os.path.join(output_dir, OPTIMIZER_DISTCP_DIR)
        self._rank0_reset_dir(ckpt_dir)

        options = StateDictOptions(full_state_dict=False)
        optimizers = self._unwrap_optimizer(optimizer)
        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state,
            self._ep_rank(parallel_context),
            self.weight_converter.is_expert_weight,
        )
        start = time.perf_counter()
        dcp.save({"optimizer": stamped}, checkpoint_id=ckpt_dir)
        logger.info_rank0("[save] optimizer state saved to %s in %.2fs", ckpt_dir, time.perf_counter() - start)

    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        checkpoint_dir: str,
        parallel_context: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )

        ckpt_dir = os.path.join(checkpoint_dir, OPTIMIZER_DISTCP_DIR)
        options = StateDictOptions(full_state_dict=False)
        optimizers = self._unwrap_optimizer(optimizer)

        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state,
            self._ep_rank(parallel_context),
            self.weight_converter.is_expert_weight,
        )
        start = time.perf_counter()
        dcp.load({"optimizer": stamped}, checkpoint_id=ckpt_dir)
        set_optimizer_state_dict(
            model, optimizers, self._strip_optim_state(stamped), options=options
        )
        logger.info_rank0("[load] optimizer state loaded from %s in %.2fs", ckpt_dir, time.perf_counter() - start)
