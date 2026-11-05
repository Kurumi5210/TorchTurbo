"""DCP (``torch.distributed.checkpoint``) state for the FSDP2 backend.

The DCP model state holds every ``model.state_dict()`` entry, including frozen
parameters. Freezing does not imply that a weight still equals its source.

Expert-parallel (EP) ranks hold different experts under the same FQN, and the
DTensor placements cover only that partition's EDP/ETP mesh dimensions. DCP keys shards by
``(fqn, global_offset)``, so every EP rank computes the same key for its own
experts and ``dedup_save_plans`` would keep the data of a single EP rank.
Entries owned by expert FSDP units get an ``::ep<rank>`` suffix before they reach
DCP. Loading passes the populated state through ``model.load_state_dict()`` so
model load hooks and extra state participate in restoration.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    init_from_local_shards,
)
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership
from torchturbo.engine.optimizers import OptimizersContainer
from torchturbo.logging import get_logger
from torchturbo.models.modules.cpu_resident import (
    CPUResidentParameter,
    NodeSharedCPUResidentParameter,
)

from ....checkpoint.distributed import collective_check

logger = get_logger(__name__)

# Top-level keys of the saved state. DCP flattens nested keys as
# ``<top-level key>.<fqn>`` in its metadata.
_MODEL_KEY = "model"
_OPTIMIZER_KEY = "optimizer"
_CPU_SHARD_SIZES_KEY = "cpu_shard_sizes"


class DcpCheckpoint:
    """Per-rank DCP save/load of model and optimizer state.

    Optimizer state follows the model-aware DTensor layout exposed by each
    optimizer's ``state_dict()``; loading lets DCP fill local tensors in place
    before the state dict is written back through the optimizer.

    Args:
        ep_rank: Rank of this process in its expert-parallel group.
        ep_size: Number of ranks in the expert-parallel group.
        ownership: Stable names of parameters owned by expert FSDP units.
    """

    _EP_SUFFIX = "::ep"

    def __init__(
        self, ep_rank: int, ep_size: int, ownership: ParameterOwnership,
        *, stamp_all_aliases: bool = False
    ) -> None:
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ownership = ownership
        self.stamp_all_aliases = stamp_all_aliases

    # ------------------------------------------------------------------ #
    # EP-unique keys (private).
    # ------------------------------------------------------------------ #
    def _ep_partitioned_fqns(self, model: nn.Module) -> frozenset[str]:
        """Return FQNs whose shards repeat the same offsets on every EP rank."""
        if self.stamp_all_aliases:
            return self.ownership.expert_fqns(model)
        return self.ownership.expert_fqns(model).intersection(
            name for name, _ in model.named_parameters()
        )

    def _stamp_fqn(self, fqn: str, ep_fqns: frozenset[str]) -> str:
        if fqn not in ep_fqns:
            return fqn
        return f"{fqn}{self._EP_SUFFIX}{self.ep_rank}"

    def _strip_fqn(self, fqn: str) -> str:
        idx = fqn.find(self._EP_SUFFIX)
        return fqn[:idx] if idx != -1 else fqn

    def _validate_ep_size(
        self,
        saved_keys: set[str],
        ep_fqns: frozenset[str],
        dcp_dir: str,
    ) -> None:
        """Validate EP layout only for entries consumed by the current model."""
        saved_ranks_by_fqn: dict[str, set[int]] = {}
        prefix = f"{_MODEL_KEY}."
        for key in saved_keys:
            fqn, separator, suffix = key.partition(self._EP_SUFFIX)
            if not separator or not fqn.startswith(prefix):
                continue
            model_fqn = fqn[len(prefix) :]
            if model_fqn not in ep_fqns:
                continue
            rank = suffix.split(".", 1)[0]
            if not rank.isdigit():
                continue
            saved_ranks_by_fqn.setdefault(model_fqn, set()).add(int(rank))
        expected_ranks = set(range(self.ep_size))
        for model_fqn, saved_ranks in saved_ranks_by_fqn.items():
            if saved_ranks == expected_ranks:
                continue
            raise ValueError(
                f"Checkpoint {dcp_dir} saved {model_fqn} on EP ranks "
                f"{sorted(saved_ranks)}, but the current run uses EP ranks "
                f"{sorted(expected_ranks)}."
            )

    def _validate_tensor_shapes(
        self, metadata: dcp.Metadata, model: nn.Module, ep_fqns: frozenset[str],
    ) -> None:
        """Check model logical shapes before live tensors change."""
        for name, parameter in model.state_dict().items():
            if not isinstance(parameter, torch.Tensor):
                continue
            stamped = self._stamp_fqn(name, ep_fqns)
            saved_key = f"{_MODEL_KEY}.{stamped}"
            if saved_key not in metadata.state_dict_metadata:
                continue
            saved = metadata.state_dict_metadata[saved_key]
            if (
                isinstance(saved, dcp.TensorStorageMetadata)
                and tuple(saved.size) != tuple(parameter.shape)
            ):
                raise ValueError(
                    f"Checkpoint tensor {saved_key} shape {tuple(saved.size)} "
                    f"differs from the current logical shape {tuple(parameter.shape)}."
                )

    def _validate_optimizer_state(
        self, metadata: dcp.Metadata, model: nn.Module,
        optimizers: torch.optim.Optimizer | tuple[torch.optim.Optimizer, ...],
    ) -> None:
        """Validate consumed AdamW leaves before native state initialization."""
        names = {parameter: name for name, parameter in model.named_parameters()}
        ep_fqns = self._ep_partitioned_fqns(model)
        saved = metadata.state_dict_metadata
        optimizers = optimizers if isinstance(optimizers, tuple) else (optimizers,)
        group_index = 0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group_prefix = f"{_OPTIMIZER_KEY}.param_groups.{group_index}"
                # PyTorch 2.10 DCP stores the list of parameter FQN strings as
                # the terminal param_groups.<i>.params key, not indexed leaves.
                required = [f"{group_prefix}.{field}" for field in group]
                group_index += 1
                missing = [key for key in required if key not in saved]
                if missing:
                    raise ValueError(
                        f"Optimizer checkpoint is missing required entries: {missing}."
                    )
                if not isinstance(optimizer, torch.optim.AdamW):
                    continue
                for parameter in group["params"]:
                    if not parameter.requires_grad and parameter not in optimizer.state:
                        continue
                    name = names[parameter]
                    fqns = (
                        [f"{name}{self._EP_SUFFIX}{rank}" for rank in range(self.ep_size)]
                        if name in ep_fqns else [name]
                    )
                    prefixes = tuple(f"{_OPTIMIZER_KEY}.state.{fqn}." for fqn in fqns)
                    if self.stamp_all_aliases and parameter.requires_grad and not any(
                        key.startswith(prefixes) for key in saved
                    ):
                        # Adam creates a slot only on the first participating update.
                        # Any present EP alias still requires every partition below.
                        continue
                    fields = ["step", "exp_avg", "exp_avg_sq"]
                    if group["amsgrad"]:
                        fields.append("max_exp_avg_sq")
                    for fqn in fqns:
                        for field in fields:
                            key = f"{_OPTIMIZER_KEY}.state.{fqn}.{field}"
                            if key not in saved:
                                raise ValueError(
                                    f"Optimizer checkpoint is missing required entry {key}."
                                )
                            if field == "step":
                                continue
                            if tuple(saved[key].size) != tuple(parameter.shape):
                                raise ValueError(
                                    f"Checkpoint tensor {key} shape {tuple(saved[key].size)} "
                                    f"differs from the current logical shape {tuple(parameter.shape)}."
                                )

    def _stamp_optim_state(self, optim_state: dict, ep_fqns: frozenset[str]) -> dict:
        new_state = {
            self._stamp_fqn(fqn, ep_fqns): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [self._stamp_fqn(p, ep_fqns) for p in g.get("params", [])]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    def _strip_optim_state(self, optim_state: dict) -> dict:
        new_state = {
            self._strip_fqn(fqn): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [self._strip_fqn(p) for p in g.get("params", [])]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    # ------------------------------------------------------------------ #
    # Model state.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _materialize_cpu_parameters(
        parameters: dict[str, CPUResidentParameter], metadata: Metadata
    ) -> None:
        def validate() -> list[tuple[CPUResidentParameter, torch.dtype]]:
            prepared = {}
            for fqn, parameter in parameters.items():
                saved = metadata.state_dict_metadata[fqn]
                if not isinstance(saved, TensorStorageMetadata):
                    raise ValueError(f"CPU checkpoint state for {fqn} must be a tensor")
                if saved.size != parameter.shape:
                    raise ValueError(f"CPU checkpoint shape mismatch for {fqn}")
                dtype = saved.properties.dtype
                identity = id(parameter)
                if identity in prepared and prepared[identity][1] != dtype:
                    raise ValueError(f"Tied CPU checkpoint dtypes differ for {fqn}")
                prepared[identity] = (parameter, dtype)
            return list(prepared.values())

        for parameter, dtype in collective_check(validate):
            parameter.materialize_(dtype=dtype)

    @staticmethod
    def _sharded_cpu_state(
        parameter: NodeSharedCPUResidentParameter,
        value: torch.Tensor,
        *,
        saving: bool,
    ) -> ShardedTensor:
        def prepare() -> list[tuple[int, int, torch.Tensor]]:
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError("Node-shared CPU state must retain the parameter shape")
            # DCP needs one chunk for an empty tensor. Source empty intervals
            # are retained separately in the parameter's logical layout.
            sizes = (0,) if parameter.ndim and parameter.shape[0] == 0 else None
            return parameter.local_shards(sizes, for_save=saving)

        shards = []
        for _, start, view in collective_check(prepare):
            if parameter.ndim == 0:
                position = []
                tensor = value
            elif view.shape[0] == 0 and parameter.shape[0] != 0:
                # Empty source intervals inside a nonempty tensor have no chunk.
                continue
            else:
                position = [start] + [0] * (parameter.ndim - 1)
                tensor = value[start : start + view.shape[0]]
            shards.append(
                Shard(
                    tensor.contiguous(),
                    ShardMetadata(position, list(tensor.shape), f"rank:{dist.get_rank()}/cpu"),
                )
            )
        group = dist.group.WORLD if saving else parameter.group
        # Loading builds one tensor per node. A layout failure on one node must
        # reach every rank before the WORLD-collective DCP call that follows.
        return collective_check(
            lambda: init_from_local_shards(shards, parameter.shape, process_group=group)
        )

    @staticmethod
    def _restore_cpu_shard_sizes(
        parameters: dict[str, CPUResidentParameter],
        shard_sizes: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> None:
        """Adopt saved source layouts; native DCP has already filled the storage."""
        def validate() -> list[tuple[CPUResidentParameter, tuple[int, ...]]]:
            saved = {}
            for fqn, sizes in shard_sizes:
                parameter = parameters.get(fqn)
                if parameter is None:
                    continue
                sizes = parameter.validate_shard_sizes(sizes)
                identity = id(parameter)
                if identity in saved and saved[identity][1] != sizes:
                    raise ValueError(f"Tied CPU checkpoint shard sizes differ for {fqn}")
                saved[identity] = (parameter, sizes)
            return list(saved.values())

        for parameter, sizes in collective_check(validate):
            parameter.shard_sizes = sizes

    def save_model(self, model: nn.Module, dcp_dir: str) -> None:
        """Save all parameters, persistent buffers, and extra state to ``dcp_dir``."""
        ep_fqns = self._ep_partitioned_fqns(model)
        state_dict = {
            self._stamp_fqn(fqn, ep_fqns): value
            for fqn, value in model.state_dict().items()
        }
        cpu_parameters = {
            f"{_MODEL_KEY}.{self._stamp_fqn(fqn, ep_fqns)}": parameter
            for fqn, parameter in model.named_parameters(remove_duplicate=False)
            if isinstance(parameter, CPUResidentParameter)
        }
        for fqn, parameter in cpu_parameters.items():
            if isinstance(parameter, NodeSharedCPUResidentParameter):
                key = fqn.removeprefix(f"{_MODEL_KEY}.")
                state_dict[key] = self._sharded_cpu_state(
                    parameter, state_dict[key], saving=True
                )
        state = {_MODEL_KEY: state_dict}
        if cpu_parameters:
            # DCP stores this tuple as one byte entry. Physical tensor chunks
            # cannot preserve logical empty shards or their original order.
            state[_CPU_SHARD_SIZES_KEY] = tuple(
                (fqn, parameter.shard_sizes)
                for fqn, parameter in cpu_parameters.items()
                if parameter.shard_sizes is not None
            )
        start = time.perf_counter()
        dcp.save(state, checkpoint_id=dcp_dir)
        logger.info_rank0(
            "[save] model DCP saved to %s in %.2fs (%d entries)",
            dcp_dir, time.perf_counter() - start, len(state_dict),
        )

    def load_model(self, model: nn.Module, dcp_dir: str) -> None:
        """Restore required model state, silently ignoring saved-only entries."""
        ep_fqns = self._ep_partitioned_fqns(model)
        state_keys = {
            self._stamp_fqn(fqn, ep_fqns) for fqn in model.state_dict()
        }

        def preflight() -> Metadata:
            metadata = FileSystemReader(dcp_dir).read_metadata()
            self._validate_ep_size(set(metadata.state_dict_metadata), ep_fqns, dcp_dir)
            if self.stamp_all_aliases:
                self._validate_tensor_shapes(metadata, model, ep_fqns)
            # Default DCP planners retain the original nested paths. Compare
            # model entries, not flattened leaf keys of dict-valued extra state.
            saved = {
                self._strip_fqn(path[1])
                for path in metadata.planner_data.values()
                if path[0] == _MODEL_KEY
            }
            expected = {self._strip_fqn(key) for key in state_keys}
            missing = sorted(expected - saved)
            if missing:
                raise RuntimeError(
                    f"Model state in {dcp_dir} does not match the current model: "
                    f"missing entries {missing}. "
                    "The checkpoint must contain all state required by the "
                    "current model. Checkpoints that omitted frozen weights "
                    "cannot restore exact model state."
                )

            return metadata

        metadata = collective_check(preflight)

        start = time.perf_counter()

        cpu_parameters = {
            f"{_MODEL_KEY}.{self._stamp_fqn(fqn, ep_fqns)}": parameter
            for fqn, parameter in model.named_parameters(remove_duplicate=False)
            if isinstance(parameter, CPUResidentParameter)
        }
        self._materialize_cpu_parameters(cpu_parameters, metadata)
        # Construct aliases only after materialization releases old storage.
        # Keep the original dictionary's module metadata for native load hooks.
        model_state = model.state_dict()
        for fqn, parameter in cpu_parameters.items():
            key = self._strip_fqn(fqn.removeprefix(f"{_MODEL_KEY}."))
            model_state[key] = parameter.detach()
        stamped = {
            self._stamp_fqn(fqn, ep_fqns): value
            for fqn, value in model_state.items()
        }
        for fqn, parameter in cpu_parameters.items():
            if isinstance(parameter, NodeSharedCPUResidentParameter):
                key = fqn.removeprefix(f"{_MODEL_KEY}.")
                stamped[key] = self._sharded_cpu_state(
                    parameter, parameter.detach(), saving=False
                )
        state = {_MODEL_KEY: stamped}
        if cpu_parameters and _CPU_SHARD_SIZES_KEY in metadata.state_dict_metadata:
            state[_CPU_SHARD_SIZES_KEY] = ()
        dcp.load(
            state,
            checkpoint_id=dcp_dir,
            # These are flat local shards. Native nested-shard expansion drops
            # keys on ranks with no shard and can misidentify extra-state format.
            planner=dcp.DefaultLoadPlanner(flatten_sharded_tensors=False),
        )
        model_state.update(
            (self._strip_fqn(fqn), value) for fqn, value in state[_MODEL_KEY].items()
            if f"{_MODEL_KEY}.{fqn}" not in cpu_parameters
        )
        collective_check(
            lambda: model.load_state_dict(model_state, strict=True, assign=False)
        )
        self._restore_cpu_shard_sizes(
            cpu_parameters, state.get(_CPU_SHARD_SIZES_KEY, ())
        )
        logger.info_rank0(
            "[load] model DCP loaded from %s in %.2fs (%d entries)",
            dcp_dir, time.perf_counter() - start, len(state_keys),
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
        if isinstance(optimizer, OptimizersContainer):
            return tuple(optimizer)
        return optimizer

    def save_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        dcp_dir: str,
    ) -> None:
        """Save the optimizer state of ``model`` to ``dcp_dir``."""
        options = StateDictOptions(full_state_dict=False)
        optimizers = self._unwrap_optimizer(optimizer)
        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state, self._ep_partitioned_fqns(model)
        )
        start = time.perf_counter()
        dcp.save({_OPTIMIZER_KEY: stamped}, checkpoint_id=dcp_dir)
        logger.info_rank0(
            "[save] optimizer state saved to %s in %.2fs",
            dcp_dir, time.perf_counter() - start,
        )

    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        dcp_dir: str,
    ) -> None:
        """Restore optimizer state saved by :meth:`save_optimizer`."""
        optimizers = self._unwrap_optimizer(optimizer)
        metadata = None

        def preflight() -> None:
            nonlocal metadata
            metadata = FileSystemReader(dcp_dir).read_metadata()
            if self.stamp_all_aliases:
                self._validate_optimizer_state(metadata, model, optimizers)

        collective_check(preflight)
        options = StateDictOptions(full_state_dict=False)

        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state, self._ep_partitioned_fqns(model)
        )
        # Native state initialization creates entries for unused parameters.
        # Keep only saved parameters, but let DCP validate every retained leaf.
        saved_fqns = {
            path[2]
            for path in metadata.planner_data.values()
            if len(path) >= 3 and path[:2] == (_OPTIMIZER_KEY, "state")
        }

        def validate_targets() -> None:
            current_fqns = {
                fqn for group in stamped["param_groups"] for fqn in group["params"]
            }
            missing = sorted((saved_fqns & current_fqns) - stamped["state"].keys())
            if missing:
                raise RuntimeError(
                    f"Optimizer state in {dcp_dir} has missing target state for "
                    f"{missing}. The current optimizer did not allocate these "
                    "state entries. Load into a fresh optimizer with matching "
                    "settings and cleared gradients."
                )

        collective_check(validate_targets)
        stamped["state"] = {
            fqn: state for fqn, state in stamped["state"].items() if fqn in saved_fqns
        }
        if self.stamp_all_aliases:
            saved = metadata.state_dict_metadata

            def validate_skeleton() -> None:
                names = {parameter: name for name, parameter in model.named_parameters()}
                ep_fqns = self._ep_partitioned_fqns(model)
                children = optimizers if isinstance(optimizers, tuple) else (optimizers,)
                for child in children:
                    for group in child.param_groups:
                        for parameter in group["params"]:
                            if not parameter.requires_grad:
                                continue
                            fqn = self._stamp_fqn(names[parameter], ep_fqns)
                            prefix = f"{_OPTIMIZER_KEY}.state.{fqn}."
                            fields = {key[len(prefix):] for key in saved if key.startswith(prefix)}
                            missing = fields - stamped["state"].get(fqn, {}).keys()
                            if missing:
                                raise ValueError(
                                    f"Native optimizer skeleton lacks saved slot {fqn} "
                                    f"fields {sorted(missing)}."
                                )

            collective_check(validate_skeleton)
        start = time.perf_counter()
        dcp.load({_OPTIMIZER_KEY: stamped}, checkpoint_id=dcp_dir)
        if self.stamp_all_aliases:
            def validate_membership() -> None:
                names = {parameter: name for name, parameter in model.named_parameters()}
                groups = [
                    {self._strip_fqn(name) for name in group["params"]}
                    for group in stamped["param_groups"]
                ]
                members = set().union(*groups)
                children = optimizers if isinstance(optimizers, tuple) else (optimizers,)
                group_index = 0
                for child in children:
                    for group in child.param_groups:
                        for parameter in group["params"]:
                            if not parameter.requires_grad:
                                continue
                            name = names[parameter]
                            if name not in members:
                                raise ValueError(
                                    f"Saved optimizer groups lack current parameter {name}."
                                )
                            # Native set maps empty state groups by index, not FQN.
                            if not stamped["state"] and name not in groups[group_index]:
                                raise ValueError(
                                    f"Empty optimizer state group {group_index} lacks "
                                    f"current parameter {name}."
                                )
                        group_index += 1

            collective_check(validate_membership)
        # Unused parameters legitimately have no state. The native setter
        # replaces the optimizer state, removing any initialized placeholders.
        set_optimizer_state_dict(
            model,
            optimizers,
            self._strip_optim_state(stamped),
            options=StateDictOptions(full_state_dict=False, strict=False),
        )
        logger.info_rank0(
            "[load] optimizer state loaded from %s in %.2fs",
            dcp_dir, time.perf_counter() - start,
        )
