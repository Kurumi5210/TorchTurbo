"""antllm_prepack plugin (v3 plugin API): pack antllm mdata data into save_path.

Pack config via --tt_plugin plugin.py:pack.yaml. The `dataset` block:
  args (data_path, save_path); pack_args (tokenizer_path + antllm_path + think_enable_datasets + ...).
`antllm_path` is the parent dir of the antllm package you cloned/installed -- added to
sys.path so `import antllm.*` resolves. You set it manually in the pack config (like
tokenizer_path); no runtime git clone. The plugin runs antllm's SFTPrePackProcesserV2
into save_path every run (no cache -- mirrors antllm's driver which always re-packs;
multi-node: global rank0 wipes save_path once, each node's local_rank0 packs its
node_rank shard), then delegates reading to ``AntllmPrePackedDataset`` from the
reader module ``antllm_prepacked_reader``: it memmaps the pack and emits the
engine's packed-input contract.
``collate_prepacked`` is served via ``get_collate_fn`` so the trainer hook uses it
(one packed row per step).
Exposed as dataset name = save_path; --dataset <save_path> selects it.
No environment variables.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import torch

from torchturbo.engine.plugins import DatasetPlugin, DatasetPrepareContext, register_plugin
from antllm_prepacked_reader import AntllmPrePackedDataset, collate_prepacked

logger = logging.getLogger("torchturbo.antllm_prepack")

_PACK_RESERVED = {"data_path", "save_path", "tokenizer_path", "max_length", "context_parallel_size", "antllm_path"}

# torch.distributed must be initialized before the distributed pack; if it is not
# ready when build_dataset runs (init racing this point), retry this many times with
# this sleep before refusing -- never silently fall back to a single-process pack.
_DIST_READY_MAX_ATTEMPTS = 3
_DIST_READY_RETRY_SLEEP = 10  # seconds


def _load_config_text(path):
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_pack_config(cfg):
    if "dataset" not in cfg:
        raise ValueError("antllm_prepack: pack config needs a top-level `dataset` block; bind via --tt_plugin plugin.py:pack.yaml")
    d = cfg["dataset"]
    args = dict(d.get("args") or {})
    pack_args = dict(d.get("pack_args") or {})
    data_path = args.get("data_path") or pack_args.get("data_path")
    save_path = args.get("save_path") or pack_args.get("save_path")
    tokenizer_path = pack_args.get("tokenizer_path")
    antllm_path = pack_args.get("antllm_path")
    if not data_path or not save_path:
        raise ValueError("antllm_prepack: dataset.args.{data_path,save_path} are required")
    if not tokenizer_path:
        raise ValueError(
            "antllm_prepack: dataset.pack_args.tokenizer_path is required "
            "(antllm-resource tokenizer dir)"
        )
    if not antllm_path:
        raise ValueError(
            "antllm_prepack: dataset.pack_args.antllm_path is required "
            "(parent dir of the antllm package you cloned/installed)"
        )
    return {
        "antllm_path": antllm_path,
        "data_path": data_path,
        "save_path": save_path,
        "pack_args": pack_args,
        "tokenizer_path": tokenizer_path,
    }


def _ensure_writable_metrics_dir(save_path):
    """Patch antllm's MDATA_METRICS_WRITE_PATH to <save_path>/pt_process/metrics.jsonl.

    antllm hardcodes MDATA_METRICS_WRITE_PATH="/home/admin/logs/pt_process/metrics.jsonl"
    as a module-level constant; callers copy it at import time. We patch the SOURCE
    module before any caller imports it. The pt_process directory itself is created
    once (global rank0) in build_dataset before the distributed pack; this function
    only sets the path (no filesystem write), so it is safe to call on every rank --
    a earlier per-rank ``os.makedirs(pt_process)`` raced to NFS ESTALE (Errno 116) on
    the shared save_path when the concurrent ``shutil.rmtree(save_path)`` invalidated
    the losers' file handles.
    """
    metrics_dir = os.path.join(save_path, "pt_process")
    metrics_file = os.path.join(metrics_dir, "metrics.jsonl")
    import antllm.metrics.metrics_writer_by_mdata as _mw
    _mw.MDATA_METRICS_WRITE_PATH = metrics_file
    logger.info("antllm_prepack: patched MDATA_METRICS_WRITE_PATH -> %s", metrics_file)


def _run_antllm_pack(data_path, save_path, tokenizer_path, max_length, pack_args):
    from antllm.data.sft_pre_pack.sft_pre_pack_data_v2 import SFTPrePackProcesserV2

    cp = int(pack_args.get("context_parallel_size", 1) or 1)
    extra = {k: v for k, v in pack_args.items() if k not in _PACK_RESERVED}
    proc = SFTPrePackProcesserV2(
        data_path=data_path,
        save_path=save_path,
        tokenizer_path=tokenizer_path,
        max_length=int(max_length),
        node_rank=int(os.environ.get("NODE_RANK", "0")),
        node_num=int(os.environ.get("NNODES", "1")),
        context_parallel_size=cp,
        **extra,
    )
    proc.run()


@register_plugin("antllm_prepack")
class AntllmPrePackPlugin(DatasetPlugin):
    """Pack antllm mdata data into save_path; reading is delegated to the reader module.

    build_dataset re-packs every run (no cache), mirroring antllm's ``sft_pack_data``:
    global rank0 wipes save_path once (barrier'd), then each node's LOCAL_RANK=0 runs
    ``SFTPrePackProcesserV2`` for its ``node_rank`` shard (antllm's FileSynchronizer
    coordinates the multi-node pack), then the result is handed to
    ``AntllmPrePackedDataset`` from the reader module ``antllm_prepacked_reader``. The
    reader recovers per-dataset sampling ratios itself from the ``data_path`` the pack
    recorded in ``meta.json`` and emits the engine's packed-input contract;
    ``collate_prepacked`` is served via ``get_collate_fn`` so the trainer hook uses
    it (one packed row per step).
    """

    def get_collate_fn(self):
        """One packed row per step: serve the reader module's collator."""
        return collate_prepacked

    def __init__(self, config=None):
        super().__init__(config)
        pack_path = self._resolve_pack_path(config)
        if not pack_path:
            raise ValueError(
                "antllm_prepack needs a pack config: bind it via --tt_plugin plugin.py:pack.yaml"
            )
        if not os.path.isfile(pack_path):
            raise FileNotFoundError(f"antllm_prepack: pack config not found: {pack_path}")
        self._pack_path = pack_path
        self._cfg = _parse_pack_config(_load_config_text(pack_path))
        self._name = self._cfg["save_path"]
        antllm_path = self._cfg["antllm_path"]
        # Stub atorch BEFORE antllm import (breaks the mpu → atorch chain).
        if "atorch" not in sys.modules:
            import types
            for mod in ["atorch", "atorch.utils", "atorch.utils.import_util"]:
                sys.modules[mod] = types.ModuleType(mod)
            from transformers.utils import is_torch_npu_available as _is_npu
            sys.modules["atorch.utils.import_util"].is_torch_npu_available = _is_npu
            logger.info("antllm_prepack: stubbed atorch (is_torch_npu_available from transformers.utils)")
        # Insert antllm path so `import antllm.*` resolves.
        if antllm_path and os.path.isdir(antllm_path) and antllm_path not in sys.path:
            sys.path.insert(0, antllm_path)
        # Patch antllm's MDATA_METRICS_WRITE_PATH AFTER sys.path.insert
        # (needs antllm importable to import the source module).
        _ensure_writable_metrics_dir(self._cfg["save_path"])
        logger.info(
            "antllm_prepack: loaded pack config %s (dataset/save_path=%s, antllm_path=%s)",
            pack_path, self._name, antllm_path,
        )

    @staticmethod
    def _resolve_pack_path(bound_config):
        return bound_config

    def dataset_names(self):
        return [self._name]

    def dataset_info(self, name):
        return {"uri": self._cfg["data_path"], "columns": {}, "tags": {}, "formatting": None}

    def provides_dataset(self, name):
        return name == self._name

    def build_dataset(self, name, ctx: DatasetPrepareContext):
        data_path = self._cfg["data_path"]
        save_path = self._cfg["save_path"]
        pack_args = self._cfg["pack_args"]
        tokenizer_path = self._cfg["tokenizer_path"]
        max_length = int(pack_args.get("max_length") or ctx.max_length or 131072)

        # antllm re-packs every run (no cache, mirroring sft_pack_data), and the pack
        # runs only under a distributed launch: the engine builds torch.distributed
        # before _prepare_dataset reaches build_dataset, so it is normally already
        # initialized here. If not (dist init racing this point), wait and re-check --
        # sleep 10s, retry up to 3 times, then give up loudly; we never silently fall
        # back to a single-process pack (antllm's sft_pack_data also relies on
        # torch.distributed.barrier, so a multi-node run that lost dist must fail loudly
        # rather than pack on one process). A barrier then coordinates the rank0 wipe
        # and the per-node pack -- without it a concurrent rmtree on another node
        # invalidated the shared save_path and the losers' makedirs(<save_path>/pt_process)
        # hit ESTALE (Errno 116).
        for _attempt in range(_DIST_READY_MAX_ATTEMPTS):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                break
            if _attempt < _DIST_READY_MAX_ATTEMPTS - 1:
                logger.warning(
                    "antllm_prepack: torch.distributed not initialized yet; "
                    "retry in %ds (attempt %d/%d)",
                    _DIST_READY_RETRY_SLEEP, _attempt + 1, _DIST_READY_MAX_ATTEMPTS,
                )
                time.sleep(_DIST_READY_RETRY_SLEEP)
        else:
            raise RuntimeError(
                f"antllm_prepack requires a distributed launch (torchrun) before the "
                f"pack: torch.distributed is still not initialized after "
                f"{_DIST_READY_MAX_ATTEMPTS} checks "
                f"(~{_DIST_READY_RETRY_SLEEP * (_DIST_READY_MAX_ATTEMPTS - 1)}s). "
                f"The engine builds the process group first; if it did not, re-run "
                f"under torch.distributed.run (--nproc_per_node / --nnodes)."
            )
        rank = torch.distributed.get_rank()
        is_local_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
        # One-time save_path wipe on global rank0 only, barrier'd before the distributed
        # pack; pt_process is created here once so _ensure_writable_metrics_dir never
        # mkdirs per rank.
        if rank == 0:
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "pt_process"), exist_ok=True)
            logger.info("antllm_prepack: rank0 wiping + preparing save_path %s (data=%s)", save_path, data_path)
        torch.distributed.barrier()
        # Distributed pack: each node's local_rank0 packs its node_rank shard; antllm's
        # FileSynchronizer coordinates the multi-node pack internally.
        if is_local_rank0:
            logger.info("antllm_prepack: local_rank0 building pack at %s (data=%s)", save_path, data_path)
            _run_antllm_pack(data_path, save_path, tokenizer_path, max_length, pack_args)
        torch.distributed.barrier()

        # The reader (antllm_prepacked_reader module) recovers per-dataset sampling
        # ratios itself from the data_path recorded in meta.json and emits the engine's
        # packed-input contract; constructing it also validates the pack (raises if
        # missing or incomplete), so no separate post-pack check is needed.
        train = AntllmPrePackedDataset(save_path)
        logger.info("antllm_prepack: built '%s' -> %d packed samples from %s", name, len(train), save_path)
        return self._finalize(train, None, ctx)

    @staticmethod
    def _finalize(train_dataset, val_dataset, ctx: DatasetPrepareContext):
        if ctx.show_dataset is not None:
            try:
                ctx.show_dataset(train_dataset, val_dataset)
            except Exception:
                logger.debug("antllm_prepack: ctx.show_dataset logging failed (non-fatal)", exc_info=True)
        return train_dataset, val_dataset


__all__ = ["AntllmPrePackPlugin"]
