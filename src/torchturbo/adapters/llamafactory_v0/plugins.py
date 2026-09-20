"""Consume loaded plugins into LlamaFactory via the engine's ``PluginManager.try_install``.

``install_plugins(manager)`` runs one ``_LFPluginConsumer`` over the manager. The
consumer dispatches each plugin to its dataset or packing handling: dataset plugins patch
``_get_merged_dataset`` to route names through ``plugin.samples`` + LF's ``align_dataset``
(non-matching fall through); packing plugins rebind ``greedy_knapsack`` in both
``processor_utils`` and ``supervised`` (by-value import). Idempotent via sentinels."""

from __future__ import annotations

from ...engine.plugins import PluginConsumer, PluginKind, PluginManager


class _LFPluginConsumer(PluginConsumer):
    """Routes each plugin to its LlamaFactory wiring (dataset -> ``_get_merged_dataset`` patch;
    packing -> ``greedy_knapsack`` rebind). One pass over the manager's plugins."""

    def matches(self, plugin) -> bool:
        return getattr(plugin, "kind", None) in (PluginKind.DATASET, PluginKind.PACKING)

    def install(self, plugin) -> None:
        kind = getattr(plugin, "kind", None)
        if kind == PluginKind.DATASET:
            self._install_dataset(plugin)
        elif kind == PluginKind.PACKING:
            self._install_packing(plugin)

    def _install_dataset(self, plugin) -> None:
        from datasets import concatenate_datasets
        from llamafactory.data import loader as lf_loader

        if getattr(lf_loader._get_merged_dataset, "_tt_dataset_plugins_patched", False):
            return  # idempotent

        original = lf_loader._get_merged_dataset

        def _patched(dataset_names, model_args, data_args, training_args, stage, return_dict=False):
            info = plugin.dataset_info()
            routed, others = [], []
            for n in (dataset_names or []):
                (routed if n in info else others).append(n)

            if not routed:
                return original(dataset_names, model_args, data_args, training_args, stage, return_dict=return_dict)

            # Once any name routes through a plugin, remaining (non-plugin) names are dropped.
            from llamafactory.data.converter import align_dataset
            from llamafactory.data.parser import DatasetAttr

            converted = []
            for n in routed:
                info_entry = info[n]  # provider-defined metadata dict; ``DatasetAttr.join`` reads its keys
                raw = plugin.samples(n, cache_dir=getattr(data_args, "cache_dir", None))
                attr = DatasetAttr(load_from="file", dataset_name=n)
                attr.join(info_entry)
                converted.append(align_dataset(raw, attr, data_args, training_args))

            _ = others  # dropped by design
            return converted[0] if len(converted) == 1 else concatenate_datasets(converted)

        _patched._tt_dataset_plugins_patched = True
        lf_loader._get_merged_dataset = _patched

        from ...logging import get_logger

        get_logger(__name__).info_rank0("dataset plugin routing enabled for LlamaFactory")

    def _install_packing(self, plugin) -> None:
        from llamafactory.data.processor import processor_utils as pu
        from llamafactory.data.processor import supervised as sm

        if getattr(pu.greedy_knapsack, "_tt_packing_patched", False):
            return  # idempotent

        def _greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
            return plugin.knapsack(numbers, capacity)

        _greedy_knapsack._tt_packing_patched = True
        pu.greedy_knapsack = _greedy_knapsack
        sm.greedy_knapsack = _greedy_knapsack  # supervised imported greedy_knapsack by value

        from ...logging import get_logger

        get_logger(__name__).info_rank0("Packing plugin '%s' enabled for LlamaFactory.", plugin.plugin_name)


def install_plugins(manager: PluginManager) -> None:
    """Wire loaded plugins into LlamaFactory (idempotent)."""
    manager.try_install(_LFPluginConsumer())


__all__ = ["install_plugins"]