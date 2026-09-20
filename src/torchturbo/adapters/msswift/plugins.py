"""Consume loaded plugins into ms-swift via the engine's ``PluginManager.try_install``.

``install_plugins(manager)`` runs one ``_SwiftPluginConsumer`` over the manager. The
consumer dispatches each plugin to its dataset or packing handling: dataset plugins are
registered as swift datasets (lazy-mount loader + MessagesPreprocessor built from the
entry's tags); packing plugins patch ``calculate_matched_group`` to call their
``pack_groups``. Idempotent (``register_dataset(exist_ok=True)`` + a patch sentinel)."""

from __future__ import annotations

from ...engine.plugins import PluginConsumer, PluginKind, PluginManager


class _SwiftPluginConsumer(PluginConsumer):
    """Routes each plugin to its ms-swift wiring (dataset -> swift dataset registration;
    packing -> ``calculate_matched_group`` patch). One pass over the manager's plugins."""

    # role-tag entry config (dataset_info tags) -> swift MessagesPreprocessor kwarg
    TAG_TO_SWIFT_KW = {
        "user_tag": "user_role",
        "assistant_tag": "assistant_role",
        "system_tag": "system_role",
        "role_tag": "role_key",
        "content_tag": "content_key",
    }

    def matches(self, plugin) -> bool:
        return getattr(plugin, "kind", None) in (PluginKind.DATASET, PluginKind.PACKING)

    def install(self, plugin) -> None:
        kind = getattr(plugin, "kind", None)
        if kind == PluginKind.DATASET:
            self._install_dataset(plugin)
        elif kind == PluginKind.PACKING:
            self._install_packing(plugin)

    # --- dataset handling ---
    @classmethod
    def _make_preprocessor(cls, entry: dict):
        """Build a swift preprocessor from one mdata ``dataset_info`` entry (keys ``tags``, ``columns``)."""
        from swift.dataset.preprocessor.core import AutoPreprocessor, MessagesPreprocessor

        role_tags = entry.get("tags") or {}
        column_map = entry.get("columns") or {}
        if not role_tags:
            return AutoPreprocessor(columns=dict(column_map))
        kwargs = {swift_kw: role_tags[tag] for tag, swift_kw in cls.TAG_TO_SWIFT_KW.items() if role_tags.get(tag)}
        return MessagesPreprocessor(**kwargs, columns=dict(column_map))

    @staticmethod
    def _make_loader(plugin, name: str):
        """Build a swift DatasetLoader that fetches samples via ``plugin.samples(name)`` and runs the entry's preprocess_func."""
        from swift.dataset.loader import DatasetLoader

        class _Loader(DatasetLoader):
            def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
                raw = plugin.samples(name, cache_dir=None)
                return dataset_meta.preprocess_func(
                    raw,
                    num_proc=getattr(self, "num_proc", 1),
                    load_from_cache_file=getattr(self, "load_from_cache_file", True),
                    strict=getattr(self, "strict", False),
                    enable_auto_mapping=not getattr(self, "disable_auto_column_mapping", False),
                )

        return _Loader

    def _install_dataset(self, plugin) -> None:
        from swift.dataset import DatasetMeta, register_dataset

        for dataset_name, info in plugin.dataset_info().items():
            register_dataset(
                DatasetMeta(
                    dataset_name=dataset_name,
                    dataset_path="torchturbo://dataset-plugin",
                    loader=self._make_loader(plugin, dataset_name),
                    preprocess_func=self._make_preprocessor(info),
                ),
                exist_ok=True,
            )

    # --- packing handling ---
    def _install_packing(self, plugin) -> None:
        import swift.dataset.packing as packing_mod

        if getattr(packing_mod.calculate_matched_group, "_tt_packing_patched", False):
            return  # idempotent

        def _calculate_matched_group(sequences, packing_length, is_finished: bool = True, strategy: str = "binpack"):
            return plugin.pack_groups(sequences, packing_length, is_finished=is_finished)

        _calculate_matched_group._tt_packing_patched = True
        packing_mod.calculate_matched_group = _calculate_matched_group

        from ...logging import get_logger

        get_logger(__name__).info_rank0("Packing plugin '%s' enabled for ms-swift.", plugin.plugin_name)


def install_plugins(manager: PluginManager) -> None:
    """Wire loaded plugins into ms-swift (idempotent)."""
    manager.try_install(_SwiftPluginConsumer())


__all__ = ["install_plugins"]