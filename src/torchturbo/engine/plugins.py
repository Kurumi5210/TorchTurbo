"""Plugin system: ``--tt_plugin`` files register ``Plugin`` subclasses; ``PluginManager``
loads them and drives plugin consumers via ``try_install``."""

from __future__ import annotations

import importlib.util
import os
import sys
from enum import Enum
from typing import Any, Callable, Iterable

from ..logging import get_logger

logger = get_logger(__name__)


class PluginKind(Enum):
    """What a plugin provides; consumers gate on this."""

    DATASET = "dataset"
    PACKING = "packing"


class Plugin:
    """Base for every ``--tt_plugin``; ``self.config`` is the bound ``plugin:config``."""

    kind: PluginKind

    def __init__(self, config: str | None = None) -> None:
        self.config: str | None = config


class PluginConsumer:
    """A framework's wiring for one :class:`PluginKind`; override ``matches`` and ``install``."""

    def matches(self, plugin: Plugin) -> bool:
        raise NotImplementedError

    def install(self, plugin: Plugin) -> None:
        raise NotImplementedError


class PluginManager:
    """Loads ``--tt_plugin`` files and routes loaded plugins to consumers via ``try_install``."""

    def __init__(self, paths: list[str] | None = None) -> None:
        self.plugins: list[Plugin] = []
        self.load(paths or [])

    def load(self, paths: list[str]) -> None:
        """Import each plugin file and instantiate its registered plugin classes (deduped by class)."""
        for entry in paths:
            plugin_path, sep, config = entry.partition(":")
            plugin_path = os.path.abspath(os.path.expanduser(plugin_path))
            if not os.path.isfile(plugin_path):
                raise FileNotFoundError(f"--tt_plugin does not point to a file: {plugin_path}")
            bound = os.path.abspath(os.path.expanduser(config)) if sep and config.strip() else None
            module = self._import_plugin(plugin_path)

            for cls in vars(module).values():
                if not (isinstance(cls, type) and issubclass(cls, Plugin) and getattr(cls, "plugin_name", None)):
                    continue
                if any(type(x) is cls for x in self.plugins):
                    continue
                self.plugins.append(cls(config=bound))
                logger.info_rank0(
                    "Initialized %s plugin '%s' (%s)",
                    getattr(cls, "kind", "?"), cls.plugin_name, cls.__name__,
                )

    def try_install(self, consumer: PluginConsumer) -> None:
        """Install each plugin the consumer claims; failures are logged and skipped."""
        for plugin in self.plugins:
            if not consumer.matches(plugin):
                continue
            try:
                consumer.install(plugin)
            except Exception:
                logger.warning_rank0(
                    "Plugin consumer %s failed on plugin '%s' (%s); skipping.",
                    type(consumer).__name__,
                    getattr(plugin, "plugin_name", "?"),
                    type(plugin).__name__,
                    exc_info=True,
                )

    @staticmethod
    def _import_plugin(path: str):
        """Import (once) the plugin .py at ``path`` and return its module object."""
        plugin_dir = os.path.dirname(path)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        module_name = f"torchturbo_plugin.{os.path.splitext(os.path.basename(path))[0]}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load plugin module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def register_plugin(name: str, kind: PluginKind) -> Callable:
    """Decorator marking a ``Plugin`` subclass under ``name`` with ``kind``."""
    def decorator(cls: type[Plugin]) -> type[Plugin]:
        if not issubclass(cls, Plugin):
            raise TypeError(f"{cls.__name__} must subclass Plugin")
        cls.plugin_name = name
        cls.kind = kind
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Plugin API surfaces (clearly-named, intent-obvious; consumers down-cast by `kind`)
# ---------------------------------------------------------------------------


class DatasetPlugin(Plugin):
    """A dataset source; ``dataset_info`` is provider-defined metadata, ``samples`` is the data."""

    kind = PluginKind.DATASET

    def dataset_info(self) -> dict[str, dict[str, Any]]:
        """``{dataset_name: <provider-defined metadata>}``; schema is owned by the provider."""
        raise NotImplementedError

    def samples(self, name: str, cache_dir: str | None = None) -> Iterable:
        """The data rows for dataset ``name``."""
        raise NotImplementedError


class PackingPlugin(Plugin):
    """A packing algorithm: assigns tokenized samples to fixed-length packs."""

    kind = PluginKind.PACKING

    def pack_groups(self, sequences, packing_length: int, *, is_finished: bool = True):
        """Group ``(idx, length)`` pairs into packs up to ``packing_length``."""
        raise NotImplementedError

    def knapsack(self, numbers: list[int], capacity: int) -> list[list[int]]:
        """Group raw lengths into packs up to ``capacity``."""
        raise NotImplementedError


__all__ = [
    "Plugin",
    "PluginKind",
    "PluginConsumer",
    "PluginManager",
    "DatasetPlugin",
    "PackingPlugin",
    "register_plugin",
]