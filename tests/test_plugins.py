"""Plugin system tests; ``torchturbo.engine.plugins`` loaded directly with a stubbed
logger to avoid the torch chain. Each test gets a fresh ``PluginManager`` (no shared
global state)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGINS_PATH = os.path.join(_REPO_ROOT, "src", "torchturbo", "engine", "plugins.py")


def _load_plugins_module():
    """Load ``torchturbo.engine.plugins`` directly with ``torchturbo.logging`` stubbed."""
    tt = types.ModuleType("torchturbo")
    tt.__path__ = [os.path.join(_REPO_ROOT, "src", "torchturbo")]
    logging_mod = types.ModuleType("torchturbo.logging")

    class _StubLogger:
        def info(self, *a, **k):
            pass

        def info_rank0(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def warning_rank0(self, *a, **k):
            pass

    logging_mod.get_logger = lambda *a, **k: _StubLogger()
    sys.modules.setdefault("torchturbo", tt)
    sys.modules.setdefault("torchturbo.logging", logging_mod)
    spec = importlib.util.spec_from_file_location("torchturbo.engine.plugins", _PLUGINS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["torchturbo.engine.plugins"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _cleanup_torchturbo_plugin_modules():
    yield
    for name in [n for n in sys.modules if n.startswith("torchturbo_plugin.")]:
        del sys.modules[n]


@pytest.fixture
def module():
    return _load_plugins_module()


@pytest.fixture
def manager(module):
    return module.PluginManager()


def _write_plugin(tmp_path, body: str, name: str = "plug.py") -> str:
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def test_register_plugin_rejects_non_plugin_subclass(module, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import Plugin, PluginKind, register_plugin\n"
        "@register_plugin('bad', PluginKind.DATASET)\nclass Bad:\n"
        "    pass\n",
    )
    with pytest.raises(TypeError):
        module.PluginManager([plugin_file])


def test_load_populates_dataset_plugin(module, manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import DatasetPlugin, PluginKind, register_plugin\n"
        "@register_plugin('ds', PluginKind.DATASET)\nclass Ds(DatasetPlugin):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        self.names_ = ['a','b']\n"
        "    def dataset_info(self): return {n: {} for n in self.names_}\n"
        "    def samples(self, name, cache_dir=None): return [name]\n",
    )
    manager.load([plugin_file])
    assert len(manager.plugins) == 1
    assert manager.plugins[0].kind == module.PluginKind.DATASET
    assert manager.plugins[0].dataset_info() == {"a": {}, "b": {}}


def test_load_populates_packing_plugin(module, manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import PackingPlugin, PluginKind, register_plugin\n"
        "@register_plugin('pk', PluginKind.PACKING)\nclass Pk(PackingPlugin):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        self.max_pack_sample_num = 7\n"
        "    def pack_groups(self, s, l, *, is_finished=True): return [], []\n"
        "    def knapsack(self, n, c): return []\n",
    )
    manager.load([plugin_file])
    assert manager.plugins[0].kind == module.PluginKind.PACKING
    assert manager.plugins[0].max_pack_sample_num == 7


def test_plugin_config_binding_passed_to_init(manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import DatasetPlugin, PluginKind, register_plugin\n"
        "RAN=[]\n"
        "@register_plugin('ds', PluginKind.DATASET)\nclass Ds(DatasetPlugin):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        RAN.append(self.config)\n"
        "    def dataset_info(self): return {}\n"
        "    def samples(self, name, cache_dir=None): return []\n",
    )
    cfg = tmp_path / "datasets.json"
    cfg.write_text("{}")
    manager.load([f"{plugin_file}:{cfg}"])
    assert manager.plugins[0].config == str(cfg)


def test_load_only_does_not_register_modules_without_decorator(manager, tmp_path):
    plugin_file = _write_plugin(tmp_path, "MARKER = 'loaded'\n")
    manager.load([plugin_file])
    assert manager.plugins == []
    assert sys.modules[f"torchturbo_plugin.{os.path.splitext(os.path.basename(plugin_file))[0]}"].MARKER == "loaded"


def test_missing_file_raises(manager, tmp_path):
    with pytest.raises(FileNotFoundError):
        manager.load([str(tmp_path / "nope.py")])


def test_try_install_runs_consumer_only_on_matching_plugins(module, manager, tmp_path):
    plugin_file = _write_plugin(
        tmp_path,
        "from torchturbo.engine.plugins import PackingPlugin, PluginKind, register_plugin\n"
        "@register_plugin('pk', PluginKind.PACKING)\nclass Pk(PackingPlugin):\n"
        "    def pack_groups(self, s, l, *, is_finished=True): return [], []\n"
        "    def knapsack(self, n, c): return [[x] for x in n]\n",
    )
    manager.load([plugin_file])

    seen: list = []

    class _C(module.PluginConsumer):
        def matches(self, plugin):
            return plugin.kind == module.PluginKind.PACKING

        def install(self, plugin):
            seen.append(plugin.knapsack([1, 2, 3], 100))

    class _NonMatching(module.PluginConsumer):
        def matches(self, plugin):
            return plugin.kind == module.PluginKind.DATASET

        def install(self, plugin):
            seen.append("nope")

    manager.try_install(_C())
    manager.try_install(_NonMatching())
    assert seen == [[[1], [2], [3]]]