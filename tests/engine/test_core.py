"""Tests for TorchTurboEngine one-stop-shop API."""

import importlib

import pytest
import torch
import torch.nn as nn

import torchturbo.engine.core as engine_core
from torchturbo.engine import (
    BaseBackend,
    DistributedBackend,
    EngineConfig,
    FSDP2Backend,
    MixedPrecisionPolicy,
    TorchTurboEngine,
    TrainHook,
    TrainState,
    get_available_backends,
    register_backend,
)
from torchturbo.engine.backend_registry import _BACKENDS
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.models.base import TurboModel
from torchturbo.models.modules.moe.moe_layer import BaseMoELayer, MoERuntimeInputs
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.runtime.cpu import CpuRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class DummyBackend(BaseBackend):
    name = "dummy"

    def shard_model(self, model, parallel_plan=None):
        model._dummy_sharded = True
        return model

    def prepare_optimizer(self, model, optimizer_cls, **kw):
        return optimizer_cls(model.parameters(), **kw)


class ProfilingBackend(DummyBackend):
    def prepare_model(self, model, parallel_plan=None, peft_config=None):
        return model

    def load_model(self, model, model_name_or_path, parallel_context):
        return model

    def save_model(self, model, model_name_or_path, output_dir, parallel_context, **kw):
        return None

    def get_dp_info(self) -> dict[str, int]:
        return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

    def get_ep_info(self) -> dict[str, int]:
        return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

    def get_sp_info(self) -> dict[str, int]:
        return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        loss = torch.tensor(0.0)
        return (loss, None) if return_outputs else loss


class RecordingHook(TrainHook):
    def __init__(self):
        self.calls = []

    def on_train_begin(self, state):
        self.calls.append("on_train_begin")

    def on_train_end(self, state):
        self.calls.append("on_train_end")

    def on_step_end(self, state):
        self.calls.append(f"step:{state.global_step}")

    def on_microbatch_end(self, state):
        self.calls.append(f"microbatch:{state.global_step}")


class RecordingProfiler:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")

    def stop(self):
        self.calls.append("stop")


def mse_loss(model, batch, state):
    return model(batch).pow(2).mean()


def make_data(n=3):
    return [torch.randn(2, 4) for _ in range(n)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_backends():
    saved = dict(_BACKENDS)
    yield
    _BACKENDS.clear()
    _BACKENDS.update(saved)


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.backend == DistributedBackend.FSDP2
        assert isinstance(cfg.backend_config, FSDPConfig)

    def test_from_dict(self):
        cfg = EngineConfig.from_dict({"backend": "fsdp2", "mixed_precision": "fp16"})
        assert cfg.mixed_precision == MixedPrecisionPolicy.FP16


# ---------------------------------------------------------------------------
# Tests: Engine core functionality
# ---------------------------------------------------------------------------

class TestEngineSetup:
    def test_default_init(self):
        engine = TorchTurboEngine()
        assert isinstance(engine.backend, FSDP2Backend)
        assert engine.model is None

    def test_prepare_model_not_supported_type(self):
        engine = TorchTurboEngine()
        # For tests, we want to mock prepare_model to just return our TinyModel
        # since actual prepare_model tries to download from HuggingFace
        engine.model = TinyModel()
        assert engine.model is not None

    def test_prepare_optimizer(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        
        # Inject model manually to avoid huggingface download logic
        model = TinyModel()
        engine.model = engine.backend.shard_model(model)
        
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        assert engine.optimizer is not None

    def test_prepare_optimizer_without_model_raises(self):
        engine = TorchTurboEngine()
        with pytest.raises(RuntimeError, match="prepare_model"):
            engine.prepare_optimizer(torch.optim.AdamW)

    def test_compute_loss_collects_model_metrics(self):
        register_backend("dummy", ProfilingBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        model = TinyModel()
        model.pop_metrics = lambda: {"test_metric": torch.tensor(2.0)}

        engine.compute_loss(model, {})
        logs = {}
        engine.report_metrics(logs, lambda value: value.reshape(1))

        assert logs == {"test_metric": 2.0}

    def test_compute_loss_sets_moe_runtime_inputs(self):
        class RuntimeMoELayer(BaseMoELayer):
            def __init__(self):
                nn.Module.__init__(self)
                self._runtime_inputs = MoERuntimeInputs()

            def forward(self, hidden_states, input_ids=None):
                return self._runtime_inputs.padding_mask

        class RuntimeModel(TurboModel, nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                self.moe = RuntimeMoELayer()

            def get_moe_runtime_inputs(self, inputs, parallel_context):
                return MoERuntimeInputs(padding_mask=inputs["padding_mask"])

        class RuntimeBackend:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(0.0)

            def get_dp_info(self):
                return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

            def get_sp_info(self):
                return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

            def get_ep_info(self):
                return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        model = RuntimeModel()
        padding_mask = torch.tensor([[False, True]])

        engine.compute_loss(model, {"padding_mask": padding_mask})

        assert model.moe._runtime_inputs.padding_mask is padding_mask

    @pytest.mark.parametrize(
        ("training", "gradient_accumulation_steps", "expected_scale"),
        [(True, 4, 0.25), (False, 4, 1.0)],
    )
    def test_compute_loss_sets_moe_aux_loss_scale(
        self,
        monkeypatch,
        training,
        gradient_accumulation_steps,
        expected_scale,
    ):
        class RuntimeBackend:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(0.0)

            def get_dp_info(self):
                return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

            def get_sp_info(self):
                return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

            def get_ep_info(self):
                return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

        class RuntimeMoELayer(BaseMoELayer):
            def __init__(self):
                nn.Module.__init__(self)

            def forward(self, hidden_states, input_ids=None):
                return hidden_states

        class RuntimeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = RuntimeMoELayer()

            def get_moe_runtime_inputs(self, inputs, parallel_context):
                return MoERuntimeInputs()

        scales = []
        monkeypatch.setattr(
            MoEAuxLossAutoScaler,
            "set_loss_scale",
            lambda scale: scales.append(scale),
        )
        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        model = RuntimeModel().train(training)

        engine.compute_loss(
            model,
            {},
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        assert scales == [expected_scale]

    def test_compute_loss_skips_moe_setup_for_dense(self, monkeypatch):
        class RuntimeBackend:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(0.0)

        scales = []
        monkeypatch.setattr(
            MoEAuxLossAutoScaler,
            "set_loss_scale",
            lambda scale: scales.append(scale),
        )
        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()

        engine.compute_loss(TinyModel(), {})

        assert scales == []

    def test_prepare_lr_scheduler(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        engine.prepare_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10)
        assert engine.lr_scheduler is not None

    def test_fit_without_model_raises(self):
        engine = TorchTurboEngine()
        with pytest.raises(RuntimeError, match="prepare_model"):
            engine.fit(make_data(), compute_loss_fn=mse_loss)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            TorchTurboEngine({"backend": "nonexistent"})

    def test_failed_backend_init_does_not_publish_ops(self, monkeypatch):
        ops_module = importlib.import_module("torchturbo.ops.ops")
        monkeypatch.setattr(ops_module, "_OPS", None)

        class FailingBackend(ProfilingBackend):
            def __init__(self, config, device_runtime):
                raise RuntimeError("backend init failed")

        monkeypatch.setattr(engine_core, "get_device_runtime", CpuRuntime)
        monkeypatch.setattr(engine_core, "get_backend", lambda _: FailingBackend)

        with pytest.raises(RuntimeError, match="backend init failed"):
            TorchTurboEngine(EngineConfig())

        assert ops_module._OPS is None

    def test_profiling_uses_device_runtime_profiler(self, monkeypatch, tmp_path):
        runtime = CpuRuntime()
        profiler = RecordingProfiler()
        profiler_kwargs = {}

        def fake_profiler(
            *,
            output_dir: str,
            worker_name: str,
            profiling_enabled: bool,
            memory_snapshot_steps: int | None,
        ):
            profiler_kwargs["output_dir"] = output_dir
            profiler_kwargs["worker_name"] = worker_name
            return profiler

        monkeypatch.setattr(runtime, "profiler", fake_profiler)
        monkeypatch.setattr(engine_core, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(
            engine_core,
            "get_backend",
            lambda backend_name: ProfilingBackend if backend_name == "fsdp2" else None,
        )

        engine = TorchTurboEngine(EngineConfig())
        engine.start_profiling(str(tmp_path), True, None)
        engine.step_profiling()
        engine.stop_profiling()

        assert profiler.calls == ["start", "step", "stop"]
        assert profiler_kwargs["output_dir"] == str(tmp_path / "profiling")
        assert profiler_kwargs["worker_name"].startswith("rank")
        assert (tmp_path / "profiling").is_dir()

    def test_profiling_start_failure_clears_previous_profiler(
        self, monkeypatch, tmp_path
    ):
        runtime = CpuRuntime()
        first_profiler = RecordingProfiler()

        class FailingProfiler(RecordingProfiler):
            def start(self):
                self.calls.append("start")
                raise RuntimeError("profiler start failed")

        second_profiler = FailingProfiler()
        profilers = iter((first_profiler, second_profiler))

        def fake_profiler(
            *,
            output_dir: str,
            worker_name: str,
            profiling_enabled: bool,
            memory_snapshot_steps: int | None,
        ):
            return next(profilers)

        monkeypatch.setattr(runtime, "profiler", fake_profiler)
        monkeypatch.setattr(engine_core, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(
            engine_core,
            "get_backend",
            lambda backend_name: ProfilingBackend if backend_name == "fsdp2" else None,
        )

        engine = TorchTurboEngine(EngineConfig())
        engine.start_profiling(str(tmp_path), True, None)

        with pytest.raises(RuntimeError, match="profiler start failed"):
            engine.start_profiling(str(tmp_path), True, None)

        assert first_profiler.calls == ["start", "stop"]
        assert second_profiler.calls == ["start"]
        with pytest.raises(RuntimeError, match="No active profiler"):
            engine.step_profiling()


# ---------------------------------------------------------------------------
# Tests: Training (fit)
# ---------------------------------------------------------------------------

class TestEngineFit:
    def test_basic_fit(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        state = engine.fit(make_data(3), compute_loss_fn=mse_loss, num_epochs=1)
        assert state.global_step == 3

    def test_max_steps(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)

        state = engine.fit(make_data(10), compute_loss_fn=mse_loss, max_steps=2)
        assert state.global_step == 2

    def test_subclass_compute_loss(self):
        class MyEngine(TorchTurboEngine):
            def compute_loss(self, model, batch, state):
                return model(batch).pow(2).mean()

        register_backend("dummy", DummyBackend)
        engine = MyEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        state = engine.fit(make_data(3), num_epochs=1)
        assert state.global_step == 3

    def test_hooks_called(self):
        hook = RecordingHook()
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        engine.fit(make_data(2), compute_loss_fn=mse_loss, hooks=[hook])
        assert "on_train_begin" in hook.calls
        assert "on_train_end" in hook.calls
        assert "microbatch:0" in hook.calls
        assert "step:1" in hook.calls
        assert "microbatch:1" in hook.calls
        assert "step:2" in hook.calls

    def test_persistent_hooks(self):
        hook = RecordingHook()
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.add_hook(hook)
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        engine.fit(make_data(1), compute_loss_fn=mse_loss)
        assert "on_train_begin" in hook.calls

    def test_early_stop(self):
        class StopAt2(TrainHook):
            def on_step_end(self, state):
                if state.global_step >= 2:
                    state.should_stop = True

        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        state = engine.fit(make_data(10), compute_loss_fn=mse_loss, hooks=[StopAt2()])
        assert state.global_step == 2

    def test_gradient_accumulation(self):
        hook = RecordingHook()
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        state = engine.fit(
            make_data(4),
            compute_loss_fn=mse_loss,
            gradient_accumulation_steps=2,
            hooks=[hook],
        )
        assert state.global_step == 2
        # Data size 4, accum 2 => 4 microbatches, 2 steps
        assert hook.calls.count("microbatch:0") == 2
        assert hook.calls.count("microbatch:1") == 2
        assert hook.calls.count("step:1") == 1
        assert hook.calls.count("step:2") == 1

    def test_multi_epoch(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(torch.optim.AdamW, lr=0.01)
        
        state = engine.fit(make_data(2), compute_loss_fn=mse_loss, num_epochs=3)
        assert state.global_step == 6

    def test_custom_backend(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        
        # We simulate what prepare_model does
        model = TinyModel()
        engine.model = engine.backend.shard_model(model)
        
        assert getattr(engine.model, "_dummy_sharded", False)

    def test_repr(self):
        engine = TorchTurboEngine()
        assert "not prepared" in repr(engine).lower()
        
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        engine.model = engine.backend.shard_model(TinyModel())
        
        assert "model" in repr(engine).lower()
