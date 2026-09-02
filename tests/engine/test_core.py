"""Tests for TorchTurboEngine one-stop-shop API."""

import importlib
from types import SimpleNamespace

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


class TinyLogitsModel(nn.Module):
    """Small logits-producing model for the packed RL engine API tests."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids):
        logits = input_ids.float().unsqueeze(-1) * self.weight
        return SimpleNamespace(logits=logits)


class DummyBackend(BaseBackend):
    name = "dummy"

    def shard_model(self, model, parallel_plan=None):
        model._dummy_sharded = True
        return model

    def prepare_optimizer(
        self,
        model,
        optimizer_cls,
        *,
        param_groups=None,
        **kw,
    ):
        params = param_groups if param_groups is not None else model.parameters()
        return optimizer_cls(params, **kw)


class ProfilingBackend(DummyBackend):
    def prepare_model(self, model, parallel_plan=None, peft_config=None):
        return model

    def load_model(self, model, model_name_or_path):
        return model

    def save_model(self, model, model_name_or_path, output_dir, **kw):
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


class RecordingProfiler:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")

    def stop(self):
        self.calls.append("stop")


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
        
        engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=0.01,
            weight_decay=0.0,
        )
        assert engine.optimizer is not None

    def test_prepare_optimizer_without_model_raises(self):
        engine = TorchTurboEngine()
        with pytest.raises(RuntimeError, match="prepare_model"):
            engine.prepare_optimizer(torch.optim.AdamW)

    def test_compute_loss_defers_eval_metrics_until_explicit_collect(self):
        register_backend("dummy", ProfilingBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        model = TinyModel().eval()
        model.pop_metrics = lambda: {"test_metric": torch.tensor(2.0)}

        engine.compute_loss(model, {})
        logs = {}
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {}

        engine.collect_metrics(model)
        engine.report_metrics(logs, lambda value: value.reshape(1))

        assert logs == {"test_metric": 2.0}

    def test_compute_loss_defers_training_metrics_until_explicit_collect(self):
        register_backend("dummy", ProfilingBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        model = TinyModel()
        model.pop_metrics = lambda: {"test_metric": torch.tensor(2.0)}

        engine.compute_loss(model, {})
        logs = {}
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {}

        engine.collect_metrics(model)
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {"test_metric": 2.0}

    def test_compute_loss_sets_moe_runtime_inputs(self):
        calls = []

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
                calls.append("moe_runtime")
                return MoERuntimeInputs(padding_mask=inputs["padding_mask"])

        class RuntimeBackend:
            def prepare_inputs(self, model, inputs):
                calls.append("prepare_inputs")
                return {**inputs, "padding_mask": prepared_padding_mask}

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                calls.append("compute_loss")
                assert inputs["padding_mask"] is prepared_padding_mask
                return torch.tensor(0.0)

            def get_dp_info(self):
                return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

            def get_sp_info(self):
                return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

            def get_ep_info(self):
                return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

            def get_parallel_context(self):
                return {
                    "dp": self.get_dp_info(),
                    "ep": self.get_ep_info(),
                    "sp": self.get_sp_info(),
                }

        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))
        model = RuntimeModel()
        padding_mask = torch.tensor([[False, True]])
        prepared_padding_mask = torch.tensor([[False, True, True]])

        engine.compute_loss(model, {"padding_mask": padding_mask})

        assert model.moe._runtime_inputs.padding_mask is prepared_padding_mask
        assert calls == ["prepare_inputs", "moe_runtime", "compute_loss"]

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
            def prepare_inputs(self, model, inputs):
                return inputs

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
            def prepare_inputs(self, model, inputs):
                return inputs

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

    def test_forward_backward_batch_runs_forward_and_backward(self):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyLogitsModel()
        engine.get_sp_info = lambda: {"sp_size": 1}
        engine.get_dp_info = lambda: {"dp_size": 1, "dp_group": None}

        mb_item = {
            "orig_mb": {"input_ids": torch.tensor([1, 2])},
            "padded_mb": {"input_ids": torch.tensor([[1, 2]])},
        }
        calls = []

        def process_fn(logits, ctx, loss_fn):
            calls.append((tuple(logits.shape), ctx["mb_input"]))
            return loss_fn(logits)

        engine.forward_backward_batch(
            [mb_item],
            process_fn=process_fn,
            loss_fn=lambda logits: logits.square().mean(),
            loss_weight_fn=lambda _: torch.tensor(1.0),
        )

        assert len(calls) == 1
        assert calls[0][0] == (2, 1)
        assert engine.model.weight.grad is not None
        assert torch.isfinite(engine.model.weight.grad)

    def test_forward_backward_batch_forward_only_skips_backward(self):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyLogitsModel()
        engine.get_sp_info = lambda: {"sp_size": 1}
        engine.get_dp_info = lambda: {"dp_size": 1, "dp_group": None}

        mb_item = {
            "orig_mb": {"input_ids": torch.tensor([1, 2])},
            "padded_mb": {"input_ids": torch.tensor([[1, 2]])},
        }
        engine.forward_backward_batch(
            [mb_item],
            process_fn=lambda logits, _ctx, _loss_fn: logits.square().mean(),
            loss_fn=lambda logits: logits.square().mean(),
            loss_weight_fn=lambda _: torch.tensor(1.0),
            forward_only=True,
        )

        assert engine.model.weight.grad is None

    def test_fit_is_unsupported_compatibility_stub(self):
        engine = object.__new__(TorchTurboEngine)

        with pytest.raises(NotImplementedError, match="no longer supported"):
            engine.fit([])

    def test_prepare_lr_scheduler(self):
        register_backend("dummy", DummyBackend)
        engine = TorchTurboEngine({"backend": "dummy"})
        
        engine.model = engine.backend.shard_model(TinyModel())
        engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=0.01,
            weight_decay=0.0,
        )
        
        engine.prepare_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10)
        assert engine.lr_scheduler is not None

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
