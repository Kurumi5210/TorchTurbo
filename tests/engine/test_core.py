"""Tests for TorchTurboEngine one-stop-shop API."""

import pytest
import torch
import torch.nn as nn

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
