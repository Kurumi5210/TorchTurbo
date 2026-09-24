from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchturbo.adapters.llamafactory_v0.trainer import (
    TorchTurboTrainerBase,
    make_torchturbo_trainer,
)


class _DiffusionEngine:
    def __init__(self) -> None:
        self.call = None

    def compute_diffusion_loss(
        self,
        model,
        inputs,
        *,
        ga,
        return_outputs,
        **kwargs,
    ):
        self.call = {
            "model": model,
            "inputs": inputs,
            "ga": ga,
            "return_outputs": return_outputs,
            "kwargs": kwargs,
        }
        return torch.tensor(1.5), SimpleNamespace()

    def step_profiling(self):
        raise AssertionError("diffusion microbatches must not advance profiling")


def test_diffusion_loss_delegates_without_advancing_profiler():
    engine = _DiffusionEngine()
    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = engine
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    model = object()
    inputs = {"input_ids": torch.tensor([[1, 2]])}

    loss, outputs = trainer._compute_diffusion_loss(
        model,
        inputs,
        return_outputs=True,
        num_items_in_batch=2,
    )

    torch.testing.assert_close(loss, torch.tensor(1.5))
    assert isinstance(outputs, SimpleNamespace)
    assert engine.call == {
        "model": model,
        "inputs": inputs,
        "ga": 4,
        "return_outputs": True,
        "kwargs": {"num_items_in_batch": 2},
    }


@pytest.mark.parametrize("loss_offset", [0.0, float("nan"), float("inf")])
def test_training_step_checks_loss_before_metrics_and_optimizer_step(loss_offset):
    model = torch.nn.Linear(1, 1, bias=False)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model.weight.sum() * 2 + loss_offset
    returned_loss = loss.detach()
    collected_models = []

    class BaseTrainer:
        def training_step(self, model, inputs):
            loss.backward()
            return returned_loss

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=SimpleNamespace(collect_metrics=collected_models.append),
        stage="sft",
    )
    trainer = object.__new__(trainer_cls)

    if loss_offset == 0.0:
        assert trainer.training_step(model, {}) is returned_loss
        assert collected_models == [model]
        optimizer.step()
        torch.testing.assert_close(model.weight, initial_weight - 0.2)
    else:
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.training_step(model, {})
            optimizer.step()
        assert collected_models == []
        torch.testing.assert_close(model.weight, initial_weight)

    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 2.0))


def test_muon_optimizer_bypasses_framework_optimizer_resolution():
    optimizer = object()

    class Engine:
        config = SimpleNamespace(
            optimizer=SimpleNamespace(name="muon", muon=SimpleNamespace(qk_clip=None)),
        )

        def prepare_optimizer(self, *args, **kwargs):
            assert args == ()
            assert kwargs == {}
            return optimizer

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.optimizer = None

    assert trainer.create_optimizer() is optimizer


def test_external_optimizer_is_rejected():
    class BaseTrainer:
        def __init__(self, *, optimizers):
            self.optimizer, self.lr_scheduler = optimizers

    trainer_cls = make_torchturbo_trainer(
        BaseTrainer,
        engine=object(),
        stage="sft",
    )

    with pytest.raises(
        ValueError,
        match="does not support externally supplied optimizers",
    ):
        trainer_cls(optimizers=(object(), None))


def test_checkpoint_hooks_use_engine_optimizer():
    raw_optimizer = object()
    wrapped_optimizer = object()
    calls = []

    class Engine:
        def __init__(self):
            self.optimizer = raw_optimizer

        def save_optimizer_and_scheduler(self, *args):
            calls.append(("save", args))

        def load_optimizer_and_scheduler(self, *args):
            calls.append(("load", args))

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = Engine()
    trainer.model = object()
    trainer.optimizer = wrapped_optimizer
    trainer.lr_scheduler = object()

    trainer._save_optimizer_and_scheduler("output")
    trainer._load_optimizer_and_scheduler("checkpoint")

    assert calls[0][1][1] is raw_optimizer
    assert calls[1][1][1] is raw_optimizer
