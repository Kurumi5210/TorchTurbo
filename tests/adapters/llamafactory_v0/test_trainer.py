from __future__ import annotations

from types import SimpleNamespace

import torch

from torchturbo.adapters.llamafactory_v0.trainer import TorchTurboTrainerBase


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
