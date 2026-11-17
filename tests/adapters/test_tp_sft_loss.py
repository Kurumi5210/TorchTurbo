"""TP SFT adapters reuse prepared training batches but prepare evaluation batches."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _Engine:
    def __init__(self, *, etp_size=1, gbs=8):
        self.etp_size = etp_size
        self.config = SimpleNamespace(gbs=gbs)
        self.calls = []

    def compute_loss(self, model, inputs, **kwargs):
        self.calls.append("prepare")
        return torch.tensor(1.), SimpleNamespace(main_loss=torch.tensor(.5))

    def compute_prepared_loss(self, model, inputs, **kwargs):
        self.calls.append("prepared")
        return torch.tensor(1.), SimpleNamespace(main_loss=torch.tensor(.5))

    def get_sp_info(self):
        return {"sp_size": 1}

    def get_tp_info(self):
        return {"tp_size": 1}

    def get_etp_info(self):
        return {"etp_size": self.etp_size}


def _llamafactory(engine, monkeypatch):
    from torchturbo.adapters.llamafactory_v0.trainer import TorchTurboTrainerBase

    trainer = object.__new__(TorchTurboTrainerBase)
    trainer.engine = engine
    return trainer, trainer._compute_causal_lm_loss


def _msswift(engine, monkeypatch):
    from torchturbo.adapters.msswift import trainer as swift_trainer

    monkeypatch.setattr(swift_trainer.engine_registry, "get_engine_or_none", lambda: engine)
    trainer = object.__new__(swift_trainer.TurboTrainerMixin)
    return trainer, trainer.compute_loss


@pytest.mark.parametrize("adapter", [_llamafactory, _msswift], ids=["llamafactory", "msswift"])
@pytest.mark.parametrize("stage", ["sft", "pt"])
def test_etp_only_activates_prepared_loss(monkeypatch, adapter, stage):
    engine = _Engine(etp_size=2)
    trainer, _ = adapter(engine, monkeypatch)
    trainer.model = SimpleNamespace(is_diffusion_model=False)
    trainer.model_accepts_loss_kwargs = False

    trainer._activate_tp_sft(engine, stage)

    assert trainer._tp_sft_enabled is True
    assert trainer.model_accepts_loss_kwargs is True


@pytest.mark.parametrize("adapter", [_llamafactory, _msswift], ids=["llamafactory", "msswift"])
@pytest.mark.parametrize(
    ("stage", "gbs", "error"),
    [("dpo", 8, "SFT/PT training only"), ("sft", None, "require explicit tt_gbs")],
)
def test_etp_only_rejects_unsupported_training(monkeypatch, adapter, stage, gbs, error):
    engine = _Engine(etp_size=2, gbs=gbs)
    trainer, _ = adapter(engine, monkeypatch)
    trainer.model = SimpleNamespace(is_diffusion_model=False)

    with pytest.raises(ValueError, match=error):
        trainer._activate_tp_sft(engine, stage)

    assert trainer._tp_sft_enabled is False


@pytest.mark.parametrize("adapter", [_llamafactory, _msswift], ids=["llamafactory", "msswift"])
def test_tp_loss_prepares_evaluation_batches(monkeypatch, adapter):
    engine = _Engine()
    trainer, compute_loss = adapter(engine, monkeypatch)
    trainer._tp_sft_enabled = True
    trainer.current_gradient_accumulation_steps = 2
    model = nn.Linear(1, 1)
    # Training batches were prepared by _get_num_items_in_batch. HF evaluation
    # passes that hook a temporary list, so its raw batch must be prepared here.
    for training, expected in ((True, "prepared"), (False, "prepare")):
        model.train(training)
        compute_loss(model, {"labels": torch.ones(1, 2)}, num_items_in_batch=3)
        assert engine.calls[-1] == expected
