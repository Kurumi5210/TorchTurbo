"""Tests for dependency-light engine utilities."""

from torchturbo.engine.utils import strip_peft_prefix


def test_strip_peft_prefix_normalizes_wrapped_names():
    name = "base_model.model.model.layers.0.mlp.experts.base_layer.weight1"

    assert strip_peft_prefix(name) == "model.layers.0.mlp.experts.weight1"
