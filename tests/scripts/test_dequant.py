from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dequant"
sys.path.insert(0, str(SCRIPT_DIR))

import converter  # noqa: E402
import dequantize  # noqa: E402
import deepseekv4  # noqa: E402
import kimik26  # noqa: E402


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index = {"metadata": {"source": "test"}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _read_index(model_dir: Path) -> dict:
    return json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )


def _read_tensor(model_dir: Path, index: dict, key: str) -> torch.Tensor:
    shard_path = model_dir / index["weight_map"][key]
    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _write_minimal_deepseek_source(
    model_dir: Path,
    *,
    include_scale: bool = True,
) -> None:
    model_dir.mkdir()
    weight_key = "model.layers.0.ffn.experts.0.w1.weight"
    scale_key = "model.layers.0.ffn.experts.0.w1.scale"
    tensors = {weight_key: torch.tensor([[0x21]], dtype=torch.uint8)}
    if include_scale:
        tensors[scale_key] = torch.tensor([[2.0]])
    save_file(tensors, model_dir / "model.safetensors")
    _write_index(
        model_dir,
        {key: "model.safetensors" for key in tensors},
    )


def test_apply_scales_to_bf16():
    result = dequantize.Dequantizer.apply_scales_to_bf16(
        torch.tensor([[1, 2]], dtype=torch.int8),
        torch.tensor([[0.5, 2.0]]),
    )

    assert result.dtype is torch.bfloat16
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.float(), torch.tensor([[0.5, 4.0]]), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="does not match"):
        dequantize.Dequantizer.apply_scales_to_bf16(
            torch.ones(1, 2),
            torch.ones(1, 1),
        )


def test_model_formats_resolve_direct_and_quantized_sources():
    assert issubclass(
        deepseekv4.DeepSeekV4Format,
        dequantize.DequantizationFormat,
    )
    assert issubclass(
        kimik26.KimiK26Format,
        dequantize.DequantizationFormat,
    )

    deepseek_keys = {
        "direct.weight": "direct.safetensors",
        "deepseek.weight": "weight.safetensors",
        "deepseek.scale": "scale.safetensors",
    }
    kimi_keys = {
        "direct.weight": "direct.safetensors",
        "kimi.weight_packed": "packed.safetensors",
        "kimi.weight_scale": "scale.safetensors",
        "kimi.weight_shape": "shape.safetensors",
    }

    deepseek_dequantizer = dequantize.Dequantizer(
        deepseekv4.DeepSeekV4Format()
    )
    deepseek_sources = deepseek_dequantizer.resolve_sources(deepseek_keys)
    kimi_sources = kimik26.KimiK26Format(32).resolve_sources(kimi_keys)

    assert not deepseek_sources["direct.weight"].quantized
    assert deepseek_sources["deepseek.weight"].quantized
    assert deepseek_sources["deepseek.weight"].physical_keys == (
        "deepseek.weight",
        "deepseek.scale",
    )
    assert not kimi_sources["direct.weight"].quantized
    assert kimi_sources["kimi.weight"].quantized
    assert kimi_sources["kimi.weight"].physical_keys == (
        "kimi.weight_packed",
        "kimi.weight_scale",
        "kimi.weight_shape",
    )


def test_resolve_checkpoint_sources_rejects_incomplete_kimi_group():
    with pytest.raises(ValueError, match="missing shape"):
        kimik26.KimiK26Format(32).resolve_sources(
            {
                "kimi.weight_packed": "packed.safetensors",
                "kimi.weight_scale": "scale.safetensors",
            }
        )


def test_dequantize_deepseek_mxfp4():
    packed = torch.tensor([[0x21]], dtype=torch.uint8)
    scale = torch.tensor([[2.0]], dtype=torch.float32)

    result = deepseekv4.DeepSeekV4Format.dequantize_weight(packed, scale)

    assert result.dtype is torch.bfloat16
    torch.testing.assert_close(
        result.float(), torch.tensor([[1.0, 2.0]]), rtol=0, atol=0
    )


def test_dequantize_deepseek_fp8():
    weight = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]], dtype=torch.float32)

    result = deepseekv4.DeepSeekV4Format.dequantize_weight(weight, scale)

    assert result.dtype is torch.bfloat16
    torch.testing.assert_close(
        result.float(), torch.tensor([[2.0]]), rtol=0, atol=0
    )


def test_dequantize_kimi_int4_with_group_scales():
    unpacked = torch.tensor(
        [[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.int8
    )

    def fake_unpack(packed, num_bits, shape):
        assert packed.dtype is torch.int32
        assert num_bits == 4
        assert shape == (2, 4)
        return unpacked

    tensor_format = kimik26.KimiK26Format(2, unpack_fn=fake_unpack)
    result = tensor_format.dequantize_int4(
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([[0.5, 2.0], [1.0, 0.25]]),
        torch.tensor([2, 4]),
    )

    expected = torch.tensor(
        [[0.5, 1.0, 6.0, 8.0], [-1.0, -2.0, -0.75, -1.0]]
    )
    assert result.dtype is torch.bfloat16
    torch.testing.assert_close(result.float(), expected, rtol=0, atol=0)


def test_deepseek_offline_conversion_handles_cross_shard_scale(tmp_path):
    model_dir = tmp_path / "deepseek"
    output_dir = tmp_path / "deepseek-bf16"
    model_dir.mkdir()
    weight_key = "model.layers.0.ffn.experts.0.w1.weight"
    scale_key = "model.layers.0.ffn.experts.0.w1.scale"
    bias_key = "model.layers.0.ffn.experts.0.w1.bias"
    save_file(
        {
            weight_key: torch.tensor([[0x21]], dtype=torch.uint8),
            bias_key: torch.tensor([3.0], dtype=torch.bfloat16),
        },
        model_dir / "model-00001-of-00002.safetensors",
    )
    save_file(
        {scale_key: torch.tensor([[2.0]])},
        model_dir / "model-00002-of-00002.safetensors",
    )
    _write_index(
        model_dir,
        {
            weight_key: "model-00001-of-00002.safetensors",
            bias_key: "model-00001-of-00002.safetensors",
            scale_key: "model-00002-of-00002.safetensors",
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.model").write_bytes(b"tokenizer")

    summary = converter.Converter(deepseekv4.DeepSeekV4Format()).convert(
        model_dir,
        output_dir,
    )

    index = _read_index(output_dir)
    assert summary.converted_tensors == 1
    assert set(index["weight_map"]) == {weight_key, bias_key}
    assert scale_key not in index["weight_map"]
    assert index["metadata"]["source"] == "test"
    assert index["metadata"]["total_size"] == 6
    torch.testing.assert_close(
        _read_tensor(output_dir, index, weight_key).float(),
        torch.tensor([[1.0, 2.0]]),
        rtol=0,
        atol=0,
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert config["torch_dtype"] == "bfloat16"
    assert (output_dir / "tokenizer.model").read_bytes() == b"tokenizer"


def test_kimi_offline_conversion_handles_components_in_different_shards(
    tmp_path,
):
    model_dir = tmp_path / "kimi"
    output_dir = tmp_path / "kimi-bf16"
    model_dir.mkdir()
    logical_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    packed_key = f"{logical_key}_packed"
    scale_key = f"{logical_key}_scale"
    shape_key = f"{logical_key}_shape"
    save_file(
        {packed_key: torch.tensor([[123]], dtype=torch.int32)},
        model_dir / "packed.safetensors",
    )
    save_file(
        {scale_key: torch.tensor([[0.5, 2.0]])},
        model_dir / "scale.safetensors",
    )
    save_file(
        {shape_key: torch.tensor([1, 4])},
        model_dir / "shape.safetensors",
    )
    _write_index(
        model_dir,
        {
            packed_key: "packed.safetensors",
            scale_key: "scale.safetensors",
            shape_key: "shape.safetensors",
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "config_groups": {
                        "group_0": {"weights": {"group_size": 2}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_unpack(packed, num_bits, shape):
        assert num_bits == 4
        assert shape == (1, 4)
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)

    tensor_format = kimik26.KimiK26Format.from_config(
        model_dir,
        unpack_fn=fake_unpack,
    )
    summary = converter.Converter(tensor_format).convert(
        model_dir,
        output_dir,
    )

    index = _read_index(output_dir)
    assert summary.converted_tensors == 1
    assert index["weight_map"] == {logical_key: "packed.safetensors"}
    assert index["metadata"]["total_size"] == 8
    result = _read_tensor(output_dir, index, logical_key)
    assert result.dtype is torch.bfloat16
    torch.testing.assert_close(
        result.float(), torch.tensor([[0.5, 1.0, 6.0, 8.0]]), rtol=0, atol=0
    )


def test_deepseek_offline_conversion_rejects_missing_scale(tmp_path):
    model_dir = tmp_path / "deepseek"
    output_dir = tmp_path / "deepseek-bf16"
    _write_minimal_deepseek_source(model_dir, include_scale=False)

    with pytest.raises(ValueError, match="no recognized quantization components"):
        converter.Converter(deepseekv4.DeepSeekV4Format()).convert(
            model_dir,
            output_dir,
        )

    assert not output_dir.exists()


def test_deepseek_offline_conversion_requires_quantized_sources(tmp_path):
    model_dir = tmp_path / "deepseek-bf16"
    output_dir = tmp_path / "deepseek-bf16-copy"
    model_dir.mkdir()
    weight_key = "model.layers.0.input_layernorm.weight"
    save_file(
        {weight_key: torch.tensor([1.0], dtype=torch.bfloat16)},
        model_dir / "model.safetensors",
    )
    _write_index(model_dir, {weight_key: "model.safetensors"})

    with pytest.raises(ValueError, match="no quantized tensor sources"):
        converter.Converter(deepseekv4.DeepSeekV4Format()).convert(
            model_dir,
            output_dir,
        )

    assert not output_dir.exists()


def test_deepseek_offline_conversion_rejects_nonempty_output(tmp_path):
    model_dir = tmp_path / "deepseek"
    output_dir = tmp_path / "deepseek-bf16"
    _write_minimal_deepseek_source(model_dir)
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output directory is not empty"):
        converter.Converter(deepseekv4.DeepSeekV4Format()).convert(
            model_dir,
            output_dir,
        )

    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_kimi_offline_conversion_requires_group_size(tmp_path):
    model_dir = tmp_path / "kimi"
    output_dir = tmp_path / "kimi-bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="group_size is missing"):
        tensor_format = kimik26.KimiK26Format.from_config(model_dir)
        converter.Converter(tensor_format).convert(
            model_dir,
            output_dir,
        )

    assert not output_dir.exists()


def test_kimi_group_size_supports_text_config(tmp_path):
    model_dir = tmp_path / "kimi"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "quantization_config": {
                        "config_groups": {
                            "group_0": {"weights": {"group_size": 64}}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert kimik26.KimiK26Format.load_group_size(model_dir) == 64


def test_kimi_offline_conversion_rejects_missing_component_before_output(tmp_path):
    model_dir = tmp_path / "kimi"
    output_dir = tmp_path / "kimi-bf16"
    model_dir.mkdir()
    logical_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    packed_key = f"{logical_key}_packed"
    scale_key = f"{logical_key}_scale"
    save_file(
        {
            packed_key: torch.tensor([[123]], dtype=torch.int32),
            scale_key: torch.tensor([[0.5]], dtype=torch.float32),
        },
        model_dir / "model.safetensors",
    )
    _write_index(
        model_dir,
        {
            packed_key: "model.safetensors",
            scale_key: "model.safetensors",
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "config_groups": {
                        "group_0": {"weights": {"group_size": 32}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing shape"):
        tensor_format = kimik26.KimiK26Format.from_config(model_dir)
        converter.Converter(tensor_format).convert(
            model_dir,
            output_dir,
        )

    assert not output_dir.exists()
