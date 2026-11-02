"""CPU dispatch contracts for GDN state layouts; no NPU numerics are simulated."""

import ast
from pathlib import Path

import pytest
import torch


_ADAPTER = (
    Path(__file__).resolve().parents[4]
    / "src/torchturbo/ops/providers/npu/_fla_gdn/chunk_gated_delta_rule_function.py"
)
_FORWARD_ALIAS = "ascendc_chunk_gated_delta_rule_fwd_h"

# FLA-NPU v26.6.0: torch_custom/fla_npu/npu_custom.yaml.
_LEGACY_SCHEMA = (
    "legacy(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, "
    "Tensor? gk=None, Tensor? initial_state=None, "
    "bool? output_final_state=False, int? chunk_size=None, "
    "bool? save_new_value=True, int[]? cu_seqlens=None, "
    "int[]? chunk_indices=None, bool? use_exp2=False, "
    "bool? transpose_state_layout=False) -> (Tensor, Tensor, Tensor)"
)
# FLA-NPU 36e67d2d: the public Python signature in ops/ascendc/_stable.py.
# The adapter calls this wrapper, which supplies the internal stream argument.
_STABLE_SCHEMA = (
    "stable(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, "
    "Tensor? gk=None, Tensor? initial_state=None, "
    "bool output_final_state=False, int? chunk_size=None, "
    "int[]? cu_seqlens=None, int[]? chunk_indices=None, "
    "bool state_v_first=False) -> (Tensor, Tensor, Tensor?)"
)


@pytest.fixture(scope="module")
def forward_ops():
    library = torch.library.Library("torchturbo_gdn_state_layout", "DEF")

    def return_inputs(k, w, u, **kwargs):
        return k, w, u

    try:
        library.define(_LEGACY_SCHEMA)
        library.define(_STABLE_SCHEMA)
        library.impl("legacy", return_inputs, "CPU")
        library.impl("stable", return_inputs, "CPU")
        yield {
            "legacy": torch.ops.torchturbo_gdn_state_layout.legacy.default,
            "stable": torch.ops.torchturbo_gdn_state_layout.stable.default,
        }
    finally:
        library._destroy()


@pytest.mark.parametrize("chain", ["_forward_chain", "_backward_chain"])
@pytest.mark.parametrize(
    "api,layout_flag",
    [("legacy", "transpose_state_layout"), ("stable", "state_v_first")],
)
def test_forward_and_recompute_accept_default_kv_layout(chain, api, layout_flag, forward_ops):
    tree = ast.parse(_ADAPTER.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == chain
    )
    call, = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _FORWARD_ALIAS
    ]
    operator = forward_ops[api]
    layout = next(arg for arg in operator._schema.arguments if arg.name == layout_flag)
    assert layout.default_value is False
    k, w, u = [torch.empty(1) for _ in range(3)]
    scope = {
        _FORWARD_ALIAS: operator,
        "k": k,
        "w": w,
        "u": u,
        "g_head": None,
        "initial_state": None,
        "output_final_state": False,
        "chunk_size": 64,
        "cu_seqlens_list": None,
        "chunk_list": None,
    }
    result = eval(compile(ast.Expression(call), str(_ADAPTER), "eval"), scope)
    assert all(actual is expected for actual, expected in zip(result, (k, w, u)))
