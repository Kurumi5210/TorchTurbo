from __future__ import annotations

import importlib
import logging

import pytest
import torch
import torch.nn.functional as F

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.ops import (
    OpDispatcher,
    OpImpl,
    OpImplKind,
    OpPolicy,
    OpRegistry,
    OpSpec,
    get_ops,
    get_or_create_ops,
)
from torchturbo.ops.providers import iter_provider_modules
from torchturbo.ops.registry import registry as ops_registry
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


def _reference_rms_norm(hidden_states, weight, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


class FakeGroupedMatmulImpl(OpImpl):
    def __call__(self, x, weight, group_list, *, trans_b=False):
        assert group_list.tolist() == [1]
        assert trans_b is True
        return x + weight


class CachedAddImpl(OpImpl):
    available_calls = 0

    def available(self) -> bool:
        type(self).available_calls += 1
        return True

    def __call__(self, x):
        return x + 1


class UnavailableAddImpl(OpImpl):
    def available(self) -> bool:
        return False

    def __call__(self, x):
        return x


class FallbackAddImpl(OpImpl):
    def __call__(self, x):
        return x + 1


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.ops"), "_OPS", None
    )
    runtime_impl.is_device_available.cache_clear()
    get_or_create_ops(
        device_runtime=CpuRuntime(),
    )
    yield
    runtime_impl.is_device_available.cache_clear()


def test_rms_norm_uses_reference_fallback_on_cpu():
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)

    actual = get_ops().rms_norm(hidden_states, weight, 1e-6)
    expected = _reference_rms_norm(hidden_states, weight, 1e-6)

    assert torch.allclose(actual, expected)


def test_provider_modules_are_discovered_from_provider_packages():
    cpu_modules = iter_provider_modules(TurboDevice.CPU)
    cuda_modules = iter_provider_modules(TurboDevice.CUDA)
    npu_modules = iter_provider_modules(TurboDevice.NPU)

    assert "torchturbo.ops.providers.native" not in cpu_modules
    assert "torchturbo.ops.providers.native.rms_norm" in cpu_modules
    assert "torchturbo.ops.providers.native.gated_activation" in cpu_modules
    assert "torchturbo.ops.providers.native.swiglu" in cpu_modules
    assert "torchturbo.ops.providers.cuda.gated_activation" in cuda_modules
    assert "torchturbo.ops.providers.cuda.swiglu" in cuda_modules
    assert "torchturbo.ops.providers.cuda.cross_entropy_loss" in cuda_modules
    assert "torchturbo.ops.providers.npu.swiglu" in npu_modules
    assert (
        "torchturbo.ops.providers.npu.matmul_add"
        in npu_modules
    )
    assert "torchturbo.ops.providers.npu.cross_entropy_loss" in npu_modules
    assert "torchturbo.ops.providers.npu.moe_token_permutation" in npu_modules
    assert "torchturbo.ops.providers.npu.moe_token_unpermutation" in npu_modules
    assert "torchturbo.ops.providers.npu.rotary_position_embedding" in npu_modules
    assert "torchturbo.ops.providers.npu.lightning_indexer" in npu_modules
    assert "torchturbo.ops.providers.npu.sparse_flash_mla" in npu_modules
    assert "torchturbo.ops.providers.native.cross_entropy_loss" in cpu_modules
    assert (
        "torchturbo.ops.providers.native.matmul_add"
        in cpu_modules
    )


def test_custom_policy_errors_when_no_cpu_custom_impl_exists():
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.ones(4, dtype=torch.float32)

    custom_ops = OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=ops_registry,
        default_policy=OpPolicy.CUSTOM,
    )

    with pytest.raises(RuntimeError, match="No implementation"):
        custom_ops.rms_norm(hidden_states, weight, 1e-6)


def test_invalid_policy_errors_when_initializing_dispatcher():
    with pytest.raises(TypeError, match="policy must be an OpPolicy"):
        OpDispatcher(
            device_runtime=CpuRuntime(),
            registry=ops_registry,
            default_policy="unknown",
        )


def test_swiglu_uses_reference_fallback_on_cpu():
    x = torch.randn(2, 8, dtype=torch.float32)
    actual = get_ops().swiglu(x)

    gate, up = torch.chunk(x, 2, dim=-1)
    expected = F.silu(gate) * up

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("activation_func", [F.silu, F.gelu])
def test_gated_activation_reference_supports_limit(activation_func):
    x = torch.randn(2, 8, dtype=torch.float32)
    actual = get_ops().gated_activation(
        x,
        activation_func=activation_func,
        limit=0.5,
    )

    gate, up = torch.chunk(x, 2, dim=-1)
    expected = activation_func(torch.clamp(gate.float(), max=0.5)) * torch.clamp(
        up.float(), min=-0.5, max=0.5
    )

    assert torch.allclose(actual, expected)


def test_swiglu_custom_policy_errors_when_no_cpu_custom_impl_exists():
    x = torch.randn(2, 8, dtype=torch.float32)

    custom_ops = OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=ops_registry,
        default_policy=OpPolicy.CUSTOM,
    )

    with pytest.raises(RuntimeError, match="No implementation"):
        custom_ops.swiglu(x)


def test_cross_entropy_loss_uses_reference_fallback_on_cpu():
    logits = torch.randn(4, 6, dtype=torch.float32)
    labels = torch.tensor([1, 4, 0, 3], dtype=torch.long)

    actual = get_ops().cross_entropy_loss(logits, labels, reduction="sum")
    expected = F.cross_entropy(logits, labels, reduction="sum")

    assert torch.allclose(actual, expected)


def test_matmul_add_uses_reference_fallback_on_cpu():
    lhs = torch.randn(3, 5, dtype=torch.bfloat16)
    rhs = torch.randn(3, 4, dtype=torch.bfloat16)
    accumulator = torch.zeros(5, 4, dtype=torch.float32)

    get_ops().matmul_add(
        accumulator,
        lhs,
        rhs,
        transpose_lhs=True,
    )
    expected = lhs.float().t().matmul(rhs.float())

    assert torch.allclose(accumulator, expected)


def test_registry_register_is_idempotent_by_impl_name():
    registry = OpRegistry()

    first = OpSpec(
        op_name="add",
        impl_name="same_name",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    replacement = OpSpec(
        op_name="add",
        impl_name="same_name",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )

    registry.register(first)
    registry.register(replacement)

    assert isinstance(registry.list(), list)
    assert isinstance(
        registry.list(spec_filter=lambda spec: spec.op_name == "add"),
        list,
    )
    assert registry.list(spec_filter=lambda spec: spec.op_name == "add")[0] is replacement
    assert registry.get_by_impl_name("same_name") is replacement


def test_registry_list_returns_highest_priority_filtered_spec():
    registry = OpRegistry()

    low = OpSpec(
        op_name="add",
        impl_name="low_priority",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    high = OpSpec(
        op_name="add",
        impl_name="high_priority",
        kind=OpImplKind.NATIVE,
        priority=10,
        devices=(),
    )

    registry.register(low)
    registry.register(high)

    assert registry.list(spec_filter=lambda spec: spec.op_name == "add") == [
        high,
        low,
    ]


def test_registry_list_prefers_device_specific_and_custom_specs_at_equal_priority():
    registry = OpRegistry()

    fallback_native = OpSpec(
        op_name="add",
        impl_name="fallback_native",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    fallback_custom = OpSpec(
        op_name="add",
        impl_name="fallback_custom",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(),
    )
    device_native = OpSpec(
        op_name="add",
        impl_name="device_native",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    device_custom = OpSpec(
        op_name="add",
        impl_name="device_custom",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.CPU,),
    )

    registry.register(fallback_native)
    registry.register(device_native)
    registry.register(fallback_custom)
    registry.register(device_custom)

    assert registry.list(spec_filter=lambda spec: spec.op_name == "add") == [
        device_custom,
        device_native,
        fallback_custom,
        fallback_native,
    ]


def test_registry_rejects_duplicate_impl_name_across_ops():
    registry = OpRegistry()

    registry.register(
        OpSpec(
            op_name="add",
            impl_name="same_name",
            kind=OpImplKind.NATIVE,
            priority=0,
            devices=(),
        )
    )

    with pytest.raises(ValueError, match="already registered for op"):
        registry.register(
            OpSpec(
                op_name="mul",
                impl_name="same_name",
                kind=OpImplKind.NATIVE,
                priority=0,
                devices=(),
            )
        )


def test_grouped_matmul_facade_dispatches_registered_impl(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="grouped_matmul",
        impl_name="fake_grouped_matmul",
        kind=OpImplKind.NATIVE,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: FakeGroupedMatmulImpl},
    )
    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())

    x = torch.ones(1)
    weight = torch.full((1,), 2.0)
    group_list = torch.tensor([1])

    actual = dispatcher.grouped_matmul(x, weight, group_list, trans_b=True)

    assert torch.equal(actual, torch.tensor([3.0]))


def test_dispatcher_caches_selected_impl_after_first_resolution(monkeypatch):
    CachedAddImpl.available_calls = 0
    registry = OpRegistry()
    spec = OpSpec(
        op_name="cached_add",
        impl_name="cached_add_impl",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: CachedAddImpl},
    )
    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())

    x = torch.tensor([1.0])

    assert torch.equal(dispatcher.cached_add(x), torch.tensor([2.0]))
    assert torch.equal(dispatcher.cached_add(x), torch.tensor([2.0]))
    assert CachedAddImpl.available_calls == 1


def test_dispatcher_uses_config_override_and_call_override(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    overridden = OpSpec(
        op_name="add",
        impl_name="overridden_add",
        kind=OpImplKind.CUSTOM,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    call_override = OpSpec(
        op_name="add",
        impl_name="call_override_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(overridden)
    registry.register(call_override)

    class OverriddenAddImpl(OpImpl):
        def __call__(self, x):
            return x + 10

    class CallOverrideAddImpl(OpImpl):
        def __call__(self, x):
            return x + 20

    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {
            overridden.impl_name: OverriddenAddImpl,
            call_override.impl_name: CallOverrideAddImpl,
        },
    )
    dispatcher = OpDispatcher(
        registry=registry,
        device_runtime=CpuRuntime(),
        default_policy=OpPolicy.NATIVE,
        overrides={"add": overridden.impl_name},
    )

    x = torch.tensor([1.0])
    assert torch.equal(dispatcher.add(x), torch.tensor([11.0]))
    assert torch.equal(
        dispatcher.add(x, impl=call_override.impl_name), torch.tensor([21.0])
    )
    assert dispatcher.overrides == {"add": "overridden_add"}
    dispatcher.log_ops_selection()
    assert "overridden" in caplog.text


def test_dispatcher_rejects_invalid_override(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="native_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)

    class NativeAddImpl(OpImpl):
        def __call__(self, x):
            return x

    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: NativeAddImpl},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*No OpSpec registered",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": "missing_add"},
        )


def test_dispatcher_rejects_impl_overridden_for_different_op(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="multiply",
        impl_name="native_multiply",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*registered for op 'multiply'",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_impl_unsupported_on_current_device(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="npu_add",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.NPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*does not support device=cpu",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_unavailable_override(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="unavailable_add",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: UnavailableAddImpl},
    )

    with pytest.raises(
        RuntimeError,
        match=r"Invalid override.*unavailable=\['unavailable_add'\]",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_override_for_unknown_op(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="native_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*not 'aardvark'",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"aardvark": spec.impl_name},
        )


def test_process_dispatcher_rejects_different_overrides():
    with pytest.raises(
        RuntimeError,
        match="incompatible configuration.*overrides",
    ):
        get_or_create_ops(
            device_runtime=CpuRuntime(),
            overrides={"rms_norm": "torch_rms_norm"},
        )


def test_dispatcher_logs_selected_ops(caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")

    dispatcher = OpDispatcher(
        registry=ops_registry,
        device_runtime=CpuRuntime(),
    )

    selected = dispatcher._candidate_impl_cache[("rms_norm", "cpu", None)]
    assert selected.spec.impl_name == "torch_rms_norm"
    assert {
        key[0] for key in dispatcher._candidate_impl_cache
    } >= {"cross_entropy_loss", "grouped_matmul", "rms_norm", "swiglu"}
    dispatcher.log_ops_selection()

    assert caplog.text.count("TorchTurbo ops selection (device=cpu, policy=auto):") == 1
    message = caplog.records[-1].message
    lines = message.splitlines()
    assert lines[2].startswith("Op")
    assert {"Implementation", "Kind", "Priority", "Status"} <= set(lines[2].split())
    assert lines[1] == lines[3] == lines[-1]
    assert "rms_norm" in message
    assert "torch_rms_norm" in message
    assert "selected" in message


def test_dispatcher_logs_unresolved_ops_without_failing_configuration(caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    registry.register(
        OpSpec(
            op_name="missing_op",
            impl_name="missing_impl",
            kind=OpImplKind.CUSTOM,
            priority=100,
            devices=(TurboDevice.CPU,),
        )
    )

    dispatcher = OpDispatcher(
        registry=registry,
        device_runtime=CpuRuntime(),
    )
    dispatcher.log_ops_selection()

    assert "missing_op" in caplog.text
    assert "unavailable" in caplog.text
    assert "Unavailable details:" in caplog.text
    with pytest.raises(RuntimeError, match="No OpImpl class registered"):
        dispatcher.missing_op()


def test_dispatcher_marks_fallback_selection(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    custom = OpSpec(
        op_name="add",
        impl_name="unavailable_add",
        kind=OpImplKind.CUSTOM,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    native = OpSpec(
        op_name="add",
        impl_name="fallback_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    registry.register(custom)
    registry.register(native)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {
            custom.impl_name: UnavailableAddImpl,
            native.impl_name: FallbackAddImpl,
        },
    )

    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())
    dispatcher.log_ops_selection()

    assert "add" in caplog.text
    assert "fallback_add" in caplog.text
    assert "fallback" in caplog.text


def test_dispatcher_can_use_injected_device_runtime(monkeypatch):
    def _boom():
        raise AssertionError("global runtime probe should not be used")

    monkeypatch.setattr(runtime_impl, "get_device_runtime", _boom)

    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)

    actual = get_ops().rms_norm(hidden_states, weight, 1e-6)
    expected = _reference_rms_norm(hidden_states, weight, 1e-6)

    assert torch.allclose(actual, expected)
