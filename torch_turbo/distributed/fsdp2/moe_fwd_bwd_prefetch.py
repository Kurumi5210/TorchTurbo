"""
MoE Parallel Prefetch Manager for FSDP2
Supports first_k_dense_replace parameter with decoupled Layer and Expert chains.

IMPORTANT: This module automatically installs FSDP hooks on import.
Make sure to import this module before calling fully_shard().
"""

from __future__ import annotations
import threading
from contextlib import nullcontext
from typing import Optional, List, Type, Any, Dict, Set

import torch
from torch import nn

from torch_turbo.utils.torch_utils import torch_version

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
else:
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule
    from torch.distributed._composable.fsdp._fsdp_state import FSDPState
    from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup


class BackwardStateTracker:
    """Global backward state tracker."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._depth = 0
        return cls._instance

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, value: int):
        self._depth = value

    @property
    def in_backward(self) -> bool:
        return self._depth > 0

    def enter_backward(self):
        with self._lock:
            self._depth += 1

    def exit_backward(self):
        with self._lock:
            self._depth = max(0, self._depth - 1)


# Global state
_backward_tracker = BackwardStateTracker()
_registered_modules: Set[int] = set()
_registered_param_groups: Set[int] = set()  # Record FSDPParamGroup IDs
_global_hooks_installed: bool = False
_global_lock = threading.Lock()
_saved_originals: Dict[str, Any] = {}


def _wrap_backward():
    """
    Wrap backward entry points to automatically track backward state.

    Two entry points need to be wrapped:
    1. torch.Tensor.backward - Direct call to tensor.backward()
    2. torch.autograd.backward - Used by gradient checkpointing
    """
    # 1. Wrap torch.Tensor.backward
    original_tensor_backward = torch.Tensor.backward

    def wrapped_tensor_backward(self, *args, **kwargs):
        _backward_tracker.enter_backward()
        try:
            return original_tensor_backward(self, *args, **kwargs)
        finally:
            _backward_tracker.exit_backward()

    torch.Tensor.backward = wrapped_tensor_backward

    # 2. Wrap torch.autograd.backward (used by gradient checkpointing)
    original_autograd_backward = torch.autograd.backward

    def wrapped_autograd_backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None):
        _backward_tracker.enter_backward()
        try:
            return original_autograd_backward(
                tensors,
                grad_tensors=grad_tensors,
                retain_graph=retain_graph,
                create_graph=create_graph,
                grad_variables=grad_variables,
                inputs=inputs
            )
        finally:
            _backward_tracker.exit_backward()

    torch.autograd.backward = wrapped_autograd_backward

    return original_tensor_backward, original_autograd_backward


# Wrap backward on module import
_original_tensor_backward, _original_autograd_backward = _wrap_backward()


def _install_global_hooks():
    """
    Install global FSDP hooks (executed only once).

    IMPORTANT: Must be called before fully_shard()!
    This function is automatically executed on module import.
    """
    global _global_hooks_installed, _saved_originals

    with _global_lock:
        if _global_hooks_installed:
            return

        _saved_originals = {
            '_pre_forward': FSDPState._pre_forward,
            '_post_forward': FSDPState._post_forward,
            '_backward_prefetch': FSDPParamGroup._backward_prefetch,
        }

        original_pre = _saved_originals['_pre_forward']

        def wrapped_pre(state: FSDPState, module: nn.Module, args, kwargs):
            # Apply custom prefetch only for registered modules during backward
            if id(module) in _registered_modules and _backward_tracker.in_backward:
                custom_prefetch = getattr(module, "_custom_backward_prefetch_states", None)
                if custom_prefetch:
                    old = state._states_to_forward_prefetch
                    state._states_to_forward_prefetch = custom_prefetch
                    res = original_pre(state, module, args, kwargs)
                    state._states_to_forward_prefetch = old
                    return res
            return original_pre(state, module, args, kwargs)

        original_post = _saved_originals['_post_forward']

        def wrapped_post(state: FSDPState, module: nn.Module, input, output):
            if id(module) in _registered_modules and _backward_tracker.in_backward:
                if getattr(module, "_custom_backward_prefetch_states", None):
                    old_mesh = state._fsdp_param_group.post_forward_mesh_info
                    state._fsdp_param_group.post_forward_mesh_info = None
                    res = original_post(state, module, input, output)
                    state._fsdp_param_group.post_forward_mesh_info = old_mesh
                    return res
            return original_post(state, module, input, output)

        original_bp = _saved_originals['_backward_prefetch']

        def wrapped_bp(group_self):
            # Directly check if FSDPParamGroup ID is registered
            if id(group_self) in _registered_param_groups:
                return None  # Disable default backward prefetch
            return original_bp(group_self)

        FSDPState._pre_forward = wrapped_pre
        FSDPState._post_forward = wrapped_post
        FSDPParamGroup._backward_prefetch = wrapped_bp
        _global_hooks_installed = True


# Automatically install hooks on module import
_install_global_hooks()


class MoEPrefetchManager:
    """
    MoE Prefetch Manager.

    Features:
    - Automatic backward detection, no context manager needed
    - Decoupled Layer and Expert chains
    - Apply prefetch logic only to registered modules
    """

    _instances: Dict[int, 'MoEPrefetchManager'] = {}

    def __init__(self, model: nn.Module, config: Optional[Any] = None):
        self.model = model
        self.config = config
        self._model_modules: List[FSDPModule] = []
        self._layer_modules: List[FSDPModule] = []
        self._expert_modules: List[FSDPModule] = []

    @classmethod
    def get_or_create(cls, model: nn.Module, config: Optional[Any] = None) -> 'MoEPrefetchManager':
        model_id = id(model)
        if model_id not in cls._instances:
            cls._instances[model_id] = cls(model, config)
        return cls._instances[model_id]

    @classmethod
    def cleanup(cls, model: nn.Module):
        """Clean up PrefetchManager associated with the model and remove registered modules."""
        model_id = id(model)
        manager = cls._instances.pop(model_id, None)
        if manager is not None:
            for module in manager._model_modules:
                _registered_modules.discard(id(module))
                state = module._get_fsdp_state()
                _registered_param_groups.discard(id(state._fsdp_param_group))
            for module in manager._layer_modules:
                _registered_modules.discard(id(module))
                state = module._get_fsdp_state()
                _registered_param_groups.discard(id(state._fsdp_param_group))
            for module in manager._expert_modules:
                _registered_modules.discard(id(module))
                state = module._get_fsdp_state()
                _registered_param_groups.discard(id(state._fsdp_param_group))

    def setup(self, layer_cls: Type[nn.Module], expert_cls: Type[nn.Module], model_cls: Type[nn.Module]):
        self._collect_modules(layer_cls, expert_cls, model_cls)
        self._setup_forward_prefetch()
        self._setup_backward_prefetch_targets()

    def _collect_modules(self, layer_cls, expert_cls, model_cls):
        """Collect all modules that need prefetch and register them globally."""
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                if isinstance(module, model_cls):
                    self._model_modules.append(module)
                    _registered_modules.add(id(module))
                    state = module._get_fsdp_state()
                    _registered_param_groups.add(id(state._fsdp_param_group))
                elif isinstance(module, layer_cls):
                    self._layer_modules.append(module)
                    _registered_modules.add(id(module))
                    state = module._get_fsdp_state()
                    _registered_param_groups.add(id(state._fsdp_param_group))
                elif isinstance(module, expert_cls):
                    self._expert_modules.append(module)
                    _registered_modules.add(id(module))
                    state = module._get_fsdp_state()
                    _registered_param_groups.add(id(state._fsdp_param_group))

        # Validation
        assert len(self._model_modules) == 1, f"Expected 1 model module, got {len(self._model_modules)}"
        if self._expert_modules:
            k = getattr(self.config, "first_k_dense_replace", 0)
            expected_experts = len(self._layer_modules) - k
            actual_experts = len(self._expert_modules)
            assert expected_experts == actual_experts, \
                f"Expert count mismatch: expected {expected_experts}, got {actual_experts}"

    def _set_forward_prefetch(self, module: FSDPModule, targets: List[FSDPModule]):
        """Set forward prefetch and record to custom attribute."""
        module.set_modules_to_forward_prefetch(targets)
        setattr(module, "_forward_prefetch_targets", targets[:])

    def _get_forward_prefetch_targets(self, module: FSDPModule) -> List[FSDPModule]:
        """Get the forward prefetch target list for a module."""
        return getattr(module, "_forward_prefetch_targets", [])

    def _setup_forward_prefetch(self):
        """Set up forward prefetch chain."""
        k = getattr(self.config, "first_k_dense_replace", 0)

        # 1. Layer chain start: Model -> Layer[0]
        if self._layer_modules:
            self._set_forward_prefetch(self._model_modules[0], [self._layer_modules[0]])

        # 2. Layer sequential prefetch: Layer[i] -> Layer[i+1]
        for i in range(len(self._layer_modules) - 1):
            self._set_forward_prefetch(self._layer_modules[i], [self._layer_modules[i + 1]])

        # 3. Expert chain start logic
        if self._expert_modules:
            if k > 0 and k <= len(self._layer_modules):
                prev_targets = self._get_forward_prefetch_targets(self._layer_modules[k - 1])
                new_targets = prev_targets + [self._expert_modules[0]]
                self._set_forward_prefetch(self._layer_modules[k - 1], new_targets)
            else:
                prev_targets = self._get_forward_prefetch_targets(self._model_modules[0])
                new_targets = prev_targets + [self._expert_modules[0]]
                self._set_forward_prefetch(self._model_modules[0], new_targets)

        # 4. Expert sequential prefetch: Expert[j] -> Expert[j+1]
        for j in range(len(self._expert_modules) - 1):
            self._set_forward_prefetch(self._expert_modules[j], [self._expert_modules[j + 1]])

    def _setup_backward_prefetch_targets(self):
        """Set up backward prefetch targets."""
        # Layer backward chain
        for i in range(1, len(self._layer_modules)):
            layer = self._layer_modules[i]
            prev_layer_state = self._layer_modules[i - 1]._get_fsdp_state()
            setattr(layer, "_custom_backward_prefetch_states", [prev_layer_state])

        # Expert backward chain
        for j in range(1, len(self._expert_modules)):
            expert = self._expert_modules[j]
            prev_expert_state = self._expert_modules[j - 1]._get_fsdp_state()
            setattr(expert, "_custom_backward_prefetch_states", [prev_expert_state])


def setup_moe_prefetch(
    model: nn.Module,
    layer_cls: Type[nn.Module],
    expert_cls: Type[nn.Module],
    model_cls: Type[nn.Module],
    config: Optional[Any] = None
) -> MoEPrefetchManager:
    """
    Configure prefetch for MoE model.

    Args:
        model: FSDP2 wrapped model
        layer_cls: Transformer layer class
        expert_cls: Expert MLP class
        model_cls: Model outermost class
        config: Model config (includes first_k_dense_replace and other parameters)

    Returns:
        Configured MoEPrefetchManager instance

    Note: Make sure to import this module before calling fully_shard()!
    """
    manager = MoEPrefetchManager.get_or_create(model, config)
    manager.setup(layer_cls, expert_cls, model_cls)
    return manager


# Backward compatibility
fsdp2_backward_prefetch_manager = nullcontext()