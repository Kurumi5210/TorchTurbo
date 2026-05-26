"""torchturbo models — each model in its own directory.
"""

from .registry import get_all_models, get_model, register_model, _register_builtins

# Auto-register built-in models
_register_builtins()

__all__ = [
    "get_all_models",
    "get_model",
    "register_model",
]
