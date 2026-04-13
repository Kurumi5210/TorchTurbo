"""Custom functools utilities for torch_turbo."""
import functools
import torch


def cache(user_function):
    """
    Simple single-item cache decorator for functions with tensor arguments.

    Uses tensor's shape, device, and data hash as the cache key.
    Suitable for deterministic functions like get_cu_seqlens_npu.

    Only keeps the most recent cache entry. When args change, old cache is cleared.

    Example:
        @cache
        def get_cu_seqlens_npu(attention_mask):
            ...

    Args:
        user_function: The function to cache.

    Returns:
        Wrapped function with single-item cache.
    """
    _cache_key = None
    _cache_value = None

    @functools.wraps(user_function)
    def wrapper(*args, **kwargs):
        nonlocal _cache_key, _cache_value

        # Generate cache key from arguments
        key_parts = []

        # Handle positional args
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_key = (
                    tuple(arg.shape),
                    str(arg.device),
                    arg.dtype,
                    arg.stride(),
                )
                key_parts.append(("tensor", tensor_key))
            else:
                key_parts.append(("other", arg))

        # Handle keyword args
        for k, v in sorted(kwargs.items()):
            if isinstance(v, torch.Tensor):
                tensor_key = (
                    tuple(v.shape),
                    str(v.device),
                    v.dtype,
                    v.stride(),
                )
                key_parts.append((k, "tensor", tensor_key))
            else:
                key_parts.append((k, "other", v))

        key = tuple(key_parts)

        # Check if key matches cached key
        if key == _cache_key:
            return _cache_value

        # Clear old cache and compute new result
        _cache_key = key
        result = user_function(*args, **kwargs)
        _cache_value = result

        return _cache_value

    # Expose cache info
    def cache_clear():
        nonlocal _cache_key, _cache_value
        _cache_key = None
        _cache_value = None

    wrapper.cache_clear = cache_clear

    return wrapper
