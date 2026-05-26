"""Pluggable logging for torchturbo.

By default, uses Python's standard ``logging`` module.  When an adapter
activates, it can swap in the framework's logger so that all torchturbo
messages go through the same logging pipeline.

Usage inside torchturbo::

    from torchturbo.logging import get_logger
    logger = get_logger(__name__)
    logger.info("hello")  # uses whatever logger the adapter set

Adapters call::

    from torchturbo.logging import set_logger_factory
    from llamafactory.extras.logging import get_logger as lf_get_logger
    set_logger_factory(lf_get_logger)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

_root = logging.getLogger("torchturbo")
if not _root.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[torchturbo] %(levelname)s %(name)s: %(message)s")
    )
    _root.addHandler(_handler)
    _root.setLevel(logging.DEBUG)

# The factory function: name -> logger-like object
_logger_factory: Callable[[str], Any] = logging.getLogger


def get_logger(name: str) -> Any:
    """Get a logger for the given module name.

    Returns whatever logger the current factory produces.  By default
    this is a standard ``logging.Logger``.  After an adapter calls
    :func:`set_logger_factory`, it returns the framework's logger.
    """
    return _logger_factory(name)


def set_logger_factory(factory: Callable[[str], Any]) -> None:
    """Replace the logger factory.

    Parameters
    ----------
    factory:
        A callable that takes a module name (str) and returns a logger-like
        object with ``.info()``, ``.warning()``, ``.debug()`` etc.

    Example::

        from llamafactory.extras.logging import get_logger
        set_logger_factory(get_logger)
    """
    global _logger_factory
    _logger_factory = factory


def reset_logger_factory() -> None:
    """Reset to Python's standard ``logging.getLogger``."""
    global _logger_factory
    _logger_factory = logging.getLogger
