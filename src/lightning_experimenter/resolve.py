"""Resolve dotted paths from config files into Python objects."""

import importlib
from typing import Any, Callable


def _resolve(spec: str) -> Any:
    module_name, _, attr = spec.rpartition(".")
    if not module_name:
        raise ValueError(f"{spec!r} is not a dotted path (expected 'package.module.Name')")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"could not import {module_name!r} while resolving {spec!r}") from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"module {module_name!r} has no attribute {attr!r}") from None


def load_class(spec: str, expected: type) -> type:
    """Resolve `spec` to a subclass of `expected`."""
    obj = _resolve(spec)
    if not isinstance(obj, type):
        raise TypeError(f"{spec!r} resolved to {type(obj).__name__}, not a class")
    if not issubclass(obj, expected):
        raise TypeError(f"{spec!r} is not a subclass of {expected.__name__}")
    return obj


def load_callable(spec: str) -> Callable:
    """Resolve `spec` to any callable (used for the `prepare` hook)."""
    obj = _resolve(spec)
    if not callable(obj):
        raise TypeError(f"{spec!r} resolved to {type(obj).__name__}, which is not callable")
    return obj
