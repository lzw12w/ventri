"""Plugin description: a plugin is a function ``(ctx, config)`` or a class ``(ctx, config)``.

Metadata is read from attributes (``name``, ``inject``, ``Config``); the ``@plugin``
decorator just sets those attributes on functions.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


def keyname(key: Any) -> str:
    return key.__name__ if isinstance(key, type) else str(key)


def _as_keys(inject: Any) -> tuple:
    if inject is None:
        return ()
    if isinstance(inject, (str, type)):
        return (inject,)
    return tuple(inject)


def _arity(fn: Callable) -> int:
    """Number of positional args (capped at 2) the callable accepts: ctx, config."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 2
    n = 0
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            return 2
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return min(n, 2)


@dataclass(frozen=True)
class PluginSpec:
    target: Any
    name: str
    inject: tuple
    config_type: Any
    is_class: bool
    arity: int

    def call(self, ctx: Any, config: Any) -> Any:
        args = (ctx, config)[: self.arity]
        return self.target(*args)


def describe(target: Any) -> PluginSpec:
    if not callable(target):
        raise TypeError(f"plugin must be a function or class, got {target!r}")
    name = getattr(target, "name", None)
    if not isinstance(name, str):
        name = getattr(target, "__name__", type(target).__name__)
    return PluginSpec(
        target=target,
        name=name,
        inject=_as_keys(getattr(target, "inject", ())),
        config_type=getattr(target, "Config", None),
        is_class=inspect.isclass(target),
        arity=_arity(target),
    )


def plugin(fn: Callable | None = None, *, name: str | None = None,
           inject: Any = (), config: Any = None) -> Any:
    """Decorator for function plugins: ``@plugin(name="tool", inject=[LLM])``."""

    def wrap(f: Callable) -> Callable:
        if name:
            f.name = name  # type: ignore[attr-defined]
        f.inject = _as_keys(inject)  # type: ignore[attr-defined]
        if config is not None:
            f.Config = config  # type: ignore[attr-defined]
        return f

    return wrap(fn) if fn is not None else wrap


MISSING: Any = type("MISSING", (), {"__repr__": lambda self: "MISSING"})()
