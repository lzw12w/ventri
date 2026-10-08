"""Plugin description.

A plugin is a function ``fn(ctx, config, *deps)`` (sync or async) or a class
``Cls(ctx, config, *deps)`` (optional ``start()`` / ``stop()``). Metadata is read
from attributes, which the ``@plugin`` decorator sets on functions:

* ``name`` -- display name (default: ``__name__``);
* ``inject`` -- explicit required dependency keys (classes or strings);
* ``Config`` -- config type; a dict config is turned into ``Config(**config)``;
* ``provides`` -- static declaration of the services the plugin provides
  (``{"llm": ModelProvider}`` or ``[ModelProvider]``); used by diagnostics and
  ``ventri stubgen``, never enforced at runtime;
* ``timeout`` -- load timeout in seconds (``None`` disables; default: kernel's);
* ``retry`` -- automatic restart policy for FAILED fibers (``Retry`` or dict);
* ``exclusive`` -- the plugin holds an exclusive resource: replacements use the
  stop-first strategy.

**Signature injection.** Parameters after ``(ctx, config)`` declare dependencies
by annotation: ``llm: ModelProvider`` is required (the fiber stays PENDING
until it can be bound), ``cal: Calendar | None`` is optional (bound if present,
``None`` otherwise; never blocks activation). ``Annotated[T, "key"]`` uses a
string key. Extra parameters that have a default and are not ``X | None`` are
not injected (they keep their default); unannotated ones without a default are
an error.
"""
from __future__ import annotations

import inspect
import types
import typing
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

MISSING: Any = type("MISSING", (), {"__repr__": lambda self: "MISSING"})()


def keyname(key: Any) -> str:
    return key.__name__ if isinstance(key, type) else str(key)


def _as_keys(inject: Any) -> tuple:
    if inject is None:
        return ()
    if isinstance(inject, (str, type)):
        return (inject,)
    return tuple(inject)


@dataclass(frozen=True)
class Retry:
    """Restart policy for FAILED fibers (load failure, load timeout or task crash).

    ``max`` automatic restarts per failure streak, delays ``base * 2**(n-1)`` capped
    at ``cap`` for ``backoff="exp"`` (``base`` for ``"fixed"``). A fiber that stayed
    ACTIVE for at least ``reset_after`` seconds before failing starts a new streak.
    Staged (uncommitted) fibers are never retried."""

    max: int = 3
    backoff: Literal["exp", "fixed"] = "exp"
    base: float = 0.5
    cap: float = 30.0
    reset_after: float = 60.0

    def delay(self, attempt: int) -> float:
        if self.backoff == "fixed":
            return self.base
        return min(self.cap, self.base * 2 ** (attempt - 1))

    @classmethod
    def coerce(cls, value: Any) -> Retry | None:
        if value is None or isinstance(value, Retry):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"retry must be a Retry, a dict or None, got {value!r}")


@dataclass(frozen=True)
class Dep:
    """One signature-injected dependency parameter."""

    param: str
    key: Any
    optional: bool
    positional: bool  # positional-only parameter


@dataclass(frozen=True)
class PluginSpec:
    target: Any
    name: str
    inject: tuple                 # required keys (explicit + inferred)
    config_type: Any
    is_class: bool
    arity: int                    # how many of (ctx, config) are passed positionally
    optional: tuple = ()          # optional keys (``X | None``)
    params: tuple[Dep, ...] = ()  # signature-injected parameters
    provides: dict[str, Any] = field(default_factory=dict)  # attribute name -> type
    provides_keys: tuple = ()     # every declared key (named or not)
    timeout: Any = MISSING        # MISSING: kernel default; None: no timeout
    retry: Retry | None = None
    exclusive: bool = False
    deps: tuple = ()              # inject + optional

    def call(self, ctx: Any, config: Any, bound: dict | None = None) -> Any:
        args: list[Any] = [ctx, config][: self.arity]
        kwargs: dict[str, Any] = {}
        for d in self.params:
            b = bound.get(d.key) if bound else None
            value = b.value if b is not None else None
            if d.positional:
                args.append(value)
            else:
                kwargs[d.param] = value
        return self.target(*args, **kwargs)


def _hints(fn: Any) -> dict[str, Any]:
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolvable string annotations: fall back to raw ones
        try:
            return dict(inspect.get_annotations(fn, eval_str=False))
        except Exception:  # noqa: BLE001
            return {}


def _dep_key(ann: Any, where: str) -> tuple[Any, bool]:
    """Annotation -> (key, optional)."""
    if typing.get_origin(ann) is typing.Annotated:
        base, *extra = typing.get_args(ann)
        keys = [m for m in extra if isinstance(m, str)]
        if keys:
            key, optional = _dep_key(base, where)
            return keys[0], optional
        return _dep_key(base, where)
    origin = typing.get_origin(ann)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(ann)) == 2:
            key, _ = _dep_key(args[0], where)
            return key, True
        raise TypeError(f"{where}: only `X | None` unions can be injected, got {ann!r}")
    if isinstance(ann, str):
        raise TypeError(f"{where}: cannot resolve annotation {ann!r} (define the type at module level)")
    if isinstance(ann, type) or typing.get_origin(ann) is not None:
        return (typing.get_origin(ann) or ann), False
    raise TypeError(f"{where}: annotation {ann!r} is not a service key")


def _signature(target: Any) -> tuple[int, tuple[Dep, ...]]:
    """(arity for ctx/config, injected parameters)."""
    fn = target.__init__ if inspect.isclass(target) else target
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return 2, ()
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params[:2]):
        return 2, ()
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    arity = min(len(positional), 2)
    rest = positional[2:] + [p for p in params if p.kind is p.KEYWORD_ONLY]
    if not rest:
        return arity, ()
    hints = _hints(fn if inspect.isclass(target) else target)
    deps = []
    name = getattr(target, "__qualname__", repr(target))
    for p in rest:
        ann = hints.get(p.name, p.annotation)
        if ann is inspect.Parameter.empty:
            if p.default is inspect.Parameter.empty:
                raise TypeError(f"{name}: parameter {p.name!r} needs a type annotation to be injected")
            continue
        if p.default is not inspect.Parameter.empty:
            try:
                key, optional = _dep_key(ann, f"{name}({p.name})")
            except TypeError:
                continue
            if not optional:
                continue  # ``x: int = 3`` is an ordinary parameter, not a dependency
        else:
            key, optional = _dep_key(ann, f"{name}({p.name})")
        deps.append(Dep(p.name, key, optional, p.kind is p.POSITIONAL_ONLY))
    return arity, tuple(deps)


def _provides(value: Any) -> tuple[dict[str, Any], tuple]:
    if value is None:
        return {}, ()
    if isinstance(value, dict):
        return dict(value), tuple(dict.fromkeys(value.values()))
    keys = _as_keys(value)
    return {}, keys


_META = ("name", "inject", "Config", "provides", "timeout", "retry", "exclusive")
_cache: weakref.WeakKeyDictionary[Any, tuple[tuple, PluginSpec]] = weakref.WeakKeyDictionary()


def describe(target: Any) -> PluginSpec:
    """PluginSpec for ``target`` (cached; re-derived if a metadata attribute changed)."""
    meta = tuple(getattr(target, a, MISSING) for a in _META)
    try:
        hit = _cache.get(target)
    except TypeError:  # unhashable / not weak-referenceable
        return _describe(target)
    if hit is not None and len(hit[0]) == len(meta) and all(a is b for a, b in zip(hit[0], meta)):
        return hit[1]
    spec = _describe(target)
    try:
        _cache[target] = (meta, spec)
    except TypeError:
        pass
    return spec


def _describe(target: Any) -> PluginSpec:
    if not callable(target):
        raise TypeError(f"plugin must be a function or class, got {target!r}")
    name = getattr(target, "name", None)
    if not isinstance(name, str):
        name = getattr(target, "__name__", type(target).__name__)
    arity, params = _signature(target)
    explicit = _as_keys(getattr(target, "inject", ()))
    required = tuple(dict.fromkeys(explicit + tuple(d.key for d in params if not d.optional)))
    optional = tuple(dict.fromkeys(d.key for d in params if d.optional and d.key not in required))
    provides, provides_keys = _provides(getattr(target, "provides", None))
    timeout = getattr(target, "timeout", MISSING)
    if timeout is not MISSING and timeout is not None and not isinstance(timeout, (int, float)):
        timeout = MISSING  # e.g. an unrelated ``timeout`` attribute on a class
    return PluginSpec(
        target=target,
        name=name,
        inject=required,
        config_type=getattr(target, "Config", None),
        is_class=inspect.isclass(target),
        arity=arity,
        optional=optional,
        params=params,
        provides=provides,
        provides_keys=provides_keys,
        timeout=timeout,
        retry=Retry.coerce(getattr(target, "retry", None)),
        exclusive=bool(getattr(target, "exclusive", False)),
        deps=required + optional,
    )


def plugin(fn: Callable | None = None, *, name: str | None = None, inject: Any = (),
           config: Any = None, provides: Any = None, timeout: Any = MISSING,
           retry: Retry | dict | None = None, exclusive: bool = False) -> Any:
    """Decorator for function plugins::

        @plugin(provides={"llm": ModelProvider}, timeout=10)
        async def deepseek(ctx, cfg: DeepSeekConfig) -> None: ...
    """

    def wrap(f: Callable) -> Callable:
        if name:
            f.name = name  # type: ignore[attr-defined]
        f.inject = _as_keys(inject)  # type: ignore[attr-defined]
        if config is not None:
            f.Config = config  # type: ignore[attr-defined]
        if provides is not None:
            f.provides = provides  # type: ignore[attr-defined]
        if timeout is not MISSING:
            f.timeout = timeout  # type: ignore[attr-defined]
        if retry is not None:
            f.retry = Retry.coerce(retry)  # type: ignore[attr-defined]
        if exclusive:
            f.exclusive = True  # type: ignore[attr-defined]
        return f

    return wrap(fn) if fn is not None else wrap
