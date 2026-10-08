"""ToolRegistry and tool descriptions (DESIGN.md 5.4).

A ``Tool`` is registered by a plugin with :meth:`ToolRegistry.register`; the
registration is an effect of that plugin's fiber, so unloading the plugin
unregisters its tools (and the next context epoch no longer lists them).

Parameters are pydantic models; :func:`tool_schema` produces JSON Schema that
satisfies DeepSeek's ``strict`` mode by default (every property required,
optional ones as ``anyOf [T, null]``, ``additionalProperties: false``, no
``minLength``/``maxLength``/``minItems``/``maxItems``, refs inlined).

Wire names: DeepSeek (and OpenAI) function names allow only ``[A-Za-z0-9_-]``,
so ``fs.read`` is sent as ``fs__read``; the registry maps both ways.
"""
from __future__ import annotations

import fnmatch
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # pragma: no cover
    from ventri import Context


class Risk(IntEnum):
    """Ordered risk classes: read < write-local < external < irreversible < spend."""

    READ = 0
    WRITE_LOCAL = 1
    EXTERNAL = 2
    IRREVERSIBLE = 3
    SPEND = 4

    @classmethod
    def parse(cls, value: Risk | str | int) -> Risk:
        if isinstance(value, Risk):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper().replace("-", "_")]

    @property
    def label(self) -> str:
        return self.name.lower().replace("_", "-")


Action = Literal["allow", "ask", "deny"]


class ToolError(Exception):
    """Raised by a handler to return a clean error message to the model.

    ``untrusted`` carries optional detail quoted from outside data (e.g. the
    file lines a failed edit matched); the loop appends it fenced as untrusted
    data, so a file cannot smuggle instructions in through an error message."""

    def __init__(self, message: str, *, untrusted: str | None = None) -> None:
        super().__init__(message)
        self.untrusted = untrusted


@dataclass
class ToolContext:
    """What a tool handler sees about the calling session."""

    session_id: str
    ctx: Context                       # the agent-loop fiber's context (session scope)
    workdir: Path                      # session directory (artifacts live below it)
    origin: str = "user"               # user | routine | evolution
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: Any, default: Any = None) -> Any:
        return self.ctx.get(key, default)


Handler = Callable[[Any, ToolContext], Awaitable[Any] | Any]


@dataclass
class Tool:
    name: str
    description: str
    handler: Handler
    params: type[BaseModel] | None = None
    risk: Risk = Risk.READ
    idempotent: bool = True
    parallel_safe: bool = False
    default_action: Action | None = None   # the tool plugin's own default (e.g. fs ``write: ask``)
    default_allow: dict[str, list[str]] = field(default_factory=dict)  # subject globs allowed by default
    grantable: bool = True                 # may "allow for this session" be granted?
    subject: Callable[[Any], dict[str, str]] | None = None  # permission-match attributes (path, domain...)
    untrusted: bool = False                # output is outside data (web pages, files): fence it
    timeout: float | None = 120.0
    source: str = ""                       # label of the registering fiber

    @property
    def wire_name(self) -> str:
        return wire_name(self.name)

    def spec(self, *, strict: bool = True) -> dict[str, Any]:
        params = tool_schema(self.params, strict=strict)
        return {"type": "function", "function": {"name": self.wire_name, "description": self.description,
                                                 "parameters": params}}

    def parse(self, arguments: dict[str, Any]) -> Any:
        """Validate model arguments. ``null`` for a field with a non-None default
        means "use the default" (strict schemas make every field required)."""
        if self.params is None:
            return arguments
        fields = self.params.model_fields
        clean = {k: v for k, v in arguments.items()
                 if not (v is None and k in fields and fields[k].default is not None)}
        return self.params.model_validate(clean)

    def describe_call(self, args: Any) -> dict[str, str]:
        if self.subject is not None:
            try:
                return {k: str(v) for k, v in self.subject(args).items()}
            except Exception:  # noqa: BLE001 - descriptive only
                return {}
        return {}


def wire_name(name: str) -> str:
    return name.replace(".", "__")


# ------------------------------------------------------------------ schemas
_DROP = {"title", "minLength", "maxLength", "minItems", "maxItems", "examples", "$defs", "definitions",
         "uniqueItems", "minProperties", "maxProperties"}
_FORMATS = {"email", "hostname", "ipv4", "ipv6", "uuid"}
EMPTY_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def tool_schema(params: type[BaseModel] | dict[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    if params is None:
        return dict(EMPTY_PARAMS)
    raw = params if isinstance(params, dict) else params.model_json_schema()
    defs = raw.get("$defs") or raw.get("definitions") or {}
    return _conv(raw, defs, strict, 0)


def _conv(s: dict[str, Any], defs: dict[str, Any], strict: bool, depth: int) -> dict[str, Any]:
    if depth > 24:
        raise ValueError("tool schema too deep (recursive models are not supported)")
    if "$ref" in s:
        name = s["$ref"].rsplit("/", 1)[-1]
        s = {**defs[name], **{k: v for k, v in s.items() if k != "$ref"}}
    out = {k: v for k, v in s.items() if k not in _DROP and not (strict and k == "default")}
    if strict and out.get("format") not in (None, *_FORMATS):
        out.pop("format")
    if "anyOf" in out:
        out["anyOf"] = [_conv(x, defs, strict, depth + 1) for x in out["anyOf"]]
    if "allOf" in out and len(out["allOf"]) == 1:  # pydantic wraps refs with descriptions this way
        inner = _conv(out.pop("allOf")[0], defs, strict, depth + 1)
        out = {**inner, **out}
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _conv(out["items"], defs, strict, depth + 1)
    if out.get("type") == "object" or "properties" in out:
        props = {k: _conv(v, defs, strict, depth + 1) for k, v in (out.get("properties") or {}).items()}
        required = list(out.get("required") or [])
        if strict:
            for k, v in props.items():
                if k not in required:
                    props[k] = _nullable(v)
            required = list(props)
            out["additionalProperties"] = False
        out["type"] = "object"
        out["properties"] = props
        out["required"] = required
    return out


def _nullable(s: dict[str, Any]) -> dict[str, Any]:
    if any(x.get("type") == "null" for x in s.get("anyOf", [])):
        return s
    if "anyOf" in s:
        return {**s, "anyOf": [*s["anyOf"], {"type": "null"}]}
    desc = s.get("description")
    inner = {k: v for k, v in s.items() if k != "description"}
    out: dict[str, Any] = {"anyOf": [inner, {"type": "null"}]}
    if desc:
        out["description"] = desc
    return out


def canonical(obj: Any) -> str:
    """Stable serialisation (sorted keys) -- used for epoch hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------- registry
class ToolRegistry:
    """Service: the set of currently registered tools. ``version`` increments on
    every change (sessions compare it to their epoch)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._wire: dict[str, str] = {}
        self.version = 0
        self._watchers: list[Callable[[], Any]] = []

    def __repr__(self) -> str:
        return f"<ToolRegistry {len(self._tools)} tools v{self.version}>"

    def register(self, ctx: Context | None, tool: Tool) -> Tool:
        """Register ``tool`` for the lifetime of ``ctx``'s fiber (``None``: until
        :meth:`unregister`)."""
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered (by {self._tools[tool.name].source})")
        w = tool.wire_name
        if w in self._wire:
            raise ValueError(f"tool {tool.name!r} collides with {self._wire[w]!r} on the wire")
        if ctx is not None and not tool.source:
            tool.source = ctx.fiber.label
        self._tools[tool.name] = tool
        self._wire[w] = tool.name
        self._changed()
        if ctx is not None:
            ctx.on_dispose(lambda: self.unregister(tool.name))
        return tool

    def unregister(self, name: str) -> None:
        t = self._tools.pop(name, None)
        if t is not None:
            self._wire.pop(t.wire_name, None)
            self._changed()

    def watch(self, fn: Callable[[], Any]) -> Callable[[], None]:
        self._watchers.append(fn)
        return lambda: self._watchers.remove(fn) if fn in self._watchers else None

    def _changed(self) -> None:
        self.version += 1
        for fn in list(self._watchers):
            try:
                fn()
            except Exception:  # noqa: BLE001, S110 - watchers are notifications only
                pass

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name) or self._tools.get(self._wire.get(name, ""))

    def names(self) -> list[str]:
        return sorted(self._tools)

    def select(self, patterns: Iterable[str] = ("*",)) -> list[Tool]:
        """Tools matching any glob in ``patterns`` (``fs``/``fs.*``, ``*``), sorted by name."""
        pats = list(patterns)
        out = []
        for n in sorted(self._tools):
            if any(fnmatch.fnmatchcase(n, p) or n.startswith(p + ".") for p in pats):
                out.append(self._tools[n])
        return out

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        return len(self._tools)


async def call_handler(tool: Tool, args: Any, tc: ToolContext) -> Any:
    r = tool.handler(args, tc)
    if inspect.isawaitable(r):
        r = await r
    return r


def render_result(value: Any) -> str:
    if value is None:
        return "ok"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=1, default=str)


def validation_message(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}" for err in e.errors())


# ------------------------------------------------------------------- plugin
def _registry_plugin(ctx: Any, config: Any) -> None:
    """``use: ventri_agent.tools.registry`` -- provides the ``ToolRegistry``."""
    ctx.provide(ToolRegistry, ToolRegistry())


_registry_plugin.name = "tool-registry"  # type: ignore[attr-defined]
_registry_plugin.provides = {"tools": ToolRegistry}  # type: ignore[attr-defined]
plugin = _registry_plugin
