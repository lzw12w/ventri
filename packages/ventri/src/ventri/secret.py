"""``Secret[T]``: a value that never shows up in reprs, snapshots or traces."""
from __future__ import annotations

import dataclasses
import typing
from typing import Any

MASK = "***"
SENSITIVE = ("key", "token", "secret", "password")


class Secret[T]:
    """Wraps a sensitive value. ``repr``/``str`` are masked; ``reveal()`` returns it.

    Works as a pydantic field type (``api_key: Secret[str]`` accepts a plain string
    or a Secret, validates the inner type and dumps as ``"***"`` in JSON mode)
    without making pydantic a kernel dependency. Equality compares the wrapped
    values, so config diffs notice a changed secret without printing it.
    """

    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    def reveal(self) -> T:
        return self._value

    get_secret_value = reveal  # pydantic.SecretStr-compatible spelling

    def __repr__(self) -> str:
        return f"Secret({MASK!r})"

    def __str__(self) -> str:
        return MASK

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("ventri.Secret", self._value))

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        args = typing.get_args(source)
        inner = handler.generate_schema(args[0] if args else Any)
        wrap = core_schema.no_info_after_validator_function(cls, inner)
        unwrap = core_schema.no_info_before_validator_function(
            lambda v: v.reveal() if isinstance(v, Secret) else v, wrap)
        return core_schema.json_or_python_schema(
            json_schema=wrap, python_schema=unwrap,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: MASK, when_used="json"))


def sensitive(key: Any) -> bool:
    k = str(key).lower()
    return any(w in k for w in SENSITIVE)


def redact(value: Any, _depth: int = 0) -> Any:
    """Copy of ``value`` with secrets masked: ``Secret`` instances anywhere, and the
    values of mapping keys / fields whose name contains key, token, secret or
    password. Recurses into dicts, lists, tuples, dataclasses and pydantic models.
    Non-guarantee: a secret embedded in a string under an innocuous name (e.g. a
    URL with credentials in ``endpoint``) is not detected."""
    if _depth > 32:
        return value
    if isinstance(value, Secret):
        return MASK
    if isinstance(value, dict):
        return {k: MASK if sensitive(k) else redact(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v, _depth + 1) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return redact({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, _depth + 1)
    fields = getattr(type(value), "model_fields", None)
    if isinstance(fields, dict) and hasattr(value, "model_dump"):
        return redact({name: getattr(value, name, None) for name in fields}, _depth + 1)
    return value
