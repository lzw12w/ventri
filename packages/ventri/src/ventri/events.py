"""Typed event keys and interceptor verdicts (DESIGN.md 4.8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Event[T]:
    """A typed event key: ``MessageIn = Event[ChannelMessage]("message.in")``.

    ``ctx.on(MessageIn, handler)`` gives ``handler`` a typed payload and
    ``ctx.emit(MessageIn, msg)`` checks the payload type statically. Two keys with
    the same name are the same channel, and the string ``"message.in"`` addresses
    it too."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Event({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Event) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("ventri.Event", self.name))


def event_name(event: Any) -> str:
    if isinstance(event, Event):
        return event.name
    if isinstance(event, str):
        return event
    raise TypeError(f"event must be an Event or a str, got {event!r}")


@dataclass(frozen=True)
class Deny:
    """Interceptor verdict: stop and refuse (``reason`` is shown to the caller)."""

    reason: str


@dataclass(frozen=True)
class Rewrite[T]:
    """Interceptor verdict: replace the value; later interceptors see the new one."""

    value: T
