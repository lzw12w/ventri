"""``va serve`` support: the long-running channel daemon.

``va serve`` provides a :class:`ServeHub` under the service key ``"va.serve"``
before it applies the configuration. Daemon channels (Feishu) only connect
when they find it (unless configured with ``run: always``), so a ``va chat``
on the same configuration does not open a second long connection that would
split the events, and they register themselves with the hub so ``va serve``
can tell whether anything is listening.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

SERVE_KEY = "va.serve"


class ServeHub:
    def __init__(self, say: Callable[[str], None] | None = None) -> None:
        self.channels: list[Any] = []
        self._say = say or (lambda line: print(line, file=sys.stderr, flush=True))

    def attach(self, channel: Any) -> Callable[[], None]:
        self.channels.append(channel)

        def detach() -> None:
            if channel in self.channels:
                self.channels.remove(channel)
        return detach

    def say(self, line: str) -> None:
        self._say(line)
