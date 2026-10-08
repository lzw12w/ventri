"""Ventri Core -- an asyncio plugin kernel: plugin tree = task tree = scope tree,
transactional plugin changes, stable observability. Only asyncio is supported
(anyio is an internal dependency)."""
from .context import Context
from .errors import (
    KernelError,
    PluginError,
    ServiceConflict,
    ServiceNotFound,
    TransactionBusy,
    TransactionConflict,
    TransactionError,
)
from .fiber import Fiber, State, TaskHandle
from .kernel import Binding, Kernel, Realm, TraceEvent
from .plugin import plugin
from .transaction import Transaction

__version__ = "0.2.0a1"

__all__ = [
    "Binding",
    "Context",
    "Fiber",
    "Kernel",
    "KernelError",
    "PluginError",
    "Realm",
    "ServiceConflict",
    "ServiceNotFound",
    "State",
    "TaskHandle",
    "TraceEvent",
    "Transaction",
    "TransactionBusy",
    "TransactionConflict",
    "TransactionError",
    "plugin",
]
