"""Ventri Core -- an asyncio plugin kernel: plugin tree = task tree = scope tree,
transactional plugin changes, stable observability. Only asyncio is supported
(anyio is an internal dependency)."""
from .context import Context
from .errors import (
    DependencyCycle,
    KernelError,
    LoadTimeout,
    PluginError,
    ServiceConflict,
    ServiceNotFound,
    TransactionBusy,
    TransactionConflict,
    TransactionError,
    TransactionTimeout,
)
from .events import Deny, Event, Rewrite
from .fiber import Fiber, State, TaskHandle
from .kernel import Binding, Kernel, Realm, TraceEvent
from .plugin import Retry, plugin
from .report import TxReport
from .secret import Secret, redact
from .trace import KERNEL_KINDS, SCHEMA_VERSION
from .transaction import Transaction

__version__ = "0.2.0a1"

__all__ = [
    "KERNEL_KINDS",
    "SCHEMA_VERSION",
    "Binding",
    "Context",
    "Deny",
    "DependencyCycle",
    "Event",
    "Fiber",
    "Kernel",
    "KernelError",
    "LoadTimeout",
    "PluginError",
    "Realm",
    "Retry",
    "Rewrite",
    "Secret",
    "ServiceConflict",
    "ServiceNotFound",
    "State",
    "TaskHandle",
    "TraceEvent",
    "Transaction",
    "TransactionBusy",
    "TransactionConflict",
    "TransactionError",
    "TransactionTimeout",
    "TxReport",
    "plugin",
    "redact",
]
