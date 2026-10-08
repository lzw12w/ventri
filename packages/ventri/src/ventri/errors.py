"""Exception hierarchy."""


class KernelError(Exception):
    """Base class for all kernel errors."""


class ServiceNotFound(KernelError, LookupError):
    """``ctx.get(key)`` found no binding for ``key``."""


class ServiceConflict(KernelError):
    """A key is already provided by another live fiber."""


class PluginError(KernelError):
    """A plugin's apply/start raised; the original error is ``__cause__``."""


class TransactionError(KernelError):
    """A transaction could not be committed and was rolled back."""


class TransactionBusy(TransactionError):
    """Another transaction is open and ``wait=False`` was requested."""


class TransactionConflict(TransactionError):
    """Commit-time validation failed (base registry changed under the transaction)."""


class TransactionTimeout(TransactionError):
    """The transaction's ``timeout`` elapsed before commit; it was rolled back."""
