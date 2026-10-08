"""Declarative configuration loader (DESIGN.md 4.7).

``ventri.yml`` (+ ordered profile patches) -> secrets -> validation -> diff
against the running tree -> **one** transaction (``origin="config"``).

Pipeline (:class:`Loader`)::

    read YAML  ->  apply profiles (patch ops, by id)  ->  resolve ${secret:..}/${env:..}
      ->  resolve ``use`` targets  ->  validate each config against the plugin's Config
      ->  plan (diff by stable id)  ->  single transaction (or dry run -> TxReport)

Rules:

* **Stable id**: explicit ``id``; otherwise the ``use`` string, suffixed ``@2``,
  ``@3``... for further siblings with the same ``use`` (groups: their name). Ids
  are unique among siblings; a fiber's config id is the ``/``-joined path
  (``tools/github``), stored in ``fiber.meta["config_id"]`` (``meta["id"]`` is
  the last segment and becomes the fiber's trace path segment).
* **Diff**: entries missing from the desired tree (or ``disabled``) are disposed;
  entries whose ``use`` target, validated config, ``timeout`` or ``retry``
  changed -- or that are FAILED -- are replaced; new entries are added. Config
  equality is computed on the *validated* model, so formatting changes and
  spelled-out defaults do not restart anything.
* **One save = one transaction**: every change of an apply commits or rolls back
  together; on failure the previous configuration keeps serving and the
  :class:`ApplyResult` names the failing fibers.
* **Secrets** never reach snapshots, traces or reports: ``${secret:name}``
  resolves to a :class:`ventri.Secret`, and the kernel redacts by key name too.

Non-guarantees: ``use`` targets are imported once per process -- editing a
plugin's *source* is not detected (dev-mode code reload is not part of M1);
stop-first replacements roll back *degraded* (see ``Transaction.replace``).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import anyio
import yaml

from ventri import Fiber, PluginError, Retry, Secret, State, TransactionError, TxReport, plugin
from ventri.plugin import MISSING, describe

SUPPORTED_VERSION = 1
ENTRY_POINT_GROUP = "ventri.plugins"
_ENTRY_KEYS = {"use", "id", "config", "disabled", "timeout", "retry"}
_GROUP_KEYS = {"group", "id", "plugins", "disabled"}
_REF = re.compile(r"\$\{(secret|env):([^}:]+)(?::-([^}]*))?\}")
_UNSET: Any = MISSING  # "not set in the YAML": the plugin's own default applies


class ConfigError(Exception):
    """The configuration cannot be loaded (syntax, schema, secrets, ``use``, validation).
    Raised before any transaction starts, so nothing changed. ``errors`` lists every
    problem found (``"<config id>: <message>"``)."""

    def __init__(self, errors: str | list[str]) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


# ------------------------------------------------------------------ secrets
class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...


class EnvSecrets:
    """``${secret:deepseek}`` -> ``$VENTRI_SECRET_DEEPSEEK``."""

    def __init__(self, prefix: str = "VENTRI_SECRET_", environ: Mapping[str, str] | None = None) -> None:
        self.prefix = prefix
        self.environ = os.environ if environ is None else environ

    def get(self, name: str) -> str | None:
        return self.environ.get(self.prefix + re.sub(r"\W", "_", name).upper())


class KeychainSecrets:
    """macOS keychain: ``security find-generic-password -s <service> -a <name> -w``."""

    def __init__(self, service: str = "ventri") -> None:
        self.service = service

    def get(self, name: str) -> str | None:
        if sys.platform != "darwin":
            return None
        try:
            out = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-s", self.service, "-a", name, "-w"],
                capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.rstrip("\n") if out.returncode == 0 else None


class DictSecrets:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = dict(values)

    def get(self, name: str) -> str | None:
        return self.values.get(name)


class ChainSecrets:
    def __init__(self, *stores: SecretStore) -> None:
        self.stores = stores

    def get(self, name: str) -> str | None:
        for s in self.stores:
            v = s.get(name)
            if v is not None:
                return v
        return None


def default_secrets() -> SecretStore:
    """Keychain (service ``ventri``) on macOS, then ``VENTRI_SECRET_<NAME>``."""
    return ChainSecrets(KeychainSecrets(), EnvSecrets())


def interpolate(value: Any, secrets: SecretStore, environ: Mapping[str, str] | None = None,
                where: str = "") -> Any:
    """Resolve ``${secret:name}`` and ``${env:NAME}`` / ``${env:NAME:-default}`` in all
    strings of ``value``. A string containing a secret reference becomes a
    ``Secret[str]``; env references are substituted as plain text."""
    env = os.environ if environ is None else environ
    if isinstance(value, dict):
        return {k: interpolate(v, secrets, env, f"{where}.{k}" if where else str(k))
                for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, secrets, env, f"{where}[{i}]") for i, v in enumerate(value)]
    if not isinstance(value, str) or "${" not in value:
        return value
    secret = False

    def sub(m: re.Match[str]) -> str:
        nonlocal secret
        kind, name, default = m.group(1), m.group(2).strip(), m.group(3)
        if kind == "secret":
            secret = True
            v = secrets.get(name)
            if v is None:
                raise ConfigError(f"{where}: secret {name!r} not found (keychain service 'ventri' "
                                  f"or ${'VENTRI_SECRET_' + re.sub(r'\W', '_', name).upper()})")
            return v
        v = env.get(name, default)
        if v is None:
            raise ConfigError(f"{where}: environment variable {name!r} is not set")
        return v

    out = _REF.sub(sub, value)
    return Secret(out) if secret else out


# ------------------------------------------------------------- use resolution
def resolve_use(use: str) -> Any:
    """``use`` -> plugin object. Tries, in order: an entry point named ``use`` in
    group ``ventri.plugins``; ``module:attr``; a dotted path (longest importable
    module prefix, then attributes). A module resolves to its ``plugin``
    attribute, or itself if it defines ``apply``."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == use:
            return _unwrap(ep.load(), use)
    if ":" in use:
        mod, _, attr = use.partition(":")
        obj: Any = importlib.import_module(mod)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return _unwrap(obj, use)
    parts = use.split(".")
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        try:
            obj = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name is not None and not modname.startswith(e.name) and e.name != modname:
                raise  # the module exists but one of *its* imports is missing
            continue
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
        except AttributeError:
            break
        return _unwrap(obj, use)
    raise ConfigError(f"cannot resolve use {use!r} (no entry point, module or attribute)")


def _unwrap(obj: Any, use: str) -> Any:
    if isinstance(obj, ModuleType):
        target = getattr(obj, "plugin", None)
        if target is not None and not isinstance(target, ModuleType) and target is not plugin:
            return target
        if hasattr(obj, "apply"):
            return obj
        raise ConfigError(f"module {use!r} defines neither 'plugin' nor 'apply'")
    return obj


# ------------------------------------------------------------- document model
@dataclass
class Entry:
    """One node of the desired tree (after profiles, secrets and validation)."""

    id: str
    path: str
    use: str | None = None  # None for a group
    config: Any = None
    disabled: bool = False
    timeout: Any = _UNSET
    retry: Any = _UNSET
    children: list[Entry] = field(default_factory=list)
    target: Any = None
    validated: Any = None

    @property
    def is_group(self) -> bool:
        return self.use is None

    def walk(self) -> Iterable[Entry]:
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class Document:
    path: Path | None
    profiles: list[str]
    entries: list[Entry]
    data: dict  # the merged raw document (``agents`` and other sections, unresolved)
    files: list[Path]  # files read (config + profiles), for the watcher

    def walk(self) -> Iterable[Entry]:
        for e in self.entries:
            yield from e.walk()

    def find(self, ref: str) -> Entry:
        return _find(self.entries, ref)


def _find(entries: list[Entry], ref: str) -> Entry:
    hits = [e for top in entries for e in top.walk() if ref in (e.path, e.id)]
    exact = [e for e in hits if e.path == ref]
    if exact:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    raise ConfigError(f"no entry with id {ref!r}" if not hits
                      else f"id {ref!r} is ambiguous ({', '.join(e.path for e in hits)}); use the path")


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            out[k] = _deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(patch)


def _raw_tree(items: Any, where: str) -> list[dict]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ConfigError(f"{where}: 'plugins' must be a list")
    out = []
    for i, item in enumerate(items):
        w = f"{where}[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{w}: plugin entry must be a mapping")
        if "group" in item:
            bad = set(item) - _GROUP_KEYS
            if bad:
                raise ConfigError(f"{w}: unknown group keys {sorted(bad)}")
            item = {**item, "plugins": _raw_tree(item.get("plugins"), f"{w}.plugins")}
        else:
            bad = set(item) - _ENTRY_KEYS
            if "use" not in item or not isinstance(item["use"], str):
                raise ConfigError(f"{w}: entry needs 'use' (a string) or 'group'")
            if bad:
                raise ConfigError(f"{w}: unknown keys {sorted(bad)}")
        out.append(dict(item))
    return out


def _assign_ids(items: list[dict], prefix: str = "") -> list[dict]:
    seen: dict[str, int] = {}
    ids: set[str] = set()
    for item in items:
        if "id" in item:
            sid = str(item["id"])
        elif "group" in item:
            sid = str(item["group"])
        else:
            n = seen[item["use"]] = seen.get(item["use"], 0) + 1
            sid = item["use"] if n == 1 else f"{item['use']}@{n}"
        if not sid or "/" in sid:
            raise ConfigError(f"{prefix or '<root>'}: invalid id {sid!r}")
        if sid in ids:
            raise ConfigError(f"{prefix or '<root>'}: duplicate id {sid!r}")
        ids.add(sid)
        item["_id"] = sid
        item["_path"] = f"{prefix}/{sid}" if prefix else sid
        if "group" in item:
            _assign_ids(item["plugins"], item["_path"])
    return items


def _iter_raw(items: list[dict]) -> Iterable[tuple[list[dict], dict]]:
    for item in items:
        yield items, item
        if "group" in item:
            yield from _iter_raw(item["plugins"])


def _locate(items: list[dict], ref: str) -> tuple[list[dict], dict]:
    hits = [(lst, it) for lst, it in _iter_raw(items) if ref in (it["_path"], it["_id"])]
    exact = [h for h in hits if h[1]["_path"] == ref]
    if exact:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    raise ConfigError(f"profile patch: no entry with id {ref!r}" if not hits else
                      f"profile patch: id {ref!r} is ambiguous; use the path")


def apply_patch(items: list[dict], ops: Any, where: str) -> list[dict]:
    """Apply profile patch ops (ordered) to a raw plugin list.

    ``{id: X, config: {...}}`` deep-merges config; ``{id: X, disabled: bool}``;
    ``{id: X, use|timeout|retry: ...}`` replaces the field;
    ``{add: {entry}, under: <group id>}`` appends (root if no ``under``);
    ``{remove: X}`` deletes the entry."""
    if not isinstance(ops, list):
        raise ConfigError(f"{where}: 'patch' must be a list")
    for i, op in enumerate(ops):
        w = f"{where}.patch[{i}]"
        if not isinstance(op, dict):
            raise ConfigError(f"{w}: patch op must be a mapping")
        _assign_ids(items)
        if "add" in op:
            entry = _raw_tree([op["add"]], f"{w}.add")[0]
            if "under" in op:
                _, grp = _locate(items, str(op["under"]))
                if "group" not in grp:
                    raise ConfigError(f"{w}: 'under' must name a group")
                grp["plugins"].append(entry)
            else:
                items.append(entry)
        elif "remove" in op:
            lst, it = _locate(items, str(op["remove"]))
            lst.remove(it)
        elif "id" in op:
            _, it = _locate(items, str(op["id"]))
            bad = set(op) - {"id", "config", "disabled", "use", "timeout", "retry"}
            if bad:
                raise ConfigError(f"{w}: unknown patch keys {sorted(bad)}")
            for k, v in op.items():
                if k == "config":
                    it["config"] = _deep_merge(it.get("config") or {}, v)
                elif k != "id":
                    it[k] = copy.deepcopy(v)
        else:
            raise ConfigError(f"{w}: patch op needs 'id', 'add' or 'remove'")
    return items


def _strip(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        it = {k: v for k, v in it.items() if not k.startswith("_")}
        if "group" in it:
            it["plugins"] = _strip(it["plugins"])
        out.append(it)
    return out


def load_document(path: str | os.PathLike | None = None, *, text: str | None = None,
                  profiles: list[str] | None = None, secrets: SecretStore | None = None,
                  environ: Mapping[str, str] | None = None,
                  resolver: Callable[[str], Any] = resolve_use) -> Document:
    """Read, merge, interpolate, resolve and validate. Raises :class:`ConfigError`
    listing every problem; never touches the running kernel."""
    p = Path(path).expanduser() if path is not None else None
    files: list[Path] = []
    if text is None:
        if p is None:
            raise ConfigError("no configuration path or text given")
        files.append(p)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"{p}: {e.strerror or e}") from e
    data = _yaml(text, str(p or "<text>"))
    if not isinstance(data, dict):
        raise ConfigError("top level must be a mapping")
    if data.get("version") != SUPPORTED_VERSION:
        raise ConfigError(f"unsupported config version {data.get('version')!r} (expected 1)")
    items = _raw_tree(data.get("plugins"), "plugins")
    chosen = list(data.get("profiles") or []) if profiles is None else list(profiles)
    for name in chosen:
        if p is None:
            raise ConfigError(f"profile {name!r}: profiles need a config file path")
        pf = p.parent / "profiles" / f"{name}.yml"
        files.append(pf)
        try:
            pdata = _yaml(pf.read_text(encoding="utf-8"), str(pf))
        except OSError as e:
            raise ConfigError(f"profile {name!r}: cannot read {pf}") from e
        if not isinstance(pdata, dict) or set(pdata) - {"patch"}:
            raise ConfigError(f"profile {name!r}: expected a mapping with only 'patch'")
        items = apply_patch(items, pdata.get("patch") or [], f"profile {name}")
    _assign_ids(items)
    secrets = secrets if secrets is not None else default_secrets()
    errors: list[str] = []
    entries = [_build(it, secrets, environ, resolver, errors) for it in items]
    if errors:
        raise ConfigError(errors)
    merged = {**data, "plugins": _strip(items), "profiles": chosen}
    return Document(p, chosen, entries, merged, files)


def _yaml(text: str, where: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{where}: invalid YAML: {e}") from e


def _build(it: dict, secrets: SecretStore, environ: Mapping[str, str] | None,
           resolver: Callable[[str], Any], errors: list[str]) -> Entry:
    e = Entry(id=it["_id"], path=it["_path"], disabled=bool(it.get("disabled", False)))
    if "group" in it:
        e.children = [_build(c, secrets, environ, resolver, errors) for c in it["plugins"]]
        return e
    e.use = it["use"]
    if "timeout" in it:
        e.timeout = it["timeout"]
    try:
        if "retry" in it:
            e.retry = Retry.coerce(it["retry"])
        e.config = interpolate(it.get("config"), secrets, environ, e.path)
        if e.disabled:
            return e  # disabled entries are neither imported nor validated
        e.target = resolver(e.use)
        e.validated = validate(e.target, e.config)
    except ConfigError as err:
        errors.extend(m if m.startswith(e.path) else f"{e.path}: {m}" for m in err.errors)
    except Exception as err:  # noqa: BLE001 - every config error is collected and reported
        errors.append(f"{e.path}: {type(err).__name__}: {err}")
    return e


def validate(target: Any, config: Any) -> Any:
    """Validate ``config`` the way the kernel will (``Config(**config)``)."""
    spec = describe(target)
    if spec.config_type is not None and (config is None or isinstance(config, dict)):
        return spec.config_type(**(config or {}))
    return config


# --------------------------------------------------------------------- plan
@dataclass
class Change:
    op: str  # add | remove | replace
    id: str
    use: str | None
    reason: str

    def __str__(self) -> str:
        sign = {"add": "+", "remove": "-", "replace": "~"}[self.op]
        return f"{sign} {self.id}" + (f" ({self.use})" if self.use else "") + f": {self.reason}"


@dataclass
class Plan:
    changes: list[Change] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.changes

    def __str__(self) -> str:
        if not self.changes:
            return "no changes"
        return "\n".join(str(c) for c in self.changes)


@dataclass
class ApplyResult:
    plan: Plan
    report: TxReport | None  # None when there was nothing to do
    document: Document

    @property
    def ok(self) -> bool:
        return self.report is None or (self.report.ok and self.report.outcome != "rolled_back")

    def __str__(self) -> str:
        return str(self.plan) + ("" if self.report is None else "\n" + str(self.report))


def _group(ctx: Any, config: Any) -> None:
    """Container fiber for a ``group:`` entry (provides nothing)."""


group_plugin = plugin(_group, name="group")


def _managed(parent: Fiber) -> dict[str, Fiber]:
    return {f.meta["config_id"]: f for f in parent.children
            if "config_id" in f.meta and f.state not in (State.UNLOADING, State.DISPOSED)}


def _changed_keys(old: Any, new: Any, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for k in sorted(set(old) | set(new), key=str):
            key = f"{prefix}.{k}" if prefix else str(k)
            if k not in old or k not in new:
                out.append(key)
            elif old[k] != new[k]:
                out += _changed_keys(old[k], new[k], key)
        return out
    return [prefix or "config"]


def plan(parent: Fiber, doc: Document) -> Plan:
    """Diff the desired tree against the config-managed fibers under ``parent``."""
    out = Plan()
    _plan(parent, doc.entries, out)
    return out


def _wanted(entries: list[Entry]) -> dict[str, Entry]:
    return {e.path: e for e in entries if not e.disabled}


def _replace_reason(f: Fiber, e: Entry) -> str | None:
    if f.state is State.FAILED:
        return f"failed ({f.error!r}); retrying"
    if f.meta.get("config_use") != e.use or f.plugin is not e.target:
        return f"use changed ({f.meta.get('config_use')} -> {e.use})"
    if f._timeout != e.timeout:
        return "timeout changed"
    if f._retry != e.retry:
        return "retry changed"
    try:
        old = f.config if f.config is not None else validate(f.plugin, f.raw_config)
    except Exception:  # noqa: BLE001 - an old config that no longer validates counts as changed
        old = _UNSET
    if old != e.validated:
        keys = _changed_keys(f.raw_config, e.config)
        return "config changed: " + ", ".join(keys[:8]) + (" ..." if len(keys) > 8 else "")
    return None


def _plan(parent: Fiber, entries: list[Entry], out: Plan) -> None:
    current = _managed(parent)
    wanted = _wanted(entries)
    for cid, f in current.items():
        e = wanted.get(cid)
        if e is None:
            reason = "disabled" if any(x.path == cid and x.disabled for x in entries) else "removed"
            out.changes.append(Change("remove", cid, f.meta.get("config_use"), reason))
        elif e.is_group != ("config_use" not in f.meta):
            out.changes.append(Change("replace", cid, e.use, "kind changed (group/plugin)"))
        elif e.is_group:
            out.unchanged.append(cid)
            _plan(f, e.children, out)
        else:
            reason = _replace_reason(f, e)
            if reason:
                out.changes.append(Change("replace", cid, e.use, reason))
            else:
                out.unchanged.append(cid)
    for cid, e in wanted.items():
        if cid not in current:
            out.changes.append(Change("add", cid, e.use, "new"))
            for sub in e.walk():
                if sub is not e and not sub.disabled:
                    out.changes.append(Change("add", sub.path, sub.use, "new (in group)"))


def _meta(e: Entry) -> dict:
    meta = {"config_id": e.path, "id": e.id}
    if e.use is not None:
        meta["config_use"] = e.use
    return meta


def _opts(e: Entry) -> dict:
    opts: dict[str, Any] = {}
    if e.timeout is not _UNSET:
        opts["timeout"] = e.timeout
    if e.retry is not _UNSET:
        opts["retry"] = e.retry
    return opts


async def _stage(tx: Any, parent: Fiber, entries: list[Entry]) -> None:
    current = _managed(parent)
    wanted = _wanted(entries)
    for cid, f in current.items():  # 1. disposals (and kind changes)
        e = wanted.get(cid)
        if e is None or e.is_group != ("config_use" not in f.meta):
            await tx.dispose(f)
    for cid, f in current.items():  # 2. replacements / recursion into groups
        e = wanted.get(cid)
        if e is None or e.is_group != ("config_use" not in f.meta):
            continue
        if e.is_group:
            await _stage(tx, f, e.children)
        elif _replace_reason(f, e):
            new = await tx.replace(f, e.target, e.config, timeout=e.timeout, retry=e.retry)
            new.meta.update(_meta(e))  # replace() copies meta; record the new use
    for cid, e in wanted.items():  # 3. additions
        f = current.get(cid)
        if f is not None and e.is_group == ("config_use" not in f.meta):
            continue
        await _add(tx, parent, e)


async def _add(tx: Any, parent: Fiber, e: Entry) -> None:
    if e.is_group:
        g = await tx.plugin(group_plugin, None, parent=parent, meta=_meta(e))
        for c in e.children:
            if not c.disabled:
                await _add(tx, g, c)
    else:
        await tx.plugin(e.target, e.config, parent=parent, meta=_meta(e), **_opts(e))


async def apply_document(parent: Fiber, doc: Document, *, dry_run: bool = False,
                         strict: bool = False, reason: str | None = None,
                         timeout: float | None = None) -> ApplyResult:
    """Apply ``doc`` under ``parent`` as one transaction (``origin="config"``).
    Returns an :class:`ApplyResult`; a failed transaction is rolled back (the
    previous configuration keeps serving) and reported, not raised."""
    p = plan(parent, doc)
    if p.empty:
        return ApplyResult(p, None, doc)
    ctx = parent.ctx
    tx = ctx.transaction(origin="config", reason=reason or _reason(doc), dry_run=dry_run,
                         strict=strict, timeout=timeout)
    try:
        async with tx:
            await _stage(tx, parent, doc.entries)
    except (TransactionError, PluginError):
        pass  # rolled back; tx.report carries the outcome and error
    return ApplyResult(p, tx.report, doc)


def _reason(doc: Document) -> str:
    src = str(doc.path) if doc.path else "<text>"
    return f"apply {src}" + (f" profiles={','.join(doc.profiles)}" if doc.profiles else "")


# ------------------------------------------------------------------- loader
class Loader:
    """Load / apply / watch one configuration file under ``parent`` (a fiber;
    pass ``kernel.fiber`` for the root)."""

    def __init__(self, parent: Fiber, path: str | os.PathLike, *, profiles: list[str] | None = None,
                 secrets: SecretStore | None = None, environ: Mapping[str, str] | None = None,
                 resolver: Callable[[str], Any] = resolve_use, strict: bool = False) -> None:
        self.parent = parent
        self.path = Path(path).expanduser()
        self.profiles = profiles
        self.secrets = secrets
        self.environ = environ
        self.resolver = resolver
        self.strict = strict
        self.document: Document | None = None
        self.last: ApplyResult | None = None
        self._lock = anyio.Lock()

    def load(self) -> Document:
        return load_document(self.path, profiles=self.profiles, secrets=self.secrets,
                             environ=self.environ, resolver=self.resolver)

    async def plan(self) -> Plan:
        return plan(self.parent, self.load())

    async def apply(self, *, dry_run: bool = False, strict: bool | None = None,
                    reason: str | None = None) -> ApplyResult:
        """Load and apply (or dry-run). Raises :class:`ConfigError` (nothing changed)."""
        async with self._lock:
            doc = self.load()
            res = await apply_document(self.parent, doc, dry_run=dry_run, reason=reason,
                                       strict=self.strict if strict is None else strict)
            if not dry_run:
                self.last = res
                if res.ok:
                    self.document = doc
            return res

    def _fingerprint(self) -> tuple:
        files = self.document.files if self.document else [self.path]
        out = []
        for f in files:
            try:
                out.append((str(f), hashlib.sha256(f.read_bytes()).hexdigest()))
            except OSError:
                out.append((str(f), None))
        return tuple(out)

    async def watch(self, *, poll: float = 0.1, debounce: float = 0.3,
                    on_result: Callable[[ApplyResult | ConfigError], Any] | None = None) -> None:
        """Poll the config and profile files; after a change, wait until they have been
        quiet for ``debounce`` seconds, then apply once (one save = one
        transaction). Errors are reported through ``on_result`` and traced
        (``config.error``); the loop keeps running."""
        seen = self._fingerprint()
        while True:
            await anyio.sleep(poll)
            fp = self._fingerprint()
            if fp == seen:
                continue
            while True:  # debounce: wait for quiet
                await anyio.sleep(debounce)
                again = self._fingerprint()
                if again == fp:
                    break
                fp = again
            seen = fp
            result: ApplyResult | ConfigError
            try:
                result = await self.apply(reason="file changed")
            except ConfigError as e:
                result = e
                self.parent.ctx.trace("config.error", errors=e.errors)
            else:
                rep = result.report
                self.parent.ctx.trace("config.apply", ok=result.ok, changes=len(result.plan.changes),
                                      tx=rep.tx if rep else None)
            seen = self._fingerprint()  # profiles may have changed the watched set
            if on_result is not None:
                on_result(result)


@dataclass
class LoaderConfig:
    path: str = "~/.ventri/ventri.yml"
    profiles: list[str] | None = None
    watch: bool = True
    poll: float = 0.1
    debounce: float = 0.3
    strict: bool = False


@plugin(name="config.loader", config=LoaderConfig, provides={"config.loader": Loader})
def loader(ctx: Any, config: LoaderConfig) -> None:
    """``use: ventri_std.config.loader`` -- apply a configuration file under this
    plugin's *parent* and keep it in sync (watch). The first apply runs in a task
    right after the loader becomes ACTIVE (a plugin cannot run a transaction
    while it is itself being loaded). The applied plugins are managed, not owned:
    unloading the loader stops watching and leaves them running. Provides the
    :class:`Loader` as ``"config.loader"``."""
    parent = ctx.fiber.parent or ctx.fiber
    ld = Loader(parent, config.path, profiles=config.profiles, strict=config.strict)
    ctx.provide("config.loader", ld)

    async def run() -> None:
        try:
            res = await ld.apply(reason="initial load")
            ctx.trace("config.apply", ok=res.ok, changes=len(res.plan.changes),
                      tx=res.report.tx if res.report else None)
        except ConfigError as e:
            ctx.trace("config.error", errors=e.errors)
        if config.watch:
            await ld.watch(poll=config.poll, debounce=config.debounce)

    ctx.spawn(run, name="apply+watch")
