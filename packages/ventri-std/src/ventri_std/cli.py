"""``ventri`` command line (DESIGN.md 7): run / apply / tree / doctor / stubgen.

M1 has no daemon control channel, so ``apply``, ``tree`` and ``doctor`` work on
a throw-away kernel: ``apply --dry-run`` loads the configuration in a dry-run
transaction and prints the TxReport (plugins are instantiated and torn down;
``--validate-only`` stops before that), ``tree`` prints the staged tree, and
``run`` hosts the configuration in the foreground and hot-applies edits.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import anyio

from ventri import Kernel, __version__

from .config import ApplyResult, ConfigError, Loader, apply_document, default_secrets, load_document, plan


def default_config() -> Path:
    home = os.environ.get("VENTRI_HOME")
    return (Path(home) if home else Path("~/.ventri")).expanduser() / "ventri.yml"


def _config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", nargs="?", type=Path, default=None,
                   help="configuration file (default: $VENTRI_HOME/ventri.yml or ~/.ventri/ventri.yml)")
    p.add_argument("--profile", action="append", dest="profiles", default=None,
                   help="profile to apply (repeatable; overrides 'profiles:' in the file)")
    p.add_argument("--strict", action="store_true", help="fail if any plugin stays PENDING")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ventri", description="Ventri kernel tools")
    ap.add_argument("--version", action="version", version=f"ventri {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="host a configuration in the foreground (hot-applies edits)")
    _config_args(run)
    run.add_argument("--no-watch", action="store_true", help="do not watch the files")
    app = sub.add_parser("apply", help="apply a configuration (M1: --dry-run only)")
    _config_args(app)
    app.add_argument("--dry-run", action="store_true", help="report what would happen; change nothing")
    app.add_argument("--validate-only", action="store_true",
                     help="stop after parsing, secrets and validation (no plugin is instantiated)")
    app.add_argument("--json", action="store_true", help="machine-readable output")
    tree = sub.add_parser("tree", help="print the plugin tree a configuration produces")
    _config_args(tree)
    doc = sub.add_parser("doctor", help="diagnose the environment and a configuration")
    _config_args(doc)
    stub = sub.add_parser("stubgen", help="generate ventri_stubs.pyi for typed ctx.<service>")
    stub.add_argument("--module", "-m", action="append", default=[], help="also scan this module")
    stub.add_argument("--out", type=Path, default=Path(), help="output directory (default: .)")
    stub.add_argument("--no-entry-points", action="store_true", help="skip ventri.plugins entry points")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "config", "unset") is None:
        args.config = default_config()
    try:
        return int(anyio.run(_COMMANDS[args.cmd], args, backend="asyncio") or 0)
    except ConfigError as e:
        print("configuration error:", file=sys.stderr)
        for err in e.errors:
            print(f"  - {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _result_json(res: ApplyResult) -> str:
    return json.dumps({"ok": res.ok, "changes": [vars(c) for c in res.plan.changes],
                       "report": res.report.to_dict() if res.report else None},
                      default=repr, ensure_ascii=False, indent=2)


async def cmd_apply(args: argparse.Namespace) -> int:
    if not (args.dry_run or args.validate_only):
        print("ventri apply: M1 has no daemon control channel; use --dry-run, or `ventri run` "
              "(which hot-applies edits to the file)", file=sys.stderr)
        return 2
    doc = load_document(args.config, profiles=args.profiles, secrets=default_secrets())
    async with Kernel() as app:
        if args.validate_only:
            p = plan(app.fiber, doc)
            if args.json:
                print(json.dumps({"ok": True, "changes": [vars(c) for c in p.changes]}, indent=2))
            else:
                print(f"configuration OK: {sum(1 for _ in doc.walk())} entries")
                print(p)
            return 0
        res = await apply_document(app.fiber, doc, dry_run=True, strict=args.strict)
    print(_result_json(res) if args.json else str(res))
    return 0 if res.ok else 1


async def cmd_tree(args: argparse.Namespace) -> int:
    doc = load_document(args.config, profiles=args.profiles, secrets=default_secrets())
    out: list[str] = []
    async with Kernel() as app:
        res = await apply_document(app.fiber, doc, dry_run=True, strict=args.strict,
                                   on_staged=lambda tx: out.append(app.tree()))
    print(out[0] if out else app.tree())
    if not res.ok and res.report is not None:
        print(str(res.report), file=sys.stderr)
    return 0 if res.ok else 1


async def cmd_doctor(args: argparse.Namespace) -> int:
    problems = 0

    def check(ok: bool, msg: str, hint: str = "") -> None:
        nonlocal problems
        problems += not ok
        print(("  ok    " if ok else "  FAIL  ") + msg + (f"  ({hint})" if hint and not ok else ""))

    def warn(msg: str) -> None:
        print("  warn  " + msg)

    print(f"ventri {__version__} -- Python {platform.python_version()} on {sys.platform}")
    check(sys.version_info >= (3, 12), "Python >= 3.12")
    if sys.platform != "darwin":
        warn("1.0 targets macOS; this platform is unsupported (the kernel itself is portable)")
    check(args.config.exists(), f"config file {args.config}", "create it or pass a path")
    if not args.config.exists():
        return 1
    try:
        doc = load_document(args.config, profiles=args.profiles, secrets=default_secrets())
    except ConfigError as e:
        for err in e.errors:
            check(False, err)
        return 1
    check(True, f"configuration valid ({sum(1 for _ in doc.walk())} entries, "
                f"profiles: {', '.join(doc.profiles) or '-'})")
    async with Kernel() as app:
        res = await apply_document(app.fiber, doc, dry_run=True, strict=False)
    rep = res.report
    if rep is not None:
        for label, err in rep.failures.items():
            check(False, f"{label} fails to load: {err}")
        for label, why in rep.pending.items():
            check(False, f"{label} would stay pending: {why}")
        if rep.error:
            check(False, f"dry run: {rep.error}")
    if problems == 0:
        print("no problems found")
    return 1 if problems else 0


async def cmd_run(args: argparse.Namespace) -> int:
    async with Kernel() as app:
        ld = Loader(app.fiber, args.config, profiles=args.profiles, strict=args.strict)
        res = await ld.apply(reason="ventri run")
        print(str(res), flush=True)
        if not res.ok:
            return 1
        print(app.tree(), flush=True)
        if args.no_watch:
            await anyio.sleep_forever()

        def report(r: Any) -> None:
            print(("configuration error: " + str(r)) if isinstance(r, ConfigError) else str(r),
                  flush=True)
        await ld.watch(on_result=report)
    return 0


async def cmd_stubgen(args: argparse.Namespace) -> int:
    from .stubgen import generate

    sys.path.insert(0, os.getcwd())
    res = generate(args.out, args.module, use_entry_points=not args.no_entry_points)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {args.out / 'ventri_stubs.pyi'} ({len(res.attrs)} attributes: "
          f"{', '.join(sorted(res.attrs)) or '-'})")
    return 0


_COMMANDS = {"run": cmd_run, "apply": cmd_apply, "tree": cmd_tree, "doctor": cmd_doctor,
             "stubgen": cmd_stubgen}

if __name__ == "__main__":
    sys.exit(main())
