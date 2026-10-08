#!/usr/bin/env bash
# Build the self-contained bundle the Harbor adapter uploads into task containers:
#   rt/        a standalone CPython 3.12 (uv-managed python-build-standalone) with
#              ventri, ventri-std, ventri-agent and their dependencies installed
#   runner.py  the in-container runner
#   VERSION    "ventri-agent <version>@<commit>"
# Usage: integrations/harbor/build_bundle.sh [OUT.tgz]   (default: integrations/harbor/dist/ventri_bundle.tgz)
# Linux x86_64 containers only (the interpreter is copied from this machine).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/dist/ventri_bundle.tgz}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PY="$(uv python find --managed-python 3.12)"
PYROOT="$(cd "$(dirname "$PY")/.." && pwd)"
cp -a "$PYROOT" "$WORK/rt"
rm -f "$WORK"/rt/lib/python3.12/EXTERNALLY-MANAGED
(cd "$REPO" && uv build --all-packages --wheel -o "$WORK/wheels" -q)
uv pip install -q --python "$WORK/rt/bin/python3.12" --no-cache "$WORK"/wheels/*.whl
cp "$HERE/runner.py" "$WORK/runner.py"
VER="$("$WORK/rt/bin/python3.12" -I -c 'import ventri_agent; print(ventri_agent.__version__)')"
echo "ventri-agent $VER@$(git -C "$REPO" rev-parse --short HEAD)" > "$WORK/VERSION"
"$WORK/rt/bin/python3.12" -I -c 'import ventri, ventri_std, ventri_agent, httpx, yaml'
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$WORK" rt runner.py VERSION
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(cat "$WORK/VERSION"))"
