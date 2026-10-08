#!/usr/bin/env python3
"""Render docs/img/*.png: every Mermaid block in DESIGN.md preceded by
``<!-- fig: NAME -->`` becomes ``docs/img/NAME.png``; ``architecture.html`` is
screenshotted to ``docs/img/architecture.png``.

Requirements (see docs/build/README.md): node + @mermaid-js/mermaid-cli (mmdc),
Google Chrome/Chromium, CJK fonts (fonts-noto-cjk).
Usage:  python docs/build/build.py [--node-modules DIR] [--chrome PATH] [--only NAME ...]
"""
from __future__ import annotations

import argparse, os, re, subprocess
from pathlib import Path

BUILD = Path(__file__).resolve().parent
DOCS = BUILD.parent
IMG = DOCS / "img"
SRC = DOCS / "DESIGN.md"
FIG = re.compile(r"<!-- fig: (?P<name>[\w-]+) -->\n```mermaid\n(?P<body>.*?)\n```", re.S)


def render_mermaid(mmdc: str, md: str, only: set[str] | None) -> None:
    IMG.mkdir(exist_ok=True)
    src = BUILD / "mmd"
    src.mkdir(exist_ok=True)
    for m in FIG.finditer(md):
        name = m["name"]
        if only and name not in only:
            continue
        f = src / f"{name}.mmd"
        f.write_text(m["body"] + "\n", encoding="utf-8")
        print("mmdc", name, flush=True)
        subprocess.run([mmdc, "-i", str(f), "-o", str(IMG / f"{name}.png"), "-s", "3", "-q",
                        "-b", "white", "-c", str(BUILD / "mermaid.json"),
                        "-p", str(BUILD / "puppeteer.json")], check=True, stdout=subprocess.DEVNULL)


def render_architecture(chrome: str) -> None:
    print("chrome architecture", flush=True)
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--window-size=1800,1250",
                    f"--screenshot={IMG / 'architecture.png'}", str(BUILD / "architecture.html")],
                   check=True, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-modules", default=os.environ.get("VENTRI_DOCS_NODE_MODULES", "/workspace/tools/node_modules"))
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--only", nargs="*", help="render only these figure names (incl. 'architecture')")
    a = ap.parse_args()
    only = set(a.only) if a.only else None
    render_mermaid(str(Path(a.node_modules) / ".bin" / "mmdc"), SRC.read_text(encoding="utf-8"), only)
    if not only or "architecture" in only:
        render_architecture(a.chrome)


if __name__ == "__main__":
    main()
