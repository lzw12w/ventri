#!/usr/bin/env python3
"""Build docs/img/*.png from the Mermaid blocks in DESIGN.md, then DESIGN.pdf.

Requirements (see docs/build/README.md): node + @mermaid-js/mermaid-cli (mmdc),
puppeteer (bundled with mmdc), Google Chrome/Chromium, python `markdown`, CJK fonts
(fonts-noto-cjk). Usage:  python docs/build/build.py [--mmdc PATH] [--node-modules DIR]
"""
from __future__ import annotations

import argparse, html, json, os, re, subprocess, sys, tempfile
from pathlib import Path

import markdown

BUILD = Path(__file__).resolve().parent
DOCS = BUILD.parent
IMG = DOCS / "img"
SRC = DOCS / "DESIGN.md"
FIG = re.compile(r"<!-- fig: (?P<name>[\w-]+) -->\n```mermaid\n(?P<body>.*?)\n```", re.S)
TALL = {"config-apply": 130, "sandbox": 130, "evolution": 175, "session-lifecycle": 150, "fiber-state": 160}


def render_mermaid(mmdc: str, md: str) -> None:
    IMG.mkdir(exist_ok=True)
    src = BUILD / "mmd"
    src.mkdir(exist_ok=True)
    for m in FIG.finditer(md):
        name, body = m["name"], m["body"]
        f = src / f"{name}.mmd"
        f.write_text(body + "\n", encoding="utf-8")
        cmd = [mmdc, "-i", str(f), "-o", str(IMG / f"{name}.png"), "-s", "3", "-q",
               "-b", "white", "-c", str(BUILD / "mermaid.json"), "-p", str(BUILD / "puppeteer.json")]
        print("mmdc", name, flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def render_architecture(chrome: str) -> None:
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--window-size=1800,1250",
                    f"--screenshot={IMG / 'architecture.png'}", str(BUILD / "architecture.html")],
                   check=True, stderr=subprocess.DEVNULL)


def to_html(md: str) -> str:
    n = 0
    heading = ""

    def fig(m: re.Match) -> str:
        nonlocal n
        n += 1
        name = m["name"]
        return (f'<figure class="{"wide" if name in ("roadmap", "layers") else ""}">'
                f'<img src="img/{name}.png" alt="{name}" style="max-height:{TALL.get(name, 225)}mm"><figcaption>图 {n}：{name}</figcaption></figure>')

    # Caption = nearest preceding heading text.
    out, last = [], 0
    for m in FIG.finditer(md):
        pre = md[last:m.start()]
        hs = re.findall(r"^#{2,4} (.+)$", pre, re.M)
        if hs:
            heading = hs[-1]
        n_before = n
        out.append(pre)
        out.append(fig(m).replace(f"图 {n_before + 1}：{m['name']}", f"图 {n_before + 1}：{heading}"))
        last = m.end()
    out.append(md[last:])
    body_md = "".join(out)
    body_md = body_md.replace("![Ventri 总体架构（内核 + Agent）](img/architecture.png)",
                              '<figure class="wide"><img src="img/architecture.png"><figcaption>图 0：Ventri 总体架构（内核 + Agent）</figcaption></figure>')
    # checkboxes
    body_md = body_md.replace("- [ ] ", "- ☐ ")
    conv = markdown.Markdown(extensions=["tables", "fenced_code", "footnotes", "toc", "attr_list", "sane_lists"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
    body = conv.convert(body_md)
    # Split title (first h1 + blockquote) into a cover with TOC.
    first_h2 = body.find("<h2")
    cover, rest = body[:first_h2], body[first_h2:]
    toc = conv.toc
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<link rel="stylesheet" href="build/style.css"><title>Ventri 设计与路线图</title></head><body>
<section class="cover">{cover}<h3>目录</h3><div class="toc">{toc}</div></section>
{rest}</body></html>"""


def to_pdf(node_modules: Path, html_path: Path, pdf_path: Path) -> None:
    js = f"""
const puppeteer = require({json.dumps(str(node_modules / 'puppeteer'))});
(async () => {{
  const b = await puppeteer.launch({{executablePath: '/usr/bin/google-chrome', args: ['--no-sandbox','--disable-gpu']}});
  const p = await b.newPage();
  await p.goto('file://{html_path}', {{waitUntil: 'networkidle0'}});
  await p.evaluateHandle('document.fonts.ready');
  await p.pdf({{path: {json.dumps(str(pdf_path))}, format: 'A4', printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:7pt;width:100%;padding:0 16mm;color:#8a93a6;font-family:Noto Sans CJK SC;text-align:right">Ventri 设计与路线图 · v1.0 · 2026-10-08</div>',
    footerTemplate: '<div style="font-size:7.5pt;width:100%;text-align:center;color:#8a93a6;font-family:Noto Sans CJK SC"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: {{top: '18mm', bottom: '16mm', left: '16mm', right: '16mm'}}}});
  await b.close();
}})();
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
    subprocess.run(["node", f.name], check=True)
    os.unlink(f.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-modules", default=os.environ.get("VENTRI_DOCS_NODE_MODULES", "/workspace/tools/node_modules"))
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--skip-diagrams", action="store_true")
    a = ap.parse_args()
    nm = Path(a.node_modules)
    md = SRC.read_text(encoding="utf-8")
    if not a.skip_diagrams:
        render_mermaid(str(nm / ".bin" / "mmdc"), md)
        render_architecture(a.chrome)
    html_path = DOCS / "DESIGN.html"
    html_path.write_text(to_html(md), encoding="utf-8")
    to_pdf(nm, html_path, DOCS / "DESIGN.pdf")
    html_path.unlink()
    print("wrote", DOCS / "DESIGN.pdf")


if __name__ == "__main__":
    main()
