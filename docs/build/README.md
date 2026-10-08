# 文档构建

`DESIGN.md` 中带 `<!-- fig: NAME -->` 标记的 Mermaid 代码块会被渲染为 `docs/img/NAME.png`；
`architecture.html` 是手工排版的总体架构图，截图为 `docs/img/architecture.png`；最后生成 `docs/DESIGN.pdf`（A4，含页码）。

依赖：Node.js、`@mermaid-js/mermaid-cli`（含 puppeteer）、Google Chrome / Chromium、Python `markdown`、CJK 字体（`fonts-noto-cjk`）。

```bash
mkdir -p /workspace/tools && cd /workspace/tools && npm init -y && PUPPETEER_SKIP_DOWNLOAD=1 npm i @mermaid-js/mermaid-cli
python3 -m venv venv && venv/bin/pip install markdown
cd /workspace/ventri && /workspace/tools/venv/bin/python docs/build/build.py \
    --node-modules /workspace/tools/node_modules [--skip-diagrams]
```

Chrome 路径在 `puppeteer.json` 与 `--chrome` 参数中配置（默认 `/usr/bin/google-chrome`）。
