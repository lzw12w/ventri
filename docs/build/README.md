# 图片构建

`DESIGN.md` 是唯一的文档交付物。文中带 `<!-- fig: NAME -->` 标记的 Mermaid 代码块会被渲染为 `docs/img/NAME.png`；
`architecture.html` 是手工排版的总体架构图，截图为 `docs/img/architecture.png`。

依赖：Node.js、`@mermaid-js/mermaid-cli`、Google Chrome / Chromium、CJK 字体（`fonts-noto-cjk`）。

```bash
mkdir -p /workspace/tools && cd /workspace/tools && npm init -y && PUPPETEER_SKIP_DOWNLOAD=1 npm i @mermaid-js/mermaid-cli
cd /workspace/ventri && python3 docs/build/build.py --node-modules /workspace/tools/node_modules [--only sandbox architecture]
```

Chrome 路径在 `puppeteer.json` 与 `--chrome` 参数中配置（默认 `/usr/bin/google-chrome`）。
