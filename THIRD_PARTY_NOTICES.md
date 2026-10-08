# Third-party notices

Ventri includes code adapted from the following third-party projects.

## Hermes Agent

- Project: Hermes Agent — https://github.com/NousResearch/hermes-agent
- Adapted from commit `a28a5d03a9fa60418db5f44f3436fa2aa029c8f2`
- Used in: `packages/ventri-agent/src/ventri_agent/tools/_hermes_fs/` (file-tool core: fuzzy
  find-and-replace, read/write/patch operations, search, read-before-write state, write guards),
  the file-tool behaviour of `packages/ventri-agent/src/ventri_agent/tools/fs.py` and `notes.py`,
  and the ported tests `tests/agent/test_hermes_fuzzy_match.py` and `tests/agent/test_fs_hermes.py`;
  `packages/ventri-agent/src/ventri_agent/tools/_url_safety.py` (SSRF / connect-time DNS-rebinding guard
  and credential-in-URL checks, from `tools/url_safety.py` and the secret prefixes of `agent/redact.py`);
  `packages/ventri-agent/src/ventri_agent/threat_patterns.py` (memory-write injection scan, from
  `tools/threat_patterns.py`); the ANSI stripping and head/tail split in
  `packages/ventri-agent/src/ventri_agent/tools/output.py` (from `tools/ansi_strip.py` and
  `tools/tool_output_truncate.py`); `packages/ventri-agent/src/ventri_agent/channels/feishu/_hermes.py`
  (Feishu inbound text/post parsing and @-mention handling, fence-aware Markdown segmentation, reply
  fallback codes, and the thread-local loop proxy / receive-loop guard for the `lark-oapi` websocket
  client, from `plugins/platforms/feishu/adapter.py`); and
  `packages/ventri-agent/src/ventri_agent/channels/feishu/_onboard.py` (the Feishu / Lark scan-to-create
  app registration: device-code init/begin/poll, Lark domain switch and bot probe, from the QR onboarding
  section of `plugins/platforms/feishu/adapter.py`).
  Each adapted file carries an attribution header naming its Hermes source files.
- License: MIT

```
MIT License

Copyright (c) 2025 Nous Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## lark-oapi (Feishu / Lark Open Platform SDK for Python)

- Project: oapi-sdk-python — https://github.com/larksuite/oapi-sdk-python
- Adapted from version 1.7.3 (`lark_oapi/scene/registration/`)
- Used in: `packages/ventri-agent/src/ventri_agent/channels/feishu/_onboard.py` (protocol details of the
  app registration flow: `slow_down` back-off, the `expires_in` spelling, the `from` / `tp` / `source`
  QR-code parameters and the gzip + URL-safe base64 `addons` encoding). The package itself is an
  optional runtime dependency (`ventri-agent[feishu]`), not vendored.
- License: MIT

```
MIT License

Copyright (c) 2023 Lark Technologies Pte. Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
