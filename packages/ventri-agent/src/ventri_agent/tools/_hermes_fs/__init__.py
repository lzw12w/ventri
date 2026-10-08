"""Local-filesystem file tools adapted from Hermes Agent.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. See THIRD_PARTY_NOTICES.md at the
repository root for the full license text. Ported from upstream commit
``a28a5d03a9fa60418db5f44f3436fa2aa029c8f2``.

What came from where (upstream paths are relative to the Hermes repository):

* ``fuzzy_match.py``  <- ``tools/fuzzy_match.py`` (strategy chain, escape-drift
  guards, multi-match locations, already-applied detection, "did you mean").
* ``common.py``       <- ``tools/file_operations_common.py``,
  ``tools/binary_extensions.py``, ``tools/tool_output_limits.py``,
  ``agent/search_policy.py``.
* ``ops.py``          <- ``tools/file_operations.py`` (the *native* local read
  path, ``write_file``, ``patch_replace``), ``tools/file_operations_lint.py``
  (in-process linters, lint delta) and the read/write parts of
  ``tools/file_tools.py``.
* ``search.py``       <- ``tools/file_operations_search.py`` (rg invocation and
  output parsing, zero-match probes, macOS TCC pruning; multi-path recovery
  is in ``fs.search_paths``).
* ``state.py``        <- ``tools/file_state.py`` and
  ``tools/file_tools_read_tracking.py`` (read stamps, full-content baselines,
  paged-read coverage, blind patches, per-path locks).
* ``guards.py``       <- ``tools/file_tools_write_guards.py`` and the device /
  special-file guards of ``tools/file_tools.py``.

Hermes runs every operation through a terminal backend (local, docker, ssh,
...); this port keeps only the local, in-process code paths and drops the
backend abstraction, LSP, document extraction, secret redaction, the
approval model and V4A patches (Ventri keeps its own permission engine and
path confinement -- see ``ventri_agent.tools.fs``).

Deliberate Ventri changes to ported behaviour: invalid-UTF-8 files that are
mostly text are readable (undecodable bytes shown as U+FFFD, counted, and the
view never counts as a full read for overwriting) instead of being refused as
binary; CRLF files display as LF; an edit whose (fuzzy) match already equals
``new_string`` is a no-op instead of a write; content search fetches one extra
row so truncation is detected; the Python search fallback replaces Hermes's
grep/find fallbacks; the read character budget is 30K (Ventri artifacts
results above ~32K chars).
"""
