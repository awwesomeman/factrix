"""Validates ``factrix/llms-full.txt`` and ``factrix/llms.txt`` against the
live public API surface.

**Approach: snapshot, not generation.** ``llms-full.txt`` is heavily
curated narrative + tables + worked examples. Generating it from
docstrings would lose editorial control over what an LLM agent sees.
Instead, this test parses the existing file for symbol references and
verifies that every reference still resolves and that every public
``__all__`` symbol gets at least one mention.

Three checks:

1. Every ``factrix.X.Y`` / ``fl.X.Y`` attribute chain in ``llms-full.txt``
   walks from the ``factrix`` package without raising ``AttributeError``.
2. Every ``from factrix... import NAME`` statement imports a name that
   actually exists at that module path.
3. Every name in ``factrix.__all__`` appears at least once in
   ``llms-full.txt`` — keeps the LLM reference in lockstep when the
   public surface widens.

Bare references like ``StatCode.IC_MEAN`` (no ``fl.`` / ``factrix.``
prefix) are intentionally not validated — too many false positives
against ``profile.X`` style attribute talk in prose.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import factrix

LLMS_FULL = pathlib.Path("factrix/llms-full.txt")
LLMS_INDEX = pathlib.Path("factrix/llms.txt")

# Negative lookbehind excludes URL paths (`github.com/awwesomeman/factrix`)
# and dotted continuations from a non-factrix root.
_REF_RE = re.compile(
    r"(?<![/.:])\b(?:factrix|fl)\."
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
)
_IMPORT_RE = re.compile(
    r"^from\s+(factrix(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$",
    re.MULTILINE,
)
# `import factrix` / `import factrix.metrics [as fl]` — bare-import form.
_BARE_IMPORT_RE = re.compile(
    r"^import\s+(factrix(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+as\s+\w+)?\s*$",
    re.MULTILINE,
)


def _referenced_chains(text: str) -> set[tuple[str, ...]]:
    return {tuple(m.group(1).split(".")) for m in _REF_RE.finditer(text)}


def _imports(text: str) -> list[tuple[str, str | None]]:
    """Return ``[(module_path, imported_name_or_None), ...]``.

    ``None`` for bare ``import factrix.X`` forms (only the module needs
    to resolve; no attribute to check).
    """
    out: list[tuple[str, str | None]] = []
    for m in _IMPORT_RE.finditer(text):
        module = m.group(1)
        # Strip trailing comments before splitting on commas.
        names_str = m.group(2).split("#", 1)[0]
        for raw in names_str.split(","):
            name = raw.strip().split(" as ")[0].strip()
            if name:
                out.append((module, name))
    for m in _BARE_IMPORT_RE.finditer(text):
        out.append((m.group(1), None))
    return out


def _resolves(chain: tuple[str, ...]) -> bool:
    """Walk attribute chain from ``factrix``, falling back to submodule
    import for lazy-loaded subpackages (e.g. ``factrix.metrics``)."""
    obj: object = factrix
    walked: list[str] = []
    for part in chain:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:
                obj = importlib.import_module("factrix." + ".".join([*walked, part]))
            except ImportError:
                return False
        walked.append(part)
    return True


def test_every_referenced_symbol_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures = sorted(
        ".".join(chain) for chain in _referenced_chains(text) if not _resolves(chain)
    )
    assert not failures, (
        "Unresolvable factrix.* references in llms-full.txt:\n  "
        + "\n  ".join(f"factrix.{f}" for f in failures)
    )


def test_every_imported_name_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures: list[str] = []
    for module_path, name in _imports(text):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            failures.append(f"{module_path} (module not importable)")
            continue
        if name is not None and not hasattr(module, name):
            failures.append(f"{module_path}.{name}")
    assert not failures, (
        "Imports in llms-full.txt that do not resolve:\n  " + "\n  ".join(failures)
    )


def test_every_public_symbol_mentioned_in_llms_full() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    # Word-boundary match: prevents `Metric` from being silently satisfied
    # by `MetricOutput` (and similarly for other short prefix names).
    missing = [
        name
        for name in sorted(factrix.__all__)
        if not re.search(rf"\b{re.escape(name)}\b", text)
    ]
    assert not missing, (
        "Public symbols in factrix.__all__ never mentioned in llms-full.txt:\n  "
        + "\n  ".join(missing)
        + "\nAdd at least one mention so LLM agents do not miss them."
    )


def test_llms_index_exists_and_nonempty() -> None:
    text = LLMS_INDEX.read_text(encoding="utf-8")
    assert text.strip(), f"{LLMS_INDEX} is empty"
    assert "factrix" in text.lower()
