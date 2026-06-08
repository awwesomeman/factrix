"""Shared helpers for doc-validation tests.

Used by ``test_docs_llms.py`` (validates ``factrix/llms-full.txt``) and
``test_docs_pages.py`` (walks all ``docs/**/*.md``). Pure regex + attribute
walking; no fixtures, intentionally not in ``conftest.py`` so it stays
out of the pytest collection path.

The leading underscore in the filename keeps pytest from collecting it
as a test module.
"""

from __future__ import annotations

import importlib
import re

import factrix

# Negative lookbehind excludes URL paths (`github.com/awwesomeman/factrix`)
# and dotted continuations from a non-factrix root.
REF_RE = re.compile(
    r"(?<![/.:])\b(?:factrix|fx)\."
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
)
FROM_IMPORT_RE = re.compile(
    r"^from\s+(factrix(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$",
    re.MULTILINE,
)
# `import factrix` / `import factrix.metrics [as fx]` — bare-import form.
BARE_IMPORT_RE = re.compile(
    r"^import\s+(factrix(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+as\s+\w+)?\s*$",
    re.MULTILINE,
)


def referenced_chains(text: str) -> set[tuple[str, ...]]:
    """Return the set of `factrix.X.Y...` attribute chains in ``text``."""
    return {tuple(m.group(1).split(".")) for m in REF_RE.finditer(text)}


def imports(text: str) -> list[tuple[str, str | None]]:
    """Return ``[(module_path, imported_name_or_None), ...]``.

    ``None`` for bare ``import factrix.X`` forms (only the module needs
    to resolve; no attribute to check).
    """
    out: list[tuple[str, str | None]] = []
    for m in FROM_IMPORT_RE.finditer(text):
        module = m.group(1)
        # Strip trailing comments before splitting on commas.
        names_str = m.group(2).split("#", 1)[0]
        for raw in names_str.split(","):
            name = raw.strip().split(" as ")[0].strip()
            if name:
                out.append((module, name))
    for m in BARE_IMPORT_RE.finditer(text):
        out.append((m.group(1), None))
    return out


def resolves(chain: tuple[str, ...]) -> bool:
    """Resolve ``factrix.<chain>`` against the live package.

    Tries the longest prefix of ``chain`` as a module path first, then
    walks the remaining parts as attributes. Going longest-first matters
    because ``factrix.metrics.__init__`` re-exports symbols like
    ``caar`` (the function) that shadow the same-named submodule —
    naive left-to-right ``getattr`` walks land on the function and then
    fail to look up ``bmp_test`` on it. Mkdocstrings cross-refs like
    ``factrix.metrics.caar.bmp_test`` mean the module path, not the
    re-exported function.
    """
    for k in range(len(chain), 0, -1):
        try:
            obj: object = importlib.import_module("factrix." + ".".join(chain[:k]))
        except ImportError:
            continue
        ok = True
        for member in chain[k:]:
            try:
                obj = getattr(obj, member)
            except AttributeError:
                ok = False
                break
        if ok:
            return True
    # Fallback: chain may live entirely on the top-level ``factrix``
    # namespace (e.g. ``factrix.metrics.ic``).
    obj = factrix
    for part in chain:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False
    return True


def import_resolves(module_path: str, name: str | None) -> bool:
    """Verify a ``from <module_path> import <name>`` statement resolves."""
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return False
    if name is None:
        return True
    return hasattr(module, name)
