"""Shared helpers for doc-validation tests.

Used by ``test_docs_llms.py`` (validates ``factrix/llms-full.txt``),
``test_docs_pages.py`` (walks all ``docs/**/*.md``), and
``test_docs_bibliography.py`` (walks citations in both docs pages and
``factrix/**/*.py``). Pure regex + attribute walking; no fixtures,
intentionally not in ``conftest.py`` so it stays out of the pytest
collection path.

The leading underscore in the filename keeps pytest from collecting it
as a test module.
"""

from __future__ import annotations

import importlib
import pathlib
import re
from collections.abc import Iterable

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


# Bibliography citations are written ``[Author (Year)][author-year]`` and
# resolve through mkdocs-autorefs to an explicit anchor in
# ``docs/reference/bibliography.md``. Requiring the 4-digit year (with an
# optional ``a``/``b`` suffix disambiguating same-year works) is what makes
# the pattern unambiguous: ordinary subscript chains such as ``values[i][0]``
# or ``params[row][j]`` cannot match it, because no subscript ends in
# ``-1234``. ``test_docs_bibliography.py`` pins the two conventions this
# relies on.
#
# The label half deliberately tolerates newlines: docstrings wrapped at 88
# columns split it across lines in 7 places, putting the author on one line
# and ``(1989)][kunsch-1989]`` on the next, and a newline-free pattern
# silently skips those. Only the slug is captured, so a label spanning
# further than intended cannot corrupt the result, and the length bound keeps
# an unclosed ``[`` in prose from scanning the rest of the file.
CITATION_RE = re.compile(r"\[[^\]]{1,200}\]\[([a-z][a-z0-9-]*-\d{4}[a-z]?)\]")
# Explicit anchor definition, e.g. ``[](){ #politis-romano-1992 }``.
ANCHOR_RE = re.compile(r"\[\]\(\)\{ #([a-z0-9-]+) \}")

DOCS_ROOT = pathlib.Path("docs")
# ``docs/plans/**`` holds fossilised planning artifacts — excluded from the
# published site via ``mkdocs.yml`` ``exclude_docs`` and from every walker here.
DOCS_EXCLUDED_PREFIXES = (DOCS_ROOT / "plans",)

# A ```python fence, possibly indented inside an admonition or tab (up to 8
# spaces); the closing fence must sit at the same indentation. The info string
# after ``python`` is ignored so ``python title="..."`` still matches.
PYTHON_FENCE_RE = re.compile(
    r"^(?P<indent> {0,8})```python[^\n]*\n(?P<body>.*?)^(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)


def docs_page_paths() -> list[pathlib.Path]:
    """Every authored ``docs/**/*.md`` page, sorted, minus the plans archive."""
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(p.is_relative_to(prefix) for prefix in DOCS_EXCLUDED_PREFIXES)
    )


def python_fences(text: str) -> list[tuple[int, str]]:
    """``(start_line, body)`` for each ```python fence in ``text``, in order.

    The body keeps its original indentation (dedent before compiling);
    ``start_line`` is the 1-based line of the opening fence, for messages.
    """
    return [
        (text.count("\n", 0, m.start()) + 1, m.group("body"))
        for m in PYTHON_FENCE_RE.finditer(text)
    ]


def citations(text: str) -> set[str]:
    """Return the bibliography anchor slugs cited in ``text``."""
    return set(CITATION_RE.findall(text))


def anchors(text: str) -> set[str]:
    """Return the explicit anchor slugs defined in ``text``."""
    return set(ANCHOR_RE.findall(text))


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


# Logger names live in the ``factrix.*`` string namespace but are NOT symbol
# paths — ``logging.getLogger("factrix.dag")`` names a node in the logging
# hierarchy, and no ``factrix.dag`` module exists or should. They appear in
# three shapes: a literal ``getLogger`` argument, an f-string one
# (``f"factrix.metric.{name}"``, truncated at the brace below), and a
# ``*_LOGGER_NAME`` constant. Prose in ``_logging.py`` also names them.
#
# These are discovered from the source rather than kept as an allowlist, so a
# newly added logger is exempt automatically instead of failing the symbol
# check until someone edits a list in the tests.
LOGGER_GETLOGGER_RE = re.compile(r'getLogger\(\s*f?"(factrix[^"{]*)')
LOGGER_CONST_RE = re.compile(r'_LOGGER_NAME\s*=\s*"(factrix[^"]*)"')


def logger_namespaces(paths: Iterable[pathlib.Path]) -> set[str]:
    """Return the ``factrix.*`` logging-hierarchy names used in ``paths``.

    A name ending in ``.`` came from an f-string truncated at its first
    placeholder (``factrix.metric.``) and matches by prefix; the rest match
    exactly. Both are stripped of the leading ``factrix.`` so they compare
    against the chains :func:`referenced_chains` produces.
    """
    names: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        names.update(LOGGER_GETLOGGER_RE.findall(text))
        names.update(LOGGER_CONST_RE.findall(text))
    return {n.removeprefix("factrix.") for n in names if n != "factrix."}


def is_logger_name(chain: tuple[str, ...], namespaces: set[str]) -> bool:
    """True when ``chain`` names a logger rather than a symbol.

    A truncated (f-string) namespace matches both its own stem and anything
    beneath it: ``getLogger(f"factrix.metric.{name}")`` yields the namespace
    ``metric.``, while the reference scanner stops at the brace and produces
    the chain ``metric`` — so the stem has to match too.
    """
    dotted = ".".join(chain)
    for ns in namespaces:
        if ns.endswith("."):
            if dotted == ns.rstrip(".") or dotted.startswith(ns):
                return True
        elif dotted == ns:
            return True
    return False


def resolves(chain: tuple[str, ...]) -> bool:
    """Resolve ``factrix.<chain>`` against the live package.

    Tries the longest prefix of ``chain`` as a module path first, then
    walks the remaining parts as attributes. Going longest-first matters
    because ``factrix.metrics.__init__`` re-exports symbols like
    ``caar`` (the function) that shadow the same-named submodule —
    naive left-to-right ``getattr`` walks land on the function and then
    fail to look up ``bmp_z`` on it. Mkdocstrings cross-refs like
    ``factrix.metrics.caar.bmp_z`` mean the module path, not the
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
