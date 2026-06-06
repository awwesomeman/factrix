"""Ratchet on legacy "the metric `evaluate()` runs" tier-name synonyms.

The project terminology settled the term to a behavioral phrase and listed the older
synonyms as legacy in ``docs/reference/glossary.md``. The synonyms are
not big-bang renamed; instead this ratchet asserts that the count can
only decrease over time, so any PR that touches a file containing one
must replace the term in that diff.

When the floor reaches a small number (~5), open a follow-up issue for
a single sweep PR that removes the remainder and deletes the legacy-
synonyms note in the glossary.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_LEGACY_PATTERN = re.compile(r"procedure-canonical|cell-canonical|canonical metric")

# Files / trees the ratchet ignores.
#
# - ``docs/plans/archive/`` is frozen historical drafts.
# - ``docs/reference/glossary.md`` defines the legacy synonyms;
#   it must mention them.
# - This test file itself names the pattern.
_EXCLUDE_PATHS = (
    _REPO_ROOT / "docs" / "plans" / "archive",
    _REPO_ROOT / "docs" / "reference" / "glossary.md",
    Path(__file__).resolve(),
)

_SCAN_DIRS = (
    _REPO_ROOT / "docs",
    _REPO_ROOT / "factrix",
)

_SCAN_SUFFIXES = {".md", ".py", ".txt"}

# Frozen baseline for terminology. PRs may only decrease.
_BASELINE = 17


def _legacy_hits() -> int:
    total = 0
    for root in _SCAN_DIRS:
        for path in root.rglob("*"):
            if path.suffix not in _SCAN_SUFFIXES:
                continue
            if any(
                path == ex or (ex.is_dir() and ex in path.parents)
                for ex in _EXCLUDE_PATHS
            ):
                continue
            text = path.read_text(encoding="utf-8")
            total += len(_LEGACY_PATTERN.findall(text))
    return total


def test_legacy_terminology_ratchet() -> None:
    hits = _legacy_hits()
    assert hits <= _BASELINE, (
        f"legacy 'evaluate()-metric' synonyms increased from {_BASELINE} "
        f"to {hits}. Replace the new occurrences with the behavioral phrase "
        f"per docs/reference/glossary.md § the metric `evaluate()` runs."
    )
    if hits < _BASELINE:
        # Soft note (not a failure) so the next PR author sees the floor
        # to decrement in tests/test_docs_terminology_ratchet.py.
        print(
            f"\nLegacy 'evaluate()-metric' synonyms: {hits} (baseline {_BASELINE}). "
            f"Decrement _BASELINE to {hits} in this PR."
        )
