"""Validates ``factrix/llms-full.txt`` and ``factrix/llms.txt`` against the
live public API surface.

**Approach: snapshot, not generation.** ``llms-full.txt`` is heavily
curated narrative + tables + worked examples. Generating it from
docstrings would lose editorial control over what an LLM agent sees.
Instead, this test parses the existing file for symbol references and
verifies that every reference still resolves and that every public
``__all__`` symbol gets at least one mention.

Three checks:

1. Every ``factrix.X.Y`` / ``fx.X.Y`` attribute chain in ``llms-full.txt``
   walks from the ``factrix`` package without raising ``AttributeError``.
2. Every ``from factrix... import NAME`` statement imports a name that
   actually exists at that module path.
3. Every name in ``factrix.__all__`` appears at least once in
   ``llms-full.txt`` — keeps the LLM reference in lockstep when the
   public surface widens.

Bare references like ``WarningCode.FEW_EVENTS`` (no ``fx.`` /
``factrix.`` prefix) are intentionally not validated — too many false
positives against ``profile.X`` style attribute talk in prose.

Sibling test ``test_docs_pages.py`` runs the resolution checks
against every ``docs/**/*.md`` page; this file remains specific to
the curated llms snapshot (because of the ``__all__``-coverage check).
"""

from __future__ import annotations

import pathlib
import re

import factrix
from factrix._codes import WarningCode

from tests._doc_validation import (
    import_resolves,
    imports,
    referenced_chains,
    resolves,
)

LLMS_FULL = pathlib.Path("factrix/llms-full.txt")
LLMS_INDEX = pathlib.Path("factrix/llms.txt")

# Preprocess / data-shape codes. They are raised outside the evaluation
# path (``factrix.preprocess`` normalizers, ``orthogonalize_factor``,
# ``compute_forward_return``), so the curated llms-full.txt table points
# at the generated reference for them instead of restating their glosses.
_PREPROCESS_CODES: frozenset[str] = frozenset(
    {
        WarningCode.ZERO_MAD_STD_FALLBACK.value,
        WarningCode.SPARSE_WINSORIZE_SKIPPED.value,
        WarningCode.INSUFFICIENT_SCALE_ASSETS.value,
        WarningCode.NON_FINITE_INPUT_DROPPED.value,
        WarningCode.INSUFFICIENT_REGRESSION_DF.value,
        WarningCode.RANK_DEFICIENT_DESIGN.value,
        WarningCode.RAGGED_PERIOD_GRID.value,
        WarningCode.UNEVEN_EVALUATION_GRID.value,
    }
)


def test_every_referenced_symbol_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures = sorted(
        ".".join(chain) for chain in referenced_chains(text) if not resolves(chain)
    )
    assert not failures, (
        "Unresolvable factrix.* references in llms-full.txt:\n  "
        + "\n  ".join(f"factrix.{f}" for f in failures)
    )


def test_every_imported_name_resolves() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    failures = [
        f"{module}.{name}" if name else f"{module} (module not importable)"
        for module, name in imports(text)
        if not import_resolves(module, name)
    ]
    assert not failures, (
        "Imports in llms-full.txt that do not resolve:\n  " + "\n  ".join(failures)
    )


def test_every_public_symbol_mentioned_in_llms_full() -> None:
    text = LLMS_FULL.read_text(encoding="utf-8")
    # Word-boundary match: prevents `Metric` from being silently satisfied
    # by `MetricResult` (and similarly for other short prefix names).
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


def test_warning_code_table_covers_every_evaluation_side_code() -> None:
    """The curated ``WarningCode`` table must cover every evaluation-side code.

    The table is hand-written (the generated
    ``docs/reference/_generated_warning_codes.md`` carries the full enum
    with its canonical glosses), and its intro states a count. Both the
    count and the membership are pinned here: a new evaluation-side code
    has to land in the table, and a new preprocess / data-shape code has
    to be declared in :data:`_PREPROCESS_CODES`.
    """
    text = LLMS_FULL.read_text(encoding="utf-8")
    section = text.split("## WarningCode reference", 1)[1]
    listed = re.findall(r"^\| `([a-z0-9_]+)` \|", section, flags=re.M)

    all_codes = {code.value for code in WarningCode}
    unknown = sorted(set(listed) - all_codes)
    assert not unknown, (
        "llms-full.txt WarningCode table lists codes that are not in "
        "WarningCode:\n  " + "\n  ".join(unknown)
    )
    missing = sorted(all_codes - set(listed) - _PREPROCESS_CODES)
    assert not missing, (
        "Evaluation-side WarningCode values missing from the llms-full.txt "
        "table:\n  " + "\n  ".join(missing)
    )
    claimed = re.search(r"The (\d+) evaluation-side codes", section)
    assert claimed is not None, (
        "llms-full.txt no longer states the evaluation-side code count; "
        "restore it or drop this assertion."
    )
    assert int(claimed.group(1)) == len(listed), (
        f"llms-full.txt claims {claimed.group(1)} evaluation-side codes but "
        f"its table has {len(listed)} rows."
    )


def test_llms_index_exists_and_nonempty() -> None:
    text = LLMS_INDEX.read_text(encoding="utf-8")
    assert text.strip(), f"{LLMS_INDEX} is empty"
    assert "factrix" in text.lower()
