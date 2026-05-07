"""Coverage test: generated registry-cell table tracks the live registry.

Asserts that ``docs/development/_generated_registry_cells.md`` exists,
contains one data row per ``_DISPATCH_REGISTRY`` entry, and lists every
registered procedure class. Skipped if the file is absent (only
meaningful after a build).
"""

from __future__ import annotations

import pathlib

import pytest
from factrix._registry import _DISPATCH_REGISTRY

GENERATED_TABLE = pathlib.Path("docs/development/_generated_registry_cells.md")


def _data_rows(text: str) -> list[str]:
    return [
        ln
        for ln in text.splitlines()
        if ln.strip().startswith("|") and not ln.strip().startswith("|---")
    ]


def test_generated_table_exists_and_nonempty() -> None:
    if not GENERATED_TABLE.exists():
        pytest.skip(
            f"{GENERATED_TABLE} not found — run "
            "'python scripts/mkdocs_hooks/gen_registry_cells.py' or 'mkdocs build' first."
        )
    rows = _data_rows(GENERATED_TABLE.read_text(encoding="utf-8"))
    # header + at least one data row
    assert len(rows) >= 2, f"{GENERATED_TABLE} has no data rows"


def test_generated_table_row_count_matches_registry() -> None:
    if not GENERATED_TABLE.exists():
        pytest.skip(f"{GENERATED_TABLE} not found")
    rows = _data_rows(GENERATED_TABLE.read_text(encoding="utf-8"))
    # rows includes the header row
    assert len(rows) - 1 == len(_DISPATCH_REGISTRY), (
        f"{GENERATED_TABLE} has {len(rows) - 1} data row(s) but registry has "
        f"{len(_DISPATCH_REGISTRY)} entries — regenerate via 'mkdocs build'."
    )


def test_every_procedure_class_appears_in_table() -> None:
    if not GENERATED_TABLE.exists():
        pytest.skip(f"{GENERATED_TABLE} not found")
    text = GENERATED_TABLE.read_text(encoding="utf-8")
    for entry in _DISPATCH_REGISTRY.values():
        cls = type(entry.procedure).__name__
        assert f"`{cls}`" in text, (
            f"procedure class {cls!r} missing from {GENERATED_TABLE}"
        )
