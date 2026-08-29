"""Docs regression checks for the allocation decision-boundary table.

The allocation guide centralizes which evidence may promote a candidate and
which evidence may not. Pin the row labels, the per-row metric roles, the
outbound links, and the five boundary statements by literal substring so the
table cannot drift away from the pages that own those contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GUIDE = "docs/guides/validating-allocation-signals.md"

ROW_LABELS = (
    "Cross-sectional asset ranking",
    "Single-asset time-series predictability",
    "Regime robustness",
    "Candidate redundancy",
    "Broad adaptive search / winner selection",
)

LINK_TARGETS = (
    "../api/metrics/ic.md",
    "../api/metrics/fm_beta.md",
    "../api/metrics/predictive_beta.md",
    "../api/metrics/spanning.md",
    "../api/by-slice.md",
    "../api/slice-test.md",
    "../api/multi-factor.md",
    "../reference/warning-codes.md",
    "../reference/statistical-methods.md#1-hac-se-under-overlapping-returns",
    "../reference/statistical-methods.md#2-multiple-testing-under-dependence",
    "../reference/statistical-methods.md"
    "#4-persistence-diagnostics-under-near-unit-root-predictors",
    "../reference/statistical-methods.md"
    "#joint-period-test-on-short-slices-known-over-rejection",
    "../reference/statistical-methods.md"
    "#persistent-per-period-series-no-hac-or-bootstrap-path-is-calibrated",
    "../api/slice-test.md#warning-contract-slice_period_joint_test",
)

BOUNDARY_STATEMENTS = (
    "`expected_warnings` affects presentation and the audit fields only",
    "A regime joint test on short slices with `K >= 3` carries a",
    "`spanning_alpha` is a fixed-base incremental-information / redundancy",
    "BHY controls the FDR of the declared hypothesis family, not the winner",
    "An uneven evaluation grid is disclosed by `uneven_evaluation_grid`",
)


def _guide_text() -> str:
    return Path(GUIDE).read_text(encoding="utf-8")


def _boundary_row(label: str) -> list[str]:
    """Return the decision-boundary cells of the row starting with ``label``."""
    prefix = f"| {label} |"
    rows = [line for line in _guide_text().splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, f"expected exactly one boundary row for {label!r}"
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 4, f"boundary row for {label!r} must have four columns"
    return cells


def test_decision_boundary_table_header_and_intro() -> None:
    text = _guide_text()
    assert "## Separate promotion evidence from diagnostics" in text
    assert (
        "| Research question / stage | Primary evidence | "
        "Supplementary or robustness evidence | Do not infer |" in text
    )


@pytest.mark.parametrize("label", ROW_LABELS)
def test_decision_boundary_rows_present(label: str) -> None:
    assert len(_boundary_row(label)) == 4


def test_predictive_beta_is_the_single_asset_primary_evidence() -> None:
    single_asset = _boundary_row("Single-asset time-series predictability")
    cross_section = _boundary_row("Cross-sectional asset ranking")
    assert "`predictive_beta`" in single_asset[1]
    assert "predictive_beta" not in cross_section[1]
    assert "cross-sectional ranking evidence" in single_asset[3]


def test_cross_sectional_ranking_keeps_ic_and_fm_as_primary_evidence() -> None:
    cells = _boundary_row("Cross-sectional asset ranking")
    assert "`ic`" in cells[1]
    assert "`fm_beta`" in cells[1]
    assert "declared hypothesis family" in cells[1]
    for diagnostic in ("`k_spread`", "`monotonicity`", "`directional_pair_accuracy`"):
        assert diagnostic in cells[2]
    assert "one diagnostic passing promotes the candidate" in cells[3]


def test_regime_robustness_row_keeps_the_joint_test_out_of_the_gate() -> None:
    cells = _boundary_row("Regime robustness")
    assert "`by_slice`" in cells[1]
    assert "`slice_period_pairwise_test`" in cells[1]
    assert "slice_period_joint_test" not in cells[1]
    assert "`slice_period_joint_test`" in cells[2]
    assert "short-slice joint p is an admission gate" in cells[3]


def test_redundancy_row_keeps_spanning_fixed_base_and_greedy_out_of_inference() -> None:
    cells = _boundary_row("Candidate redundancy")
    assert "fixed-base [`spanning_alpha`]" in cells[1]
    assert "`greedy_forward_selection` t-stats are post-selection inference" in cells[3]


def test_search_wide_row_keeps_bhy_out_of_the_primary_column() -> None:
    cells = _boundary_row("Broad adaptive search / winner selection")
    assert "Held-out evaluation" in cells[1]
    assert "external search-wide procedure" in cells[1]
    assert "BHY" not in cells[1]
    assert "BHY" in cells[2]
    assert "BHY has controlled the whole research search" in cells[3]


@pytest.mark.parametrize("target", LINK_TARGETS)
def test_decision_boundary_links_are_present(target: str) -> None:
    assert f"]({target})" in _guide_text()


@pytest.mark.parametrize("statement", BOUNDARY_STATEMENTS)
def test_boundary_statements_are_present(statement: str) -> None:
    assert statement in _guide_text()


def test_predictive_beta_stays_estimand_neutral() -> None:
    text = _guide_text()
    assert "The metric follows the research question" in text
    assert (
        "`predictive_beta` is the primary evidence for single-asset\n"
        "time-series predictability and is not cross-sectional ranking evidence"
    ) in text
    assert "Hansen SPA / White Reality Check" in text
