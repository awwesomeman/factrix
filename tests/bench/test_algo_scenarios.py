"""End-to-end smoke for the greedy forward selection scenario."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from bench.scenarios.algo import SCENARIOS, s4_greedy_forward_selection
from bench.validator import validate_file


@pytest.fixture(autouse=True)
def _silence_sample_floor_warnings():
    # Tiny scale trips factrix's sample-floor warnings; the algo
    # itself also emits a snooping warning that the scenario already
    # suppresses inside `compute`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def test_s4_runs_and_validates(tmp_path: Path):
    out = tmp_path / "S4.jsonl"
    records = s4_greedy_forward_selection(out, preset="tiny")
    rep = validate_file(out)
    assert rep.ok, rep.failures
    assert len(records) == 2
    assert records[0].is_warmup is True
    assert records[1].is_warmup is False
    assert all(r.scenario_id == "S4" for r in records)
    assert all(r.metric_set == "algo" for r in records)


def test_s4_setup_compute_split(tmp_path: Path):
    """Spread series construction is `setup`, greedy loop is `compute` —
    both phases must report non-zero timings."""
    out = tmp_path / "S4.jsonl"
    records = s4_greedy_forward_selection(out, preset="tiny")
    measured = records[1]
    assert measured.setup_s is not None and measured.setup_s > 0
    assert measured.compute_s is not None and measured.compute_s > 0


def test_scenarios_dict_exports_s4():
    assert set(SCENARIOS) == {"S4"}
