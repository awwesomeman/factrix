"""End-to-end smoke for the greedy forward selection scenario."""

from __future__ import annotations

from pathlib import Path

from bench.scenarios.algo import SCENARIOS, s4_greedy_forward_selection
from bench.validator import validate_file


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
    assert all(r.status == "ok" for r in records)


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


def test_seed_determinism(tmp_path: Path):
    """Same seed must produce identical scale + label fields across runs."""
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    a = s4_greedy_forward_selection(out_a, preset="tiny", seed=42)
    b = s4_greedy_forward_selection(out_b, preset="tiny", seed=42)
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra.scenario_id == rb.scenario_id
        assert ra.scale == rb.scale
        assert ra.metric_set == rb.metric_set
        assert ra.axis_cell == rb.axis_cell
