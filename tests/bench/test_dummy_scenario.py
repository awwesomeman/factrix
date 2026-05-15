"""End-to-end smoke for the dummy scenario."""

from __future__ import annotations

from pathlib import Path

from bench.scenarios.dummy import run
from bench.validator import validate_file


def test_dummy_scenario_round_trip(tmp_path: Path):
    out = tmp_path / "dummy.jsonl"
    records = run(output=out, n_factors=3, n_assets=10, n_dates=30)
    assert len(records) == 2  # warmup + measured
    rep = validate_file(out)
    assert rep.ok, rep.failures
    assert rep.n_rows == 2
