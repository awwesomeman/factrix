"""End-to-end smoke for every Cont × Ind scenario at `tiny` scale.

Each scenario runs, writes JSONL, self-validates; the test verifies
record count, scenario_id labelling, and validator pass. Wall-time
budget on a CI runner: under 30 s for the whole module.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from bench.scenarios.continuous import (
    SCENARIOS,
    m_ic_bootstrap,
    p1_scaling_probe,
    s1_evaluate,
)
from bench.validator import validate_file


@pytest.fixture(autouse=True)
def _silence_sample_floor_warnings():
    # Tiny scale trips factrix's "median assets per group" warning;
    # the bench harness intentionally runs under-spec sizes for CI.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def test_every_scenario_runs_and_validates(tmp_path: Path):
    for sid, fn in SCENARIOS.items():
        out = tmp_path / f"{sid}.jsonl"
        records = fn(out, preset="tiny")
        rep = validate_file(out)
        assert rep.ok, (sid, rep.failures)
        assert records, sid
        assert all(r.scenario_id == sid for r in records)


def test_s1_emits_warmup_and_measured(tmp_path: Path):
    out = tmp_path / "s1.jsonl"
    records = s1_evaluate(out, preset="tiny")
    assert len(records) == 2
    assert records[0].is_warmup is True
    assert records[1].is_warmup is False


def test_p1_emits_one_record_pair_per_scale_step(tmp_path: Path):
    out = tmp_path / "p1.jsonl"
    records = p1_scaling_probe(out, preset="tiny")
    # 3 steps × (warmup + measured) = 6
    assert len(records) == 6
    n_factors_seen = {r.scale.n_factors for r in records}
    assert len(n_factors_seen) == 3, sorted(n_factors_seen)


def test_m_ic_boot_uses_heavy_set_label(tmp_path: Path):
    out = tmp_path / "m.jsonl"
    records = m_ic_bootstrap(out, preset="tiny")
    assert all(r.metric_set == "heavy" for r in records)


def test_seed_determinism_across_runs(tmp_path: Path):
    """Two runs with the same seed produce identical scale + label
    fields (timings differ; we don't compare those)."""
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    records_a = s1_evaluate(out_a, preset="tiny", seed=42)
    records_b = s1_evaluate(out_b, preset="tiny", seed=42)
    assert len(records_a) == len(records_b)
    for ra, rb in zip(records_a, records_b, strict=True):
        assert ra.scenario_id == rb.scenario_id
        assert ra.scale == rb.scale
        assert ra.metric_set == rb.metric_set
        assert ra.axis_cell == rb.axis_cell
