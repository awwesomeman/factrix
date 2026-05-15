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
    m_ic,
    m_ic_bootstrap,
    p1_scaling_probe,
    s1_evaluate,
    s2_screen_50,
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


def test_micros_label_by_single_metric_not_bundle(tmp_path: Path):
    """Micros use the metric name as `metric_set` so downstream
    aggregation can distinguish single-metric attribution from
    whole-bundle runs (S2/S3 emit metric_set='core'). M-ic-boot
    labels the bootstrap path explicitly."""
    out = tmp_path / "m.jsonl"
    records = m_ic_bootstrap(out, preset="tiny")
    assert all(r.metric_set == "ic_bootstrap" for r in records)


def test_micro_vs_bundle_metric_set_labels_are_disjoint(tmp_path: Path):
    """A micro (M-ic) and a bundle run (S2) on the same panel must
    carry different `metric_set` labels — otherwise a downstream
    aggregator that sums by metric_set will double-count M-ic
    (subset of S2's per-factor work) into the bundle total."""
    s2_out = tmp_path / "s2.jsonl"
    m_out = tmp_path / "m_ic.jsonl"
    s2_records = s2_screen_50(s2_out, preset="tiny")
    m_records = m_ic(m_out, preset="tiny")
    s2_labels = {r.metric_set for r in s2_records}
    m_labels = {r.metric_set for r in m_records}
    assert s2_labels == {"core"}
    assert m_labels == {"ic"}
    assert not (s2_labels & m_labels)


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
