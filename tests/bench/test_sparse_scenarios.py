"""End-to-end smoke for the Sparse × Individual scenarios."""

from __future__ import annotations

from pathlib import Path

from bench.scenarios.sparse import SCENARIOS, m_corrado, s5_event_study
from bench.validator import validate_file


def test_every_sparse_scenario_runs_and_validates(tmp_path: Path):
    for sid, fn in SCENARIOS.items():
        out = tmp_path / f"{sid}.jsonl"
        records = fn(out, preset="tiny")
        rep = validate_file(out)
        assert rep.ok, (sid, rep.failures)
        assert records, sid
        assert all(r.scenario_id == sid for r in records)
        assert all(r.axis_cell == "sparse_individual_panel" for r in records)


def test_s5_uses_event_bundle_label(tmp_path: Path):
    out = tmp_path / "S5.jsonl"
    records = s5_event_study(out, preset="tiny")
    assert all(r.metric_set == "event" for r in records)


def test_m_corrado_labels_by_single_metric(tmp_path: Path):
    """Like Continuous-cell micros, M-corrado labels by metric name —
    not by the `event` bundle — so downstream aggregation can
    distinguish single-metric attribution from whole-bundle runs."""
    out = tmp_path / "M.jsonl"
    records = m_corrado(out, preset="tiny")
    assert all(r.metric_set == "corrado_rank_test" for r in records)


def test_scale_records_realised_event_count(tmp_path: Path):
    """`n_events` reflects the realised count from the seeded panel,
    not a configured value — `make_event_panel` produces Binomial
    events whose count depends on (n_dates × n_assets × event_rate)
    and the seed."""
    out = tmp_path / "S5.jsonl"
    records = s5_event_study(out, preset="tiny", seed=0)
    assert all(r.scale.n_events > 0 for r in records)
    # Determinism: same seed → same n_events.
    out2 = tmp_path / "S5b.jsonl"
    records2 = s5_event_study(out2, preset="tiny", seed=0)
    assert records[0].scale.n_events == records2[0].scale.n_events
