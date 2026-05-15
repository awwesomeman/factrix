"""bench.wrapper — measure, status paths, JSONL round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.preflight import collect_env
from bench.validator import validate_file
from bench.wrapper import measure, read_records, write_records


def _scale() -> dict:
    return {"n_factors": 2, "n_dates": 10, "n_assets": 5}


def test_measure_ok_path_populates_timings():
    env = collect_env()
    rec = measure(
        setup=lambda: [1, 2, 3],
        compute=lambda x: sum(x),
        scenario_id="t",
        axis_cell="continuous_individual_panel",
        scale=_scale(),
        metric_set="core",
        env=env,
    )
    assert rec.status == "ok"
    assert rec.setup_s is not None and rec.setup_s >= 0
    assert rec.compute_s is not None and rec.compute_s >= 0
    assert rec.wall_s is not None
    assert rec.peak_rss_mb is not None
    assert rec.peak_alloc_mb is not None
    assert rec.error_message is None


def test_measure_error_path_records_exception():
    env = collect_env()

    def boom(_):
        raise RuntimeError("kaboom")

    rec = measure(
        setup=lambda: None,
        compute=boom,
        scenario_id="t",
        axis_cell="continuous_individual_panel",
        scale=_scale(),
        metric_set="core",
        env=env,
    )
    assert rec.status == "error"
    assert "kaboom" in (rec.error_message or "")
    assert rec.wall_s is not None  # wall still recorded


def test_warmup_record_is_preserved(tmp_path: Path):
    env = collect_env()
    warm = measure(
        setup=lambda: 0,
        compute=lambda _: 0,
        scenario_id="t",
        axis_cell="continuous_individual_panel",
        scale=_scale(),
        metric_set="core",
        is_warmup=True,
        env=env,
    )
    main = measure(
        setup=lambda: 0,
        compute=lambda _: 0,
        scenario_id="t",
        axis_cell="continuous_individual_panel",
        scale=_scale(),
        metric_set="core",
        is_warmup=False,
        env=env,
    )
    out = tmp_path / "run.jsonl"
    write_records(out, [warm, main])

    rows = read_records(out)
    assert len(rows) == 2
    assert rows[0]["is_warmup"] is True
    assert rows[1]["is_warmup"] is False

    report = validate_file(out)
    assert report.ok, report.failures


def test_scale_axis_cell_mismatch_fails_loudly():
    env = collect_env()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        measure(
            setup=lambda: 0,
            compute=lambda _: 0,
            scenario_id="t",
            axis_cell="sparse_individual_panel",
            scale=_scale(),  # continuous shape
            metric_set="core",
            env=env,
        )
