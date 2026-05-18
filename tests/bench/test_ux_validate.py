"""``bench.ux_validate`` CLI — happy / red flag / OOM / unknown rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.ux_targets import UX_TARGETS, UX_TARGETS_VERSION
from bench.ux_validate import evaluate_dir, main, render_markdown

_BASE_ROW: dict = {
    "schema_version": "1",
    "metric_set": "core",
    "metric_set_version": "1",
    "axis_cell": "continuous_individual_panel",
    "run_idx": 0,
    "is_warmup": False,
    "cache_state": "warm",
    "error_message": None,
    "started_at": "2026-05-16T00:00:00Z",
    "setup_s": 0.1,
    "compute_s": 0.1,
    "cpu_s": 0.2,
    "peak_rss_mb": 256.0,
    "peak_alloc_mb": 64.0,
    "env": {
        "git_sha": "abc1234",
        "factrix_version": "0.13.0",
        "dataset_spec_version": "1",
        "python": "3.13.0",
        "numpy": "2.0.0",
        "blas": "openblas",
        "omp_threads": 4,
        "cpu_model": "x86",
        "cpu_cores": 8,
        "ram_gb": 32.0,
        "os": "linux-x86_64",
    },
}


def _make_row(
    scenario_id: str,
    *,
    wall_s: float | None,
    status: str = "ok",
    is_warmup: bool = False,
    n_factors: int = 50,
    n_assets: int = 100,
    n_dates: int = 60,
    error_message: str | None = None,
) -> dict:
    row = dict(_BASE_ROW)
    row.update(
        {
            "scenario_id": scenario_id,
            "scale": {
                "axis_cell": "continuous_individual_panel",
                "n_factors": n_factors,
                "n_assets": n_assets,
                "n_dates": n_dates,
            },
            "wall_s": wall_s,
            "status": status,
            "is_warmup": is_warmup,
            "error_message": error_message,
        }
    )
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_happy_path_all_pass(tmp_path: Path):
    _write_jsonl(
        tmp_path / "S2.jsonl",
        [
            _make_row("S2", wall_s=1.0, is_warmup=True),
            _make_row("S2", wall_s=2.0),
        ],
    )
    rc = main([str(tmp_path)])
    assert rc == 0


def test_red_flag_overshoot_exit_nonzero(tmp_path: Path):
    overshoot = UX_TARGETS["S2"].wall_s_max + 1.0
    _write_jsonl(tmp_path / "S2.jsonl", [_make_row("S2", wall_s=overshoot)])
    rc = main([str(tmp_path)])
    assert rc != 0


def test_warmup_rows_ignored(tmp_path: Path):
    # An overshooting warmup must not flip the verdict to fail.
    overshoot = UX_TARGETS["S2"].wall_s_max + 100.0
    _write_jsonl(
        tmp_path / "S2.jsonl",
        [
            _make_row("S2", wall_s=overshoot, is_warmup=True),
            _make_row("S2", wall_s=1.0),
        ],
    )
    rc = main([str(tmp_path)])
    assert rc == 0


def test_oom_is_non_blocking_incident(tmp_path: Path):
    _write_jsonl(
        tmp_path / "S2.jsonl",
        [
            _make_row("S2", wall_s=1.0),
            _make_row(
                "S2", wall_s=None, status="oom", error_message="MemoryError(...)"
            ),
        ],
    )
    rows, incidents = evaluate_dir(tmp_path)
    assert len(rows) == 1 and rows[0].verdict == "pass"
    assert len(incidents) == 1 and incidents[0].status == "oom"
    assert main([str(tmp_path)]) == 0


def test_stretch_n_factors_recorded_but_not_asserted(tmp_path: Path):
    # n_factors >= STRETCH_N_FACTORS demotes to stretch — wildly
    # overshooting the per-scenario target must not fail.
    _write_jsonl(
        tmp_path / "S2.jsonl",
        [_make_row("S2", wall_s=9999.0, n_factors=1000)],
    )
    rows, _ = evaluate_dir(tmp_path)
    assert rows[0].verdict == "stretch"
    assert main([str(tmp_path)]) == 0


def test_unknown_scenario_is_skip(tmp_path: Path):
    _write_jsonl(tmp_path / "X.jsonl", [_make_row("X-unknown", wall_s=1.0)])
    rows, _ = evaluate_dir(tmp_path)
    assert rows[0].verdict == "skip"
    assert main([str(tmp_path)]) == 0


def test_render_markdown_includes_version_and_columns(tmp_path: Path):
    _write_jsonl(tmp_path / "S2.jsonl", [_make_row("S2", wall_s=1.0)])
    rows, incidents = evaluate_dir(tmp_path)
    md = render_markdown(rows, incidents)
    assert f"UX_TARGETS_VERSION={UX_TARGETS_VERSION}" in md
    assert "wall_s" in md and "target" in md and "n_threads" in md
    assert "S2" in md


def test_missing_jsonl_field_fails_loud(tmp_path: Path):
    # ``scenario_id`` missing — raises KeyError on read rather than
    # silently passing.
    row = _make_row("S2", wall_s=1.0)
    del row["scenario_id"]
    _write_jsonl(tmp_path / "broken.jsonl", [row])
    with pytest.raises(KeyError):
        evaluate_dir(tmp_path)
