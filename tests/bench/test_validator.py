"""bench.validator — fail-loud on bad rows, CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path

from bench.validator import main, validate_file


def _good_row() -> dict:
    return {
        "schema_version": "1",
        "scenario_id": "S2",
        "metric_set": "core",
        "metric_set_version": "1",
        "axis_cell": "continuous_individual_panel",
        "scale": {
            "axis_cell": "continuous_individual_panel",
            "n_factors": 50,
            "n_dates": 1250,
            "n_assets": 1000,
        },
        "run_idx": 0,
        "is_warmup": False,
        "cache_state": "cold",
        "status": "ok",
        "error_message": None,
        "started_at": "2026-05-15T08:30:00Z",
        "wall_s": 1.0,
        "setup_s": 0.1,
        "compute_s": 0.9,
        "cpu_s": 0.95,
        "peak_rss_mb": 100.0,
        "peak_alloc_mb": 90.0,
        "env": {
            "git_sha": "abc1234",
            "factrix_version": "0.13.0",
            "dataset_spec_version": "1",
            "python": "3.12.0",
            "numpy": "2.0.0",
            "blas": "openblas",
            "omp_threads": 1,
            "cpu_model": "x86_64",
            "cpu_cores": 4,
            "ram_gb": 16.0,
            "os": "linux-x86_64",
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_validate_clean_file(tmp_path: Path):
    p = tmp_path / "ok.jsonl"
    _write(p, [_good_row()])
    rep = validate_file(p)
    assert rep.ok
    assert rep.n_rows == 1


def test_validate_missing_field(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    row = _good_row()
    del row["status"]
    _write(p, [row])
    rep = validate_file(p)
    assert not rep.ok
    assert rep.failures[0].line_no == 1


def test_validate_enum_out_of_range(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    row = _good_row()
    row["cache_state"] = "lukewarm"
    _write(p, [row])
    rep = validate_file(p)
    assert not rep.ok


def test_validate_schema_version_mismatch(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    row = _good_row()
    row["schema_version"] = "0"
    _write(p, [row])
    rep = validate_file(p)
    assert not rep.ok
    assert "schema_version" in rep.failures[0].error


def test_validate_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n", encoding="utf-8")
    rep = validate_file(p)
    assert not rep.ok


def test_cli_returns_zero_on_clean(tmp_path: Path, capsys):
    p = tmp_path / "ok.jsonl"
    _write(p, [_good_row()])
    code = main([str(p)])
    assert code == 0


def test_cli_returns_nonzero_on_bad(tmp_path: Path, capsys):
    p = tmp_path / "bad.jsonl"
    row = _good_row()
    del row["status"]
    _write(p, [row])
    code = main([str(p)])
    assert code == 1
