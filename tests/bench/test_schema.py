"""bench.schema — pydantic model and JSON Schema export."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bench.schema import SCHEMA_VERSION, BenchRecord, record_json_schema


def _env_dict() -> dict:
    return {
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
    }


def _ok_record() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
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
        "wall_s": 12.345,
        "setup_s": 1.23,
        "compute_s": 11.115,
        "cpu_s": 11.987,
        "peak_rss_mb": 2048.0,
        "peak_alloc_mb": 1820.0,
        "env": _env_dict(),
    }


def test_valid_record_round_trip():
    rec = BenchRecord.model_validate(_ok_record())
    assert rec.scenario_id == "S2"
    assert rec.scale.n_factors == 50


def test_scale_discriminator_rejects_mismatch():
    bad = _ok_record()
    bad["axis_cell"] = "sparse_individual_panel"
    # scale block still carries continuous fields
    with pytest.raises(ValidationError):
        BenchRecord.model_validate(bad)


def test_status_enum_strict():
    bad = _ok_record()
    bad["status"] = "broken"
    with pytest.raises(ValidationError):
        BenchRecord.model_validate(bad)


def test_extra_fields_forbidden():
    bad = _ok_record()
    bad["bogus_field"] = 1
    with pytest.raises(ValidationError):
        BenchRecord.model_validate(bad)


def test_missing_required_field():
    bad = _ok_record()
    del bad["started_at"]
    with pytest.raises(ValidationError):
        BenchRecord.model_validate(bad)


def test_sparse_scale_shape():
    rec = _ok_record()
    rec["axis_cell"] = "sparse_individual_panel"
    rec["scale"] = {
        "axis_cell": "sparse_individual_panel",
        "n_events": 20,
        "n_assets": 200,
        "n_dates": 1250,
        "window_pre": 5,
        "window_post": 10,
    }
    parsed = BenchRecord.model_validate(rec)
    assert parsed.scale.n_events == 20


def test_json_schema_exports():
    schema = record_json_schema()
    assert schema["title"] == "BenchRecord"
    assert "schema_version" in schema["properties"]
