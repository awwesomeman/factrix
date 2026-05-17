"""Tests for ``scripts/bench_diff.py`` (#389)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import bench_diff  # noqa: E402


def _record(
    *,
    scenario_id: str = "S2",
    n_factors: int = 50,
    compute_s: float = 1.0,
    peak_rss_mb: float = 100.0,
    schema_version: str = "1",
    metric_set_version: str = "1",
    cache_state: str = "cold",
    dataset_spec_version: str = "1",
    is_warmup: bool = False,
    status: str = "ok",
    setup_s: float = 0.5,
    wall_s: float | None = None,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "scenario_id": scenario_id,
        "metric_set": "core",
        "metric_set_version": metric_set_version,
        "axis_cell": "continuous_individual_panel",
        "scale": {
            "axis_cell": "continuous_individual_panel",
            "n_factors": n_factors,
            "n_dates": 250,
            "n_assets": 100,
        },
        "run_idx": 0,
        "is_warmup": is_warmup,
        "cache_state": cache_state,
        "status": status,
        "error_message": None,
        "started_at": "2026-05-17T00:00:00Z",
        "wall_s": wall_s if wall_s is not None else setup_s + compute_s,
        "setup_s": setup_s,
        "compute_s": compute_s,
        "cpu_s": compute_s,
        "peak_rss_mb": peak_rss_mb,
        "peak_alloc_mb": peak_rss_mb / 2,
        "env": {
            "git_sha": "deadbeef",
            "factrix_version": "0.14.0",
            "dataset_spec_version": dataset_spec_version,
            "python": "3.13.0",
            "numpy": "2.0.0",
            "blas": "openblas",
            "omp_threads": 1,
            "cpu_model": "x86_64",
            "cpu_cores": 1,
            "ram_gb": 16.0,
            "os": "linux",
        },
    }


def _write_dir(path: Path, records: list[dict[str, object]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    by_scenario: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario_id"], []).append(r)  # type: ignore[index]
    for scenario_id, rs in by_scenario.items():
        (path / f"{scenario_id}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rs) + "\n"
        )
    return path


def test_happy_path_renders_ratio_table(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [
            _record(scenario_id="S2", compute_s=2.0, peak_rss_mb=100.0),
            _record(scenario_id="M-ic", compute_s=1.0, peak_rss_mb=80.0),
        ],
    )
    after = _write_dir(
        tmp_path / "after",
        [
            _record(scenario_id="S2", compute_s=1.0, peak_rss_mb=110.0),
            _record(scenario_id="M-ic", compute_s=0.5, peak_rss_mb=80.0),
        ],
    )
    table = bench_diff.diff(before, after)
    assert "| scenario_id | scale | compute_s_before" in table
    assert "| M-ic |" in table
    assert "0.500" in table  # M-ic ratio
    assert "| S2 |" in table
    assert "1.100" in table  # S2 peak_rss_mb ratio


def test_warmup_and_failed_records_ignored(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [
            _record(scenario_id="S2", compute_s=2.0, is_warmup=True),
            _record(scenario_id="S2", compute_s=2.0),
            _record(scenario_id="S3", compute_s=5.0, status="error"),
            _record(scenario_id="S3", compute_s=5.0),
        ],
    )
    after = _write_dir(
        tmp_path / "after",
        [
            _record(scenario_id="S2", compute_s=1.0),
            _record(scenario_id="S3", compute_s=2.5),
        ],
    )
    table = bench_diff.diff(before, after)
    assert "S2" in table
    assert "S3" in table


def test_schema_version_mismatch_fails_loud(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [_record(scenario_id="S2", schema_version="1")],
    )
    after = _write_dir(
        tmp_path / "after",
        [_record(scenario_id="S2", schema_version="2")],
    )
    with pytest.raises(SystemExit, match="compatibility mismatch"):
        bench_diff.diff(before, after)


def test_cache_state_mismatch_fails_loud(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [_record(scenario_id="S2", cache_state="cold")],
    )
    after = _write_dir(
        tmp_path / "after",
        [_record(scenario_id="S2", cache_state="warm")],
    )
    with pytest.raises(SystemExit, match="cache_state"):
        bench_diff.diff(before, after)


def test_metric_set_version_mismatch_fails_loud(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [_record(scenario_id="S2", metric_set_version="1")],
    )
    after = _write_dir(
        tmp_path / "after",
        [_record(scenario_id="S2", metric_set_version="2")],
    )
    with pytest.raises(SystemExit, match="metric_set_version"):
        bench_diff.diff(before, after)


def test_axis_cell_mismatch_fails_loud(tmp_path: Path) -> None:
    """Sparse vs continuous axis cells on the same scenario_id are incomparable."""
    before = _write_dir(
        tmp_path / "before",
        [_record(scenario_id="S2")],
    )
    # Manually rewrite axis_cell on the after record (skip schema's
    # nested scale validator — this is a fail-loud test fixture, the
    # script's job is to refuse the comparison even if the input is
    # inconsistent).
    after_rec = _record(scenario_id="S2")
    after_rec["axis_cell"] = "sparse_individual_panel"
    after = _write_dir(tmp_path / "after", [after_rec])
    with pytest.raises(SystemExit, match="axis_cell"):
        bench_diff.diff(before, after)


def test_dataset_spec_version_mismatch_fails_loud(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [_record(scenario_id="S2", dataset_spec_version="1")],
    )
    after = _write_dir(
        tmp_path / "after",
        [_record(scenario_id="S2", dataset_spec_version="2")],
    )
    with pytest.raises(SystemExit, match="dataset_spec_version"):
        bench_diff.diff(before, after)


def test_scenario_set_mismatch_fails_loud(tmp_path: Path) -> None:
    before = _write_dir(
        tmp_path / "before",
        [
            _record(scenario_id="S2"),
            _record(scenario_id="S3"),
        ],
    )
    after = _write_dir(
        tmp_path / "after",
        [_record(scenario_id="S2")],
    )
    with pytest.raises(SystemExit, match="scenario set mismatch"):
        bench_diff.diff(before, after)


def test_multi_scale_scenarios_aligned_by_scale(tmp_path: Path) -> None:
    """P1-style: same scenario_id, different n_factors → separate rows."""
    before = _write_dir(
        tmp_path / "before",
        [
            _record(scenario_id="P1", n_factors=10, compute_s=1.0),
            _record(scenario_id="P1", n_factors=100, compute_s=10.0),
        ],
    )
    after = _write_dir(
        tmp_path / "after",
        [
            _record(scenario_id="P1", n_factors=10, compute_s=0.5),
            _record(scenario_id="P1", n_factors=100, compute_s=8.0),
        ],
    )
    table = bench_diff.diff(before, after)
    lines = [line for line in table.splitlines() if line.startswith("| P1 |")]
    assert len(lines) == 2
    assert "n_factors=10" in lines[0]
    assert "n_factors=100" in lines[1]
