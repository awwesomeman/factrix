"""CLI dispatcher smoke (``python -m bench``).

The full continuous + algo + sparse coverage at `tiny` runs in
~20 seconds on a CI runner. Cold-cache structure re-execs Python per
scenario; tested on a 2-scenario subset (`event` target) so the
subprocess fork cost is bounded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench.__main__ import main
from bench.validator import validate_file


def test_tiny_target_runs_full_scenario_set(tmp_path: Path):
    rc = main(["--target", "tiny", "--output", str(tmp_path)])
    assert rc == 0
    files = sorted(tmp_path.glob("*.jsonl"))
    file_names = {f.name for f in files}
    # Mandatory #380 §4 coverage.
    expected = {
        "S1.jsonl",
        "S2.jsonl",
        "S3.jsonl",
        "S4.jsonl",
        "S5.jsonl",
        "P1.jsonl",
        "M-ic.jsonl",
        "M-ic-boot.jsonl",
        "M-quantile.jsonl",
        "M-mono.jsonl",
        "M-corrado.jsonl",
    }
    assert expected.issubset(file_names), file_names
    for f in files:
        rep = validate_file(f)
        assert rep.ok, (f.name, rep.failures)


def test_cloud_targets_registered():
    """Cloud-only presets must be reachable via --target; otherwise
    UX validation runs fail at arg-parse time with no way to invoke
    the preset short of --run-one. Regression: an earlier draft
    added xlarge / user-realistic-high to PRESETS but forgot the
    TARGETS dispatcher entry."""
    from bench.__main__ import TARGETS

    for name in ("xlarge", "user-realistic-high"):
        assert name in TARGETS, name
        assert TARGETS[name]["preset"] == name


def test_event_target_runs_only_sparse_scenarios(tmp_path: Path):
    rc = main(["--target", "event", "--output", str(tmp_path)])
    assert rc == 0
    files = {f.name for f in tmp_path.glob("*.jsonl")}
    assert files == {"S5.jsonl", "M-corrado.jsonl"}


def test_cold_cache_labels_jsonl_cold(tmp_path: Path):
    rc = main(["--target", "event", "--output", str(tmp_path), "--cold-cache"])
    assert rc == 0
    with (tmp_path / "S5.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            assert row["cache_state"] == "cold"


def test_threads_flag_stamps_omp_threads_into_env(tmp_path: Path):
    """``--threads N`` must show up in ``env.omp_threads`` of the
    measured rows. We use ``--run-one`` against a single short
    scenario so the test stays bounded."""
    rc = main(
        [
            "--run-one",
            "M-ic",
            "--preset",
            "tiny",
            "--output",
            str(tmp_path),
            "--threads",
            "4",
        ]
    )
    assert rc == 0
    with (tmp_path / "M-ic.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            assert row["env"]["omp_threads"] == 4


def test_threads_default_is_one(tmp_path: Path):
    rc = main(["--run-one", "M-ic", "--preset", "tiny", "--output", str(tmp_path)])
    assert rc == 0
    with (tmp_path / "M-ic.jsonl").open() as fh:
        first = json.loads(fh.readline())
    assert first["env"]["omp_threads"] == 1


def test_cold_cache_labels_continuous_scenarios_cold(tmp_path: Path):
    """Regression: an earlier draft of the CLI silently dropped
    `cache_state` for Continuous + algo scenarios because their
    signatures lacked the kwarg. Every cell must label cold."""
    rc = main(
        [
            "--run-one",
            "S2",
            "--preset",
            "tiny",
            "--output",
            str(tmp_path),
            "--cache-state",
            "cold",
        ]
    )
    assert rc == 0
    with (tmp_path / "S2.jsonl").open() as fh:
        for line in fh:
            assert json.loads(line)["cache_state"] == "cold"


def test_run_one_invocation(tmp_path: Path):
    """The --run-one entry point used internally by --cold-cache is
    also useful for ad-hoc single-scenario reruns."""
    rc = main(
        [
            "--run-one",
            "S2",
            "--preset",
            "tiny",
            "--output",
            str(tmp_path),
        ]
    )
    assert rc == 0
    rep = validate_file(tmp_path / "S2.jsonl")
    assert rep.ok


def test_target_required_when_no_run_one(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path)])


def test_scenario_id_matches_registry_key(tmp_path: Path):
    """Every scenario must stamp its `scenario_id` field to match the
    key it is registered under. Drift between the two means a future
    CLI dispatch by `scenario_id` (e.g. `--run-one`) would land on
    the wrong function."""
    from bench.__main__ import ALL_SCENARIOS

    for sid, fn in ALL_SCENARIOS.items():
        out = tmp_path / f"{sid}.jsonl"
        records = fn(out, preset="tiny")
        assert records, sid
        assert all(r.scenario_id == sid for r in records), (
            sid,
            [r.scenario_id for r in records],
        )
