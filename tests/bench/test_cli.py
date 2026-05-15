"""CLI dispatcher smoke (``python -m bench``).

The full continuous + algo + sparse coverage at `tiny` runs in
~20 seconds on a CI runner. Cold-cache mode re-execs Python per
scenario; tested on a 2-scenario subset (`event` target) so the
subprocess fork cost is bounded.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from bench.__main__ import main
from bench.validator import validate_file


@pytest.fixture(autouse=True)
def _silence_sample_floor_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


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
