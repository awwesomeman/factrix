"""``python -m bench`` — run scenarios at a chosen scale preset.

Usage::

    python -m bench --target tiny --output out/
    python -m bench --target small --output out/baselines/v0.13.1/
    python -m bench --target small --output out/ --cold-cache

Each scenario writes ``<scenario_id>.jsonl`` under ``--output``. The
harness self-validates every file as it lands; CI smoke jobs and
reference-baseline reruns share the same entry.

`--cold-cache` re-execs ``python -m bench --run-one <id>`` per
scenario in a fresh subprocess so OS-page-cache / numpy-import /
BLAS-thread state is reset between scenarios. Reference-baseline
runs must use this mode; ad-hoc warm-cache runs (default) skip the
subprocess overhead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.scenarios.algo import SCENARIOS as ALGO_SCENARIOS
from bench.scenarios.continuous import SCENARIOS as CONTINUOUS_SCENARIOS
from bench.scenarios.sparse import SCENARIOS as SPARSE_SCENARIOS

# All scenarios, indexed by `scenario_id`. Targets pick subsets by name.
ALL_SCENARIOS: dict[str, Callable[..., Any]] = {
    **CONTINUOUS_SCENARIOS,
    **ALGO_SCENARIOS,
    **SPARSE_SCENARIOS,
}
_CONT_ALGO_IDS = list(CONTINUOUS_SCENARIOS) + list(ALGO_SCENARIOS)
_SPARSE_IDS = list(SPARSE_SCENARIOS)

TARGETS: dict[str, dict[str, Any]] = {
    "tiny": {"preset": "tiny", "scenarios": _CONT_ALGO_IDS + _SPARSE_IDS},
    "small": {"preset": "small", "scenarios": _CONT_ALGO_IDS + _SPARSE_IDS},
    "large": {"preset": "large", "scenarios": _CONT_ALGO_IDS + _SPARSE_IDS},
    "event": {"preset": "small", "scenarios": _SPARSE_IDS},
}


def _run_one(
    scenario_id: str,
    *,
    preset: str,
    output_dir: Path,
    cache_state: str,
) -> Path:
    """Invoke one scenario directly (in this process)."""
    fn = ALL_SCENARIOS[scenario_id]
    output = output_dir / f"{scenario_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    fn(output, preset=preset, cache_state=cache_state)
    return output


def _run_cold_cache(scenario_ids: list[str], *, preset: str, output_dir: Path) -> None:
    """Re-exec one subprocess per scenario for cold-cache mode."""
    for sid in scenario_ids:
        cmd = [
            sys.executable,
            "-m",
            "bench",
            "--run-one",
            sid,
            "--preset",
            preset,
            "--output",
            str(output_dir),
            "--cache-state",
            "cold",
        ]
        # Inherit env (preflight thread locks already in env via
        # lock_threads); fail-fast on any subprocess error.
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args and dispatch to one of the modes."""
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Run benchmark scenarios at a chosen scale preset.",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        help="Named target (selects preset + scenario set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for <scenario_id>.jsonl files.",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help=(
            "Re-exec one subprocess per scenario so OS page cache, "
            "numpy import, and BLAS thread state reset between "
            "scenarios. Required for reference-baseline runs."
        ),
    )
    # Internal flags used by --cold-cache subprocess invocations.
    parser.add_argument("--run-one", help=argparse.SUPPRESS, default=None)
    parser.add_argument("--preset", help=argparse.SUPPRESS, default=None)
    parser.add_argument(
        "--cache-state",
        choices=("cold", "warm", "unknown"),
        default="warm",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output

    if args.run_one is not None:
        if args.preset is None:
            parser.error("--run-one requires --preset")
        _run_one(
            args.run_one,
            preset=args.preset,
            output_dir=output_dir,
            cache_state=args.cache_state,
        )
        return 0

    if not args.target:
        parser.error("--target is required (or use --run-one for one scenario)")

    target = TARGETS[args.target]
    preset = target["preset"]
    scenarios = target["scenarios"]

    if args.cold_cache:
        _run_cold_cache(scenarios, preset=preset, output_dir=output_dir)
    else:
        for sid in scenarios:
            _run_one(sid, preset=preset, output_dir=output_dir, cache_state="warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
