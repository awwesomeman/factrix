"""End-to-end smoke scenario — proves wrapper → JSONL → validator.

Used by tests and by CI smoke jobs to prevent harness rot. The real
benchmark workloads live in sibling modules under ``bench.scenarios``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from factrix.datasets import make_multi_factor_panel

from bench.preflight import preflight
from bench.schema import BenchRecord
from bench.validator import validate_file
from bench.wrapper import measure, write_records


def run(
    *,
    output: Path,
    n_factors: int = 4,
    n_assets: int = 20,
    n_dates: int = 60,
    warmup: bool = True,
) -> list[BenchRecord]:
    """Run the dummy scenario, write JSONL, self-validate."""
    pre = preflight(threads=1, seed=0)
    scale = {"n_factors": n_factors, "n_dates": n_dates, "n_assets": n_assets}

    def setup():
        return make_multi_factor_panel(
            n_factors=n_factors,
            n_assets=n_assets,
            n_dates=n_dates,
            seed=0,
        )

    def compute(df):
        # Stand-in for real metric work: touch every factor column.
        factor_cols = [c for c in df.columns if c.startswith("factor_")]
        return float(df.select(factor_cols).sum().to_numpy().sum())

    records: list[BenchRecord] = []
    if warmup:
        records.append(
            measure(
                setup,
                compute,
                scenario_id="dummy",
                axis_cell="continuous_individual_panel",
                scale=scale,
                metric_set="core",
                run_idx=0,
                is_warmup=True,
                cache_state="cold",
                env=pre.env,
            )
        )
    records.append(
        measure(
            setup,
            compute,
            scenario_id="dummy",
            axis_cell="continuous_individual_panel",
            scale=scale,
            metric_set="core",
            run_idx=0,
            is_warmup=False,
            cache_state="warm",
            env=pre.env,
        )
    )
    write_records(output, records)
    report = validate_file(output)
    if not report.ok:
        raise RuntimeError(f"self-validation failed: {report.failures}")
    return records


def main() -> int:
    """CLI entry point for the dummy scenario."""
    p = argparse.ArgumentParser(description="Dummy bench scenario")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-factors", type=int, default=4)
    p.add_argument("--n-assets", type=int, default=20)
    p.add_argument("--n-dates", type=int, default=60)
    args = p.parse_args()
    run(
        output=args.output,
        n_factors=args.n_factors,
        n_assets=args.n_assets,
        n_dates=args.n_dates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
