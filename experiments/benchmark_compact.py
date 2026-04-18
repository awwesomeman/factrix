"""Benchmark: compact=True memory savings on full TW panel × many factors.

Question: does compact=True save enough memory to justify keeping the
API, or is it <30% and better dropped (reduce surface area)?

Methodology:
  - load full TW panel (2017-2025, ~2M rows, ~2000 assets)
  - build 30 CS factors via lookback sweeps on price-only generators
  - run evaluate_batch 3 ways; measure RSS delta + artifact sizes
  - compare peak memory and retained-artifact footprint

Interpretation guide:
  - If compact saves < 30% of retained memory → drop the API surface
  - If 30-70% → keep with documented use case (≥ hundreds of factors)
  - If > 70% → promote in README / make default for large batches
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import polars as pl
import psutil

import factorlib as fl
from factorlib.factors import (
    generate_idiosyncratic_vol,
    generate_max_effect,
    generate_mean_reversion,
    generate_momentum,
    generate_volatility,
)

DATA = Path(__file__).parent.parent / "tw_stock_daily_2017_2025.parquet"


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_full_panel() -> pl.DataFrame:
    raw = pl.read_parquet(DATA)
    panel = fl.adapt(raw, date="date", asset_id="ticker", price="close_adj")
    return panel.with_columns(pl.col("date").cast(pl.Datetime("ms")))


def build_factor_sweep(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """30 factors from lookback sweeps on 5 generators × 6 lookbacks."""
    lookbacks = [5, 10, 20, 40, 60, 120]
    factors: dict[str, pl.DataFrame] = {}
    for lb in lookbacks:
        factors[f"mom_{lb}"] = generate_momentum(panel, lookback=lb)
        factors[f"vol_{lb}"] = generate_volatility(panel, lookback=lb)
        factors[f"mean_rev_{lb}"] = generate_mean_reversion(panel, lookback=lb)
        factors[f"max_effect_{lb}"] = generate_max_effect(panel, lookback=lb)
        factors[f"idio_vol_{lb}"] = generate_idiosyncratic_vol(panel, lookback=lb)
    return factors


def artifact_nbytes(art) -> int:
    """Estimate Artifacts memory footprint.

    prepared DataFrame contributes the bulk when not compact; the
    intermediates are small (ic_series, spread_series — O(T) rows).
    """
    total = 0
    # prepared: either real DataFrame or _CompactedPrepared sentinel
    prepared = object.__getattribute__(art, "prepared")
    if isinstance(prepared, pl.DataFrame):
        total += prepared.estimated_size()
    for df in art.intermediates.values():
        total += df.estimated_size()
    return total


def bench_mode(
    name: str, factors: dict[str, pl.DataFrame],
    *, keep_artifacts: bool, compact: bool,
) -> dict[str, float]:
    gc.collect()
    rss_before = rss_mb()
    t0 = time.perf_counter()
    result = fl.evaluate_batch(
        factors, factor_type="cross_sectional",
        keep_artifacts=keep_artifacts, compact=compact,
    )
    elapsed = time.perf_counter() - t0
    rss_after = rss_mb()
    rss_delta = rss_after - rss_before

    arts_nbytes = 0
    if keep_artifacts:
        _ps, arts = result
        arts_nbytes = sum(artifact_nbytes(a) for a in arts.values())
        # Hold the reference so GC doesn't reclaim before we measure.
        _keepalive = arts  # noqa: F841
    else:
        _keepalive = result  # noqa: F841

    rss_hold = rss_mb()

    # Release and re-measure; allows reading steady-state vs peak
    del result
    gc.collect()
    rss_released = rss_mb()

    print(
        f"  [{name:>24s}]  elapsed={elapsed:5.2f}s  "
        f"ΔRSS={rss_delta:+7.1f}MB  hold={rss_hold:7.1f}MB  "
        f"released={rss_released:7.1f}MB  "
        f"artifacts_est={arts_nbytes / (1024**2):7.1f}MB"
    )
    return {
        "elapsed": elapsed,
        "rss_delta_mb": rss_delta,
        "rss_hold_mb": rss_hold,
        "rss_released_mb": rss_released,
        "artifacts_est_mb": arts_nbytes / (1024 ** 2),
    }


def main() -> None:
    print("=" * 80)
    print("Benchmark: compact=True memory impact")
    print("=" * 80)
    print(f"  Python rss at start: {rss_mb():.1f} MB")

    t0 = time.perf_counter()
    panel = load_full_panel()
    print(
        f"  [{time.perf_counter() - t0:5.2f}s] full panel: "
        f"{panel.height:,} rows × {panel['asset_id'].n_unique():,} assets  "
        f"({panel.estimated_size() / (1024 ** 2):.1f} MB)"
    )

    t0 = time.perf_counter()
    factors = build_factor_sweep(panel)
    factor_bytes = sum(df.estimated_size() for df in factors.values())
    print(
        f"  [{time.perf_counter() - t0:5.2f}s] {len(factors)} factors built  "
        f"(~{factor_bytes / (1024 ** 2):.1f} MB raw)"
    )

    print()
    print("  Mode comparisons:")

    control = bench_mode(
        "baseline (no retain)", factors,
        keep_artifacts=False, compact=False,
    )
    full = bench_mode(
        "retain full", factors,
        keep_artifacts=True, compact=False,
    )
    compact = bench_mode(
        "retain compact", factors,
        keep_artifacts=True, compact=True,
    )

    full_mb = full["artifacts_est_mb"]
    comp_mb = compact["artifacts_est_mb"]
    if full_mb > 0:
        ratio = 1 - comp_mb / full_mb
        print()
        print("=" * 80)
        print(
            f"  Retained-artifact footprint:  full={full_mb:.1f} MB  "
            f"compact={comp_mb:.1f} MB  savings={ratio:.1%}"
        )

    # RSS-based comparison (while artifacts held)
    hold_full = full["rss_hold_mb"]
    hold_compact = compact["rss_hold_mb"]
    hold_base = control["rss_hold_mb"]
    print(
        f"  Process RSS while holding:   full={hold_full:.1f} MB  "
        f"compact={hold_compact:.1f} MB  baseline={hold_base:.1f} MB"
    )
    if hold_full > hold_base:
        overhead_full = hold_full - hold_base
        overhead_compact = hold_compact - hold_base
        if overhead_full > 0:
            rss_savings = 1 - overhead_compact / overhead_full
            print(
                f"  RSS-based artifact overhead: full=+{overhead_full:.1f} MB  "
                f"compact=+{overhead_compact:.1f} MB  savings={rss_savings:.1%}"
            )

    print()
    print("Interpretation:")
    if full_mb > 0:
        if ratio > 0.7:
            verdict = "KEEP + promote (savings > 70%)"
        elif ratio > 0.3:
            verdict = "KEEP (savings 30-70%) — document for large batches"
        else:
            verdict = "CONSIDER DROPPING (savings < 30% vs added API surface)"
        print(f"  → {verdict}")


if __name__ == "__main__":
    main()
