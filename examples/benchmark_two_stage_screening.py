"""Microbench for two-stage screening: IC-only vs Full profile per-factor.

Backs the "Large-batch screening pattern" section in README — when
surveying hundreds of candidate factors, users can pre-screen cheaply
with just IC then run the full pipeline on survivors. This script
measures how much that actually saves.

Methodology:
  - full TW panel (2017-2025) × 30-factor sweep on price-only generators
  - warm-up once per factor, then time two paths:
      * Full    — build_artifacts + CrossSectionalProfile.from_artifacts
                  (same cost evaluate_batch pays per factor)
      * IC-only — preprocess + compute_ic + ic metric (what the user
                  writes in the README recipe's Stage 1 loop)

Interpretation: if the IC-only ratio is meaningfully < 1.0, the
two-stage pattern is worth the extra 20 lines of user-space loop.

Remember the statistical caveat: IC-only screening then correcting IC
p-values is a conditional filter — pair with an orthogonal gate (IC
magnitude threshold, turnover cap) if you want BHY's marginal FDR
interpretation to hold. See ProfileSet.multiple_testing_correct
docstring.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from statistics import mean, median

import polars as pl

import factorlib as fl
from factorlib.config import CrossSectionalConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles.cross_sectional import CrossSectionalProfile
from factorlib.factors import (
    generate_idiosyncratic_vol,
    generate_max_effect,
    generate_mean_reversion,
    generate_momentum,
    generate_volatility,
)
from factorlib.metrics.ic import compute_ic, ic as ic_metric

DATA = Path(__file__).parent.parent / "tw_stock_daily_2017_2025.parquet"


def load_full_panel() -> pl.DataFrame:
    raw = pl.read_parquet(DATA)
    panel = fl.adapt(raw, date="date", asset_id="ticker", price="close_adj")
    return panel.with_columns(pl.col("date").cast(pl.Datetime("ms")))


def build_factor_sweep(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    lookbacks = [5, 10, 20, 40, 60, 120]
    factors: dict[str, pl.DataFrame] = {}
    for lb in lookbacks:
        factors[f"mom_{lb}"] = generate_momentum(panel, lookback=lb)
        factors[f"vol_{lb}"] = generate_volatility(panel, lookback=lb)
        factors[f"mean_rev_{lb}"] = generate_mean_reversion(panel, lookback=lb)
        factors[f"max_effect_{lb}"] = generate_max_effect(panel, lookback=lb)
        factors[f"idio_vol_{lb}"] = generate_idiosyncratic_vol(panel, lookback=lb)
    return factors


def prepare(df: pl.DataFrame, config: CrossSectionalConfig) -> pl.DataFrame:
    return fl.preprocess(df, config=config)


def time_full(prepared: pl.DataFrame, config: CrossSectionalConfig) -> float:
    """Full pipeline: build_artifacts + CrossSectionalProfile.from_artifacts.

    Equivalent to what evaluate_batch pays per factor.
    """
    t0 = time.perf_counter()
    arts = build_artifacts(prepared, config)
    _profile = CrossSectionalProfile.from_artifacts(arts)
    return time.perf_counter() - t0


def time_ic_only(prepared: pl.DataFrame, config: CrossSectionalConfig) -> float:
    """IC-only screen: compute_ic + ic metric on the prepared panel.

    Matches the README recipe's Stage 1 per-factor body. Skips
    spread_series, monotonicity, quantile, tradability, OOS splits.
    """
    fp = config.forward_periods
    t0 = time.perf_counter()
    ic_series = compute_ic(prepared)
    _ic_m = ic_metric(ic_series, forward_periods=fp)
    return time.perf_counter() - t0


def run_one_factor(
    name: str, factor_df: pl.DataFrame, config: CrossSectionalConfig,
) -> dict[str, float]:
    prepared = prepare(factor_df, config)
    # Warm-up: first call pays a polars planning cost; discard it so
    # the reported numbers reflect steady-state per-factor work. GC once
    # after warmup; not between timed calls (that biases the 1000-factor
    # extrapolation below by ~5-10%).
    _ = time_full(prepared, config)
    gc.collect()

    full_s = time_full(prepared, config)
    ic_s = time_ic_only(prepared, config)

    return {"full": full_s, "ic_only": ic_s}


def report(results: list[dict[str, float]]) -> None:
    full = [r["full"] for r in results]
    ic = [r["ic_only"] for r in results]

    print()
    print("=" * 70)
    print("Per-factor latency (seconds)")
    print("=" * 70)
    print(f"  {'variant':<16s}  {'mean':>8s}  {'median':>8s}  {'ratio':>10s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*10}")
    mean_full = mean(full)
    med_full = median(full)
    print(f"  {'Full':<16s}  {mean_full:>8.3f}  {med_full:>8.3f}  {'1.000':>10s}")

    m_ic = mean(ic)
    md_ic = median(ic)
    print(
        f"  {'IC-only':<16s}  {m_ic:>8.3f}  {md_ic:>8.3f}  "
        f"{m_ic / mean_full:>10.3f}"
    )

    print()
    print("=" * 70)
    speedup = mean_full / m_ic
    n = len(results)
    # Extrapolation for the README recipe: 1000 candidates → top-50
    # IC-only screen, then full pipeline on survivors.
    est_1000 = 1000 * m_ic + 50 * mean_full
    est_single_stage = 1000 * mean_full
    print(
        f"  IC-only is {speedup:.1f}x faster than Full "
        f"(n={n} factors averaged)"
    )
    print(
        f"  1000-candidate extrapolation (top-50 cutoff):"
    )
    print(
        f"    single-stage evaluate_batch  ≈ {est_single_stage / 60:.1f} min"
    )
    print(
        f"    two-stage screen + evaluate  ≈ {est_1000 / 60:.1f} min  "
        f"({est_single_stage / est_1000:.1f}x overall speedup)"
    )


def main() -> None:
    print("=" * 70)
    print("Microbench — two-stage screening: IC-only vs Full")
    print("=" * 70)

    t0 = time.perf_counter()
    panel = load_full_panel()
    print(
        f"  [{time.perf_counter() - t0:5.2f}s] panel: "
        f"{panel.height:,} rows × {panel['asset_id'].n_unique():,} assets"
    )

    t0 = time.perf_counter()
    factors = build_factor_sweep(panel)
    print(
        f"  [{time.perf_counter() - t0:5.2f}s] {len(factors)} factors built"
    )

    config = CrossSectionalConfig()

    print()
    print(f"  Timing {len(factors)} factors (first call per factor is "
          f"warm-up, discarded) ...")
    results: list[dict[str, float]] = []
    for i, (name, df) in enumerate(factors.items(), 1):
        r = run_one_factor(name, df, config)
        results.append(r)
        print(
            f"    [{i:2d}/{len(factors)}] {name:<18s}  "
            f"full={r['full']:.3f}s  ic-only={r['ic_only']:.3f}s"
        )

    report(results)


if __name__ == "__main__":
    main()
