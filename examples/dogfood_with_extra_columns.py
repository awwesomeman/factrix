"""Dogfood: with_extra_columns on 10+ TW factors.

Runs a realistic end-to-end flow to surface API friction. Reports
are printed inline; findings feed back into a follow-up PR if any
rough edges emerge.

Scope:
  - load TW panel, slice 2023-01 to 2024-06 for speed
  - build 12 CS factors via factorlib.factors generators
  - evaluate_batch with keep_artifacts=True
  - compute 3 custom metrics per factor using Artifacts:
      * ic_worst_quarter  (min quarterly mean IC — regime-like stress)
      * ic_best_quarter   (max quarterly mean IC)
      * ic_half_life      (OLS slope of |IC| decay, crude)
  - with_extra_columns to attach all three
  - combined filter + rank + BHY
  - register_rule for a factorlib-idiomatic custom diagnose
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

import factorlib as fl
from factorlib.evaluation.diagnostics import Rule, register_rule
from factorlib.factors import (
    generate_idiosyncratic_vol,
    generate_max_effect,
    generate_mean_reversion,
    generate_momentum,
    generate_momentum_60d,
    generate_volatility,
)

DATA = Path(__file__).parent.parent / "tw_stock_daily_2017_2025.parquet"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_panel() -> pl.DataFrame:
    raw = pl.read_parquet(DATA)
    panel = fl.adapt(raw, date="date", asset_id="ticker", price="close_adj")
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("ms")))
    return panel.filter(
        (pl.col("date") >= pl.datetime(2023, 1, 1))
        & (pl.col("date") < pl.datetime(2024, 7, 1))
    )


def build_factors(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """12 CS factors spanning momentum / reversal / volatility families.

    Keeps to price-only generators so the flow runs on canonical panels
    without pulling in high/low/volume columns.
    """
    return {
        "mom_10d": generate_momentum(panel, lookback=10),
        "mom_20d": generate_momentum(panel, lookback=20),
        "mom_60d": generate_momentum(panel, lookback=60),
        "mom_60_5": generate_momentum_60d(panel),
        "vol_10d": generate_volatility(panel, lookback=10),
        "vol_20d": generate_volatility(panel, lookback=20),
        "vol_60d": generate_volatility(panel, lookback=60),
        "idio_vol_60": generate_idiosyncratic_vol(panel, lookback=60),
        "mean_rev_5": generate_mean_reversion(panel, lookback=5),
        "mean_rev_20": generate_mean_reversion(panel, lookback=20),
        "max_effect_20": generate_max_effect(panel, lookback=20),
        "max_effect_60": generate_max_effect(panel, lookback=60),
    }


# ---------------------------------------------------------------------------
# Custom metrics pulled from Artifacts
# ---------------------------------------------------------------------------

def ic_quarter_stats(artifacts) -> tuple[float, float]:
    """Return (worst_q_ic, best_q_ic) — quarterly-bucketed mean IC.

    Regime-stress proxy: if even the worst quarter stays positive, the
    factor is robust; if the best quarter is 3× the worst, it's
    concentrated in regimes.
    """
    ic_series = artifacts.intermediates["ic_series"]
    if ic_series.height == 0:
        return float("nan"), float("nan")
    quarters = (
        ic_series.with_columns(
            pl.col("date").dt.truncate("3mo").alias("quarter")
        )
        .group_by("quarter")
        .agg(pl.col("ic").mean())
        .drop_nulls("ic")
    )
    if quarters.height == 0:
        return float("nan"), float("nan")
    return float(quarters["ic"].min()), float(quarters["ic"].max())


def ic_half_life(artifacts) -> float:
    """Crude OLS slope of |IC| over time (per-day).

    Negative ≈ signal decaying; positive ≈ signal persisting / improving.
    """
    ic_series = artifacts.intermediates["ic_series"]
    ic = ic_series["ic"].drop_nulls().abs().to_numpy()
    if len(ic) < 10:
        return float("nan")
    x = np.arange(len(ic), dtype=float)
    # slope = cov(x, y) / var(x)
    return float(np.polyfit(x, ic, 1)[0])


# ---------------------------------------------------------------------------
# Register a custom diagnose rule (dogfooding register_rule)
# ---------------------------------------------------------------------------

def wire_custom_diagnostic() -> None:
    register_rule(
        "cross_sectional",
        Rule(
            code="custom.ic_regime_concentrated",
            severity="warn",
            message=(
                "Quarterly IC spread is wide (max - min > 0.08) — alpha "
                "may be concentrated in specific regimes rather than "
                "broadly reliable."
            ),
            # predicate inspects the Profile, not Artifacts — Profile
            # alone can't see ic_quarter_stats, so this rule uses a
            # coarser proxy (ic_ir < 0.3 combined with high turnover).
            predicate=lambda p: p.ic_ir < 0.3 and p.turnover > 1.5,
        ),
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Dogfood: with_extra_columns on 12 TW CS factors")
    print("=" * 72)

    wire_custom_diagnostic()

    t0 = time.perf_counter()
    panel = load_panel()
    print(
        f"[{time.perf_counter() - t0:5.2f}s] panel loaded: "
        f"{panel.height:,} rows × {panel['asset_id'].n_unique():,} assets"
    )

    t0 = time.perf_counter()
    factors = build_factors(panel)
    print(
        f"[{time.perf_counter() - t0:5.2f}s] built {len(factors)} factors: "
        f"{list(factors)}"
    )

    t0 = time.perf_counter()
    ps, arts = fl.evaluate_batch(
        factors, factor_type="cross_sectional", keep_artifacts=True,
    )
    print(f"[{time.perf_counter() - t0:5.2f}s] evaluate_batch + retain")

    # --- Custom metrics ---
    t0 = time.perf_counter()
    worst_q, best_q, half_life = [], [], []
    for p in ps:
        a = arts[p.factor_name]
        w, b = ic_quarter_stats(a)
        worst_q.append(w)
        best_q.append(b)
        half_life.append(ic_half_life(a))
    print(
        f"[{time.perf_counter() - t0:5.2f}s] computed 3 custom metrics × 12 factors"
    )

    # --- Attach ---
    scored = ps.with_extra_columns({
        "ic_worst_quarter": worst_q,
        "ic_best_quarter": best_q,
        "ic_half_life_slope": half_life,
    })

    # --- BHY on top + combined filter/sort ---
    t0 = time.perf_counter()
    ranked = (
        scored
        .multiple_testing_correct(p_source="canonical_p", fdr=0.10)
        .rank_by("ic_ir", descending=True)
    )
    print(f"[{time.perf_counter() - t0:5.2f}s] BHY + rank")

    # --- Display ---
    view = ranked.to_polars().select([
        "factor_name", "ic_mean", "ic_tstat", "ic_ir", "canonical_p",
        "p_adjusted", "bhy_significant",
        "ic_worst_quarter", "ic_best_quarter", "ic_half_life_slope",
    ])
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(4)
    print("\n--- Full ranking ---")
    print(view)

    # --- Combined filter: built-in + custom on the same expression ---
    survivors = ranked.filter(
        (pl.col("bhy_significant"))
        & (pl.col("ic_worst_quarter") > 0)
        & (pl.col("ic_best_quarter") < pl.col("ic_worst_quarter") * 3.0)
    )
    print(f"\n--- Survivors (BHY-sig + regime-stable): {len(survivors)} / {len(ranked)} ---")
    if len(survivors) > 0:
        print(survivors.to_polars().select([
            "factor_name", "ic_ir", "p_adjusted",
            "ic_worst_quarter", "ic_best_quarter",
        ]))

    # --- Custom diagnose rule sightings ---
    print("\n--- Custom diagnose fires ---")
    hits = 0
    for p in ranked:
        for d in p.diagnose():
            if d.code == "custom.ic_regime_concentrated":
                print(f"  {p.factor_name}: {d.message}")
                hits += 1
    if hits == 0:
        print("  (none fired)")

    # --- Findings check list ---
    print("\n" + "=" * 72)
    print("API dark-corner checklist")
    print("=" * 72)

    # 1. Can the extra column be used inside filter(pl.Expr) AND in
    #    combination with built-in columns?
    check1 = ranked.filter(pl.col("ic_best_quarter") > 0).to_polars().height
    print(f"  [1] filter on extra col works: {check1} rows passed")

    # 2. Does rank_by on an extra column work?
    by_custom = ranked.rank_by("ic_worst_quarter", descending=True)
    print(
        f"  [2] rank_by(extra col) works: top is {by_custom.to_polars()['factor_name'][0]!r}"
    )

    # 3. Does multiple_testing_correct survive hstack? (columns are there?)
    assert "p_adjusted" in ranked.to_polars().columns
    assert "ic_worst_quarter" in ranked.to_polars().columns
    print("  [3] p_adjusted + custom cols coexist: OK")

    # 4. Does chained with_extra_columns work?
    chained = ranked.with_extra_columns({
        "rank_bucket": [i // 4 for i in range(len(ranked))],
    })
    print(f"  [4] chained with_extra_columns: added rank_bucket ({len(chained)} rows)")

    # 5. Collision detection — should raise
    try:
        ranked.with_extra_columns({"ic_mean": [0.0] * len(ranked)})
        print("  [5] collision detection: FAILED (should have raised)")
    except ValueError:
        print("  [5] collision detection: OK")

    print("\nDone.")


if __name__ == "__main__":
    main()
