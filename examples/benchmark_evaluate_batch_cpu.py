"""Benchmark: evaluate_batch CPU utilization.

Blocks T3.S4 §3 (n_jobs implementation). Decision rule:
  > 70%  : Polars already saturates CPU → do not add n_jobs, close spike
  40-70% : partial, write microbench for n_jobs=2/4/8 before deciding
  < 40%  : IO/Python overhead bound → parallelism may help

Setup: full TW panel × 30 CS factors (lookback sweeps on 5 generators),
same as benchmark_compact.py for comparability.
"""

from __future__ import annotations

import os
import statistics
import threading
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


class CPUSampler:
    """Background sampler of total-system CPU utilization."""

    def __init__(self, interval: float = 0.2) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        # psutil.cpu_percent with interval blocks — we drive it ourselves
        psutil.cpu_percent(interval=None)  # prime
        while not self._stop.is_set():
            time.sleep(self._interval)
            self._samples.append(psutil.cpu_percent(interval=None))

    def __enter__(self) -> "CPUSampler":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "n": 0}
        samples = sorted(self._samples)
        return {
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "p95": samples[int(0.95 * (len(samples) - 1))],
            "n": len(samples),
        }


def load_panel() -> pl.DataFrame:
    raw = pl.read_parquet(DATA)
    panel = fl.adapt(raw, date="date", asset_id="ticker", price="close_adj")
    return panel.with_columns(pl.col("date").cast(pl.Datetime("ms")))


def build_factors(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    lookbacks = [5, 10, 20, 40, 60, 120]
    factors: dict[str, pl.DataFrame] = {}
    for lb in lookbacks:
        factors[f"mom_{lb}"] = generate_momentum(panel, lookback=lb)
        factors[f"vol_{lb}"] = generate_volatility(panel, lookback=lb)
        factors[f"mean_rev_{lb}"] = generate_mean_reversion(panel, lookback=lb)
        factors[f"max_effect_{lb}"] = generate_max_effect(panel, lookback=lb)
        factors[f"idio_vol_{lb}"] = generate_idiosyncratic_vol(panel, lookback=lb)
    return factors


def main() -> None:
    print("=" * 80)
    print("Benchmark: evaluate_batch CPU utilization (T3.S4 §2.3)")
    print("=" * 80)
    print(f"  Host cores: {psutil.cpu_count(logical=False)} physical, "
          f"{psutil.cpu_count(logical=True)} logical")
    polars_threads = os.environ.get("POLARS_MAX_THREADS")
    print(f"  POLARS_MAX_THREADS: {polars_threads or 'not set (polars default)'}")

    panel = load_panel()
    factors = build_factors(panel)
    print(f"  Panel: {panel.height:,} rows × {panel['asset_id'].n_unique():,} assets")
    print(f"  Factors: {len(factors)}")
    print()

    # --- Single-factor CPU ---
    print("[1] Single-factor evaluate CPU:")
    one_name, one_df = next(iter(factors.items()))
    # warm-up (prime numpy / polars caches)
    _ = fl.evaluate(one_df, one_name, factor_type="cross_sectional")
    with CPUSampler() as s:
        t0 = time.perf_counter()
        _ = fl.evaluate(one_df, one_name, factor_type="cross_sectional")
        t_one = time.perf_counter() - t0
    one_stats = s.summary()
    print(
        f"  elapsed={t_one:.2f}s  "
        f"CPU mean={one_stats['mean']:.1f}%  "
        f"median={one_stats['median']:.1f}%  "
        f"p95={one_stats['p95']:.1f}%  "
        f"(n={one_stats['n']} samples)"
    )

    # --- Batch CPU ---
    print()
    print("[2] Batch evaluate_batch CPU (sequential loop):")
    with CPUSampler() as s:
        t0 = time.perf_counter()
        _ = fl.evaluate_batch(factors, factor_type="cross_sectional")
        t_batch = time.perf_counter() - t0
    batch_stats = s.summary()
    per_factor = t_batch / len(factors)
    print(
        f"  elapsed={t_batch:.2f}s  per_factor={per_factor:.2f}s  "
        f"CPU mean={batch_stats['mean']:.1f}%  "
        f"median={batch_stats['median']:.1f}%  "
        f"p95={batch_stats['p95']:.1f}%  "
        f"(n={batch_stats['n']} samples)"
    )

    # --- Interpretation ---
    cpu_mean = batch_stats["mean"]
    print()
    print("=" * 80)
    print("Decision")
    print("=" * 80)
    logical = psutil.cpu_count(logical=True)
    # Normalize: psutil returns total system % already (0 - 100 × n_cores on macOS,
    # or 0-100 normalized on Linux).  Check platform behaviour.
    print(f"  Mean CPU during batch: {cpu_mean:.1f}%")
    print(f"  (n_cores={logical}; psutil.cpu_percent returns 0-100 normalized)")
    if cpu_mean > 70:
        verdict = "SATURATED — do NOT add n_jobs, close T3.S4"
    elif cpu_mean >= 40:
        verdict = "PARTIAL — write microbench for n_jobs=2/4/8 before committing"
    else:
        verdict = "UNDERSATURATED — parallelism likely helps, proceed to T3.S4 §3"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
