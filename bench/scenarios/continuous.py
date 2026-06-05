"""Continuous × Individual scenarios.

Each scenario is a small function with the same shape::

    def <name>(output: Path, preset: str = "tiny", ...) -> list[BenchRecord]

— the helper in ``_helpers`` handles preflight, timing, writing, and
self-validation. Scenarios only declare *what* to compute on the
prepared panel.

Algo scenarios (``greedy_forward_selection``) live in ``algo.py``;
sparse-cell scenarios live in ``sparse.py``.
"""

from __future__ import annotations

from pathlib import Path

import factrix as fx
import numpy as np
import polars as pl
from factrix.metrics.ic import compute_ic
from factrix.stats import bootstrap_mean_ci, bootstrap_mean_ci_batch

from bench import metric_sets
from bench.metric_sets import MetricSet
from bench.scenarios._helpers import (
    DEFAULT_FORWARD_PERIODS,
    factor_columns,
    resolve_scale,
    run_continuous_scenario,
    write_and_validate,
)
from bench.schema import BenchRecord, CacheState

# Bootstrap resample count for the heavy / bootstrap scenarios.
# Pinned here (not at call site) so tuning compute cost does not
# silently drift the baseline workload. The value matches factrix's
# BlockBootstrap default.
BOOTSTRAP_N = 999


# ---------------------------------------------------------------------------
# Multi-factor compute primitives
#
# Single-factor scenarios (S1, micros) iterate over one column; multi-
# factor scenarios (S2/S3/P1) loop and aggregate. Each helper returns
# a tally we don't actually inspect — we time the call.
# ---------------------------------------------------------------------------


def _bootstrap_ic_per_factor(
    panel: pl.DataFrame,
    factors: list[str],
    *,
    seed: int,
) -> int:
    # Batch the IC compute across factors (one polars query) and the
    # bootstrap (one shared block-index matrix). compute_ic is called
    # directly (not via fx.evaluate) because evaluate's IC stage-1
    # is shaped for the t-test consumers (ic / ic_newey_west / ic_ir),
    # not for downstream non-metric numpy work like bootstrap.
    ic_results = compute_ic(panel, factor_cols=factors)
    ic_matrix = np.stack([ic_results[col]["ic"].to_numpy() for col in factors])
    bootstrap_mean_ci_batch(ic_matrix, n_bootstrap=BOOTSTRAP_N, seed=seed)
    return len(factors)


# ---------------------------------------------------------------------------
# S1 — single factor: evaluate(heavy)
# ---------------------------------------------------------------------------


def s1_evaluate(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """S1: single factor through ``evaluate(heavy)``.

    Fixed scale: 1 factor; dates / assets follow the preset.
    """
    scale = resolve_scale(preset, n_factors=1)
    heavy = metric_sets.HEAVY

    def compute(panel: pl.DataFrame, specs: tuple[fx.MetricSpec, ...]) -> int:
        col = factor_columns(panel)[0]
        metric_instances = {s.name: getattr(fx.metrics, s.name)() for s in specs}
        fx.evaluate(
            panel,
            metrics=metric_instances,
            factor_cols=[col],
            forward_periods=DEFAULT_FORWARD_PERIODS,
        )
        ic_df = compute_ic(panel, factor_cols=[col])[col]
        bootstrap_mean_ci(ic_df["ic"].to_numpy(), n_bootstrap=BOOTSTRAP_N, seed=seed)
        return 1

    return run_continuous_scenario(
        scenario_id="S1",
        metric_set=heavy,
        scale=scale,
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


# ---------------------------------------------------------------------------
# S2 / S3 — fixed-N factor screening with `core`
# ---------------------------------------------------------------------------


def _screen(
    output: Path | None,
    *,
    scenario_id: str,
    n_factors: int,
    preset: str,
    seed: int,
    cache_state: CacheState,
    threads: int = 1,
) -> list[BenchRecord]:
    scale = resolve_scale(preset, n_factors=n_factors)
    core = metric_sets.CORE

    def compute(panel: pl.DataFrame, specs: tuple[fx.MetricSpec, ...]) -> int:
        metric_instances = {s.name: getattr(fx.metrics, s.name)() for s in specs}
        results = fx.evaluate(
            panel,
            metrics=metric_instances,
            factor_cols=factor_columns(panel),
            forward_periods=DEFAULT_FORWARD_PERIODS,
        )
        return len(results)

    return run_continuous_scenario(
        scenario_id=scenario_id,
        metric_set=core,
        scale=scale,
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def s2_screen_50(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """S2: 50-factor screening with the `core` metric set."""
    n = min(50, resolve_scale(preset).n_factors)  # tiny preset caps at 8
    return _screen(
        output,
        scenario_id="S2",
        n_factors=n,
        preset=preset,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def s3_screen_200(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """S3: 200-factor screening with the `core` metric set."""
    n = min(200, resolve_scale(preset).n_factors)
    return _screen(
        output,
        scenario_id="S3",
        n_factors=n,
        preset=preset,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


# ---------------------------------------------------------------------------
# P1 — scaling probe at 100 / 200 / 500 factors
# ---------------------------------------------------------------------------


def p1_scaling_probe(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """Scaling probe — emits one record per scale step.

    Step values are 100 / 200 / 500 factors at `small` and above;
    `tiny` proportionally shrinks to keep the probe under a second.
    """
    max_factors = resolve_scale(preset).n_factors
    if max_factors >= 500:
        steps = [100, 200, 500]
    elif max_factors >= 50:
        steps = [10, 25, max_factors]
    else:
        # `tiny` (8 factors) — three observable points within the cap.
        steps = [max(2, max_factors // 4), max(3, max_factors // 2), max_factors]

    all_records: list[BenchRecord] = []
    for step in steps:
        # Collect records without writing; aggregate and write once
        # at the end so the JSONL holds all sub-runs and the harness
        # self-validates the aggregated file exactly once.
        records = _screen(
            None,
            scenario_id="P1",
            n_factors=step,
            preset=preset,
            seed=seed,
            cache_state=cache_state,
            threads=threads,
        )
        all_records.extend(records)

    write_and_validate(output, all_records)
    return all_records


# ---------------------------------------------------------------------------
# S6 — evaluate batch scaling (IC cell, cross-factor stage-1 share)
# ---------------------------------------------------------------------------


def _evaluate_batch(
    output: Path | None,
    *,
    scenario_id: str,
    n_factors: int,
    preset: str,
    seed: int,
    cache_state: CacheState,
    threads: int = 1,
) -> list[BenchRecord]:
    scale = resolve_scale(preset, n_factors=n_factors)

    def compute(panel: pl.DataFrame, specs: tuple[fx.MetricSpec, ...]) -> int:
        metric_instances = {s.name: getattr(fx.metrics, s.name)() for s in specs}
        results = fx.evaluate(
            panel,
            metrics=metric_instances,
            factor_cols=factor_columns(panel),
            forward_periods=DEFAULT_FORWARD_PERIODS,
        )
        return len(results)

    return run_continuous_scenario(
        scenario_id=scenario_id,
        metric_set=metric_sets.CORE,
        scale=scale,
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def s6_evaluate_batch(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """S6: ``fx.evaluate`` batch over K factors at the IC cell.

    Scales like ``P1`` (one record per K step) so the cross-factor
    IC stage-1 share added in #426 surfaces as a sub-linear
    ``compute_s`` curve vs the pre-#426 per-factor loop baseline.
    """
    max_factors = resolve_scale(preset).n_factors
    if max_factors >= 500:
        steps = [50, 100, 500]
    elif max_factors >= 50:
        steps = [10, 25, max_factors]
    else:
        steps = [max(2, max_factors // 4), max(3, max_factors // 2), max_factors]

    all_records: list[BenchRecord] = []
    for step in steps:
        records = _evaluate_batch(
            None,
            scenario_id="S6",
            n_factors=step,
            preset=preset,
            seed=seed,
            cache_state=cache_state,
            threads=threads,
        )
        all_records.extend(records)

    write_and_validate(output, all_records)
    return all_records


# ---------------------------------------------------------------------------
# Per-metric micros — attribute compute cost to a single metric
# ---------------------------------------------------------------------------

_MICRO_FACTORS = 50  # capped by preset for tiny smoke runs


def _micro(
    output: Path,
    *,
    scenario_id: str,
    metric_name: str,
    preset: str,
    seed: int,
    cache_state: CacheState,
    threads: int = 1,
) -> list[BenchRecord]:
    max_factors = resolve_scale(preset).n_factors
    n = min(_MICRO_FACTORS, max_factors)
    scale = resolve_scale(preset, n_factors=n)
    # Micros use the metric name as `metric_set` label so downstream
    # aggregation can distinguish "ran the whole `core` bundle" (S2/S3)
    # from "ran a single metric to attribute its cost". Stuffing both
    # under `core` would let an aggregator double-count M-ic + S2 as
    # combined core cost when M-ic is a strict subset of S2's work.
    label = MetricSet(name=metric_name, metric_specs=(fx.spec_by_name()[metric_name],))

    def compute(panel: pl.DataFrame, specs: tuple[fx.MetricSpec, ...]) -> int:
        metric_instances = {s.name: getattr(fx.metrics, s.name)() for s in specs}
        results = fx.evaluate(
            panel,
            metrics=metric_instances,
            factor_cols=factor_columns(panel),
            forward_periods=DEFAULT_FORWARD_PERIODS,
        )
        return len(results)

    return run_continuous_scenario(
        scenario_id=scenario_id,
        metric_set=label,
        scale=scale,
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def m_ic(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """M-ic: cost of ``ic`` alone (no bootstrap)."""
    return _micro(
        output,
        scenario_id="M-ic",
        metric_name="ic",
        preset=preset,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def m_quantile(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """M-quantile: cost of ``quantile_spread`` alone."""
    return _micro(
        output,
        scenario_id="M-quantile",
        metric_name="quantile_spread",
        preset=preset,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def m_monotonicity(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """M-mono: cost of ``monotonicity`` alone."""
    return _micro(
        output,
        scenario_id="M-mono",
        metric_name="monotonicity",
        preset=preset,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def m_ic_bootstrap(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """Cost of the bootstrap path on a per-factor IC series.

    Unlike the other single-metric scenarios this one does **not** go
    through ``evaluate``; it times ``compute_ic`` +
    ``bootstrap_mean_ci`` directly so the bootstrap cost is
    attributable separately from the IC computation.
    """
    max_factors = resolve_scale(preset).n_factors
    n = min(_MICRO_FACTORS, max_factors)
    scale = resolve_scale(preset, n_factors=n)
    # Single-metric attribution: label by what is actually being
    # timed (compute_ic + bootstrap_mean_ci), parallel to the other
    # micros. The `heavy` bundle is reserved for S1 which times the
    # full evaluate + bootstrap path together.
    label = MetricSet(name="ic_bootstrap", metric_specs=())

    def compute(panel: pl.DataFrame, _specs: tuple[fx.MetricSpec, ...]) -> int:
        return _bootstrap_ic_per_factor(panel, factor_columns(panel), seed=seed)

    return run_continuous_scenario(
        scenario_id="M-ic-boot",
        metric_set=label,
        scale=scale,
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


SCENARIOS = {
    "S1": s1_evaluate,
    "S2": s2_screen_50,
    "S3": s3_screen_200,
    "S6": s6_evaluate_batch,
    "P1": p1_scaling_probe,
    "M-ic": m_ic,
    "M-ic-boot": m_ic_bootstrap,
    "M-quantile": m_quantile,
    "M-mono": m_monotonicity,
}


__all__ = [
    "SCENARIOS",
    "m_ic",
    "m_ic_bootstrap",
    "m_monotonicity",
    "m_quantile",
    "p1_scaling_probe",
    "s1_evaluate",
    "s2_screen_50",
    "s3_screen_200",
    "s6_evaluate_batch",
]
