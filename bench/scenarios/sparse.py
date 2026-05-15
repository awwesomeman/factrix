"""Sparse × Individual scenarios (#380 §4 S5 + M-corrado).

The sparse cell exercises a different factrix code path from the
continuous one: ``corrado_rank_test`` is loop-heavy with a
permutation-style bootstrap, ``compute_caar`` and ``compute_mfe_mae``
are direct (not behind ``run_metrics``) and run on the same panel.

`compute_caar` / `compute_mfe_mae` consume the canonical event panel
(``date, asset_id, factor, forward_return`` with sparse ``factor``)
directly — not the per-event-row ``compute_event_returns`` output —
so the setup phase is just panel construction; the compute phase
runs all three metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import factrix as fx
import polars as pl
from factrix.metrics.caar import caar, compute_caar
from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae_summary

from bench.metric_sets import EVENT, MetricSet
from bench.preflight import preflight
from bench.scenarios._helpers import (
    AXIS_CELL_SPARSE_IND,
    SparseScale,
    build_event_panel,
    count_events,
    resolve_sparse_scale,
    run_scenario,
)
from bench.schema import BenchRecord, CacheState

# MFE/MAE window parameters — pinned so a #378 sub-task tuning the
# windows cannot silently shift the workload. Values match factrix's
# function defaults at small/large preset; tiny shrinks via scenario
# overrides below.
_MFE_WINDOW = 10
_MFE_ESTIMATION_WINDOW = 30


def _run_event_bundle(panel: pl.DataFrame, cfg: fx.AnalysisConfig) -> int:
    """Run the three event-cell metrics on the panel."""
    fx.run_metrics(panel, cfg, metrics=["corrado_rank_test"])
    caar(compute_caar(panel))
    mfe_mae_summary(
        compute_mfe_mae(
            panel, window=_MFE_WINDOW, estimation_window=_MFE_ESTIMATION_WINDOW
        )
    )
    return 3


def _run_sparse_scenario(
    output: Path,
    *,
    scenario_id: str,
    metric_set: MetricSet,
    scale: SparseScale,
    compute: Callable[[pl.DataFrame, fx.AnalysisConfig], int],
    seed: int,
    cache_state: CacheState,
) -> list[BenchRecord]:
    pre = preflight(threads=1, seed=seed)

    # Probe build once to read the realised event count for the scale
    # field; measure rebuilds the panel inside `setup` for accurate
    # `setup_s` timing. The probe is deterministic given the seed, so
    # the count matches the measured panel's count by construction.
    n_events = count_events(build_event_panel(scale, seed=seed))
    scale_dict = scale.as_scale_field(n_events=n_events)

    def setup() -> tuple[pl.DataFrame, fx.AnalysisConfig]:
        panel = build_event_panel(scale, seed=seed)
        return panel, fx.AnalysisConfig.individual_sparse(forward_periods=5)

    def compute_step(artifact: tuple[pl.DataFrame, fx.AnalysisConfig]) -> int:
        panel, cfg = artifact
        return compute(panel, cfg)

    return run_scenario(
        scenario_id=scenario_id,
        axis_cell=AXIS_CELL_SPARSE_IND,
        metric_set=metric_set,
        scale=scale_dict,
        setup=setup,
        compute=compute_step,
        output=output,
        env=pre.env,
        cache_state=cache_state,
    )


def s5_event_study(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
) -> list[BenchRecord]:
    """S5: event-study bundle (corrado + caar + mfe_mae)."""
    scale = resolve_sparse_scale(preset)
    return _run_sparse_scenario(
        output,
        scenario_id="S5",
        metric_set=EVENT,
        scale=scale,
        compute=_run_event_bundle,
        seed=seed,
        cache_state=cache_state,
    )


def m_corrado(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
) -> list[BenchRecord]:
    """M-corrado: cost of ``corrado_rank_test`` alone on the sparse cell.

    Attributes the permutation-bootstrap cost to a single metric,
    parallel to the M-ic / M-quantile / M-mono pattern on the
    continuous cell.
    """
    scale = resolve_sparse_scale(preset)
    label = MetricSet(
        name="corrado_rank_test", run_metrics_names=("corrado_rank_test",)
    )

    def compute(panel: pl.DataFrame, cfg: fx.AnalysisConfig) -> int:
        fx.run_metrics(panel, cfg, metrics=["corrado_rank_test"])
        return 1

    return _run_sparse_scenario(
        output,
        scenario_id="M-corrado",
        metric_set=label,
        scale=scale,
        compute=compute,
        seed=seed,
        cache_state=cache_state,
    )


SCENARIOS = {"S5": s5_event_study, "M-corrado": m_corrado}


__all__ = ["SCENARIOS", "m_corrado", "s5_event_study"]
