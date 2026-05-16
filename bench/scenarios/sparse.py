"""Sparse × Individual scenarios — event-study workloads.

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

from pathlib import Path

import factrix as fx
import polars as pl
from factrix.metrics.caar import caar, compute_caar
from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae_summary

from bench.metric_sets import EVENT, MetricSet
from bench.scenarios._helpers import (
    resolve_sparse_scale,
    run_sparse_scenario,
)
from bench.schema import BenchRecord, CacheState

# MFE/MAE window parameters — pinned so tuning the windows cannot
# silently shift the workload. Values match factrix's function
# defaults at small/large preset.
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


def s5_event_study(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """Event-study bundle: corrado rank test + CAAR + MFE/MAE."""
    return run_sparse_scenario(
        scenario_id="S5",
        metric_set=EVENT,
        scale=resolve_sparse_scale(preset),
        compute=_run_event_bundle,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


def m_corrado(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
    threads: int = 1,
) -> list[BenchRecord]:
    """Cost of ``corrado_rank_test`` alone on the sparse cell.

    Attributes the permutation-bootstrap cost to a single metric,
    parallel to the single-metric attribution scenarios on the
    continuous cell.
    """
    label = MetricSet(
        name="corrado_rank_test", run_metrics_names=("corrado_rank_test",)
    )

    def compute(panel: pl.DataFrame, cfg: fx.AnalysisConfig) -> int:
        fx.run_metrics(panel, cfg, metrics=["corrado_rank_test"])
        return 1

    return run_sparse_scenario(
        scenario_id="M-corrado",
        metric_set=label,
        scale=resolve_sparse_scale(preset),
        compute=compute,
        output=output,
        seed=seed,
        cache_state=cache_state,
        threads=threads,
    )


SCENARIOS = {"S5": s5_event_study, "M-corrado": m_corrado}


__all__ = ["SCENARIOS", "m_corrado", "s5_event_study"]
