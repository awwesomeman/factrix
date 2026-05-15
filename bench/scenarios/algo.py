"""Algo scenarios — ``greedy_forward_selection`` (#380 §4 S4).

`greedy_forward_selection` does not live behind ``factrix.run_metrics``;
it consumes a ``dict[str, pl.DataFrame]`` of per-factor spread series.
The scenario therefore splits the work explicitly:

- **setup** builds the panel, then computes each factor's spread series
  via ``factrix.metrics.quantile.compute_spread_series``. Spread
  construction (rank → bucket → spread per date) is itself non-trivial
  but is the algorithm's prerequisite, not its hotspot — accounting
  it under ``setup_s`` matches the §9 distinction between
  preparation cost and the work under measurement.
- **compute** runs the greedy + backward-elimination loop. This is
  the per-iteration O(K²) hotspot #378 is targeting.

`suppress_snooping_warning=True` is set unconditionally — the bench
runs are not inference, and emitting one warning per scenario per
preset would flood the harness log.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import factrix as fx
import polars as pl
from factrix.metrics.quantile import compute_spread_series

from bench.metric_sets import ALGO
from bench.preflight import preflight
from bench.scenarios._helpers import (
    AXIS_CELL_CONT_IND,
    DEFAULT_FORWARD_PERIODS,
    build_panel,
    factor_columns,
    resolve_scale,
    run_scenario,
)
from bench.schema import BenchRecord, CacheState

# Greedy selection runs to convergence by default; cap candidates so
# the inner loop terminates predictably under the benchmark budget.
# #380 §4 pins S4 at "50 candidates"; capped further by preset for
# tiny smoke runs.
_S4_CANDIDATES = 50

# Long-short bucket count for the spread series fed into spanning.
# Pinned (not at call site) so a #378 sub-task cannot silently shift
# the workload by tuning quantile granularity.
_N_GROUPS = 5


def _build_spread_dict(
    panel: pl.DataFrame, factors: list[str]
) -> dict[str, pl.DataFrame]:
    """Per-factor ``(date, spread)`` series, ready for spanning."""
    spreads: dict[str, pl.DataFrame] = {}
    for col in factors:
        series = compute_spread_series(
            panel,
            forward_periods=DEFAULT_FORWARD_PERIODS,
            factor_col=col,
            n_groups=_N_GROUPS,
        )
        spreads[col] = series.select(["date", "spread"])
    return spreads


def s4_greedy_forward_selection(
    output: Path,
    *,
    preset: str = "tiny",
    seed: int = 0,
    cache_state: CacheState = "warm",
) -> list[BenchRecord]:
    """S4: greedy forward selection over a candidate factor pool.

    Candidate pool size follows the preset (capped at 50 per #380 §4).
    The base set is empty — every selected factor competes on its
    own merit, matching the §4 "greedy forward selection" framing.
    """
    max_factors = resolve_scale(preset).n_factors
    n = min(_S4_CANDIDATES, max_factors)
    scale = resolve_scale(preset, n_factors=n)
    pre = preflight(threads=1, seed=seed)

    def setup() -> dict[str, pl.DataFrame]:
        panel = build_panel(scale, seed=seed)
        return _build_spread_dict(panel, factor_columns(panel))

    def compute(spreads: dict[str, pl.DataFrame]) -> int:
        with warnings.catch_warnings():
            # Snooping warning is one-shot per process; the algo
            # also emits sample-floor warnings on tiny presets that
            # are not the work under measurement.
            warnings.simplefilter("ignore", UserWarning)
            result = fx.metrics.spanning.greedy_forward_selection(
                spreads,
                significance_threshold=2.0,
                max_factors=n,
                suppress_snooping_warning=True,
            )
        return len(result.selected_factors)

    return run_scenario(
        scenario_id="S4",
        axis_cell=AXIS_CELL_CONT_IND,
        metric_set=ALGO,
        scale=scale.as_scale_field(),
        setup=setup,
        compute=compute,
        output=output,
        env=pre.env,
        cache_state=cache_state,
    )


SCENARIOS = {"S4": s4_greedy_forward_selection}


__all__ = ["SCENARIOS", "s4_greedy_forward_selection"]
