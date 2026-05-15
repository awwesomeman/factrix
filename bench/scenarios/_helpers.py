"""Shared scaffolding for scenario modules.

Centralises:

- Scale presets (``tiny`` for tests / CI smoke; ``small`` / ``large``
  baseline-grade scales pinned per #380 §7). Per-scenario overrides
  win over preset defaults so fixed-scale scenarios (S2 = 50 factors)
  stay fixed regardless of preset.
- Continuous multi-factor panel construction + forward-return
  attachment, so every Cont × Ind scenario sees the same seed
  discipline.
- ``run_scenario`` — generic measure / write / self-validate loop,
  independent of axis cell.
- ``run_continuous_scenario`` — Cont × Ind specialisation that
  builds the panel + config and delegates to ``run_scenario``.
  Algo scenarios call ``run_scenario`` directly because their
  setup phase produces a per-factor spread dict, not the canonical
  (panel, cfg) pair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import factrix as fx
import polars as pl
from factrix.datasets import make_multi_factor_panel
from factrix.preprocess import compute_forward_return

from bench.metric_sets import MetricSet
from bench.preflight import preflight
from bench.schema import BenchRecord, CacheState, Env
from bench.validator import validate_file
from bench.wrapper import measure, write_records

AXIS_CELL_CONT_IND = "continuous_individual_panel"


@dataclass(frozen=True)
class ContinuousScale:
    """Scale specification for the continuous individual panel cell."""

    n_factors: int
    n_dates: int
    n_assets: int

    def as_scale_field(self) -> dict[str, int]:
        return {
            "n_factors": self.n_factors,
            "n_dates": self.n_dates,
            "n_assets": self.n_assets,
        }


# Scale presets per #380 §7. Every scenario module reads these so
# preset-named runs are guaranteed identical in dimensionality
# regardless of which scenarios are exercised.
PRESETS: dict[str, ContinuousScale] = {
    # `tiny` is for tests and the eventual CI bench-tiny smoke — must
    # complete in seconds on a CI runner.
    "tiny": ContinuousScale(n_factors=8, n_assets=20, n_dates=60),
    # `small` — 16 GB laptop baseline (#380 §7).
    "small": ContinuousScale(n_factors=100, n_assets=1000, n_dates=1250),
    # `large` — 32 GB cloud baseline (#380 §7); opt-in, not mandatory.
    "large": ContinuousScale(n_factors=500, n_assets=1000, n_dates=1250),
}


def resolve_scale(
    preset: str,
    *,
    n_factors: int | None = None,
    n_assets: int | None = None,
    n_dates: int | None = None,
) -> ContinuousScale:
    """Pick a preset and apply per-scenario overrides.

    Fixed-scale scenarios (e.g. S2 always 50 factors) pass their
    pinned dimension as an override so the preset only controls the
    free dimensions.
    """
    base = PRESETS[preset]
    return replace(
        base,
        n_factors=n_factors if n_factors is not None else base.n_factors,
        n_assets=n_assets if n_assets is not None else base.n_assets,
        n_dates=n_dates if n_dates is not None else base.n_dates,
    )


# A single horizon is sufficient for benchmark purposes — sweeping
# across horizons is a separate user story not covered by #380.
DEFAULT_FORWARD_PERIODS = 5


def build_panel(scale: ContinuousScale, *, seed: int = 0) -> pl.DataFrame:
    """Generate a continuous multi-factor panel with forward return attached.

    Returned panel is wide (one column per factor) with
    ``forward_return`` already computed against the synthetic
    ``signal_horizon`` so each ``factor_NNNN`` realises the calibrated
    IC at ``forward_periods == DEFAULT_FORWARD_PERIODS``.
    """
    raw = make_multi_factor_panel(
        n_factors=scale.n_factors,
        n_assets=scale.n_assets,
        n_dates=scale.n_dates,
        signal_horizon=DEFAULT_FORWARD_PERIODS,
        seed=seed,
    )
    return compute_forward_return(raw, forward_periods=DEFAULT_FORWARD_PERIODS)


def factor_columns(panel: pl.DataFrame) -> list[str]:
    """Return factor column names in stable iteration order."""
    return sorted(c for c in panel.columns if c.startswith("factor_"))


def make_cfg() -> fx.AnalysisConfig:
    """Canonical continuous × individual config used across scenarios."""
    return fx.AnalysisConfig.individual_continuous(
        forward_periods=DEFAULT_FORWARD_PERIODS
    )


def run_scenario(
    *,
    scenario_id: str,
    axis_cell: str,
    metric_set: MetricSet,
    scale: dict[str, Any],
    setup: Callable[[], Any],
    compute: Callable[[Any], Any],
    output: Path,
    env: Env,
    warmup: bool = True,
    cache_state: CacheState = "warm",
) -> list[BenchRecord]:
    """Run ``setup`` then ``compute`` (warmup + measured), write JSONL,
    self-validate.

    Generic over axis_cell / scale shape — the cell-specific helper
    (e.g. ``run_continuous_scenario``) chooses panel construction,
    config, and scale serialisation, then delegates here for the
    measure / write / validate loop. Mirrors the §9.1 self-validation
    invariant exactly once instead of per-cell.
    """
    records: list[BenchRecord] = []
    if warmup:
        records.append(
            measure(
                setup,
                compute,
                scenario_id=scenario_id,
                axis_cell=axis_cell,
                scale=scale,
                metric_set=metric_set.name,
                is_warmup=True,
                cache_state=cache_state,
                env=env,
            )
        )
    records.append(
        measure(
            setup,
            compute,
            scenario_id=scenario_id,
            axis_cell=axis_cell,
            scale=scale,
            metric_set=metric_set.name,
            is_warmup=False,
            cache_state=cache_state,
            env=env,
        )
    )
    write_records(output, records)
    report = validate_file(output)
    if not report.ok:
        raise RuntimeError(f"self-validation failed: {report.failures}")
    return records


def run_continuous_scenario(
    *,
    scenario_id: str,
    metric_set: MetricSet,
    scale: ContinuousScale,
    compute: Callable[[pl.DataFrame, fx.AnalysisConfig], Any],
    output: Path,
    warmup: bool = True,
    cache_state: CacheState = "warm",
    seed: int = 0,
    threads: int = 1,
) -> list[BenchRecord]:
    """Run a Continuous × Individual scenario end-to-end.

    ``compute`` receives the prepared panel and a default config; what
    it does inside (``run_metrics(metrics=...)``, direct algo call,
    bootstrap primitives) is up to the scenario — the helper only
    knows about timing + JSONL + validation.
    """
    pre = preflight(threads=threads, seed=seed)

    def setup() -> tuple[pl.DataFrame, fx.AnalysisConfig]:
        return build_panel(scale, seed=seed), make_cfg()

    def compute_step(artifact: tuple[pl.DataFrame, fx.AnalysisConfig]) -> Any:
        panel, cfg = artifact
        return compute(panel, cfg)

    return run_scenario(
        scenario_id=scenario_id,
        axis_cell=AXIS_CELL_CONT_IND,
        metric_set=metric_set,
        scale=scale.as_scale_field(),
        setup=setup,
        compute=compute_step,
        output=output,
        env=pre.env,
        warmup=warmup,
        cache_state=cache_state,
    )
