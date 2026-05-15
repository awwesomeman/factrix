"""Shared scaffolding for the continuous-cell scenarios.

Centralises:

- Scale presets (``tiny`` for tests / CI smoke; ``small`` / ``large``
  baseline-grade scales pinned per #380 §7). Per-scenario overrides
  win over preset defaults so fixed-scale scenarios (S2 = 50 factors)
  stay fixed regardless of preset.
- Panel construction: continuous multi-factor panel + forward return
  attachment in one place, so every scenario sees the same seed
  discipline.
- ``run_continuous_scenario`` — single entry that wires preflight →
  measure → write + self-validate. Each scenario module just declares
  ``metric_set``, ``scale``, and a ``compute`` callable.
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
from bench.schema import BenchRecord, CacheState
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


# Scale presets per #380 §7. Concrete `small` / `large` values land
# here so #382-A / #382-C share a single source of truth; the CLI
# (sub-PR #382-C) merely picks a preset by name.
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
    """Run a continuous × individual scenario end-to-end.

    ``compute`` receives the prepared panel and a default config; what
    it does inside (``run_metrics(metrics=...)``, direct algo call,
    bootstrap primitives) is up to the scenario — the helper only
    knows about timing + JSONL + validation.

    Returns the produced records. The harness writes JSONL to
    ``output`` and self-validates; a validation failure raises.
    """
    pre = preflight(threads=threads, seed=seed)
    scale_dict = scale.as_scale_field()

    def setup() -> tuple[pl.DataFrame, fx.AnalysisConfig]:
        return build_panel(scale, seed=seed), make_cfg()

    def compute_step(artifact: tuple[pl.DataFrame, fx.AnalysisConfig]) -> Any:
        panel, cfg = artifact
        return compute(panel, cfg)

    records: list[BenchRecord] = []
    if warmup:
        records.append(
            measure(
                setup,
                compute_step,
                scenario_id=scenario_id,
                axis_cell=AXIS_CELL_CONT_IND,
                scale=scale_dict,
                metric_set=metric_set.name,
                is_warmup=True,
                cache_state=cache_state,
                env=pre.env,
            )
        )
    records.append(
        measure(
            setup,
            compute_step,
            scenario_id=scenario_id,
            axis_cell=AXIS_CELL_CONT_IND,
            scale=scale_dict,
            metric_set=metric_set.name,
            is_warmup=False,
            cache_state=cache_state,
            env=pre.env,
        )
    )
    write_records(output, records)
    report = validate_file(output)
    if not report.ok:
        raise RuntimeError(f"self-validation failed: {report.failures}")
    return records
