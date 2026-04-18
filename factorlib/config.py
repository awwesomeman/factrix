"""Per-type pipeline configuration.

Gate thresholds (significance_threshold, oos_decay_threshold) are NOT here.
They live on the gate functions and are bound via ``functools.partial``.
This keeps per-gate tuning separate from data/pipeline settings.

Users should instantiate one of the concrete subclasses
(CrossSectionalConfig, EventConfig, ...), never BaseConfig directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from factorlib._types import FactorType

if TYPE_CHECKING:
    import polars as pl

ClusteringAdjustment = Literal[
    "none", "calendar_block_bootstrap", "kolari_pynnonen"
]


@dataclass(kw_only=True)
class BaseConfig:
    """Shared settings across all factor types.

    Not user-facing — use a concrete subclass.
    """

    forward_periods: int = 5
    n_groups: int = 5
    estimated_cost_bps: float = 30.0

    def __post_init__(self) -> None:
        if type(self) is BaseConfig:
            raise TypeError(
                "BaseConfig cannot be used directly. "
                "Use a concrete subclass: CrossSectionalConfig(), "
                "EventConfig(), MacroPanelConfig(), MacroCommonConfig()."
            )


@dataclass(kw_only=True)
class CrossSectionalConfig(BaseConfig):
    """Config for cross-sectional factors (individual stock selection)."""

    factor_type: ClassVar[FactorType] = FactorType.CROSS_SECTIONAL

    n_groups: int = 10  # override BaseConfig default
    q_top: float = 0.2
    mad_n: float = 3.0
    return_clip_pct: tuple[float, float] = (0.01, 0.99)

    # --- Orthogonalization (Step 6, opt-in) ---
    # Pass a DataFrame with (date, asset_id, *base_cols). When supplied,
    # the z-scored factor is regressed per-date against base_cols and
    # replaced by the (MAD-winsorized, re-z-scored) residual. See
    # docs/spike_orthogonalize.md for the full policy rationale.
    orthogonalize: "pl.DataFrame | None" = None
    orthogonalize_cols: list[str] | None = None
    # Fraction of factor rows that must have basis coverage after the
    # inner join; below this, the pipeline raises rather than silently
    # mixing residuals with originals. 1.0 = require every row; 0.0
    # disables the gate (not recommended).
    orthogonalize_min_coverage: float = 0.95

    # --- Level 2 metric inputs (T3.S2, opt-in) ---
    # Consumed by _build_cs_artifacts; None leaves the corresponding
    # Profile fields as None. See docs/spike_level2_profile_integration.md.
    #
    # regime_labels: DataFrame with (date, regime). Set this explicitly
    # (e.g. labels for {bull, bear, high_vol}) to populate
    # regime_ic_min_tstat / regime_ic_consistent. None = skip;
    # regime_ic's own time-bisection fallback is Level 2 escape only.
    regime_labels: "pl.DataFrame | None" = None
    # multi_horizon_periods: horizons passed to multi_horizon_ic.
    # None = skip (opt-in). Pass an explicit list (e.g. [1, 5, 10, 20])
    # to populate multi_horizon_ic_retention / _monotonic on Profile.
    # Cost: one full IC pass per horizon — not free on large panels.
    multi_horizon_periods: list[int] | None = None
    # spanning_base_spreads: {name -> DataFrame(date, spread)} for the
    # base factor set. None = skip spanning regression.
    spanning_base_spreads: "dict[str, pl.DataFrame] | None" = None


@dataclass(kw_only=True)
class EventConfig(BaseConfig):
    """Config for event signal factors (event-driven trading)."""

    factor_type: ClassVar[FactorType] = FactorType.EVENT_SIGNAL

    event_window_pre: int = 5
    event_window_post: int = 20
    cluster_window: int = 3
    adjust_clustering: ClusteringAdjustment = "none"


@dataclass(kw_only=True)
class MacroPanelConfig(BaseConfig):
    """Config for macro panel factors (cross-country allocation)."""

    factor_type: ClassVar[FactorType] = FactorType.MACRO_PANEL

    n_groups: int = 3  # override: small N needs fewer groups
    demean_cross_section: bool = False
    min_cross_section: int = 10


@dataclass(kw_only=True)
class MacroCommonConfig(BaseConfig):
    """Config for macro common factors (risk attribution)."""

    factor_type: ClassVar[FactorType] = FactorType.MACRO_COMMON

    ts_window: int = 60
    tradable: bool = False


MARKET_DEFAULTS: dict[str, dict[str, object]] = {
    "tw": {"estimated_cost_bps": 30},
    "us": {"estimated_cost_bps": 5},
}
