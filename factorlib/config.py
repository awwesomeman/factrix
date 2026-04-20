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

# Quantile-bucket tie-break policy. See _assign_quantile_groups docstring
# in factorlib/metrics/_helpers.py for semantics. Kept narrow (2 values)
# so primitives accept plain ``str`` and point at this alias in docstrings
# — polars' rank(method=...) does the real validation at the leaf.
TiePolicy = Literal["ordinal", "average"]


@dataclass(kw_only=True)
class BaseConfig:
    """Shared settings across all factor types.

    Not user-facing — use a concrete subclass.
    """

    forward_periods: int = 5
    n_groups: int = 5
    estimated_cost_bps: float = 30.0
    # Quantile-bucket tie-break policy; see _assign_quantile_groups.
    tie_policy: TiePolicy = "ordinal"

    def __post_init__(self) -> None:
        if type(self) is BaseConfig:
            raise TypeError(
                "BaseConfig cannot be used directly. "
                "Use a concrete subclass: CrossSectionalConfig(), "
                "EventConfig(), MacroPanelConfig(), MacroCommonConfig()."
            )


@dataclass(kw_only=True)
class OrthoConfig:
    """Orthogonalization inputs for CrossSectionalConfig (Step 6, opt-in).

    When attached to a ``CrossSectionalConfig.ortho``, the z-scored factor
    is regressed per-date against ``base_factors[cols]`` and replaced by
    the (MAD-winsorized, re-z-scored) residual. See
    ``docs/spike_orthogonalize.md`` for the full policy rationale.

    Usage: for the common case (residualize against a basis, defaults
    for everything else), pass the DataFrame directly as
    ``CrossSectionalConfig(ortho=basis_df)`` — ``CrossSectionalConfig``
    normalizes it into ``OrthoConfig(base_factors=basis_df)`` in
    ``__post_init__``. Construct ``OrthoConfig`` explicitly when you
    need to tune ``cols`` or ``min_coverage``.
    """

    # DataFrame with (date, asset_id, *cols). Required — the whole point
    # of constructing an OrthoConfig is to supply these.
    base_factors: "pl.DataFrame"
    # Column subset to regress against; None = all non-key columns.
    cols: list[str] | None = None
    # Fraction of factor rows that must have basis coverage after the
    # inner join; below this, the pipeline raises rather than silently
    # mixing residuals with originals. 1.0 = require every row; 0.0
    # disables the gate (not recommended).
    min_coverage: float = 0.95


@dataclass(kw_only=True)
class CrossSectionalConfig(BaseConfig):
    """Config for cross-sectional factors (individual stock selection).

    Key knobs (inherited from ``BaseConfig``):
        forward_periods: N-period forward return horizon (default 5).
        n_groups: Quantile bucket count for long-short spread (default 10).
        estimated_cost_bps: Per-side trading cost for breakeven analysis.
        tie_policy: ``"ordinal"`` (default) breaks ties by row order →
            balanced group sizes but injects sorting-artifact noise on
            low-cardinality factors (binary signals, ESG buckets, sector
            dummies). Switch to ``"average"`` when a UserWarning fires
            about ``tie_ratio`` — tied assets share a bucket at the cost
            of slightly unbalanced group sizes.

    CS-specific:
        mad_n: MAD winsorize cutoff (default 3σ-equivalent).
        return_clip_pct: Forward-return percentile clip.
        ortho: Optional OrthoConfig (or bare DataFrame) to residualize
            the factor against a basis.
        regime_labels / multi_horizon_periods / spanning_base_spreads:
            Level-2 opt-in metrics (None → field stays None on Profile).
    """

    factor_type: ClassVar[FactorType] = FactorType.CROSS_SECTIONAL

    n_groups: int = 10  # override BaseConfig default
    mad_n: float = 3.0
    return_clip_pct: tuple[float, float] = (0.01, 0.99)

    # See OrthoConfig. Accepts the DataFrame shortcut; __post_init__
    # normalizes so downstream pipeline code sees only OrthoConfig.
    ortho: "OrthoConfig | pl.DataFrame | None" = None

    # Level 2 metrics (T3.S2, opt-in). Each is independently enabled by
    # supplying its input; None leaves the corresponding Profile fields
    # as None. Flat because these three are mutually independent and
    # the sweep pattern ``dataclasses.replace(cfg, regime_labels=…)`` is
    # common — nesting them would force double-``replace`` gymnastics.
    regime_labels: "pl.DataFrame | None" = None
    multi_horizon_periods: list[int] | None = None
    spanning_base_spreads: "dict[str, pl.DataFrame] | None" = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ortho is None or isinstance(self.ortho, OrthoConfig):
            return
        # Local import — avoids paying polars at module-import time while
        # still giving a type-safe DataFrame check here (by the time any
        # Config is constructed, polars is loaded anyway).
        import polars as pl
        if isinstance(self.ortho, pl.DataFrame):
            self.ortho = OrthoConfig(base_factors=self.ortho)
        else:
            raise TypeError(
                f"CrossSectionalConfig.ortho expects OrthoConfig, "
                f"pl.DataFrame, or None; got {type(self.ortho).__name__}."
            )


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
