"""Pipeline configuration and market presets.

Gate thresholds (significance_threshold, oos_decay_threshold) are NOT here.
They live on the gate functions and are bound via ``functools.partial``.
This keeps per-gate tuning separate from data/pipeline settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class PipelineConfig:
    """Settings for ``evaluate_factor``.

    Attributes:
        forward_periods: Forward return horizon (default 5).
        return_clip_pct: Percentile bounds for return winsorization.
        mad_n: Number of MADs for factor winsorization (0 = disabled).
        orthogonalize: Whether Step 6 (factor orthogonalization) was applied.
        base_factors: Base factor panel for orthogonalization (if applicable).
        n_groups: Number of quantile groups for spread / monotonicity.
        q_top: Fraction of stocks in Q1 for concentration.
        multi_horizon_periods: Forward horizons for multi-horizon IC.
        estimated_cost_bps: Estimated single-leg trading cost in bps.
    """

    forward_periods: int = 5
    return_clip_pct: tuple[float, float] = (0.01, 0.99)
    mad_n: float = 3.0
    orthogonalize: bool = False
    base_factors: pl.DataFrame | None = None
    n_groups: int = 10
    q_top: float = 0.2
    multi_horizon_periods: list[int] = field(
        default_factory=lambda: [1, 5, 10, 20],
    )
    estimated_cost_bps: float = 30.0


MARKET_DEFAULTS: dict[str, dict[str, object]] = {
    "tw": {
        "estimated_cost_bps": 30,
        "ortho_factors": [
            "size", "value", "momentum", "industry_tse30",
        ],
    },
    "us": {
        "estimated_cost_bps": 5,
        "ortho_factors": [
            "size", "value", "momentum", "industry_gics",
        ],
    },
}
