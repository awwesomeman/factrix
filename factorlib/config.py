"""Per-type pipeline configuration.

Gate thresholds (significance_threshold, oos_decay_threshold) are NOT here.
They live on the gate functions and are bound via ``functools.partial``.
This keeps per-gate tuning separate from data/pipeline settings.

Users should instantiate one of the concrete subclasses
(CrossSectionalConfig, EventConfig, ...), never BaseConfig directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from factorlib._types import FactorType


@dataclass(kw_only=True)
class BaseConfig:
    """Shared settings across all factor types.

    Not user-facing — use a concrete subclass.
    """

    forward_periods: int = 5
    n_groups: int = 5
    estimated_cost_bps: float = 30.0
    multi_horizon_periods: list[int] = field(
        default_factory=lambda: [1, 5, 10, 20],
    )

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
    orthogonalize: bool = False
    mad_n: float = 3.0
    return_clip_pct: tuple[float, float] = (0.01, 0.99)


@dataclass(kw_only=True)
class EventConfig(BaseConfig):
    """Config for event signal factors (event-driven trading)."""

    factor_type: ClassVar[FactorType] = FactorType.EVENT_SIGNAL

    event_window_pre: int = 5
    event_window_post: int = 20
    cluster_window: int = 3
    adjust_clustering: Literal[
        "none", "calendar_block_bootstrap", "kolari_pynnonen"
    ] = "none"


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
