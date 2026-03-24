"""Factor generators — reusable factor construction functions.

Each module groups related factor types. All generators follow the same
contract: ``(df: pl.DataFrame, ...) -> pl.DataFrame`` with a ``factor``
column appended.
"""

from factorlib.factors.momentum import (
    generate_momentum,
    generate_momentum_60d,
)
from factorlib.factors.volatility import (
    generate_volatility,
    generate_idiosyncratic_vol,
)
from factorlib.factors.liquidity import generate_amihud
from factorlib.factors.technical import (
    generate_mean_reversion,
    generate_52w_high_ratio,
    generate_overnight_return,
    generate_intraday_range,
    generate_rsi,
    generate_volume_price_trend,
    generate_market_beta,
    generate_max_effect,
)
from factorlib.factors.event import generate_event_signal_mock

__all__ = [
    "generate_momentum",
    "generate_momentum_60d",
    "generate_volatility",
    "generate_idiosyncratic_vol",
    "generate_amihud",
    "generate_mean_reversion",
    "generate_52w_high_ratio",
    "generate_overnight_return",
    "generate_intraday_range",
    "generate_rsi",
    "generate_volume_price_trend",
    "generate_market_beta",
    "generate_max_effect",
    "generate_event_signal_mock",
]
