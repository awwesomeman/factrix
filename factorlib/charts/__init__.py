"""factorlib.charts — Plotly chart builders for factor analysis.

Requires: ``pip install factorlib[charts]``

Each function accepts tool outputs and returns a ``plotly.graph_objects.Figure``.
Charts do NO computation — metrics compute, charts visualize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from factorlib.evaluation._protocol import EvaluationResult


def _require_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "Charts require plotly. Install with: pip install factorlib[charts]"
        ) from None


def report_charts(result: EvaluationResult) -> dict[str, go.Figure]:
    """Generate standard charts from an evaluation result.

    Args:
        result: Output of ``evaluate()`` — must have ``artifacts``.

    Returns:
        Mapping of chart name → plotly Figure.
    """
    _require_plotly()

    from factorlib.charts.ic import cumulative_ic_chart, ic_distribution_chart
    from factorlib.charts.quantile import (
        quantile_return_chart,
        spread_time_series_chart,
    )
    from factorlib.metrics.quantile import compute_group_returns

    if result.artifacts is None:
        raise ValueError(
            "result.artifacts is None — cannot generate charts. "
            "Use fl.evaluate() which attaches artifacts."
        )

    figs: dict[str, go.Figure] = {}

    ic_series = result.artifacts.get("ic_series")
    spread_series = result.artifacts.get("spread_series")
    config = result.artifacts.config

    figs["cumulative_ic"] = cumulative_ic_chart(ic_series)
    figs["ic_distribution"] = ic_distribution_chart(ic_series)
    figs["spread_ts"] = spread_time_series_chart(spread_series)

    from factorlib.config import CrossSectionalConfig
    if isinstance(config, CrossSectionalConfig):
        group_returns = compute_group_returns(
            result.artifacts.prepared,
            forward_periods=config.forward_periods,
            n_groups=config.n_groups,
        )
        figs["quantile_returns"] = quantile_return_chart(group_returns)

    return figs
