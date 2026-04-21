"""Quantile-related charts: group returns, spread time series."""

from __future__ import annotations

import math

import polars as pl
import plotly.graph_objects as go


def _fmt_bar_label(v: float) -> str:
    # NaN → em-dash matches reporting.describe_profile_values so the two
    # renderers agree; "< 0" check would crash on NaN otherwise.
    return "—" if v is None or math.isnan(v) else f"{v:.4f}"


def quantile_return_chart(group_returns: pl.DataFrame) -> go.Figure:
    """Bar chart of mean return per quantile group.

    Args:
        group_returns: Output of ``compute_group_returns()``
            — columns ``group, mean_return``.
    """
    df = group_returns.sort("group")
    labels = [f"Q{g + 1}" for g in df["group"].to_list()]
    values = df["mean_return"].to_list()

    colors = [
        "gray" if v is None or math.isnan(v)
        else ("#EF553B" if v < 0 else "#636EFA")
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[_fmt_bar_label(v) for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="Mean Return by Quantile Group",
        yaxis_title="Mean Return (per period)",
        height=380,
        margin=dict(t=40, b=30),
    )
    return fig


def spread_time_series_chart(spread_series: pl.DataFrame) -> go.Figure:
    """Long-short spread over time with top and bottom bucket returns.

    Args:
        spread_series: Output of ``compute_spread_series()``
            — columns ``date, spread, top_return, bottom_return, universe_return``.
    """
    df = spread_series.sort("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["spread"],
        name="Long-Short Spread", line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["top_return"],
        name="Top", line=dict(color="#00CC96", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bottom_return"],
        name="Bottom", line=dict(color="#EF553B", width=1, dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="Long-Short Spread Time Series",
        yaxis_title="Return (per period)",
        height=380,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig
