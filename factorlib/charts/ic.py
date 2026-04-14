"""IC-related charts: cumulative, rolling, distribution."""

from __future__ import annotations

import polars as pl
import plotly.graph_objects as go


def cumulative_ic_chart(
    ic_series: pl.DataFrame,
    oos_ratio: float = 0.2,
) -> go.Figure:
    """Cumulative IC curve with IS/OOS split.

    Args:
        ic_series: Output of ``compute_ic()`` — columns ``date, ic``.
        oos_ratio: Fraction of data reserved for OOS (default 0.2).
    """
    df = ic_series.sort("date")
    cum_ic = df["ic"].cum_sum()
    dates = df["date"]

    split_idx = int(len(df) * (1 - oos_ratio))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates[:split_idx + 1], y=cum_ic[:split_idx + 1],
        name="IS", line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates[split_idx:], y=cum_ic[split_idx:],
        name="OOS", line=dict(color="#EF553B", width=2),
    ))
    fig.add_shape(
        type="line", x0=dates[split_idx], x1=dates[split_idx],
        y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dash", width=1),
    )
    fig.add_annotation(
        x=dates[split_idx], y=1, yref="paper",
        text="OOS", showarrow=False, font=dict(color="red", size=11),
    )
    fig.update_layout(
        title="Cumulative IC",
        yaxis_title="Cumulative IC",
        height=380,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def rolling_ic_chart(
    ic_series: pl.DataFrame,
    window: int = 63,
) -> go.Figure:
    """Rolling mean IC with zero reference line.

    Args:
        ic_series: Output of ``compute_ic()`` — columns ``date, ic``.
        window: Rolling window size (default 63 ~ 3 months).
    """
    df = ic_series.sort("date")
    rolling = df.with_columns(
        pl.col("ic").rolling_mean(window_size=window).alias("rolling_ic"),
    ).drop_nulls("rolling_ic")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling["date"], y=rolling["rolling_ic"],
        name=f"Rolling IC ({window}D)",
        line=dict(color="#00CC96", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,150,0.08)",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title=f"Rolling IC ({window}-day mean)",
        yaxis_title="IC",
        height=350,
        margin=dict(t=40, b=30),
    )
    return fig


def ic_distribution_chart(ic_series: pl.DataFrame) -> go.Figure:
    """IC value histogram with mean/median markers.

    Args:
        ic_series: Output of ``compute_ic()`` — columns ``date, ic``.
    """
    ic_col = ic_series["ic"].drop_nulls()
    mean_ic = float(ic_col.mean())
    median_ic = float(ic_col.median())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ic_col, nbinsx=40, name="IC",
        marker_color="rgba(99,110,250,0.6)",
    ))
    fig.add_vline(x=mean_ic, line=dict(color="red", width=2),
                  annotation_text=f"mean={mean_ic:.3f}")
    fig.add_vline(x=median_ic, line=dict(color="orange", dash="dash", width=2),
                  annotation_text=f"median={median_ic:.3f}")
    fig.add_vline(x=0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(
        title="IC Distribution",
        xaxis_title="IC", yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=30),
    )
    return fig
