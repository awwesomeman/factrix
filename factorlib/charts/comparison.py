"""Generic comparison charts for multi-group analysis.

Works with any grouping dimension (universe, time period, factor).
Each function accepts ``dict[str, pl.DataFrame]`` — group name to data.
"""

from __future__ import annotations

import polars as pl
import plotly.graph_objects as go

# Plotly default qualitative color sequence
_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]


def compare_line_chart(
    groups: dict[str, pl.DataFrame],
    x_col: str = "date",
    y_col: str = "value",
    cumulative: bool = False,
    title: str = "",
) -> go.Figure:
    """Overlay line chart comparing multiple groups.

    Args:
        groups: Mapping of group name → DataFrame with ``x_col`` and ``y_col``.
        x_col: Column for x-axis (default ``"date"``).
        y_col: Column for y-axis (default ``"value"``).
        cumulative: If True, plot cumulative sum of ``y_col``.
        title: Chart title.

    Returns:
        Plotly Figure with one line per group.
    """
    fig = go.Figure()

    for i, (name, df) in enumerate(groups.items()):
        x = df[x_col]
        y = df[y_col].cum_sum() if cumulative else df[y_col]
        color = _COLORS[i % len(_COLORS)]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=name,
            line=dict(color=color, width=2),
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title=title,
        yaxis_title=f"{'Cumulative ' if cumulative else ''}{y_col}",
        height=400,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def compare_bar_chart(
    groups: dict[str, pl.DataFrame],
    x_col: str = "group",
    y_col: str = "mean_return",
    title: str = "",
) -> go.Figure:
    """Grouped bar chart comparing multiple groups.

    Args:
        groups: Mapping of group name → DataFrame with ``x_col`` and ``y_col``.
        x_col: Column for x-axis categories (default ``"group"``).
        y_col: Column for bar values (default ``"mean_return"``).
        title: Chart title.

    Returns:
        Plotly Figure with grouped bars.
    """
    fig = go.Figure()

    for i, (name, df) in enumerate(groups.items()):
        x_vals = df[x_col].to_list()
        x_labels = [f"Q{v + 1}" if isinstance(v, int) else str(v) for v in x_vals]
        y_vals = df[y_col]
        color = _COLORS[i % len(_COLORS)]

        fig.add_trace(go.Bar(
            x=x_labels, y=y_vals,
            name=name,
            marker_color=color,
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title=y_col,
        height=400,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig
