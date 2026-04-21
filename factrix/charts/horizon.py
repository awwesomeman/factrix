"""Multi-horizon IC chart."""

from __future__ import annotations

import plotly.graph_objects as go


def multi_horizon_ic_chart(per_horizon: dict[int, dict]) -> go.Figure:
    """Bar chart of mean IC at each forward horizon.

    Args:
        per_horizon: Mapping of period → detail dict (with ``mean_ic`` key).
            Typically from ``multi_horizon_ic().metadata["per_horizon"]``.
    """
    periods = sorted(per_horizon.keys())
    labels = [str(p) for p in periods]
    values = [per_horizon[p]["mean_ic"] for p in periods]

    colors = ["#EF553B" if v < 0 else "#636EFA" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="Multi-Horizon IC",
        xaxis_title="Forward Period",
        yaxis_title="Mean IC",
        height=380,
        margin=dict(t=40, b=30),
    )
    return fig
