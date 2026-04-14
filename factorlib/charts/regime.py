"""Regime IC chart."""

from __future__ import annotations

import math

import plotly.graph_objects as go


def regime_ic_chart(per_regime: dict[str, dict]) -> go.Figure:
    """Mean IC per regime with 95% confidence interval error bars.

    Args:
        per_regime: Mapping of regime name -> stats dict.
            Typically from ``regime_ic().metadata["per_regime"]``.
            Each value must contain ``mean_ic``, ``n_periods``, and
            optionally ``std_ic``.
    """
    names = sorted(per_regime.keys())
    means = [per_regime[n]["mean_ic"] for n in names]
    n_periods = [per_regime[n]["n_periods"] for n in names]

    # WHY: compute SE = std / sqrt(n) directly from per-regime stats
    # when std_ic is available; otherwise derive from mean_ic and t_stat
    errors = []
    for n in names:
        d = per_regime[n]
        np_ = d["n_periods"]
        if "std_ic" in d and np_ > 1:
            se = d["std_ic"] / math.sqrt(np_)
        elif abs(d.get("t_stat", 0)) > 0.01 and np_ > 1:
            se = abs(d["mean_ic"] / d["t_stat"])
        else:
            se = 0.0
        errors.append(1.96 * se)

    colors = ["#EF553B" if m < 0 else "#636EFA" for m in means]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=errors, visible=True),
        marker_color=colors,
        text=[f"n={n}" for n in n_periods],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="IC by Regime",
        xaxis_title="Regime",
        yaxis_title="Mean IC",
        height=380,
        margin=dict(t=40, b=30),
    )
    return fig
