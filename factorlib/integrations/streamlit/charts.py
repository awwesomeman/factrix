"""Dashboard chart builders — all Plotly figure constructors."""

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_EVENT_COLORS = {"Buy": "rgba(0, 204, 150, 0.15)", "Sell": "rgba(239, 83, 59, 0.15)"}
_SIGNAL_RGB = {"Buy": (0, 204, 150), "Sell": (239, 83, 59)}


def radar_chart(metrics: dict, metric_names: list[str], labels: list[str] | None = None) -> go.Figure:
    """Radar chart from MLflow metrics dict."""
    display_labels = labels if labels else metric_names
    values = [metrics.get(m, 0) for m in metric_names]
    values.append(values[0])
    theta = list(display_labels) + [display_labels[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=theta, fill="toself",
        line=dict(color="#636EFA", width=2),
        fillcolor="rgba(99, 110, 250, 0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=10)),
        ),
        showlegend=False,
        height=380,
        margin=dict(t=40, b=40, l=60, r=60),
    )
    return fig


def ic_chart(ic_df: pl.DataFrame, oos_ratio: float = 0.2) -> go.Figure:
    """2-panel IC chart: cumulative IC (top) + rolling IC (bottom).

    WHY: 累積 IC 呈現整體 alpha 累積趨勢；rolling IC 直接呈現「近期因子有效性」
    是否在衰退，兩者互補，同一 x 軸易於對照時間點。
    """
    pdf = ic_df.to_pandas().sort_values("date")
    split_idx = int(len(pdf) * (1 - oos_ratio))
    split_date_str = str(pdf["date"].iloc[split_idx])

    is_data = pdf.iloc[:split_idx + 1]
    oos_data = pdf.iloc[split_idx:]

    has_rolling = "rolling_ic" in pdf.columns and pdf["rolling_ic"].notna().any()

    if has_rolling:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=["Cumulative IC", "Rolling IC (63-day)"],
        )
        row_cum, row_roll = 1, 2
        height = 500
    else:
        fig = make_subplots(rows=1, cols=1)
        row_cum = 1
        height = 350

    # Panel 1: Cumulative IC
    fig.add_trace(go.Scatter(
        x=is_data["date"], y=is_data["cum_ic"],
        name="IS Cum IC", line=dict(color="#636EFA", width=2),
    ), row=row_cum, col=1)
    fig.add_trace(go.Scatter(
        x=oos_data["date"], y=oos_data["cum_ic"],
        name="OOS Cum IC", line=dict(color="#EF553B", width=2),
    ), row=row_cum, col=1)

    # Panel 2: Rolling IC
    if has_rolling:
        rolling = pdf.dropna(subset=["rolling_ic"])
        fig.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_ic"],
            name="Rolling IC", line=dict(color="#00CC96", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0,204,150,0.08)",
        ), row=row_roll, col=1)
        fig.add_hline(y=0, row=row_roll, col=1,
                      line=dict(color="gray", dash="dash", width=1))

    # IS/OOS boundary across both panels (yref="paper" spans full figure height)
    fig.add_shape(
        type="line", x0=split_date_str, x1=split_date_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dash", width=1),
    )
    fig.add_annotation(
        x=split_date_str, y=1, yref="paper",
        text="OOS", showarrow=False,
        font=dict(color="red", size=11), yanchor="bottom",
    )
    fig.update_layout(
        height=height,
        margin=dict(t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


def nav_chart(nav_df: pl.DataFrame) -> go.Figure:
    pdf = nav_df.to_pandas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf["date"], y=pdf["Q1_NAV"],
        name="Q1 (Top 20%)", line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pdf["date"], y=pdf["Universe_NAV"],
        name="Universe", line=dict(color="#888", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=pdf["date"], y=pdf["Excess_NAV"],
        name="Excess (Q1/Univ)", line=dict(color="#00CC96", width=2, dash="dash"),
    ))
    fig.update_layout(
        yaxis_title="NAV",
        height=350,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _event_signal_rows(event_df: pl.DataFrame) -> pd.DataFrame:
    """De-dup to one row per signal; returns pandas DataFrame."""
    pdf = event_df.to_pandas()
    # WHY: groupby().first() is safer than drop_duplicates() for list/array columns
    return pdf.groupby("signal", sort=False).first().reset_index()


def event_caar_chart(event_df: pl.DataFrame) -> go.Figure:
    """CAAR path with 95% CI bands (+/-1.96 sigma/sqrt(n)), Buy/Sell overlaid.

    WHY: The confidence band must reflect uncertainty of the MEAN (SEM = sigma/sqrt(n)),
    not the dispersion of individual events (sigma). A narrow CI on a 300-event
    signal is meaningful; showing +/-sigma would swamp the signal with noise.
    """
    unique_signals = _event_signal_rows(event_df)
    fig = go.Figure()

    for _, row in unique_signals.iterrows():
        car_path = np.array(row["car_path"])
        std_path = np.array(row["car_std_path"]) if row.get("car_std_path") is not None else np.zeros_like(car_path)
        n_path = np.array(row["n_path"]) if "n_path" in row and row.get("n_path") is not None else np.ones_like(car_path)
        x_vals = np.array(row["calc_horizons"]) if "calc_horizons" in row else np.arange(1, len(car_path) + 1)
        x_labels = [f"{h}D" for h in x_vals]

        # WHY: SEM = sigma/sqrt(n) -> 95% CI = +/-1.96 SEM
        sem = np.where(n_path > 0, std_path / np.sqrt(np.maximum(n_path, 1)), 0)
        ci_upper = car_path + 1.96 * sem
        ci_lower = car_path - 1.96 * sem

        n_total = int(n_path[0]) if len(n_path) > 0 and n_path[0] > 0 else "?"
        color = "#00CC96" if row["signal"] == "Buy" else "#EF553B"
        fill_color = _EVENT_COLORS.get(row["signal"], "rgba(100,100,100,0.12)")

        # CI band
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=np.concatenate([ci_upper, ci_lower[::-1]]).tolist(),
            fill="toself", fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        # CAAR line
        fig.add_trace(go.Scatter(
            x=x_labels, y=car_path.tolist(),
            name=f"{row['signal']} (n={n_total})",
            line=dict(color=color, width=2.5),
            mode="lines+markers",
            marker=dict(size=6),
        ))

    # WHY: y=0 基準線是事件研究的必備元素，標示「無超額報酬」的虛無假設
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        yaxis_title="CAAR", xaxis_title="Days after event",
        height=340, margin=dict(t=30, b=40, l=60, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def event_hitrate_chart(event_df: pl.DataFrame) -> go.Figure:
    """Win Rate by horizon, Buy/Sell overlaid with 50% reference line.

    WHY: Separated from CAAR to avoid dual-y-axis scale confusion.
    The 50% line is the null hypothesis (random direction).
    """
    unique_signals = _event_signal_rows(event_df)
    fig = go.Figure()

    for _, row in unique_signals.iterrows():
        hr_path = np.array(row["hit_rate_path"]) if row.get("hit_rate_path") is not None else np.array([])
        x_vals = np.array(row["calc_horizons"]) if "calc_horizons" in row else np.arange(1, len(hr_path) + 1)
        x_labels = [f"{h}D" for h in x_vals]
        color = "#00CC96" if row["signal"] == "Buy" else "#EF553B"

        fig.add_trace(go.Scatter(
            x=x_labels, y=(hr_path * 100).tolist(),
            name=row["signal"],
            line=dict(color=color, width=2.5),
            mode="lines+markers",
            marker=dict(size=6),
        ))

    # WHY: 50% = coin-flip baseline；高於此線才有方向預測價值
    fig.add_hline(y=50, line=dict(color="gray", dash="dash", width=1),
                  annotation_text="50% (random)", annotation_position="bottom right")
    # WHY: 動態 y 範圍讓極端值（>70% 或 <30%）可見；對稱延伸確保 50% 基準線居中
    all_hr = []
    for _, row in unique_signals.iterrows():
        if row.get("hit_rate_path") is not None:
            all_hr.extend(np.array(row["hit_rate_path"]) * 100)
    if all_hr:
        lo = max(0, min(all_hr) - 8)
        hi = min(100, max(all_hr) + 8)
        # Keep 50% roughly centred
        half = max(50 - lo, hi - 50) + 2
        lo, hi = max(0, 50 - half), min(100, 50 + half)
    else:
        lo, hi = 30, 70
    fig.update_layout(
        yaxis_title="Win Rate (%)", xaxis_title="Days after event",
        yaxis=dict(range=[lo, hi]),
        height=340, margin=dict(t=30, b=40, l=60, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def event_ks_chart(event_df: pl.DataFrame) -> go.Figure:
    """K-S D-statistic grouped by horizon (Buy/Sell), p-value as secondary axis.

    WHY: K-S tests distribution divergence at discrete horizons — it is NOT
    a time-series path. Showing it as grouped bars makes the horizon-by-horizon
    comparison explicit and separates it from the CAAR path interpretation.
    p-value is shown on secondary y-axis so readers can cross-reference
    statistical significance without conflating it with the D-statistic scale.
    """
    pdf = event_df.to_pandas()[["signal", "horizon", "ks_stat", "p_value"]].copy()
    pdf["horizon_label"] = pdf["horizon"].apply(lambda h: f"{h}D")
    horizons = sorted(pdf["horizon"].unique())
    x_labels = [f"{h}D" for h in horizons]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for signal in ["Buy", "Sell"]:
        r, g, b = _SIGNAL_RGB[signal]
        solid = f"rgba({r},{g},{b},1.0)"
        faded = f"rgba({r},{g},{b},0.30)"

        subset = pdf[pdf["signal"] == signal].set_index("horizon")
        d_vals = [float(subset.loc[h, "ks_stat"]) if h in subset.index else 0.0 for h in horizons]
        p_vals = [float(subset.loc[h, "p_value"]) if h in subset.index else 1.0 for h in horizons]

        # WHY: p<0.05 實色、p>=0.05 透明，視覺化區分統計顯著性
        bar_colors = [solid if p < 0.05 else faded for p in p_vals]

        fig.add_trace(go.Bar(
            name=f"{signal} D-stat", x=x_labels, y=d_vals,
            marker_color=bar_colors,
            text=[f"p={p:.3f}" for p in p_vals],
            textposition="outside",
            offsetgroup=signal,
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            name=f"{signal} p-value", x=x_labels, y=p_vals,
            mode="markers", marker=dict(color=solid, size=9, symbol="diamond"),
            showlegend=True,
        ), secondary_y=True)

    # Significance threshold lines on p-value axis
    fig.add_hline(y=0.05, secondary_y=True,
                  line=dict(color="orange", dash="dot", width=1.2),
                  annotation_text="p=0.05", annotation_position="top right")
    fig.add_hline(y=0.01, secondary_y=True,
                  line=dict(color="red", dash="dot", width=1.2),
                  annotation_text="p=0.01", annotation_position="top right")

    fig.update_layout(
        barmode="group",
        yaxis_title="K-S D-statistic", xaxis_title="Horizon",
        height=340, margin=dict(t=30, b=40, l=60, r=60),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title_text="p-value", secondary_y=True, range=[0, 1.05])
    return fig


def event_temporal_chart(event_temporal_df: pl.DataFrame) -> go.Figure:
    """Quarterly signed CAAR bar chart with 95% CI, analogous to rolling IC.

    WHY: 事件型因子沒有連續 IC 序列，最能呈現 temporal stability 的方式是
    按季度分組的 signed CAAR（含 95% CI 誤差棒）。每根 bar 代表「該季所有
    事件的平均方向性獲利」，配合 Hit Rate 輔助線，一眼判斷哪個時段因子失效。
    """
    pdf = event_temporal_df.to_pandas()
    pdf["period"] = pd.to_datetime(pdf["period"])

    def _quarter_label(d: pd.Timestamp) -> str:
        return f"Q{(d.month - 1) // 3 + 1} {d.year}"

    pdf["label"] = pdf["period"].apply(_quarter_label)

    mean_car_pct = pdf["mean_signed_car"] * 100
    ci_pct = pdf["ci_95"] * 100
    hit_pct = pdf["hit_rate"] * 100

    r, g, b_buy = _SIGNAL_RGB["Buy"]
    _, _, b_sell = _SIGNAL_RGB["Sell"]
    r_sell, g_sell, _ = _SIGNAL_RGB["Sell"]
    bar_colors = [
        f"rgba({r},{g},{b_buy},0.85)" if v >= 0 else f"rgba({r_sell},{g_sell},{0},0.85)"
        for v in mean_car_pct
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Signed CAAR bars with 95% CI error bars
    fig.add_trace(go.Bar(
        x=pdf["label"],
        y=mean_car_pct,
        error_y=dict(type="data", array=ci_pct.tolist(), visible=True,
                     color="rgba(100,100,100,0.5)", thickness=1.5),
        marker_color=bar_colors,
        name="Signed CAAR (%)",
        text=[f"n={n}" for n in pdf["n_events"]],
        textposition="outside",
        textfont=dict(size=10),
    ), secondary_y=False)

    # Hit Rate line
    fig.add_trace(go.Scatter(
        x=pdf["label"], y=hit_pct,
        name="Hit Rate (%)",
        mode="lines+markers",
        line=dict(color="rgba(120,120,120,0.7)", dash="dot", width=1.5),
        marker=dict(size=5),
    ), secondary_y=True)

    # Reference lines
    fig.add_hline(y=0, secondary_y=False,
                  line=dict(color="gray", dash="dash", width=1))
    fig.add_hline(y=50, secondary_y=True,
                  line=dict(color="gray", dash="dash", width=1),
                  annotation_text="50%", annotation_position="bottom right")

    # Dynamic y-range for hit rate
    hr_min = max(0, hit_pct.min() - 8)
    hr_max = min(100, hit_pct.max() + 8)
    half = max(50 - hr_min, hr_max - 50) + 2
    fig.update_yaxes(title_text="Signed CAAR (%)", secondary_y=False)
    fig.update_yaxes(title_text="Hit Rate (%)", secondary_y=True,
                     range=[max(0, 50 - half), min(100, 50 + half)])
    fig.update_layout(
        xaxis_title="Quarter",
        height=360,
        margin=dict(t=30, b=50, l=60, r=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
