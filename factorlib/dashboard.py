"""
Layer 5: Streamlit Dashboard — Leaderboard + Drill-down.
Run: streamlit run factorlib/dashboard.py
"""

import streamlit as st
import mlflow
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_leaderboard(experiment_name: str = "Factor_Zoo") -> pd.DataFrame:
    try:
        runs = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception:
        return pd.DataFrame()

    if runs.empty:
        return pd.DataFrame()

    display_cols = {
        "tags.mlflow.runName": "Factor",
        "tags.status": "Status",
        "metrics.Total_Score": "Total",
    }

    dim_cols = [
        c for c in runs.columns
        if c.startswith("metrics.") and c.endswith("_Score") and c != "metrics.Total_Score"
    ]
    for c in sorted(dim_cols):
        display_cols[c] = c.replace("metrics.", "").replace("_Score", "")

    available = {k: v for k, v in display_cols.items() if k in runs.columns}
    df = runs[list(available.keys()) + ["run_id"]].rename(columns=available)

    if "Total" in df.columns:
        df = df.sort_values("Total", ascending=False).reset_index(drop=True)

    return df


def fetch_artifact(run_id: str, artifact_name: str) -> pl.DataFrame | None:
    try:
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_name)
        return pl.read_parquet(local_path)
    except Exception:
        return None


def fetch_run_metrics(run_id: str) -> dict:
    """Fetch all individual metric values for a run (for drill-down detail)."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def radar_chart(scoring_row: pd.Series, dim_columns: list[str]) -> go.Figure:
    values = [scoring_row.get(c, 0) for c in dim_columns]
    values.append(values[0])
    labels = dim_columns + [dim_columns[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=labels, fill="toself",
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
    pdf = ic_df.to_pandas().sort_values("date")
    split_idx = int(len(pdf) * (1 - oos_ratio))
    split_date_str = str(pdf["date"].iloc[split_idx])

    is_data = pdf.iloc[:split_idx + 1]
    oos_data = pdf.iloc[split_idx:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=is_data["date"], y=is_data["cum_ic"],
        name="In-Sample", line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=oos_data["date"], y=oos_data["cum_ic"],
        name="Out-of-Sample", line=dict(color="#EF553B", width=2),
    ))
    fig.add_shape(type="line", x0=split_date_str, x1=split_date_str,
                  y0=0, y1=1, yref="paper", line=dict(color="red", dash="dash", width=1))
    fig.add_annotation(x=split_date_str, y=1, yref="paper", text="OOS Start",
                       showarrow=False, font=dict(color="red", size=11), yanchor="bottom")
    fig.update_layout(
        yaxis_title="Cumulative IC",
        height=350,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.15),
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


# ---------------------------------------------------------------------------
# Leaderboard styling
# ---------------------------------------------------------------------------

def style_leaderboard(df: pd.DataFrame):
    """Apply conditional formatting to leaderboard."""
    def color_status(val):
        if val == "PASS":
            return "color: #28a745; font-weight: bold"
        elif val == "VETOED":
            return "color: #dc3545; font-weight: bold"
        return ""

    def color_score(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 70:
            return "background-color: rgba(40, 167, 69, 0.15)"
        elif v <= 30:
            return "background-color: rgba(220, 53, 69, 0.10)"
        return ""

    score_cols = [c for c in df.columns if c not in ("Factor", "Status")]
    styler = df.style
    if "Status" in df.columns:
        styler = styler.map(color_status, subset=["Status"])
    styler = styler.map(color_score, subset=score_cols)
    styler = styler.format("{:.1f}", subset=score_cols, na_rep="-")
    return styler


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Factor Lab", layout="wide")
    st.title("Factor Lab")
    st.caption("Quantitative Factor Evaluation Dashboard")

    # --- Leaderboard ---
    df = fetch_leaderboard()

    if df.empty:
        st.warning("No experiment runs found. Run `python main.py` first.")
        return

    st.subheader("Leaderboard")

    dim_columns = [c for c in df.columns if c not in ("Factor", "Status", "Total", "run_id")]

    display_df = df.drop(columns=["run_id"])
    st.dataframe(
        style_leaderboard(display_df),
        use_container_width=True,
        hide_index=True,
        height=min(len(df) * 38 + 40, 400),
    )

    # --- Factor Selection ---
    factor_names = df["Factor"].tolist() if "Factor" in df.columns else []
    if not factor_names:
        return

    st.divider()
    selected = st.selectbox("Drill-down", factor_names)
    selected_row = df[df["Factor"] == selected].iloc[0]
    run_id = selected_row["run_id"]

    # --- Drill-down: Radar + Metrics ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dimension Radar")
        if dim_columns:
            st.plotly_chart(radar_chart(selected_row, dim_columns), use_container_width=True)

    with col2:
        st.subheader("Score Breakdown")

        total = selected_row.get("Total", 0)
        status = selected_row.get("Status", "N/A")
        delta_color = "normal" if status == "PASS" else "inverse"
        st.metric("Total Score", f"{total:.1f}", delta=status, delta_color=delta_color)

        # Show individual metrics from MLflow
        metrics = fetch_run_metrics(run_id)
        for dim in dim_columns:
            with st.expander(f"{dim}: {selected_row.get(dim, 0):.1f}", expanded=True):
                dim_metrics = {
                    k: v for k, v in metrics.items()
                    if k not in ("Total_Score",) and not k.endswith("_Score")
                }
                # Filter metrics that belong to this dimension (heuristic)
                dim_key_map = {
                    "Alpha": ["Rank_IC", "IC_IR", "Long_Only_Alpha"],
                    "Robustness": ["Internal_OOS_Decay", "IC_Stability"],
                    "Risk": ["Turnover", "MDD"],
                }
                keys = dim_key_map.get(dim, [])
                metric_cols = st.columns(len(keys)) if keys else []
                for i, k in enumerate(keys):
                    val = metrics.get(k, 0)
                    with metric_cols[i]:
                        st.metric(k, f"{val:.1f}")

    # --- Artifact Charts ---
    ic_df = fetch_artifact(run_id, "ic_series.parquet")
    nav_df = fetch_artifact(run_id, "nav_series.parquet")

    if ic_df is not None or nav_df is not None:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if ic_df is not None:
                st.subheader("Cumulative IC")
                st.plotly_chart(ic_chart(ic_df), use_container_width=True)
        with c2:
            if nav_df is not None:
                st.subheader("Long-Only Q1 vs Universe")
                st.plotly_chart(nav_chart(nav_df), use_container_width=True)


if __name__ == "__main__":
    main()
