"""Dashboard Streamlit layout — leaderboard + drill-down pages."""

import streamlit as st
import pandas as pd

from factorlib.scoring.config import DIMENSIONS
from factorlib.dashboard.data import (
    fetch_experiment_names,
    fetch_leaderboard,
    fetch_artifact,
    fetch_run_data,
)
from factorlib.dashboard.charts import (
    radar_chart,
    ic_chart,
    nav_chart,
    event_caar_chart,
    event_hitrate_chart,
    event_ks_chart,
    event_temporal_chart,
)

DIMENSION_LABELS = [d.capitalize() for d in DIMENSIONS]
DIMENSION_KEYS = [f"{label}_Score" for label in DIMENSION_LABELS]

_LEADING_COLS = ["Factor", "Status"]
_TRAILING_COLS = ["Type", "Sample Period", "Asset Pool"]
_NON_SCORE_COLS = set(_LEADING_COLS + _TRAILING_COLS)


# ---------------------------------------------------------------------------
# Leaderboard styling
# ---------------------------------------------------------------------------

def style_leaderboard(df: pd.DataFrame):
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

    score_cols = [c for c in df.columns if c not in _NON_SCORE_COLS]
    styler = df.style
    if "Status" in df.columns:
        styler = styler.map(color_status, subset=["Status"])
    styler = styler.map(color_score, subset=score_cols)
    styler = styler.format("{:.1f}", subset=score_cols, na_rep="-")
    return styler


# ---------------------------------------------------------------------------
# Metric detail table builder
# ---------------------------------------------------------------------------

DISPLAY_GROUPS = {
    "Predictability": ["IC", "IC_IR", "Monotonicity",
                       "Event_CAAR", "Event_KS", "Event_CAR_Dispersion"],
    "Profitability": ["Long_Alpha", "MDD",
                      "Profit_Factor", "Event_Skewness"],
    "Robustness": ["OOS_Decay", "IC_Stability", "Hit_Rate",
                   "Cross_Consistency",
                   "Event_Decay", "Event_Stability", "Event_Hit_Rate"],
    "Tradability": ["Turnover", "Capacity", "Slippage"],
}

_RAW_FORMAT: dict[str, str] = {
    "IC":                  ".4f",
    "IC_IR":               ".3f",
    "Long_Alpha":          ".1%",
    "Monotonicity":        ".3f",
    "Hit_Rate":            ".1%",
    "OOS_Decay":           ".3f",
    "IC_Stability":        ".3f",
    "MDD":                 ".1%",
    "Turnover":            ".3f",
    "Event_CAAR":          ".3%",
    "Event_KS":            ".4f",
    "Event_CAR_Dispersion":".4f",
    "Event_Decay":         ".3f",
    "Event_Stability":     ".3f",
    "Event_Hit_Rate":      ".1%",
    "Profit_Factor":       ".3f",
    "Event_Skewness":      ".3f",
}


def _fmt_raw(metric: str, value: float | None) -> str | None:
    if value is None:
        return None
    fmt = _RAW_FORMAT.get(metric, ".3f")
    return format(value, fmt)


def _extract_min_thresholds(params: dict) -> dict[str, float]:
    """Extract min_threshold values from MLflow params like 'signal.Event_CAAR.min_threshold'."""
    thresholds = {}
    for k, v in params.items():
        if k.endswith(".min_threshold"):
            metric_name = k.rsplit(".", 2)[-2]
            try:
                thresholds[metric_name] = float(v)
            except ValueError:
                pass
    return thresholds


def build_metric_table(metrics: dict, params: dict | None = None) -> pd.DataFrame:
    """Build a single detail table for all dimensions with a Metric Type column."""
    thresholds = _extract_min_thresholds(params) if params else {}
    rows = []
    for dim, metric_names in DISPLAY_GROUPS.items():
        for m in metric_names:
            if m not in metrics:
                continue
            raw_val = metrics.get(f"{m}_raw")
            row = {
                "Metric Type": dim,
                "Metric": m,
                "Raw": _fmt_raw(m, raw_val),
                "Score": metrics.get(m, 0),
                "Threshold": thresholds.get(m),
                "t-stat": metrics.get(f"{m}_t_stat"),
                "Adaptive W": metrics.get(f"{m}_adaptive_w"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Factor Lab", layout="wide")
    st.title("Factor Lab")
    st.caption("Quantitative Factor Evaluation Dashboard")

    # --- Sidebar: Experiment Selector ---
    st.sidebar.header("Experiment")
    all_experiments = fetch_experiment_names()
    if not all_experiments:
        st.warning("No experiments found. Run `python main.py` first.")
        return
    selected_experiments = st.sidebar.multiselect(
        "Experiments", all_experiments, default=all_experiments,
    )
    if not selected_experiments:
        st.warning("Select at least one experiment.")
        return

    # --- Leaderboard ---
    df = fetch_leaderboard(tuple(selected_experiments))

    if df.empty:
        st.warning("No runs found in selected experiments.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    for col_name, label in [
        ("Type", "Factor Type"),
        ("Sample Period", "Sample Period"),
        ("Asset Pool", "Asset Pool"),
    ]:
        if col_name in df.columns:
            filled = df[col_name].fillna("N/A")
            options = sorted(filled.unique().tolist())
            selected = st.sidebar.multiselect(label, options, default=options)
            df = df[filled.isin(selected)]

    # --- Factor Selection in Sidebar ---
    factor_names = df["Factor"].tolist() if "Factor" in df.columns else []
    if not factor_names:
        st.subheader("Leaderboard")
        st.warning("No factors match current filters.")
        return

    selected = st.sidebar.selectbox("Factor Name", factor_names)

    st.subheader("Leaderboard")

    display_df = df.drop(columns=["run_id"])
    score_order = [c for c in display_df.columns if c not in _NON_SCORE_COLS]
    ordered = [c for c in _LEADING_COLS + score_order + _TRAILING_COLS if c in display_df.columns]
    display_df = display_df[ordered]
    st.dataframe(
        style_leaderboard(display_df),
        use_container_width=True,
        hide_index=True,
        height=min(len(df) * 38 + 40, 400),
    )

    # --- Drill-down Detail ---
    st.divider()
    selected_row = df[df["Factor"] == selected].iloc[0]
    run_id = selected_row["run_id"]

    metrics, params = fetch_run_data(run_id)

    # --- Top-level scores with dimension weights ---
    total = selected_row.get("Total", 0)
    status = selected_row.get("Status", "N/A")
    delta_color = "normal" if status == "PASS" else "inverse"

    cols = st.columns(1 + len(DIMENSION_LABELS))
    with cols[0]:
        st.metric("Total Score", f"{total:.1f}", delta=status, delta_color=delta_color)
    for i, dim in enumerate(DIMENSION_LABELS):
        with cols[i + 1]:
            val = selected_row.get(dim)
            w = params.get(f"routing.{dim.lower()}")
            label = f"{dim} ({float(w):.0%})" if w else dim
            st.metric(label, f"{val:.1f}" if pd.notna(val) else "-")

    # --- Drill-down: Radar + Metric Tables ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dimension Radar")
        st.plotly_chart(
            radar_chart(metrics, DIMENSION_KEYS, DIMENSION_LABELS),
            use_container_width=True,
        )

    with col2:
        st.subheader("Metric Scorecard")
        tbl = build_metric_table(metrics, params)
        if not tbl.empty:
            st.dataframe(
                tbl.style.format({
                    "Score": "{:.1f}",
                    "Threshold": "{:.0f}",
                    "t-stat": "{:.2f}",
                    "Adaptive W": "{:.3f}",
                }, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )

    # --- Artifact Charts ---
    factor_type = selected_row.get("Type", "")
    ic_df = fetch_artifact(run_id, "ic_series.parquet")
    nav_df = fetch_artifact(run_id, "nav_series.parquet")
    event_df = fetch_artifact(run_id, "event_matrix.parquet")
    event_temporal_df = fetch_artifact(run_id, "event_temporal.parquet")

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

    if event_df is not None and not event_df.is_empty():
        st.divider()
        st.header("Event Signal Analysis")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("CAAR Path (95% CI)")
            st.plotly_chart(event_caar_chart(event_df), use_container_width=True)
        with c2:
            st.subheader("Win Rate by Horizon")
            st.plotly_chart(event_hitrate_chart(event_df), use_container_width=True)

        st.subheader("Distribution Divergence by Horizon (K-S Test)")
        st.plotly_chart(event_ks_chart(event_df), use_container_width=True)

        if event_temporal_df is not None and not event_temporal_df.is_empty():
            st.subheader("Temporal Stability — Quarterly Signed CAAR")
            st.plotly_chart(event_temporal_chart(event_temporal_df), use_container_width=True)
    elif factor_type == "event_signal":
        st.divider()
        st.warning(
            "Event analysis artifacts not found for this run. "
            "Re-run `python main.py` to regenerate `event_matrix.parquet`."
        )


if __name__ == "__main__":
    main()
