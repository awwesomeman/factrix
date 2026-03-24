"""Dashboard data layer — MLflow fetching and caching."""

import logging

import streamlit as st
import mlflow
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


@st.cache_data(ttl=30)
def fetch_leaderboard(experiment_name: str = "Factor_Zoo") -> pd.DataFrame:
    try:
        runs = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception as e:
        logger.warning("Failed to fetch leaderboard: %s", e)
        return pd.DataFrame()

    if runs.empty:
        return pd.DataFrame()

    display_cols = {
        "tags.mlflow.runName": "Factor",
        "tags.factor_type": "Type",
        "tags.status": "Status",
        "metrics.Total_Score": "Total",
        "metrics.Signal_Score": "Signal",
        "metrics.Performance_Score": "Performance",
        "metrics.Robustness_Score": "Robustness",
        "metrics.Efficiency_Score": "Efficiency",
    }

    # WHY: Keep only the most recent run per factor name — historical runs may
    # lack newer artifacts (e.g. event_matrix.parquet), causing silent failures
    # in drill-down. Dedup before score-sort so the displayed row always matches
    # the artifact that will be fetched.
    if "tags.mlflow.runName" in runs.columns and "start_time" in runs.columns:
        runs = (
            runs.sort_values("start_time", ascending=False)
            .drop_duplicates(subset=["tags.mlflow.runName"])
        )

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
    except Exception as e:
        logger.debug("Artifact %s not found for run %s: %s", artifact_name, run_id, e)
        return None


def fetch_run_metrics(run_id: str) -> dict:
    """Fetch all metric values for a run."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        logger.warning("Failed to fetch metrics for run %s: %s", run_id, e)
        return {}
