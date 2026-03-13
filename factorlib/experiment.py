"""
Layer 4: Experiment Tracking — MLflow integration.
Logs scoring results, IC series, and NAV curves as structured experiments.
"""

import tempfile
from pathlib import Path

import mlflow
import polars as pl
import numpy as np


class FactorTracker:
    def __init__(self, experiment_name: str = "Factor_Zoo"):
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        factor_name: str,
        scoring_results: dict,
        config: dict,
        ic_series: pl.DataFrame | None = None,
        nav_series: pl.DataFrame | None = None,
        logic_desc: str = "",
    ) -> str:
        """
        Log a complete factor evaluation run to MLflow.
        Returns the run_id.
        """
        with mlflow.start_run(run_name=factor_name) as run:
            # Tags
            has_veto = len(scoring_results.get("penalties", [])) > 0
            mlflow.set_tag("status", "VETOED" if has_veto else "PASS")
            mlflow.set_tag("logic_description", logic_desc)
            if scoring_results["penalties"]:
                mlflow.set_tag("veto_reasons", "; ".join(scoring_results["penalties"]))

            # Params (serializable config snapshot)
            flat_params = self._flatten_config(config)
            mlflow.log_params(flat_params)

            # Metrics: total + per-dimension + per-metric
            mlflow.log_metric("Total_Score", scoring_results["total"])
            for dim_name, dim_data in scoring_results["dimensions"].items():
                mlflow.log_metric(f"{dim_name}_Score", dim_data["score"])
                for m_name, m_score in dim_data["metrics"].items():
                    mlflow.log_metric(m_name, m_score)

            # Artifacts: IC series & NAV curve for Streamlit drill-down
            with tempfile.TemporaryDirectory() as tmpdir:
                if ic_series is not None:
                    ic_path = Path(tmpdir) / "ic_series.parquet"
                    ic_series.write_parquet(ic_path)
                    mlflow.log_artifact(str(ic_path))

                if nav_series is not None:
                    nav_path = Path(tmpdir) / "nav_series.parquet"
                    nav_series.write_parquet(nav_path)
                    mlflow.log_artifact(str(nav_path))

            return run.info.run_id

    @staticmethod
    def _flatten_config(config: dict, prefix: str = "") -> dict[str, str]:
        """Flatten nested config dict for MLflow params (string values only)."""
        flat = {}
        for k, v in config.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(FactorTracker._flatten_config(v, prefix=f"{key}."))
            else:
                flat[key] = str(v)
        return flat


def build_ic_artifact(prepared_data: pl.DataFrame, min_assets: int = 10) -> pl.DataFrame:
    """Build IC time series artifact from prepared factor data.

    Uses method='average' for Spearman rank and filters dates with fewer than
    min_assets stocks, consistent with the scoring IC computation in registry.py.
    """
    ranked = prepared_data.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("rank_factor"),
        pl.col("forward_return").rank(method="average").over("date").alias("rank_return"),
    )
    ic_df = (
        ranked.group_by("date")
        .agg(
            pl.corr("rank_factor", "rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_assets)
        .sort("date")
        .select("date", "ic")
    )
    return ic_df.with_columns(pl.col("ic").cum_sum().alias("cum_ic"))


def build_nav_artifact(
    prepared_data: pl.DataFrame,
    step: int = 5,
) -> pl.DataFrame:
    """Build Long-Only Q1 vs Universe NAV curves (non-overlapping).

    Uses method='average' for rank consistency with scoring metrics.
    Drops nulls before computing NAV to guarantee aligned lengths.
    """
    dates = prepared_data["date"].unique().sort()
    sampled = dates.gather_every(step)
    filtered = prepared_data.filter(pl.col("date").is_in(sampled.implode()))

    returns_df = (
        filtered.with_columns(
            (pl.col("factor").rank(method="average").over("date") / pl.len().over("date"))
            .alias("pct_rank")
        )
        .group_by("date")
        .agg(
            pl.col("forward_return")
            .filter(pl.col("pct_rank") >= 0.8)
            .mean()
            .alias("q1_return"),
            pl.col("forward_return").mean().alias("universe_return"),
        )
        .sort("date")
    )

    # Drop nulls FIRST, then compute NAV from the same clean DataFrame
    valid = returns_df.drop_nulls()

    q1_nav = np.cumprod(1 + valid["q1_return"].to_numpy())
    univ_nav = np.cumprod(1 + valid["universe_return"].to_numpy())

    return pl.DataFrame({
        "date": valid["date"],
        "Q1_NAV": q1_nav,
        "Universe_NAV": univ_nav,
        "Excess_NAV": q1_nav / univ_nav,
    })
