"""
Layer 4: Experiment Tracking — MLflow integration.
Logs scoring results, IC series, and NAV curves as structured experiments.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import polars as pl

from factorlib.evaluation._protocol import EvaluationResult


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
        event_matrix: pl.DataFrame | None = None,
        event_temporal: pl.DataFrame | None = None,
        logic_desc: str = "",
        factor_type: str = "individual_stock",
        sample_period: str = "",
        asset_pool: str = "",
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
            self._set_context_tags(factor_type, sample_period, asset_pool)
            if scoring_results["penalties"]:
                mlflow.set_tag("veto_reasons", "; ".join(scoring_results["penalties"]))

            # Params (serializable config snapshot)
            flat_params = self._flatten_config(config)
            mlflow.log_params(flat_params)

            # Metrics: total + dimension scores + per-metric (score + adaptive_w)
            mlflow.log_metric("Total_Score", scoring_results["total"])

            for dim_name in scoring_results.get("dimension_weights", {}):
                mlflow.log_metric(
                    f"{dim_name.capitalize()}_Score",
                    scoring_results.get(f"{dim_name}_score", 0),
                )

            for dim_detail in scoring_results.get("dimensions", {}).values():
                for m_name, m_detail in dim_detail.get("metrics", {}).items():
                    mlflow.log_metric(m_name, m_detail["score"])
                    mlflow.log_metric(f"{m_name}_adaptive_w", m_detail["adaptive_w"])
                    if m_detail["t_stat"] is not None:
                        mlflow.log_metric(f"{m_name}_t_stat", m_detail["t_stat"])
                    if m_detail.get("raw_value") is not None:
                        mlflow.log_metric(f"{m_name}_raw", m_detail["raw_value"])

            # Artifacts: IC series, NAV curve, Event Matrix
            with tempfile.TemporaryDirectory() as tmpdir:
                if ic_series is not None:
                    ic_path = Path(tmpdir) / "ic_series.parquet"
                    ic_series.write_parquet(ic_path)
                    mlflow.log_artifact(str(ic_path))

                if nav_series is not None:
                    nav_path = Path(tmpdir) / "nav_series.parquet"
                    nav_series.write_parquet(nav_path)
                    mlflow.log_artifact(str(nav_path))

                if event_matrix is not None and not event_matrix.is_empty():
                    event_path = Path(tmpdir) / "event_matrix.parquet"
                    event_matrix.write_parquet(event_path)
                    mlflow.log_artifact(str(event_path))

                if event_temporal is not None and not event_temporal.is_empty():
                    et_path = Path(tmpdir) / "event_temporal.parquet"
                    event_temporal.write_parquet(et_path)
                    mlflow.log_artifact(str(et_path))

            return run.info.run_id

    def log_evaluation(
        self,
        result: EvaluationResult,
        config: dict | None = None,
        factor_type: str = "individual_stock",
        sample_period: str = "",
        asset_pool: str = "",
    ) -> str:
        """Log a gate-based factor evaluation to MLflow.

        Args:
            result: Output of ``evaluate_factor()``.
            config: Optional config snapshot to log as params.
            factor_type: Factor category tag.
            sample_period: Date range tag.
            asset_pool: Universe tag.

        Returns:
            MLflow run_id.
        """
        with mlflow.start_run(run_name=result.factor_name) as run:
            mlflow.set_tag("status", result.status)
            self._set_context_tags(factor_type, sample_period, asset_pool)

            if result.caution_reasons:
                mlflow.set_tag(
                    "caution_reasons", "; ".join(result.caution_reasons),
                )

            if config:
                mlflow.log_params(self._flatten_config(config))

            # Per-gate results
            for gr in result.gate_results:
                mlflow.set_tag(f"gate.{gr.name}.status", gr.status)
                for k, v in gr.detail.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"gate.{gr.name}.{k}", v)

            # Profile metrics
            if result.profile:
                self._log_metrics(result.profile.reliability)
                self._log_metrics(result.profile.profitability)

            return run.info.run_id

    @staticmethod
    def _log_metrics(metrics: list) -> None:
        for metric in metrics:
            mlflow.log_metric(metric.name, metric.value)
            if metric.stat is not None:
                mlflow.log_metric(f"{metric.name}_t_stat", metric.stat)
            if metric.significance:
                mlflow.set_tag(f"{metric.name}_sig", metric.significance)

    def log_failed_run(
        self,
        factor_name: str,
        error: str,
        factor_type: str = "individual_stock",
        sample_period: str = "",
        asset_pool: str = "",
    ) -> str:
        """Log a failed factor evaluation run to MLflow.

        WHY: quant/multiple-testing §6 要求所有實驗（含失敗）都必須記錄，
        避免僅報告倖存者的選擇偏差。
        """
        with mlflow.start_run(run_name=factor_name) as run:
            mlflow.set_tag("status", "FAILED")
            self._set_context_tags(factor_type, sample_period, asset_pool)
            mlflow.set_tag("error", error[:250])
            mlflow.log_metric("Total_Score", 0.0)
            return run.info.run_id

    @staticmethod
    def _set_context_tags(factor_type: str, sample_period: str, asset_pool: str) -> None:
        mlflow.set_tag("factor_type", factor_type)
        if sample_period:
            mlflow.set_tag("sample_period", sample_period)
        if asset_pool:
            mlflow.set_tag("asset_pool", asset_pool)

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
