"""FactorScorer — 4-dimension scoring with adaptive weighting.

Dimensions: Signal, Performance, Robustness, Efficiency.
"""

import logging
import math
import polars as pl
from typing import Any

from factorlib.scoring.registry import METRIC_REGISTRY, adaptive_weight
from factorlib.scoring._utils import MetricResult, _ic_series
from factorlib.scoring.config import (
    FACTOR_CONFIGS, DIMENSIONS,
    DEFAULT_ADAPTIVE_TAU, DEFAULT_ADAPTIVE_K, DEFAULT_VETO_PENALTY,
)

logger = logging.getLogger(__name__)


class FactorScorer:
    def __init__(
        self,
        prepared_data: pl.DataFrame,
        forward_periods: int = 5,
        factor_type: str = "individual_stock",
    ):
        self.data = prepared_data
        self.forward_periods = forward_periods
        self.factor_type = factor_type
        self._ic_cache: pl.DataFrame | None = None

    @property
    def ic_series(self) -> pl.DataFrame:
        if self._ic_cache is None:
            self._ic_cache = _ic_series(self.data)
        return self._ic_cache

    def _score_dimension(
        self, metrics_config: dict, tau: float, k: float,
    ) -> tuple[float, dict, list[str]]:
        """Score a dimension: equal base weight, adaptive sigmoid adjusts by t-stat.

        Returns (dimension_score, metrics_detail, penalties).
        metrics_detail: {name: {"score", "t_stat", "adaptive_w"}}
        """
        metrics_detail: dict[str, Any] = {}
        penalties: list[str] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for m_name, m_params in metrics_config.items():
            func = METRIC_REGISTRY.get(m_name)
            if func is None:
                raise ValueError(f"Unknown metric: {m_name}")

            extra = {k_: v for k_, v in m_params.items() if k_ != "min_threshold"}
            result: MetricResult | None = func(
                self.data,
                _ic_cache=self.ic_series,
                _forward_periods=self.forward_periods,
                **extra,
            )

            if result is None:
                continue

            score, t_stat, raw_value = result.score, result.t_stat, result.raw_value

            # WHY: some metrics return valid float that is NaN when data is
            # degenerate (e.g. IC on event signals with no cross-section)
            if math.isnan(score):
                continue
            if t_stat is not None and math.isnan(t_stat):
                t_stat = None
            if raw_value is not None and (math.isnan(raw_value) or math.isinf(raw_value)):
                raw_value = None

            w = adaptive_weight(1.0, t_stat, tau=tau, k=k)

            metrics_detail[m_name] = {
                "score": round(score, 2),
                "t_stat": round(t_stat, 2) if t_stat is not None else None,
                "adaptive_w": round(w, 3),
                "raw_value": raw_value,
            }

            threshold = m_params.get("min_threshold")
            if threshold is not None and score < threshold:
                penalties.append(f"VETO: {m_name} score {score:.1f} below {threshold}")

            weighted_sum += score * w
            total_weight += w

        facet_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return (facet_score, metrics_detail, penalties)

    def compute(self) -> dict[str, Any]:
        if self.factor_type not in FACTOR_CONFIGS:
            logger.warning(
                "Unknown factor_type %r, falling back to 'individual_stock'",
                self.factor_type,
            )
        config = FACTOR_CONFIGS.get(self.factor_type, FACTOR_CONFIGS["individual_stock"])
        routing = config["routing"]
        tau = config.get("adaptive_tau", DEFAULT_ADAPTIVE_TAU)
        k = config.get("adaptive_k", DEFAULT_ADAPTIVE_K)
        veto_penalty = config.get("veto_penalty", DEFAULT_VETO_PENALTY)

        unknown_dims = set(routing) - set(DIMENSIONS)
        if unknown_dims:
            raise ValueError(
                f"Routing contains unknown dimensions {unknown_dims}; "
                f"expected subset of {DIMENSIONS}"
            )

        all_penalties: list[str] = []
        dim_scores: dict[str, float] = {}
        dim_details: dict[str, dict] = {}

        for dim_name, dim_weight in routing.items():
            dim_config = config.get(dim_name, {})
            if not dim_config:
                dim_scores[dim_name] = 0.0
                dim_details[dim_name] = {"metrics": {}, "penalties": []}
                continue
            score, metrics, penalties = self._score_dimension(dim_config, tau=tau, k=k)
            dim_scores[dim_name] = round(score, 2)
            dim_details[dim_name] = {"metrics": metrics, "penalties": penalties}
            all_penalties.extend(penalties)

        raw_total = sum(dim_scores[d] * routing[d] for d in routing)
        total = (round(raw_total * veto_penalty, 2)
                 if all_penalties else round(raw_total, 2))

        return {
            "total": total,
            "dimension_weights": dict(routing),
            "dimensions": dim_details,
            "penalties": all_penalties,
            **{f"{d}_score": dim_scores[d] for d in routing},
        }
