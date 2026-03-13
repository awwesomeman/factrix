"""FactorScorer — orchestrates metric computation and scoring."""

import polars as pl
from typing import Any

from factorlib.scoring.registry import METRIC_REGISTRY, _ic_series
from factorlib.scoring.config import SCORING_CONFIG


class FactorScorer:
    def __init__(
        self,
        prepared_data: pl.DataFrame,
        config: dict | None = None,
        forward_periods: int = 5,
    ):
        self.data = prepared_data
        self.config = config or SCORING_CONFIG
        self.forward_periods = forward_periods
        self._ic_cache: pl.DataFrame | None = None

    @property
    def ic_series(self) -> pl.DataFrame:
        """Cached IC series — computed once, reused across metrics."""
        if self._ic_cache is None:
            self._ic_cache = _ic_series(self.data)
        return self._ic_cache

    def compute(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "total": 0.0,
            "dimensions": {},
            "penalties": [],
        }
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, dim_cfg in self.config.items():
            dim_w_sum = 0.0
            active_w = 0.0
            metric_scores = {}

            for m_name, m_cfg in dim_cfg["metrics"].items():
                func = METRIC_REGISTRY.get(m_name)
                if func is None:
                    raise ValueError(f"Unknown metric: {m_name}")

                extra = {k: v for k, v in m_cfg.items() if k not in ("weight", "min_threshold")}
                score = func(
                    self.data,
                    _ic_cache=self.ic_series,
                    _forward_periods=self.forward_periods,
                    **extra,
                )

                if score is None:
                    continue

                metric_scores[m_name] = round(score, 2)

                threshold = m_cfg.get("min_threshold")
                if threshold is not None and score < threshold:
                    results["penalties"].append(
                        f"VETO: {m_name} score {score:.1f} below {threshold}"
                    )

                dim_w_sum += score * m_cfg["weight"]
                active_w += m_cfg["weight"]

            if active_w == 0:
                continue

            dim_final = dim_w_sum / active_w
            results["dimensions"][dim_name] = {
                "score": round(dim_final, 2),
                "metrics": metric_scores,
            }
            weighted_sum += dim_final * dim_cfg["weight"]
            total_weight += dim_cfg["weight"]

        raw_total = weighted_sum / total_weight if total_weight > 0 else 0
        results["total"] = round(raw_total * 0.2, 2) if results["penalties"] else round(raw_total, 2)

        return results
