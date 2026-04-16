"""High-level convenience API.

quick_check     — one-call factor screening (preprocess + evaluate)
batch_evaluate  — evaluate multiple factors
compare         — tabular comparison of results
split_by_group  — sub-universe splitting with N-aware config
FACTOR_TYPES    — registry of factor type → config class
describe_factor_types — print factor type descriptions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import polars as pl

from factorlib._types import FactorType
from factorlib.config import (
    BaseConfig,
    CrossSectionalConfig,
    EventConfig,
    MacroCommonConfig,
    MacroPanelConfig,
)
from factorlib.evaluation._protocol import EvaluationResult, GateFn
from factorlib.metrics._helpers import _median_universe_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor type registry
# ---------------------------------------------------------------------------

FACTOR_TYPES: dict[FactorType, type[BaseConfig]] = {
    FactorType.CROSS_SECTIONAL: CrossSectionalConfig,
    FactorType.EVENT_SIGNAL: EventConfig,
    FactorType.MACRO_PANEL: MacroPanelConfig,
    FactorType.MACRO_COMMON: MacroCommonConfig,
}

_DESCRIPTIONS: dict[FactorType, str] = {
    FactorType.CROSS_SECTIONAL: "截面因子（每期每資產有 signal，N ≥ 30）→ 選股",
    FactorType.EVENT_SIGNAL: "事件訊號（離散觸發）→ 事件交易",
    FactorType.MACRO_PANEL: "宏觀 panel（小截面 N < 30）→ 跨國配置",
    FactorType.MACRO_COMMON: "宏觀共用（單一時序）→ 風險歸因",
}


def describe_factor_types() -> None:
    """Print supported factor types with descriptions."""
    for ft, desc in _DESCRIPTIONS.items():
        print(f"  {ft.value:<20s}: {desc}")


def _config_for_type(
    factor_type: str | FactorType,
    **overrides: Any,
) -> BaseConfig:
    if isinstance(factor_type, str):
        try:
            factor_type = FactorType(factor_type)
        except ValueError:
            raise ValueError(
                f"Unknown factor_type '{factor_type}'. "
                f"Supported: {', '.join(ft.value for ft in FactorType)}. "
                f"Use fl.describe_factor_types() for details."
            ) from None

    config_cls = FACTOR_TYPES[factor_type]
    return config_cls(**overrides)


# ---------------------------------------------------------------------------
# quick_check
# ---------------------------------------------------------------------------

def quick_check(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = "cross_sectional",
    config: BaseConfig | None = None,
    **config_overrides: Any,
) -> EvaluationResult:
    """One-call factor screening: preprocess + evaluate.

    Args:
        df: Raw data with ``date, asset_id, price, factor``.
        factor_name: Factor identifier.
        factor_type: Factor type string or FactorType enum.
        config: Explicit config (overrides factor_type).
        **config_overrides: Passed to the config constructor.

    Returns:
        EvaluationResult with profile and artifacts.
    """
    from factorlib.evaluation.pipeline import evaluate

    if config is None:
        config = _config_for_type(factor_type, **config_overrides)

    return evaluate(df, factor_name, config=config, preprocess=True)


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------

def batch_evaluate(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = "cross_sectional",
    config: BaseConfig | None = None,
    gates: list[GateFn] | None = None,
    preprocess: bool = True,
    on_result: Callable[[str, EvaluationResult], None] | None = None,
    stop_on_error: bool = False,
    **config_overrides: Any,
) -> dict[str, EvaluationResult]:
    """Evaluate multiple factors.

    Args:
        factors: Mapping or list of (name, DataFrame) pairs.
        factor_type: Default factor type for all factors.
        config: Shared config (overrides factor_type).
        gates: Gate functions (None = type default).
        preprocess: Whether to preprocess each factor.
        on_result: Callback after each evaluation, e.g. ``tracker.log``.
        stop_on_error: If True, raise on first error. If False, log and skip.
        **config_overrides: Passed to config constructor.

    Returns:
        Mapping of factor name → EvaluationResult.
    """
    from factorlib.evaluation.pipeline import evaluate

    if isinstance(factors, dict):
        factors = list(factors.items())

    if config is None:
        config = _config_for_type(factor_type, **config_overrides)

    results: dict[str, EvaluationResult] = {}
    for name, df in factors:
        try:
            result = evaluate(
                df, name, config=config, gates=gates, preprocess=preprocess,
            )
        except Exception:
            if stop_on_error:
                raise
            logger.exception("batch_evaluate: failed on '%s'", name)
            continue

        results[name] = result
        if on_result is not None:
            on_result(name, result)

    return results


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare(
    results: dict[str, EvaluationResult] | list[EvaluationResult],
    *,
    sort_by: str = "ic",
    ascending: bool = False,
) -> pl.DataFrame:
    """Create comparison table from multiple evaluation results.

    Args:
        results: Dict or list of EvaluationResult.
        sort_by: Metric name to sort by.
        ascending: Sort direction.

    Returns:
        DataFrame with one row per factor, columns for each metric.
    """
    import warnings

    if isinstance(results, list):
        results = {r.factor_name: r for r in results}

    factor_types = set()
    for r in results.values():
        if r.artifacts is not None:
            factor_types.add(type(r.artifacts.config).factor_type)
    if len(factor_types) > 1:
        warnings.warn(
            f"Comparing results across different factor types: {factor_types}. "
            f"Missing metrics will be filled with None.",
            UserWarning,
            stacklevel=2,
        )

    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        row: dict[str, Any] = {
            "factor": name,
            "status": result.status,
        }
        if result.profile:
            for m in result.profile.metrics:
                row[m.name] = m.value
                if m.stat is not None:
                    row[f"{m.name}_stat"] = m.stat
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort(sort_by, descending=not ascending)
    return df


# ---------------------------------------------------------------------------
# split_by_group
# ---------------------------------------------------------------------------

_N_GROUP_TIERS = [
    (1000, 10),
    (200, 5),
    (0, 3),
]


def _recommend_n_groups(median_n: int) -> int:
    for min_n, groups in _N_GROUP_TIERS:
        if median_n >= min_n:
            return groups
    return 3


def split_by_group(
    df: pl.DataFrame,
    definitions: dict[str, pl.Expr],
    base_config: CrossSectionalConfig,
    auto_n_groups: bool = True,
    unify_n_groups: bool = True,
) -> dict[str, tuple[pl.DataFrame, CrossSectionalConfig]]:
    """Split DataFrame by filter conditions with N-aware config adjustment.

    Args:
        df: Preprocessed panel with ``date, asset_id, ...``.
        definitions: Mapping of group name → Polars filter expression.
        base_config: Base pipeline configuration.
        auto_n_groups: Adjust n_groups per group based on median N.
        unify_n_groups: All groups use same n_groups (minimum).

    Returns:
        Mapping of group name → (filtered DataFrame, adjusted config).
    """
    per_group: dict[str, tuple[pl.DataFrame, int, int]] = {}

    for name, expr in definitions.items():
        filtered = df.filter(expr)

        if filtered.is_empty():
            logger.warning("split_by_group: group '%s' is empty after filtering", name)
            continue

        median_n = _median_universe_size(filtered)
        recommended = _recommend_n_groups(median_n)
        per_group[name] = (filtered, median_n, recommended)

    if not per_group:
        return {}

    if auto_n_groups:
        if unify_n_groups:
            per_group_rec = {n: rec for n, (_, _, rec) in per_group.items()}
            unified = min(per_group_rec.values())
            if unified != base_config.n_groups:
                bottleneck = [n for n, r in per_group_rec.items() if r == unified]
                logger.warning(
                    "split_by_group: n_groups reduced %d → %d to unify "
                    "across groups (bottleneck: %s, per-group recommended: %s). "
                    "Pass unify_n_groups=False to keep independent n_groups.",
                    base_config.n_groups, unified,
                    bottleneck, per_group_rec,
                )
        else:
            unified = None
    else:
        unified = None

    for name, (_, median_n, _) in per_group.items():
        if median_n < 50:
            logger.warning(
                "split_by_group: group '%s' has median N=%d (< 50) "
                "— quantile results may be unstable",
                name, median_n,
            )

    result: dict[str, tuple[pl.DataFrame, CrossSectionalConfig]] = {}
    for name, (filtered, median_n, recommended) in per_group.items():
        if auto_n_groups:
            target = unified if unified is not None else recommended
            if target != base_config.n_groups:
                cfg = replace(base_config, n_groups=target)
            else:
                cfg = base_config
        else:
            cfg = base_config
            if base_config.n_groups > recommended:
                logger.warning(
                    "split_by_group: group '%s' median N=%d, "
                    "n_groups=%d may be too high (recommended: %d)",
                    name, median_n, base_config.n_groups, recommended,
                )

        result[name] = (filtered, cfg)

    return result
