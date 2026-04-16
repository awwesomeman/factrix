"""Factor evaluation pipeline.

``evaluate`` is the main entry point:

1. (Optional) Preprocess raw data.
2. Build ``Artifacts`` (intermediates computed once, shared).
3. Run gates sequentially — short-circuit on FAILED/VETOED.
4. If all gates pass → compute profile + check CAUTION conditions.

Usage::

    import factorlib as fl

    result = fl.evaluate(prepared, "Mom_20D",
                         config=fl.CrossSectionalConfig())
"""

from __future__ import annotations

import polars as pl

from factorlib.config import BaseConfig, CrossSectionalConfig
from factorlib.evaluation._caution import check_caution, warn_small_n
from factorlib.evaluation._protocol import (
    Artifacts,
    EvaluationResult,
    GateFn,
    GateResult,
)
from factorlib.evaluation.presets import default_gates_for
from factorlib.evaluation.profile import compute_profile
from factorlib.metrics.ic import compute_ic
from factorlib.metrics.quantile import compute_spread_series


def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    config: BaseConfig | None = None,
    gates: list[GateFn] | None = None,
    preprocess: bool = False,
) -> EvaluationResult:
    """Run gate-based factor evaluation.

    Args:
        df: Panel data. If ``preprocess=False`` (default), must already
            contain ``forward_return``. If ``preprocess=True``, must
            contain ``price``.
        factor_name: Identifier for the factor being evaluated.
        config: Pipeline configuration. Defaults to CrossSectionalConfig().
        gates: Gate functions. Defaults to type-appropriate preset.
        preprocess: If True, run preprocessing before evaluation.

    Returns:
        EvaluationResult with status, gate details, profile, and artifacts.
    """
    if config is None:
        config = CrossSectionalConfig()

    if preprocess:
        df = _preprocess(df, config)

    if gates is None:
        gates = default_gates_for(config)

    warn_small_n(df, config)

    artifacts = build_artifacts(df, config)

    gate_results: list[GateResult] = []
    for gate_fn in gates:
        result = gate_fn(artifacts)
        gate_results.append(result)
        if not result.passed:
            return EvaluationResult(
                factor_name=factor_name,
                status=result.status,
                gate_results=gate_results,
                artifacts=artifacts,
            )

    profile = compute_profile(artifacts)
    caution_reasons = check_caution(artifacts, gate_results, profile)

    return EvaluationResult(
        factor_name=factor_name,
        status="CAUTION" if caution_reasons else "PASS",
        gate_results=gate_results,
        profile=profile,
        artifacts=artifacts,
        caution_reasons=caution_reasons,
    )


def build_artifacts(df: pl.DataFrame, config: BaseConfig) -> Artifacts:
    """Pre-compute shared intermediate results."""
    match config:
        case CrossSectionalConfig():
            return _build_cs_artifacts(df, config)
        case _:
            ft = type(config).factor_type
            raise NotImplementedError(
                f"build_artifacts not yet implemented for {ft}"
            )


def _preprocess(df: pl.DataFrame, config: BaseConfig) -> pl.DataFrame:
    from factorlib.preprocess.pipeline import preprocess
    return preprocess(df, config=config)


def _build_cs_artifacts(
    df: pl.DataFrame, config: CrossSectionalConfig,
) -> Artifacts:
    required = {"date", "asset_id", "factor", "forward_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cross_sectional requires columns {required}. "
            f"Missing: {missing}.\n\n"
            f"Expected DataFrame schema:\n"
            f"  date           Datetime[ms]   — 交易日期\n"
            f"  asset_id       String         — 資產代碼\n"
            f"  factor         Float64        — 因子值（z-scored）\n"
            f"  forward_return Float64        — N 期前瞻報酬\n\n"
            f"Hint: call fl.preprocess(df) first, or set preprocess=True."
        )

    ic_series = compute_ic(df)
    ic_values = ic_series.rename({"ic": "value"})
    spread_series = compute_spread_series(
        df, config.forward_periods, config.n_groups,
    )
    return Artifacts(
        prepared=df,
        config=config,
        intermediates={
            "ic_series": ic_series,
            "ic_values": ic_values,
            "spread_series": spread_series,
        },
    )
