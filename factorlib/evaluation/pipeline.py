"""Factor evaluation pipeline — the main runner.

``evaluate_factor`` is the single entry point:

1. Build ``Artifacts`` (intermediates computed once, shared).
2. Run gates sequentially — short-circuit on FAILED/VETOED.
3. If all gates pass → compute profile + check CAUTION conditions.

Usage::

    from factorlib.evaluation.pipeline import evaluate_factor
    from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES
    from factorlib.config import CrossSectionalConfig

    result = evaluate_factor(df, "Mom_20D", CROSS_SECTIONAL_GATES,
                             CrossSectionalConfig())
"""

from __future__ import annotations

import polars as pl

from factorlib.config import BaseConfig, CrossSectionalConfig
from factorlib.evaluation._protocol import (
    Artifacts,
    EvaluationResult,
    FactorProfile,
    GateFn,
    GateResult,
)
from factorlib.evaluation.profile import compute_profile
from factorlib.metrics._helpers import _median_universe_size
from factorlib.metrics.ic import compute_ic
from factorlib.metrics.quantile import compute_spread_series


def evaluate_factor(
    df: pl.DataFrame,
    factor_name: str,
    gates: list[GateFn],
    config: BaseConfig,
) -> EvaluationResult:
    """Run gate-based factor evaluation.

    Args:
        df: Preprocessed panel with ``date, asset_id, factor, forward_return``.
        factor_name: Identifier for the factor being evaluated.
        gates: Ordered list of gate functions. Use presets or compose your own.
        config: Pipeline configuration (use a concrete subclass).

    Returns:
        EvaluationResult with overall status, per-gate details, profile,
        and artifacts.
    """
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
    caution_reasons = _check_caution(artifacts, gate_results, profile)

    return EvaluationResult(
        factor_name=factor_name,
        status="CAUTION" if caution_reasons else "PASS",
        gate_results=gate_results,
        profile=profile,
        artifacts=artifacts,
        caution_reasons=caution_reasons,
    )


def build_artifacts(df: pl.DataFrame, config: BaseConfig) -> Artifacts:
    """Pre-compute shared intermediate results.

    Called once before any gate runs. All gates and profile read from
    the same Artifacts instance.
    """
    match config:
        case CrossSectionalConfig():
            return _build_cs_artifacts(df, config)
        case _:
            ft = type(config).factor_type
            raise NotImplementedError(
                f"build_artifacts not yet implemented for {ft}"
            )


def _build_cs_artifacts(
    df: pl.DataFrame, config: CrossSectionalConfig,
) -> Artifacts:
    required = {"date", "asset_id", "factor", "forward_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cross_sectional requires columns {required}. "
            f"Missing: {missing}. "
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


def _check_caution(
    artifacts: Artifacts,
    gate_results: list[GateResult],
    profile: FactorProfile,
) -> list[str]:
    """Check CAUTION conditions."""
    reasons: list[str] = []
    cfg = artifacts.config

    if isinstance(cfg, CrossSectionalConfig) and not cfg.orthogonalize:
        reasons.append(
            "Step 6 (orthogonalize) not applied"
            " — factor exposures may overlap known risk factors"
        )

    median_n = _median_universe_size(artifacts.prepared)
    if median_n < 200:
        reasons.append(
            f"Median universe size = {median_n:.0f} (< 200)"
            " — quantile analysis may be unstable"
        )

    for gr in gate_results:
        if gr.name == "significance":
            via = gr.detail.get("via", [])
            if via == ["Q1-Q5_spread"]:
                reasons.append(
                    "Significance passed only via Q1-Q5 spread, not IC"
                    " — may be driven by outlier stocks"
                )
            break

    ic_trend = profile.get("ic_trend")
    if ic_trend is not None:
        ci_excludes_zero = ic_trend.metadata.get("ci_excludes_zero", False)
        if ic_trend.value < 0 and ci_excludes_zero:
            reasons.append("IC trend shows significant decay")

    q1_conc = profile.get("q1_concentration")
    if q1_conc is not None:
        ratio = q1_conc.metadata.get("ratio_eff_to_total", 1.0)
        if ratio < 0.5:
            reasons.append(
                f"Q1 concentration too high (effective N / total = {ratio:.2f})"
                " — alpha may be driven by a few stocks"
            )

    return reasons
