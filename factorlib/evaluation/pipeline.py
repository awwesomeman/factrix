"""Factor evaluation pipeline — the main runner.

``evaluate_factor`` is the single entry point:

1. Build ``Artifacts`` (IC series, spread series) — computed once, shared.
2. Run gates sequentially — short-circuit on FAILED/VETOED.
3. If all gates pass → compute profile + check CAUTION conditions.

Usage::

    from factorlib.evaluation.pipeline import evaluate_factor
    from factorlib.evaluation.presets import CROSS_SECTIONAL_GATES
    from factorlib.config import PipelineConfig

    result = evaluate_factor(df, "Mom_20D", CROSS_SECTIONAL_GATES, PipelineConfig())
"""

from __future__ import annotations

import polars as pl

from factorlib.evaluation._protocol import (
    Artifacts,
    EvaluationResult,
    FactorProfile,
    GateFn,
    GateResult,
)
from factorlib.config import PipelineConfig
from factorlib.evaluation.profile import compute_profile
from factorlib.metrics._helpers import _median_universe_size
from factorlib.metrics.ic import compute_ic
from factorlib.metrics.quantile import compute_spread_series


def evaluate_factor(
    df: pl.DataFrame,
    factor_name: str,
    gates: list[GateFn],
    config: PipelineConfig,
) -> EvaluationResult:
    """Run gate-based factor evaluation.

    Args:
        df: Preprocessed panel with ``date, asset_id, factor, forward_return``.
            Should already have been through ``preprocess_cs_factor``
            (and optionally ``orthogonalize_factor``).
        factor_name: Identifier for the factor being evaluated.
        gates: Ordered list of gate functions. Use presets or compose your own.
        config: Pipeline configuration.

    Returns:
        EvaluationResult with overall status, per-gate details, and profile.
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
                profile=None,
            )

    # All gates passed → compute profile, then check caution using profile
    profile = compute_profile(artifacts)
    caution_reasons = _check_caution(artifacts, gate_results, profile)

    return EvaluationResult(
        factor_name=factor_name,
        status="CAUTION" if caution_reasons else "PASS",
        gate_results=gate_results,
        profile=profile,
        caution_reasons=caution_reasons,
    )


def build_artifacts(df: pl.DataFrame, config: PipelineConfig) -> Artifacts:
    """Pre-compute shared intermediate results.

    Called once before any gate runs. All gates and profile read from
    the same Artifacts instance.
    """
    required = {"date", "asset_id", "factor", "forward_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_artifacts: missing required columns {missing}. "
            f"Available: {df.columns}"
        )

    ic_series = compute_ic(df)
    ic_values = ic_series.rename({"ic": "value"})
    spread_series = compute_spread_series(
        df, config.forward_periods, config.n_groups,
    )
    return Artifacts(
        prepared=df,
        ic_series=ic_series,
        ic_values=ic_values,
        spread_series=spread_series,
        config=config,
    )


def _check_caution(
    artifacts: Artifacts,
    gate_results: list[GateResult],
    profile: FactorProfile,
) -> list[str]:
    """Check CAUTION conditions per v3 spec section 5.

    Args:
        artifacts: Pipeline artifacts.
        gate_results: Results from all passed gates.
        profile: Computed factor profile (for IC trend check).

    Returns:
        List of human-readable caution reasons (empty = no caution).
    """
    reasons: list[str] = []
    cfg = artifacts.config

    # 1. Orthogonalization not applied
    if not cfg.orthogonalize:
        reasons.append(
            "Step 6 (orthogonalize) not applied"
            " — factor exposures may overlap known risk factors"
        )

    # 2. Universe too small
    median_n = _median_universe_size(artifacts.prepared)
    if median_n < 200:
        reasons.append(
            f"Median universe size = {median_n:.0f} (< 200)"
            " — quantile analysis may be unstable"
        )

    # 3. Gate 1 passed only via Q1-Q5 spread (not IC)
    for gr in gate_results:
        if gr.name == "significance":
            via = gr.detail.get("via", [])
            if via == ["Q1-Q5_spread"]:
                reasons.append(
                    "Significance passed only via Q1-Q5 spread, not IC"
                    " — may be driven by outlier stocks"
                )
            break

    # 4. IC trend shows significant decay
    for metric in profile.reliability:
        if metric.name == "ic_trend":
            ci_excludes_zero = metric.metadata.get("ci_excludes_zero", False)
            if metric.value < 0 and ci_excludes_zero:
                reasons.append("IC trend shows significant decay")
            break

    # 5. Q1 too concentrated
    for metric in profile.profitability:
        if metric.name == "q1_concentration":
            # WHY: eff_n / n_q1 < 0.5 means >50% of Q1 weight is in a few stocks
            ratio = metric.metadata.get("ratio_eff_to_total", 1.0)
            if ratio < 0.5:
                reasons.append(
                    f"Q1 concentration too high (effective N / total = {ratio:.2f})"
                    " — alpha may be driven by a few stocks"
                )
            break

    return reasons
