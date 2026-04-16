"""CAUTION condition checks after all gates pass."""

from __future__ import annotations

import warnings

from factorlib.config import BaseConfig, CrossSectionalConfig
from factorlib.evaluation._protocol import (
    Artifacts,
    FactorProfile,
    GateResult,
)
from factorlib.metrics._helpers import _median_universe_size


def check_caution(
    artifacts: Artifacts,
    gate_results: list[GateResult],
    profile: FactorProfile,
) -> list[str]:
    """Check CAUTION conditions. Returns list of human-readable reasons."""
    match artifacts.config:
        case CrossSectionalConfig():
            return _cs_caution(artifacts, gate_results, profile)
        case _:
            return []


def warn_small_n(df_or_artifacts: object, config: BaseConfig) -> None:
    """Emit UserWarning if median cross-section N < 30 with a CS config."""
    if not isinstance(config, CrossSectionalConfig):
        return

    import polars as pl

    if isinstance(df_or_artifacts, Artifacts):
        prepared = df_or_artifacts.prepared
    elif isinstance(df_or_artifacts, pl.DataFrame):
        prepared = df_or_artifacts
    else:
        return

    if "asset_id" not in prepared.columns or "date" not in prepared.columns:
        return

    median_n = _median_universe_size(prepared)
    if median_n < 30:
        warnings.warn(
            f"Median cross-section size = {median_n} (< 30). "
            f"Consider using MacroPanelConfig instead of CrossSectionalConfig. "
            f"IC-based metrics may be unreliable at this N.",
            UserWarning,
            stacklevel=3,
        )


def _cs_caution(
    artifacts: Artifacts,
    gate_results: list[GateResult],
    profile: FactorProfile,
) -> list[str]:
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
