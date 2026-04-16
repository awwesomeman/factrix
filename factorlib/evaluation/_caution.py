"""CAUTION condition checks after all gates pass."""

from __future__ import annotations

import warnings

from factorlib.config import BaseConfig, CrossSectionalConfig, EventConfig, MacroCommonConfig, MacroPanelConfig
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
        case EventConfig():
            return _event_signal_caution(artifacts, gate_results, profile)
        case MacroPanelConfig():
            return _macro_panel_caution(artifacts, profile)
        case MacroCommonConfig():
            return _macro_common_caution(artifacts, profile)
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


def _event_signal_caution(
    artifacts: Artifacts,
    gate_results: list[GateResult],
    profile: FactorProfile,
) -> list[str]:
    reasons: list[str] = []
    cfg: EventConfig = artifacts.config  # type: ignore[assignment]

    bmp_m = profile.get("bmp_sar")
    n_events = bmp_m.metadata.get("n_events", 0) if bmp_m else 0
    if n_events < 30:
        reasons.append(
            f"Only {n_events} events (< 30) — statistical power is low"
        )

    for gr in gate_results:
        if gr.name == "event_significance":
            via = gr.detail.get("via", [])
            if via and via == ["hit_rate"]:
                reasons.append(
                    "Significance passed only via hit_rate, not CAAR or BMP"
                    " — direction is correct but economic magnitude may be weak"
                )
            break

    caar_m = profile.get("caar")
    if caar_m is not None and bmp_m is not None:
        # WHY: if CAAR and BMP disagree on direction, CAAR may be
        # spuriously inflated by event-induced variance.
        if (caar_m.value * bmp_m.value) < 0 and abs(bmp_m.value) > 1e-9:
            reasons.append(
                "CAAR and BMP have opposite signs"
                " — CAAR may be inflated by event-induced variance"
            )
        # WHY: if CAAR is significant but BMP fails to confirm,
        # event-induced variance may be inflating the CAAR t-stat.
        caar_t = abs(caar_m.stat or 0.0)
        bmp_z = abs(bmp_m.stat or 0.0)
        if caar_t >= 2.0 and bmp_z < 1.5:
            reasons.append(
                "CAAR is significant but BMP does not confirm"
                " — CAAR t-stat may be inflated by event-induced variance"
            )

    n_assets = artifacts.prepared["asset_id"].n_unique()
    clust = profile.get("clustering_hhi")
    if clust is not None and n_assets > 1:
        hhi_norm = clust.metadata.get("hhi_normalized", 0.0)
        if hhi_norm > 0.3 and cfg.adjust_clustering == "none":
            reasons.append(
                f"Event clustering HHI_normalized = {hhi_norm:.2f} (> 0.30)"
                " — independence assumption may be violated."
                " Consider setting adjust_clustering='kolari_pynnonen'"
            )

    reasons.extend(_check_trend_decay(profile, "caar_trend", "CAAR"))

    return reasons


def _macro_panel_caution(
    artifacts: Artifacts,
    profile: FactorProfile,
) -> list[str]:
    reasons: list[str] = []
    cfg = artifacts.config

    if isinstance(cfg, MacroPanelConfig):
        median_n = _median_universe_size(artifacts.prepared)
        if median_n < cfg.min_cross_section:
            reasons.append(
                f"Median cross-section N = {median_n} (< {cfg.min_cross_section})"
                " — FM β estimates may be unreliable"
            )

    fm = profile.get("fm_beta")
    pooled = profile.get("pooled_beta")
    if fm is not None and pooled is not None:
        if fm.value * pooled.value < 0:
            reasons.append(
                "FM β and Pooled β have opposite signs"
                " — robustness check failed"
            )

    reasons.extend(_check_trend_decay(profile))

    return reasons


def _macro_common_caution(
    artifacts: Artifacts,
    profile: FactorProfile,
) -> list[str]:
    reasons: list[str] = []

    r2 = profile.get("mean_r_squared")
    if r2 is not None and r2.value < 0.01:
        reasons.append(
            f"Mean R² = {r2.value:.4f} — common factor explains very little"
            " return variation across assets"
        )

    sign_cons = profile.get("ts_beta_sign_consistency")
    if sign_cons is not None and sign_cons.value < 0.6:
        reasons.append(
            f"β sign consistency = {sign_cons.value:.0%}"
            " — assets disagree on exposure direction"
        )

    reasons.extend(_check_trend_decay(profile))

    return reasons


def _check_trend_decay(
    profile: FactorProfile,
    key: str = "beta_trend",
    label: str = "β",
) -> list[str]:
    trn = profile.get(key)
    if trn is not None:
        ci_excludes_zero = trn.metadata.get("ci_excludes_zero", False)
        if trn.value < 0 and ci_excludes_zero:
            return [f"{label} trend shows significant decay"]
    return []
