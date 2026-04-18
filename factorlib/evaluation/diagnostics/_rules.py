"""Diagnostic rules for per-type FactorProfile.

Rules are pure-function predicates over a typed Profile. Each rule
encapsulates a single condition -- "IC is not significant but spread
is", "FM β and pooled β disagree on sign", etc. -- plus a severity
and a human-readable message.

Design notes:
- Rules live outside the Profile classes so the rule set can grow and
  evolve without churning the dataclass schema.
- Dispatch is by isinstance rather than a registry. The four profile
  types are a closed set for Phase A; if we add more types later, we
  add a branch here. Closed-world dispatch is simpler and self-
  documenting.
- Rules intentionally use plain lambdas instead of a Protocol/ABC.
  They are trivial enough that the indirection would cost readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from factorlib._types import Diagnostic, DiagnosticSeverity

if TYPE_CHECKING:
    from factorlib.evaluation.profiles.cross_sectional import (
        CrossSectionalProfile,
    )
    from factorlib.evaluation.profiles.event import EventProfile
    from factorlib.evaluation.profiles.macro_common import MacroCommonProfile
    from factorlib.evaluation.profiles.macro_panel import MacroPanelProfile


P = TypeVar("P")


@dataclass(frozen=True, slots=True)
class Rule(Generic[P]):
    """A single diagnostic condition over a Profile.

    Args:
        code: Stable machine-readable id (for AI / programmatic triage).
        severity: One of ``info``, ``warn``, ``veto``.
        message: Human-readable explanation of the condition.
        predicate: Called with the profile; when True, the rule emits a
            Diagnostic. Must be pure (no IO, no mutation).
    """

    code: str
    severity: DiagnosticSeverity
    message: str
    predicate: Callable[[P], bool]

    def evaluate(self, profile: P) -> Diagnostic | None:
        return (
            Diagnostic(
                severity=self.severity,
                message=self.message,
                code=self.code,
            )
            if self.predicate(profile)
            else None
        )


# ---------------------------------------------------------------------------
# Cross-sectional
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_RULES: list[Rule["CrossSectionalProfile"]] = [
    # TODO: replace with a measured residual-exposure diagnostic (e.g.
    # FM beta of the factor against size / value / momentum) once that
    # metric lands. Today this rule fires on every factor with the default
    # orthogonalize=False, which is why severity is 'info'.
    Rule(
        code="cs.orthogonalize_not_applied",
        severity="info",
        message=(
            "orthogonalize=False — factor exposures were not regressed "
            "against the market's standard risk factors (size / value / "
            "momentum / industry). Any alpha observed here may be a "
            "repackaging of a known risk premium."
        ),
        predicate=lambda p: not p.orthogonalize_applied,
    ),
    Rule(
        code="cs.ic_weak_spread_strong",
        severity="warn",
        message=(
            "IC not significant (p > 0.05) but Q1-Q5 spread is "
            "— alpha may be driven by outlier stocks rather than the "
            "broad cross-section."
        ),
        predicate=lambda p: p.ic_p > 0.05 and p.spread_p <= 0.05,
    ),
    Rule(
        code="cs.small_universe",
        severity="warn",
        message=(
            "Median universe size < 200 — quantile analysis may be "
            "unstable; consider MacroPanelConfig (smaller-N workflows)."
        ),
        predicate=lambda p: p.median_universe_n < 200,
    ),
    Rule(
        code="cs.ic_trend_decay",
        severity="warn",
        message="IC trend is significantly negative — signal may be decaying.",
        predicate=lambda p: p.ic_trend < 0 and p.ic_trend_p <= 0.05,
    ),
    Rule(
        code="cs.q1_concentration_high",
        severity="veto",
        message=(
            "Q1 effective-N / total-N < 0.5 — long leg concentrated "
            "in a few stocks; alpha is not broadly spread."
        ),
        predicate=lambda p: p.q1_concentration_eff_ratio < 0.5,
    ),
    Rule(
        code="cs.oos_sign_flipped",
        severity="veto",
        message=(
            "OOS sample shows a sign flip vs IS — overfitting risk; "
            "the factor did not generalize."
        ),
        predicate=lambda p: p.oos_sign_flipped,
    ),
    Rule(
        code="cs.oos_survival_low",
        severity="warn",
        message=(
            "OOS survival ratio < 0.5 (|mean_OOS| / |mean_IS| < 0.5) — "
            "half or more of the signal did not carry over to the "
            "out-of-sample split."
        ),
        predicate=lambda p: (
            p.oos_survival_ratio < 0.5 and not p.oos_sign_flipped
        ),
    ),
    Rule(
        code="cs.high_turnover",
        severity="warn",
        message=(
            "Turnover > 1.0 (average per-period rank churn) — "
            "implementation costs may erode live alpha."
        ),
        predicate=lambda p: p.turnover > 1.0,
    ),
]


# ---------------------------------------------------------------------------
# Event signal
# ---------------------------------------------------------------------------

EVENT_RULES: list[Rule["EventProfile"]] = [
    Rule(
        code="event.n_events_low",
        severity="warn",
        message=(
            "Fewer than 30 events — statistical power is low and "
            "CAAR / BMP t-stats are sensitive to individual outcomes."
        ),
        predicate=lambda p: p.n_events < 30,
    ),
    Rule(
        code="event.caar_bmp_sign_mismatch",
        severity="veto",
        message=(
            "CAAR and BMP have opposite signs — CAAR may be spuriously "
            "inflated by event-induced variance; use BMP as the "
            "trustworthy signal."
        ),
        predicate=lambda p: (
            (p.caar_mean * p.bmp_sar_mean) < 0 and abs(p.bmp_sar_mean) > 1e-9
        ),
    ),
    # Thresholds: |t|≥2 is the CAAR 95% single-test boundary; |z|<1.5
    # for BMP is looser than its own 95% cut so that "BMP barely fails
    # to confirm" still fires the diagnose. A |z|<2 gate would miss
    # the grey zone where BMP is at 0.075-ish — exactly the cases the
    # event-induced-variance warning is built for.
    Rule(
        code="event.caar_sig_bmp_not_confirm",
        severity="veto",
        message=(
            "CAAR t >= 2.0 but BMP z < 1.5 — CAAR significance is not "
            "confirmed after standardizing by event-window variance; "
            "likely event-induced variance inflation."
        ),
        predicate=lambda p: (
            abs(p.caar_tstat) >= 2.0 and abs(p.bmp_zstat) < 1.5
        ),
    ),
    Rule(
        code="event.hit_rate_only",
        severity="warn",
        message=(
            "Hit-rate is significant but CAAR and BMP are not — "
            "direction is correct on average, but economic magnitude "
            "is weak."
        ),
        predicate=lambda p: (
            p.caar_p > 0.05 and p.bmp_p > 0.05 and p.event_hit_rate_p <= 0.05
        ),
    ),
    Rule(
        code="event.clustering_high",
        severity="warn",
        message=(
            "Event clustering HHI_normalized > 0.30 and "
            "adjust_clustering='none' — event independence assumption is "
            "likely violated and no correction was applied. Set "
            "adjust_clustering to 'kolari_pynnonen' or "
            "'calendar_block_bootstrap'. Suppressed once an adjustment "
            "is in effect, since the t-stat already accounts for "
            "dependence."
        ),
        predicate=lambda p: (
            p.clustering_hhi_normalized is not None
            and p.clustering_hhi_normalized > 0.30
            and p.clustering_adjustment == "none"
        ),
    ),
    Rule(
        code="event.caar_trend_decay",
        severity="warn",
        message="CAAR trend is significantly negative — signal may be decaying.",
        predicate=lambda p: p.caar_trend < 0 and p.caar_trend_p <= 0.05,
    ),
    Rule(
        code="event.oos_sign_flipped",
        severity="veto",
        message=(
            "OOS sample shows a sign flip vs IS — overfitting risk."
        ),
        predicate=lambda p: p.oos_sign_flipped,
    ),
]


# ---------------------------------------------------------------------------
# Macro panel
# ---------------------------------------------------------------------------

MACRO_PANEL_RULES: list[Rule["MacroPanelProfile"]] = [
    Rule(
        code="macro_panel.small_cross_section",
        severity="warn",
        message=(
            "Median cross-section N below the configured "
            "min_cross_section floor — Fama-MacBeth β estimates are "
            "noisy; interpret λ t-stat with caution."
        ),
        predicate=lambda p: p.cross_section_below_min,
    ),
    Rule(
        code="macro_panel.fm_pooled_sign_mismatch",
        severity="veto",
        message=(
            "FM λ and pooled β have opposite signs — robustness check "
            "failed; the effect is not stable across FM and pooled methods."
        ),
        predicate=lambda p: (
            (p.fm_beta_mean * p.pooled_beta) < 0
            and abs(p.pooled_beta) > 1e-9
        ),
    ),
    Rule(
        code="macro_panel.beta_trend_decay",
        severity="warn",
        message=(
            "β trend is significantly negative — the risk-premium "
            "link may be weakening over time."
        ),
        predicate=lambda p: p.beta_trend < 0 and p.beta_trend_p <= 0.05,
    ),
    Rule(
        code="macro_panel.oos_sign_flipped",
        severity="veto",
        message=(
            "OOS β sample shows a sign flip vs IS — overfitting risk."
        ),
        predicate=lambda p: p.oos_sign_flipped,
    ),
]


# ---------------------------------------------------------------------------
# Macro common
# ---------------------------------------------------------------------------

MACRO_COMMON_RULES: list[Rule["MacroCommonProfile"]] = [
    Rule(
        code="macro_common.single_asset",
        severity="info",
        message=(
            "Single-asset panel — cross-sectional β t-test degenerates; "
            "reporting single-asset TS t-stat instead."
        ),
        predicate=lambda p: p.n_assets == 1,
    ),
    Rule(
        code="macro_common.r2_very_low",
        severity="warn",
        message=(
            "Mean R² < 1% — the common factor explains very little "
            "variance; economic relevance is weak even if β is significant."
        ),
        predicate=lambda p: p.mean_r_squared < 0.01,
    ),
    Rule(
        code="macro_common.sign_inconsistent",
        severity="warn",
        message=(
            "β sign consistency < 60% — assets disagree on exposure "
            "direction; the 'common' factor is not uniformly directional."
        ),
        predicate=lambda p: p.ts_beta_sign_consistency < 0.60,
    ),
    Rule(
        code="macro_common.beta_trend_decay",
        severity="warn",
        message="β trend is significantly negative — exposure may be fading.",
        predicate=lambda p: p.beta_trend < 0 and p.beta_trend_p <= 0.05,
    ),
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def diagnose_profile(profile: object) -> list[Diagnostic]:
    """Run the rule list for a profile's concrete type.

    Imports the profile modules lazily to avoid circular imports at
    package init time (profiles import from diagnostics via .diagnose()).
    """
    # Local imports: profile modules import from _base, which is fine;
    # but to keep this dispatch not at module-import time, we lazy-load.
    from factorlib.evaluation.profiles.cross_sectional import (
        CrossSectionalProfile,
    )
    from factorlib.evaluation.profiles.event import EventProfile
    from factorlib.evaluation.profiles.macro_common import MacroCommonProfile
    from factorlib.evaluation.profiles.macro_panel import MacroPanelProfile

    if isinstance(profile, CrossSectionalProfile):
        rules = CROSS_SECTIONAL_RULES
    elif isinstance(profile, EventProfile):
        rules = EVENT_RULES
    elif isinstance(profile, MacroPanelProfile):
        rules = MACRO_PANEL_RULES
    elif isinstance(profile, MacroCommonProfile):
        rules = MACRO_COMMON_RULES
    else:
        raise TypeError(
            f"No diagnostic rule set for profile type {type(profile).__name__}. "
            f"Supported: CrossSectionalProfile / EventProfile / "
            f"MacroPanelProfile / MacroCommonProfile."
        )

    out: list[Diagnostic] = []
    for rule in rules:
        hit = rule.evaluate(profile)
        if hit is not None:
            out.append(hit)
    return out
