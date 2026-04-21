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
- User-defined rules join via ``register_rule(factor_type, Rule)``.
  They append to the per-type built-in list so built-ins stay
  authoritative; custom rules run after them in order of registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from factorlib._types import (
    Diagnostic,
    DiagnosticSeverity,
    FactorType,
    coerce_factor_type,
)

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
        message: Human-readable explanation of the condition. May be
            either a plain string or a ``Callable[[profile], str]`` when
            the message depends on runtime state (e.g. listing which
            metrics short-circuited).
        predicate: Called with the profile; when True, the rule emits a
            Diagnostic. Must be pure (no IO, no mutation).
    """

    code: str
    severity: DiagnosticSeverity
    message: "str | Callable[[P], str]"
    predicate: Callable[[P], bool]
    recommended_p_source: str | None = None

    def evaluate(self, profile: P) -> Diagnostic | None:
        if not self.predicate(profile):
            return None
        msg = self.message(profile) if callable(self.message) else self.message
        return Diagnostic(
            severity=self.severity,
            message=msg,
            code=self.code,
            recommended_p_source=self.recommended_p_source,
        )


# ---------------------------------------------------------------------------
# Cross-type rules (apply to every profile regardless of factor type)
# ---------------------------------------------------------------------------

def _insufficient_message(profile: object) -> str:
    names = getattr(profile, "insufficient_metrics", ())
    joined = ", ".join(names) if names else "?"
    return (
        f"One or more metrics short-circuited due to insufficient data "
        f"({joined}). The verdict and canonical p-value are technically "
        f"computable but may not be trustworthy — collect more data or "
        f"re-scope the factor. See the metric's metadata['reason'] for "
        f"the exact threshold it missed."
    )


CROSS_TYPE_RULES: list[Rule] = [
    Rule(
        code="data.insufficient",
        severity="warn",
        message=_insufficient_message,
        predicate=lambda p: bool(getattr(p, "insufficient_metrics", ())),
    ),
]


# ---------------------------------------------------------------------------
# Cross-sectional
# ---------------------------------------------------------------------------

def _ortho_absorbed_message(profile: "CrossSectionalProfile") -> str:
    r2 = profile.orthogonalize_r2_mean
    return (
        f"Basis factors absorbed {r2:.1%} of variance (R²>0.7). The "
        "residual signal is small; IC / spread reflect it, not the "
        "original factor. Check whether the basis set is appropriate."
    )


CROSS_SECTIONAL_RULES: list[Rule["CrossSectionalProfile"]] = [
    Rule(
        code="cs.orthogonalize_absorbed_most",
        severity="warn",
        message=_ortho_absorbed_message,
        predicate=lambda p: (
            p.orthogonalize_r2_mean is not None
            and p.orthogonalize_r2_mean > 0.7
        ),
    ),
    Rule(
        code="cs.regime_ic_inconsistent",
        severity="warn",
        message=(
            "IC direction flips across regimes — factor is "
            "regime-dependent; behaviour in bull may reverse in bear."
        ),
        predicate=lambda p: p.regime_ic_consistent is False,
    ),
    Rule(
        code="cs.multi_horizon_decay_fast",
        severity="warn",
        message=(
            "|IC| retains <30% from shortest to longest horizon — signal "
            "is short-lived, likely overfitting to the shortest horizon."
        ),
        # WHY: abs() — sign-flip (retention < 0) is a separate pathology
        # reported by multi_horizon_ic_monotonic=False; this rule is
        # about magnitude decay only, so we compare the absolute ratio.
        predicate=lambda p: (
            p.multi_horizon_ic_retention is not None
            and abs(p.multi_horizon_ic_retention) < 0.3
        ),
    ),
    Rule(
        code="cs.spanning_alpha_absorbed",
        severity="warn",
        message=(
            "Spanning alpha not significant (p > 0.10) — the factor may "
            "be a repackaging of supplied base factors rather than a "
            "new source of alpha."
        ),
        predicate=lambda p: (
            p.spanning_alpha_p is not None and p.spanning_alpha_p > 0.10
        ),
    ),
    Rule(
        code="cs.ic_weak_spread_strong",
        severity="warn",
        message=(
            "IC not significant (p > 0.05) but long-short spread is "
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
        code="cs.top_concentration_high",
        severity="veto",
        message=(
            "top-bucket effective-N / total-N < 0.5 — long leg concentrated "
            "in a few stocks; alpha is not broadly spread."
        ),
        predicate=lambda p: p.top_concentration_eff_ratio < 0.5,
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
    # WHY: when ``ic_p`` rejects H0 but ``ic_nw_p`` does not, the
    # non-overlapping t-test's significance is likely inflated by
    # autocorrelation that the HAC version correctly deflates.
    # ``recommended_p_source="ic_nw_p"`` surfaces the defensible
    # alternative without auto-switching canonical.
    Rule(
        code="cs.overlapping_returns_inflates_ic",
        severity="warn",
        message=(
            "ic_p ≤ 0.05 but ic_nw_p > 0.10 — overlap-induced "
            "autocorrelation is likely inflating IC significance. "
            "Consider BHY on ic_nw_p instead."
        ),
        predicate=lambda p: p.ic_p <= 0.05 and p.ic_nw_p > 0.10,
        recommended_p_source="ic_nw_p",
    ),
]


# ---------------------------------------------------------------------------
# Event signal
# ---------------------------------------------------------------------------

EVENT_RULES: list[Rule["EventProfile"]] = [
    Rule(
        code="event.single_asset",
        severity="info",
        message=(
            "Single-asset panel — CAAR degrades from a cross-sectional "
            "average of abnormal returns to a per-event time average on "
            "one asset; BMP uses Patell-style per-event SE (no "
            "cross-sectional std); clustering_hhi is disabled (needs N≥2). "
            "Verify the signal is an event study on this specific instrument, "
            "not a misrouted cross-sectional factor."
        ),
        # clustering_hhi is None iff n_assets == 1 (see _build_event_artifacts).
        predicate=lambda p: p.clustering_hhi is None,
    ),
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
            (p.caar_mean * p.bmp_test_mean) < 0 and abs(p.bmp_test_mean) > 1e-9
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
    # WHY: ADF fail-to-reject signals the factor likely has a unit root,
    # so per-asset β estimates carry Stambaugh (1999) bias. No
    # ``recommended_p_source`` — the proper remediation is a user-level
    # Stambaugh reverse-regression or first-difference of the factor,
    # not a simple alternative p-value inside this library.
    Rule(
        code="macro_common.factor_persistent",
        severity="warn",
        message=(
            "Factor ADF p > 0.10 — likely unit root; per-asset β and "
            "β t-stats are susceptible to Stambaugh bias. Consider "
            "first-differencing the factor or applying a reverse-"
            "regression correction before acting on ts_beta_p."
        ),
        predicate=lambda p: p.factor_adf_p > 0.10,
    ),
]


# ---------------------------------------------------------------------------
# Custom (user-registered) rules
# ---------------------------------------------------------------------------

# Keyed by FactorType, appended after built-ins at dispatch time.
#
# Thread-safety: this registry is a plain dict without locking. factorlib
# today is single-threaded end-to-end (Polars handles its own parallelism
# inside vectorized ops; our Python control flow is linear). If batch
# evaluation ever spawns worker processes/threads, each worker should
# register its own rules in isolation, or a lock must be added here.
_CUSTOM_RULES: dict[FactorType, list[Rule]] = {}


def register_rule(factor_type: FactorType | str, rule: Rule) -> None:
    """Register a user-defined diagnostic rule for a factor type.

    The rule runs after built-in rules on every ``profile.diagnose()``
    call whose profile matches ``factor_type``. Use this to plug in
    domain-specific checks (earnings-window sensitivity, liquidity
    filters, custom regime signals) without forking factorlib.

    Args:
        factor_type: Which factor type the rule applies to. Accepts
            either a ``FactorType`` enum or its string value
            (``"cross_sectional"`` etc.).
        rule: A ``Rule`` instance. Its predicate receives the matching
            Profile dataclass; it must be pure.
    """
    _CUSTOM_RULES.setdefault(coerce_factor_type(factor_type), []).append(rule)


def clear_custom_rules(factor_type: FactorType | str | None = None) -> None:
    """Remove registered custom rules.

    Args:
        factor_type: If given, clear only that factor type. If ``None``
            (default), clear every registered custom rule. Useful as a
            test fixture — never call from production code paths.
    """
    if factor_type is None:
        _CUSTOM_RULES.clear()
    else:
        _CUSTOM_RULES.pop(coerce_factor_type(factor_type), None)


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
        factor_type = FactorType.CROSS_SECTIONAL
    elif isinstance(profile, EventProfile):
        rules = EVENT_RULES
        factor_type = FactorType.EVENT_SIGNAL
    elif isinstance(profile, MacroPanelProfile):
        rules = MACRO_PANEL_RULES
        factor_type = FactorType.MACRO_PANEL
    elif isinstance(profile, MacroCommonProfile):
        rules = MACRO_COMMON_RULES
        factor_type = FactorType.MACRO_COMMON
    else:
        raise TypeError(
            f"No diagnostic rule set for profile type {type(profile).__name__}. "
            f"Supported: CrossSectionalProfile / EventProfile / "
            f"MacroPanelProfile / MacroCommonProfile."
        )

    out: list[Diagnostic] = []
    # Cross-type rules fire first — data-quality signals should precede
    # factor-type-specific diagnostics so the reader sees the warning
    # about trustworthiness before the interpretation notes.
    for rule in CROSS_TYPE_RULES:
        hit = rule.evaluate(profile)
        if hit is not None:
            out.append(hit)
    for rule in rules:
        hit = rule.evaluate(profile)
        if hit is not None:
            out.append(hit)
    # User-registered rules run last so they can reference cross-type
    # diagnostics without reordering built-ins.
    for rule in _CUSTOM_RULES.get(factor_type, ()):
        hit = rule.evaluate(profile)
        if hit is not None:
            out.append(hit)
    return out
