"""User-facing reporting helpers for factor profiles.

``describe_profile_values(profile, artifacts)`` prints both the scalar
metric table and — when ``include_detail=True`` — per-regime /
per-horizon / spanning-alpha detail sections auto-discovered from
``artifacts.metric_outputs``. The Profile scalar dataclass stays
unchanged (frozen / slots / polars-native); this module is the place
where raw ``MetricOutput`` objects are rendered for humans.

For a batch of factors, pair each profile with its artifacts via
``arts_map[profile.factor_name]`` (see ``experiments/demo.ipynb``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from factorlib._types import MetricOutput
    from factorlib.evaluation._protocol import Artifacts
    from factorlib.evaluation.profiles._base import FactorProfile


__all__ = ["describe_profile_values"]


def describe_profile_values(
    profile: "FactorProfile",
    artifacts: "Artifacts",
    *,
    include_detail: bool = True,
) -> None:
    """Print a scalar summary + optional detail views for a factor profile.

    Args:
        profile: A typed ``FactorProfile`` returned by ``fl.evaluate(...)``.
        artifacts: The matching ``Artifacts`` returned when
            ``fl.evaluate(..., return_artifacts=True)`` (or looked up from
            ``evaluate_batch(..., keep_artifacts=True)`` via
            ``arts_map[profile.factor_name]``). Its ``metric_outputs`` is
            populated by the pipeline and ``Profile.from_artifacts``.
        include_detail: When True (default), after the scalar table, print
            per-regime / per-horizon / spanning-alpha breakdowns for each
            opt-in metric actually present in ``artifacts.metric_outputs``.

    Raises:
        ValueError: ``artifacts.metric_outputs`` is empty — likely because
            the caller hand-built the ``Artifacts`` rather than going
            through ``fl.evaluate(..., return_artifacts=True)``.
    """
    outputs = artifacts.metric_outputs
    if not outputs:
        raise ValueError(
            "artifacts.metric_outputs is empty. Pass an Artifacts returned "
            "by fl.evaluate(..., return_artifacts=True) or by "
            "fl.evaluate_batch(..., keep_artifacts=True)[profile.factor_name]."
        )

    _print_header(profile)
    _print_scalar_table(outputs)
    if include_detail:
        _print_detail_sections(outputs)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _print_header(profile: "FactorProfile") -> None:
    cls_name = type(profile).__name__
    try:
        verdict = profile.verdict()
    except Exception:  # noqa: BLE001
        verdict = "?"
    n_periods = getattr(profile, "n_periods", "?")
    print()
    print(
        f"  {profile.factor_name} — {cls_name}  "
        f"(n_periods={n_periods}, verdict={verdict})"
    )
    print(f"  {'-' * 60}")


# ---------------------------------------------------------------------------
# Scalar table
# ---------------------------------------------------------------------------

def _print_scalar_table(outputs: "dict[str, MetricOutput]") -> None:
    # Detail metrics are rendered separately; excluding them from the
    # scalar table avoids printing a tersely-summarized "regime_ic" line
    # that duplicates the detail section below.
    scalar_names = [n for n in outputs if n not in _DETAIL_RENDERERS]
    if not scalar_names:
        return
    name_w = max(len(n) for n in scalar_names) + 2
    print("  Metrics:")
    for name in scalar_names:
        m = outputs[name]
        value = _fmt_number(m.value)
        stat = _fmt_number(m.stat) if m.stat is not None else "-"
        sig = m.significance if m.significance else ""
        p = _fmt_number(m.metadata.get("p_value")) if "p_value" in m.metadata else "-"
        print(
            f"    {name:<{name_w}} value={value:<10s} stat={stat:<8s} "
            f"sig={sig:<4s} p={p}"
        )
    print()


# ---------------------------------------------------------------------------
# Detail sections
# ---------------------------------------------------------------------------

def _print_detail_sections(outputs: "dict[str, MetricOutput]") -> None:
    present = [k for k in _DETAIL_RENDERERS if k in outputs]
    if not present:
        return
    print("  Detail:")
    for key in present:
        _DETAIL_RENDERERS[key](outputs[key])
        print()


def _render_regime_detail(m: "MetricOutput") -> None:
    per_regime = m.metadata.get("per_regime") or {}
    consistent = m.metadata.get("direction_consistent")
    print(f"    regime_ic  [direction_consistent={consistent}]")
    if not per_regime:
        reason = m.metadata.get("reason") or "empty"
        print(f"      (no per-regime breakdown: {reason})")
        return
    # Column widths sized for the regime labels actually present.
    label_w = max(len(str(r)) for r in per_regime) + 2
    print(
        f"      {'regime':<{label_w}} {'mean_ic':>10s} {'stat':>8s} "
        f"{'p':>8s}  {'sig':<4s} n"
    )
    for regime, d in per_regime.items():
        print(
            f"      {str(regime):<{label_w}} "
            f"{_fmt_number(d.get('mean_ic')):>10s} "
            f"{_fmt_number(d.get('stat')):>8s} "
            f"{_fmt_number(d.get('p_value')):>8s}  "
            f"{str(d.get('significance') or ''):<4s} "
            f"{d.get('n_periods', '-')}"
        )


def _render_multi_horizon_detail(m: "MetricOutput") -> None:
    per_horizon = m.metadata.get("per_horizon") or {}
    print("    multi_horizon_ic")
    if not per_horizon:
        reason = m.metadata.get("reason") or "empty"
        print(f"      (no per-horizon breakdown: {reason})")
        return
    print(
        f"      {'horizon':>8s} {'mean_ic':>10s} {'stat':>8s} "
        f"{'p':>8s}  n"
    )
    for h, d in per_horizon.items():
        mic = d.get("mean_ic")
        mic_s = "nan" if (isinstance(mic, float) and math.isnan(mic)) else _fmt_number(mic)
        print(
            f"      {h:>8} {mic_s:>10s} "
            f"{_fmt_number(d.get('stat')):>8s} "
            f"{_fmt_number(d.get('p_value')):>8s}  "
            f"{d.get('n_periods', '-')}"
        )


def _render_spanning_detail(m: "MetricOutput") -> None:
    reason = m.metadata.get("reason")
    if reason:
        print(f"    spanning_alpha  (skipped: {reason})")
        return
    alpha = m.value
    t = m.stat
    p = m.metadata.get("p_value")
    sig = m.significance or ""
    r2 = m.metadata.get("r_squared")
    n_obs = m.metadata.get("n_obs")
    print("    spanning_alpha")
    print(
        f"      alpha={_fmt_number(alpha)}  t={_fmt_number(t)}  "
        f"p={_fmt_number(p)}  sig={sig}  R²={_fmt_number(r2)}  n={n_obs}"
    )
    betas = m.metadata.get("betas") or {}
    if betas:
        print("      betas:")
        for name, beta in betas.items():
            print(f"        {name:<20s} {_fmt_number(beta)}")


# Opt-in metrics that carry rich per-bucket detail in metadata. Order here
# defines render order of detail sections; defined after the renderers so
# the dict can hold the function objects directly.
_DETAIL_RENDERERS: "dict[str, Callable[[MetricOutput], None]]" = {
    "regime_ic": _render_regime_detail,
    "multi_horizon_ic": _render_multi_horizon_detail,
    "spanning_alpha": _render_spanning_detail,
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_number(x: Any) -> str:
    """Render floats with 4 decimals; ints unadorned; None/missing as '-'."""
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4f}"
    return str(x)
