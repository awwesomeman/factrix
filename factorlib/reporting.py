"""User-facing reporting helpers for factor profiles.

``describe_profile_values(profile)`` prints a summary table rendered
directly from the Profile dataclass — no ``Artifacts`` handle required.
The Profile scalar dataclass stays unchanged (frozen / slots / polars-
native); this module is the place where its fields are formatted for
humans.

Per-regime / per-horizon / spanning-beta breakdowns live in
``artifacts.metric_outputs[key].metadata`` (dict of per-bucket stats).
When you need those, iterate them directly — e.g.
``arts.metric_outputs["regime_ic"].metadata["per_regime"]``.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from factorlib.evaluation.profiles._base import FactorProfile


__all__ = ["describe_profile_values"]


# Fields handled in the header / not useful as a row in the value table.
_SKIP_FIELDS: frozenset[str] = frozenset({
    "factor_name", "n_periods", "insufficient_metrics",
})


def describe_profile_values(profile: "FactorProfile") -> None:
    """Print a summary of every non-None value on ``profile``.

    Args:
        profile: A typed ``FactorProfile`` returned by ``fl.evaluate(...)``
            (or ``f.evaluate()`` on a ``fl.factor()`` session).

    Renders:
        - Header: factor name, profile class, n_periods, verdict.
        - One row per dataclass field with a non-None value, so L2
          opt-in summaries (``regime_ic_min_tstat`` etc.) appear only
          when the corresponding config was enabled.

    For per-regime / per-horizon / spanning-beta drill-down, iterate
    ``artifacts.metric_outputs[key].metadata`` directly — available from
    ``fl.evaluate(..., return_artifacts=True)`` or ``fl.factor()``.
    """
    _print_header(profile)
    _print_value_table(profile)


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


def _print_value_table(profile: "FactorProfile") -> None:
    rows: list[tuple[str, Any]] = []
    for f in fields(profile):
        if f.name in _SKIP_FIELDS:
            continue
        value = getattr(profile, f.name)
        if value is None:
            continue
        rows.append((f.name, value))
    if not rows:
        return
    name_w = max(len(n) for n, _ in rows) + 2
    print("  Values:")
    for name, value in rows:
        print(f"    {name:<{name_w}} {_fmt_value(value)}")
    print()


def _fmt_value(x: Any) -> str:
    """Render floats with 4 decimals; ints / bools unadorned; tuple/list short."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4f}"
    if isinstance(x, (tuple, list)):
        return f"[{len(x)} items]" if len(x) > 4 else str(tuple(x))
    return str(x)
