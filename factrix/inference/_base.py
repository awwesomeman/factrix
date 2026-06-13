"""Inference base — dimension-agnostic protocol + result shape.

``Inference`` is the single user-facing noun for a statistical inference
method: a frozen dataclass that carries its own ``compute`` plus identity
labels. The Protocol constrains only the base identity ClassVars
(``test`` / ``summary``); ``compute`` is deliberately **not** in the
Protocol because its signature varies by target shape (series-mean /
slice / panel) and a single Protocol cannot honestly cover all of them.
Derived ClassVars (e.g. ``se``) are declared by downstream dataclasses as
needed, not hoisted into the base Protocol.

``InferenceResult`` is the harmonized return shape compute methods emit.
``stat_name`` / ``p_name`` are ``None`` for a metric-internal inference
unit that claims no ``profile.stats`` ``StatCode`` key — its
``stat`` / ``p_value`` feed a ``MetricResult`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from factrix._codes import StatCode, WarningCode


@runtime_checkable
class Inference(Protocol):
    """Statistical inference method identity.

    Implementations are frozen dataclasses that additionally carry a
    ``compute`` whose signature is fixed per target shape (series-mean
    members share ``compute(df, *, value_col, forward_periods)``).
    ``compute`` is intentionally absent here — see module docstring.
    """

    test: str
    """Test-statistic family label (e.g. ``"t"``)."""

    summary: str
    """One-line human-readable description of the method."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Harmonized return shape for an ``Inference.compute`` call.

    ``stat`` / ``p_value`` are the test statistic and two-sided p-value.
    ``stat_name`` / ``p_name`` key the values into ``FactorProfile.stats``
    when the method is a discoverable profile estimator; they are ``None``
    for metric-internal units (the ``stat`` / ``p_value`` then feed a
    ``MetricResult`` directly). ``metadata`` is a flat ``str -> Any`` map
    (non-overlapping emits ``stride`` / sample counts; Newey-West emits
    ``nw_lags``). ``warnings`` carries soft-floor / kernel-clamp signals.
    """

    stat: float
    p_value: float
    stat_name: StatCode | None
    p_name: StatCode | None
    metadata: Mapping[str, Any]
    warnings: frozenset[WarningCode]
