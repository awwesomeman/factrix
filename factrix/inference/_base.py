"""Inference base — dimension-agnostic protocol + result shape.

``Inference`` is the single user-facing noun for a statistical inference
method: a frozen dataclass that carries its own ``compute`` plus identity
labels. The Protocol constrains only the base identity ClassVars
(``test`` / ``summary``); ``compute`` is deliberately **not** in the
Protocol because its signature varies by target shape (series-mean /
slice / panel) and a single Protocol cannot honestly cover all of them.
Derived ClassVars (e.g. ``se``, or the series-mean family's
``consumes_full_series``) are declared by downstream dataclasses as
needed, not hoisted into the base Protocol.

``InferenceResult`` is the harmonized return shape compute methods emit:
its ``stat`` / ``p_value`` feed a ``MetricResult`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from factrix._codes import WarningCode


@runtime_checkable
class Inference(Protocol):
    """Statistical inference method identity.

    Implementations are frozen dataclasses that additionally carry a
    ``compute`` whose signature is fixed per target shape (series-mean
    members share ``compute(data, *, value_col, overlap_periods,
    alternative="two-sided")``).
    ``compute`` is intentionally absent here — see module docstring.
    """

    test: str
    """Test-statistic family label (e.g. ``"t"``)."""

    summary: str
    """One-line human-readable description of the method."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Harmonized return shape for an ``Inference.compute`` call.

    ``stat`` / ``p_value`` are the test statistic and its requested-tail p-value;
    they feed a ``MetricResult`` directly. ``metadata`` is a flat
    ``str -> Any`` map (non-overlapping emits ``stride`` / sample counts;
    Newey-West emits ``newey_west_lags``). ``warnings`` carries soft-floor /
    kernel-clamp signals.
    """

    stat: float
    p_value: float
    metadata: Mapping[str, Any]
    warnings: frozenset[WarningCode]
    # Point estimate and sample size of the series the test actually ran on.
    # A stride-based method tests a subsample, so the headline ``value`` /
    # ``n_obs`` a metric reports must come from here, not from the full
    # input — otherwise value, stat and n describe different samples.
    estimate: float | None = None
    n_obs: int | None = None
