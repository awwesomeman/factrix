"""Pinned metric sets — independent of ``factrix.evaluate`` defaults
so a default-set tweak in factrix cannot silently shift the baseline.

Each ``MetricSet`` declares the names ``evaluate`` should dispatch
(``metric_specs``) and an optional ``custom`` callable for paths
that don't live behind ``evaluate`` (e.g. ``greedy_forward_selection``,
direct ``bootstrap_mean_ci``). Scenarios pick a set + cell and the
helper in ``bench.scenarios._helpers`` does the dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from factrix import MetricSpec, spec_by_name

# Bumped when a metric set's membership / parameterization changes in
# a way that should refuse silent comparison against earlier baselines.
METRIC_SET_VERSION = "1"


@dataclass(frozen=True)
class MetricSet:
    """A pinned bundle of metric calls.

    ``metric_specs`` dictates the bundle dispatched through ``factrix.evaluate``.
    ``custom`` is run on the same panel afterwards; its return value is
    discarded — the harness times the call, not the result. Scenarios
    that want to time a path without metrics (algo / bootstrap primitives)
    declare it here.
    """

    name: str
    metric_specs: tuple[MetricSpec, ...]
    custom: Callable[..., Any] | None = field(default=None, repr=False)


CORE = MetricSet(
    name="core",
    metric_specs=tuple(
        spec_by_name()[n] for n in ("ic", "quantile_spread", "monotonicity")
    ),
)

HEAVY = MetricSet(
    name="heavy",
    # Share CORE's tuple by reference so adding a metric to `core`
    # automatically extends `heavy` — `heavy = core + bootstrap` is
    # the conceptual relationship, not two parallel literals.
    # Bootstrap supplement is layered at the scenario level (S1 /
    # M-ic-boot) rather than via evaluate, so it lives outside
    # this tuple.
    metric_specs=CORE.metric_specs,
)

ALGO = MetricSet(
    name="algo",
    metric_specs=(),
)

EVENT = MetricSet(
    name="event",
    # Only `corrado_rank` dispatches through ``evaluate``;
    # ``caar`` and ``mfe_mae_summary`` require pre-computed event-row
    # inputs and are called directly by the sparse scenario. The set
    # name is the JSONL label for the conceptual bundle; the
    # ``metric_specs`` tuple reflects what ``evaluate`` can
    # actually take.
    metric_specs=(spec_by_name()["corrado_rank"],),
)


SETS: dict[str, MetricSet] = {
    "core": CORE,
    "heavy": HEAVY,
    "algo": ALGO,
    "event": EVENT,
}


def get(name: str) -> MetricSet:
    """Lookup a metric set by name; raises KeyError on miss."""
    try:
        return SETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric_set {name!r}; known: {sorted(SETS)}") from exc
