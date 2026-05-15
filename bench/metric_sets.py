"""Pinned metric sets — independent of ``factrix.run_metrics`` defaults
so a default-set tweak in factrix cannot silently shift the baseline
(#380 §3).

Each ``MetricSet`` declares the names ``run_metrics`` should dispatch
(``run_metrics_names``) and an optional ``custom`` callable for paths
that don't live behind ``run_metrics`` (e.g. ``greedy_forward_selection``,
direct ``bootstrap_mean_ci``). Scenarios pick a set + cell and the
helper in ``bench.scenarios._helpers`` does the dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Bumped when a metric set's membership / parameterization changes in
# a way that should refuse silent comparison against earlier baselines.
METRIC_SET_VERSION = "1"


@dataclass(frozen=True)
class MetricSet:
    """A pinned bundle of metric calls.

    ``run_metrics_names`` is passed straight to
    ``factrix.run_metrics(metrics=...)``. ``custom`` is run on the same
    panel afterwards; its return value is discarded — the harness times
    the call, not the result. Scenarios that want to time a non-
    ``run_metrics`` path (algo / bootstrap primitives) declare it here.
    """

    name: str
    run_metrics_names: tuple[str, ...]
    custom: Callable[..., Any] | None = field(default=None, repr=False)


CORE = MetricSet(
    name="core",
    run_metrics_names=("ic", "quantile_spread", "monotonicity"),
)

HEAVY = MetricSet(
    name="heavy",
    # Share CORE's tuple by reference so adding a metric to `core`
    # automatically extends `heavy` — `heavy = core + bootstrap` is
    # the conceptual relationship, not two parallel literals.
    # Bootstrap supplement is layered at the scenario level (S1 /
    # M-ic-boot) rather than via run_metrics, so it lives outside
    # this tuple.
    run_metrics_names=CORE.run_metrics_names,
)

ALGO = MetricSet(
    name="algo",
    run_metrics_names=(),
)

EVENT = MetricSet(
    name="event",
    run_metrics_names=("corrado_rank_test", "caar", "mfe_mae_summary"),
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
