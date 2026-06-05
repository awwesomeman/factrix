"""bench.metric_sets — pinned set membership + version."""

from __future__ import annotations

import pytest

from bench import metric_sets


def test_version_pinned():
    assert metric_sets.METRIC_SET_VERSION == "1"


def test_core_membership():
    assert tuple(s.name for s in metric_sets.CORE.metric_specs) == (
        "ic",
        "quantile_spread",
        "monotonicity",
    )


def test_heavy_extends_core():
    # `heavy` shares the core metric list; the bootstrap supplement
    # is layered on at the scenario level (see continuous.s1_evaluate
    # / m_ic_bootstrap) so the JSONL `metric_set` label stays clean.
    assert metric_sets.HEAVY.metric_specs == metric_sets.CORE.metric_specs


def test_event_set_lists_only_run_metrics_dispatchable():
    # `caar` and `mfe_mae_summary` are part of the event bundle
    # conceptually but require pre-computed event-row inputs;
    # `metric_specs` must contain only metrics `evaluate`
    # can dispatch directly.
    assert tuple(s.name for s in metric_sets.EVENT.metric_specs) == ("corrado_rank",)


def test_algo_is_run_metrics_empty():
    # `algo` (`greedy_forward_selection`) does not live behind
    # evaluate; the set is a label slot and the scenario calls the
    # function directly.
    assert metric_sets.ALGO.metric_specs == ()


def test_get_known_and_unknown():
    assert metric_sets.get("core") is metric_sets.CORE
    with pytest.raises(KeyError, match="unknown metric_set"):
        metric_sets.get("nope")
