"""Dynamic sample-threshold resolver.

A metric whose floor is a function of its own parameters (e.g. ``ic``'s periods
floor scales with ``forward_periods``) declares ``sample_threshold`` as a
``Callable[[MetricBase], SampleThreshold]`` rather than a constant. The decorator
normalizes both forms into one resolver; the floor baked at the default config
lands on ``cls.sample_threshold`` so ``spec()`` / ``inspect_data`` can pre-flight
it, while the run-time path re-resolves the same resolver against the actual
instance.
"""

from __future__ import annotations

import factrix as fx
import pytest
from factrix._inspect import DataInspection, inspect_data
from factrix._types import MIN_IC_PERIODS
from factrix.metrics._registry import REGISTRY
from factrix.metrics.ic import ic


def _by_name(info: DataInspection, name: str):
    for m in info.metrics:
        if m.spec.name == name:
            return m
    raise KeyError(name)


class TestHookResolution:
    def test_spec_resolves_dynamic_floor_from_default_params(self):
        # ic default forward_periods=5, non-overlapping → MIN_IC_PERIODS * 5.
        st = ic.spec().sample_threshold
        assert st.min_periods == MIN_IC_PERIODS * 5

    def test_forward_periods_is_not_a_metric_param(self):
        # forward_periods is the data's overlap horizon, not a per-metric knob:
        # constructing a metric with it is rejected (the floor scales off the
        # signature default, resolved at spec() time — see the test above).
        with pytest.raises(fx.UserInputError):
            REGISTRY["ic"](forward_periods=10)

    def test_static_metric_keeps_empty_spec_threshold(self):
        # A constant-floor metric resolves to that constant (here empty); the
        # decorator normalizes it without ever constructing the (required-param,
        # non-default-constructible) instance.
        st = REGISTRY["net_spread"].spec().sample_threshold
        assert st.min_periods is None
        assert REGISTRY["net_spread"].sample_threshold.min_periods is None


class TestInspectDataPreflight:
    def test_dynamic_floor_gates_short_panel(self):
        # forward_periods=5 default needs MIN_IC_PERIODS*5 = 50 input periods;
        # 30 dates is short, so inspect_data now flags ic unusable — the floor
        # that used to be invisible at pre-flight.
        short = fx.datasets.make_cs_panel(n_assets=40, n_dates=30, seed=0)
        assert _by_name(inspect_data(short), "ic").usable is False

    def test_dynamic_floor_passes_long_panel(self):
        long_panel = fx.datasets.make_cs_panel(n_assets=40, n_dates=120, seed=0)
        assert _by_name(inspect_data(long_panel), "ic").usable is True


# Periods-axis metrics that sub-sample at ``forward_periods``: the resolver and
# the in-body gate must agree on one stride-scaled floor.


def _cs_panel(n_dates: int, *, n_assets: int = 60, market_cap: bool = False):
    import polars as pl
    from factrix.preprocess import compute_forward_return

    panel = compute_forward_return(
        fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=0),
        forward_periods=5,
    )
    if market_cap:
        panel = panel.with_columns(pl.lit(1e6).alias("market_cap"))
    return panel


def _hit_rate_series(n_dates: int):
    from datetime import datetime, timedelta

    import polars as pl

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    return pl.DataFrame({"date": dates, "value": [0.01] * n_dates}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


def _scalar(result):
    return result["factor"] if isinstance(result, dict) else result


# Each entry: metric name, base periods constant, a builder taking ``n_dates``,
# and per-call kwargs.
def _build_stride_sampled_cases():
    from factrix._types import (
        MIN_IC_PERIODS,
        MIN_MONOTONICITY_PERIODS,
        MIN_PORTFOLIO_PERIODS_HARD,
    )
    from factrix.metrics.concentration import top_concentration
    from factrix.metrics.hit_rate import hit_rate
    from factrix.metrics.k_spread import k_spread
    from factrix.metrics.monotonicity import monotonicity
    from factrix.metrics.quantile import quantile_spread, quantile_spread_vw

    return [
        ("quantile_spread", quantile_spread, MIN_PORTFOLIO_PERIODS_HARD, _cs_panel, {}),
        (
            "quantile_spread_vw",
            quantile_spread_vw,
            MIN_PORTFOLIO_PERIODS_HARD,
            lambda n: _cs_panel(n, market_cap=True),
            {},
        ),
        ("k_spread", k_spread, MIN_PORTFOLIO_PERIODS_HARD, _cs_panel, {"k": 3}),
        (
            "top_concentration",
            top_concentration,
            MIN_PORTFOLIO_PERIODS_HARD,
            _cs_panel,
            {},
        ),
        ("monotonicity", monotonicity, MIN_MONOTONICITY_PERIODS, _cs_panel, {}),
        ("hit_rate", hit_rate, MIN_IC_PERIODS, _hit_rate_series, {}),
    ]


_STRIDE_SAMPLED_CASES = _build_stride_sampled_cases()
_STRIDE_SAMPLED_IDS = [c[0] for c in _STRIDE_SAMPLED_CASES]


class TestStrideSampledPeriodsFloorConsistency:
    """For every stride-sampling metric, the pre-flight floor (``spec()``,
    resolved at default ``forward_periods=5``) and the in-body run-time floor
    are one numerically identical stride-scaled value."""

    @pytest.mark.parametrize(
        "name,metric,base,_build,_kwargs",
        _STRIDE_SAMPLED_CASES,
        ids=_STRIDE_SAMPLED_IDS,
    )
    def test_spec_floor_is_stride_scaled(self, name, metric, base, _build, _kwargs):
        from factrix.metrics._helpers import _scaled_min_periods

        # Default forward_periods=5 (a stride > 1): the floor scales to base * 5,
        # not the raw base — without scaling, pre-flight is too loose.
        assert metric.spec().sample_threshold.min_periods == _scaled_min_periods(
            base, 5
        )
        assert metric.spec().sample_threshold.min_periods == base * 5

    @pytest.mark.parametrize(
        "name,metric,base,build,kwargs", _STRIDE_SAMPLED_CASES, ids=_STRIDE_SAMPLED_IDS
    )
    def test_runtime_floor_equals_preflight_floor(
        self, name, metric, base, build, kwargs
    ):
        from factrix.metrics._helpers import _scaled_min_periods

        spec_floor = metric.spec().sample_threshold.min_periods
        # Raw input one date below the scaled floor → the body short-circuits and
        # reports the very floor the spec pre-flights (same _scaled_min_periods).
        sub = build(spec_floor - 1)
        res = _scalar(metric(sub, forward_periods=5, **kwargs))
        assert res.metadata["reason"].startswith("insufficient")
        assert res.metadata["min_required"] == spec_floor
        assert spec_floor == _scaled_min_periods(base, 5)

    def test_runtime_floor_tracks_actual_forward_periods(self):
        # The in-body floor re-derives from the *actual* forward_periods, not the
        # default baked into spec(): 24 raw dates clear base*5=15 at stride 5 but
        # trip base*10=30 at stride 10 (same _scaled_min_periods source).
        from factrix._types import MIN_PORTFOLIO_PERIODS_HARD
        from factrix.metrics.quantile import quantile_spread

        panel = _cs_panel(30)
        raw = panel["date"].n_unique()
        assert MIN_PORTFOLIO_PERIODS_HARD * 5 <= raw < MIN_PORTFOLIO_PERIODS_HARD * 10
        ok = quantile_spread(panel, forward_periods=5)["factor"]
        assert ok.metadata.get("reason") is None
        gated = quantile_spread(panel, forward_periods=10)["factor"]
        assert gated.metadata["min_required"] == MIN_PORTFOLIO_PERIODS_HARD * 10

    def test_inspect_and_runtime_agree_on_subfloor_panel(self):
        # End-to-end: on a raw panel below the scaled floor, inspect_data marks
        # the metric UNUSABLE and the run-time call short-circuits — one floor,
        # both gates.
        from factrix.metrics.quantile import quantile_spread

        sub = _cs_panel(quantile_spread.spec().sample_threshold.min_periods - 1)
        assert _by_name(inspect_data(sub), "quantile_spread").usable is False
        assert (
            quantile_spread(sub, forward_periods=5)["factor"]
            .metadata["reason"]
            .startswith("insufficient")
        )
