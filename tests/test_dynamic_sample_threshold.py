"""Dynamic sample-threshold hook.

A metric whose floor is a function of its own parameters (e.g. ``ic``'s periods
floor scales with ``forward_periods``) cannot express it as a static
``SampleThreshold``. It declares ``sample_threshold_for`` instead; ``spec()``
resolves the hook against a default-constructed instance so the spec carries a
concrete floor and ``inspect_data`` can pre-flight it — previously these metrics
declared an empty ``SampleThreshold()`` and hid the floor in the body.
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
        # A hookless metric resolves to its static (here empty) threshold.
        assert REGISTRY["net_spread"].sample_threshold_for is None
        st = REGISTRY["net_spread"].spec().sample_threshold
        assert st.min_periods is None


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
