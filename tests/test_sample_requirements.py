"""``sample_requirements``: a configured metric's floor at the panel's horizon.

The catalog surfaces (``list_metrics`` / ``metrics_summary`` /
``spec().sample_threshold``) report the default-configuration floor;
``evaluate`` gates in-body on the injected ``overlap_periods``. This public
resolver returns the run-time floor for the instance as configured, at the
stamped or declared horizon, so a coverage audit can be planned against the
number the run will actually apply.
"""

from __future__ import annotations

import factrix as fx
import pytest
from factrix._errors import UserInputError
from factrix.metrics import ic, positive_rate


def _stamped(overlap_periods: int):
    raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120, seed=3)
    return fx.preprocess.compute_forward_return(raw, forward_periods=overlap_periods)


class TestConfiguration:
    def test_ic_default_inference_is_stride_scaled(self):
        assert fx.sample_requirements(ic()).min_periods == 50
        assert fx.sample_requirements(ic(), overlap_periods=1).min_periods == 10

    def test_ic_newey_west_is_a_fixed_hac_bound(self):
        nw = ic(inference=fx.inference.NEWEY_WEST)
        assert fx.sample_requirements(nw).min_periods == 20
        assert fx.sample_requirements(nw, overlap_periods=1).min_periods == 20

    def test_positive_rate_scales_with_horizon(self):
        assert (
            fx.sample_requirements(positive_rate(), overlap_periods=1).min_periods == 10
        )
        assert (
            fx.sample_requirements(positive_rate(), overlap_periods=5).min_periods == 50
        )

    def test_default_matches_spec_threshold(self):
        for m in (ic(), positive_rate()):
            assert fx.sample_requirements(m) == type(m).spec().sample_threshold


class TestHorizonResolution:
    def test_reads_panel_stamp(self):
        assert (
            fx.sample_requirements(positive_rate(), data=_stamped(1)).min_periods == 10
        )
        assert (
            fx.sample_requirements(positive_rate(), data=_stamped(5)).min_periods == 50
        )

    def test_explicit_horizon_must_match_stamp(self):
        with pytest.raises(UserInputError, match="stamped evaluation-grid overlap"):
            fx.sample_requirements(positive_rate(), data=_stamped(5), overlap_periods=1)

    def test_unstamped_panel_requires_explicit_horizon(self):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120, seed=3)
        with pytest.raises(UserInputError, match="forward_periods"):
            fx.sample_requirements(positive_rate(), data=raw)
        assert (
            fx.sample_requirements(
                positive_rate(), data=raw, overlap_periods=2
            ).min_periods
            == 20
        )

    def test_rejects_an_invalid_explicit_overlap(self):
        """Validated as ``evaluate`` validates a declared overlap: a positive
        int, ``bool`` rejected — never silently clamped to 1."""
        for bad in (0, -1, True, "5"):
            with pytest.raises(UserInputError, match="overlap_periods") as excinfo:
                fx.sample_requirements(positive_rate(), overlap_periods=bad)
            assert excinfo.value.func_name == "sample_requirements"
            assert excinfo.value.field == "overlap_periods"
        assert (
            fx.sample_requirements(positive_rate(), overlap_periods=1).min_periods == 10
        )

    def test_rejects_class_and_non_metric(self):
        with pytest.raises(UserInputError, match="metric instance"):
            fx.sample_requirements(positive_rate)
        with pytest.raises(UserInputError, match="metric instance"):
            fx.sample_requirements("ic")

    def test_instance_is_not_mutated(self):
        m = positive_rate()
        fx.sample_requirements(m, overlap_periods=1)
        assert m.overlap_periods == 5
