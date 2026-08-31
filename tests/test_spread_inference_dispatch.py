"""Spread-metric inference dispatch: golden values, allowlist and bootstrap keys.

The golden block pins ``value / stat / p_value / n_obs`` for the three
spread metrics under ``NON_OVERLAPPING`` and ``NEWEY_WEST`` to within
platform rounding (``rel=1e-12``; ``n_obs`` exact). The numbers were
recorded from the pre-refactor hard-branching dispatch on a fixed panel;
the polymorphic dispatch that replaced it reproduces ``stat`` / ``p_value``
/ ``n_obs`` up to last-ULP accumulation order across the CI platform
matrix, since it is a routing change and not a statistical one.

The three ``NEWEY_WEST`` ``value`` entries moved by 1-2 units in the last
place when re-recorded, and only those: the full-series mean is now taken
by the inference member (numpy pairwise summation over the values it
tested) rather than by the metric (polars summation over the same
values), so the accumulation order differs while the sample does not.
Nothing else in any cell moved. A failure here is a regression, not a
tolerance to loosen.
"""

from __future__ import annotations

import factrix as fx
import polars as pl
import pytest
from factrix._errors import IncompatibleInferenceError
from factrix._results import MetricResult
from factrix.inference import (
    NEWEY_WEST,
    NON_OVERLAPPING,
    STATIONARY_BOOTSTRAP,
    StationaryBootstrap,
)
from factrix.inference.series_mean import HANSEN_HODRICK
from factrix.metrics._helpers import _spread_significance_with_inference
from factrix.metrics.k_spread import k_spread
from factrix.metrics.quantile import quantile_spread, quantile_spread_vw
from factrix.preprocess import compute_forward_return

N_ASSETS = 80
N_DATES = 240
SEED = 0
OVERLAP_PERIODS = 5

# (metric, inference) -> (value, stat, p_value, n_obs), recorded pre-refactor.
GOLDEN: dict[tuple[str, str], tuple[float, float, float, int]] = {
    ("quantile_spread", "NON_OVERLAPPING"): (
        0.000991985503647292,
        1.890311625944073,
        0.06502597796665252,
        47,
    ),
    ("quantile_spread_vw", "NON_OVERLAPPING"): (
        0.0011151858973103643,
        1.9631851002473892,
        0.05582295855549759,
        46,
    ),
    ("k_spread", "NON_OVERLAPPING"): (
        0.0014547737220652647,
        1.6757497371084795,
        0.10057252627870207,
        47,
    ),
    ("quantile_spread", "NEWEY_WEST"): (
        0.0011755210341221423,
        5.804893767353654,
        2.3482153730009554e-05,
        234,
    ),
    ("quantile_spread_vw", "NEWEY_WEST"): (
        0.0012255349044815469,
        5.944071669395953,
        1.8237784819543553e-05,
        233,
    ),
    ("k_spread", "NEWEY_WEST"): (
        0.0015584384553266602,
        4.087355731532199,
        0.0008064807797969355,
        234,
    ),
}

INFERENCES = {"NON_OVERLAPPING": NON_OVERLAPPING, "NEWEY_WEST": NEWEY_WEST}


@pytest.fixture(scope="module")
def panel():
    """Fixed panel the golden numbers were recorded on."""
    raw = fx.datasets.make_cs_panel(n_assets=N_ASSETS, n_dates=N_DATES, rng=SEED)
    return compute_forward_return(raw, forward_periods=OVERLAP_PERIODS)


def _run(metric_name: str, panel, inference) -> MetricResult:
    if metric_name == "quantile_spread":
        return quantile_spread(
            panel, overlap_periods=OVERLAP_PERIODS, inference=inference
        )["factor"]
    if metric_name == "quantile_spread_vw":
        return quantile_spread_vw(
            panel,
            overlap_periods=OVERLAP_PERIODS,
            weight_col="price",
            inference=inference,
        )
    return k_spread(panel, overlap_periods=OVERLAP_PERIODS, inference=inference)


def test_full_series_member_rejects_missing_full_series():
    """An internal routing error must not change the requested estimator."""
    with pytest.raises(RuntimeError, match="requires the full spread series"):
        _spread_significance_with_inference(
            NEWEY_WEST,
            strided_spread=pl.DataFrame({"spread": [0.01, -0.02, 0.03]}),
            full_spread=None,
            overlap_periods=5,
            n_assets=80,
            metric_name="quantile_spread",
        )


@pytest.mark.parametrize(("metric_name", "inference_name"), sorted(GOLDEN))
def test_golden_spread_dispatch(panel, metric_name, inference_name):
    """Polymorphic dispatch reproduces the pre-refactor numbers.

    Floats are compared at ``rel=1e-12``, not ``==``: the last few ULPs of
    a mean / HAC t / p differ across platforms and numpy / polars builds
    (accumulation order), and CI runs a macOS / Linux / Windows matrix plus
    a declared-dependency-floor lane. A real change to the dispatch moves
    these numbers by orders of magnitude more than that. ``n_obs`` is exact.
    """
    value, stat, p_value, n_obs = GOLDEN[(metric_name, inference_name)]
    result = _run(metric_name, panel, INFERENCES[inference_name])
    assert result.value == pytest.approx(value, rel=1e-12, abs=0)
    assert result.stat == pytest.approx(stat, rel=1e-12, abs=0)
    assert result.p_value == pytest.approx(p_value, rel=1e-12, abs=0)
    assert result.n_obs == n_obs


@pytest.mark.parametrize(
    "metric_name", ["quantile_spread", "quantile_spread_vw", "k_spread"]
)
def test_hansen_hodrick_rejected(panel, metric_name):
    """The rectangular kernel stays out of every spread allowlist."""
    with pytest.raises(IncompatibleInferenceError):
        _run(metric_name, panel, HANSEN_HODRICK)


@pytest.mark.parametrize(
    ("inference_name", "expected_stat_type"),
    [
        ("NON_OVERLAPPING", "t"),
        ("NEWEY_WEST", "t"),
        ("STATIONARY_BOOTSTRAP", "bootstrap-mean"),
    ],
)
@pytest.mark.parametrize(
    "metric_name", ["quantile_spread", "quantile_spread_vw", "k_spread"]
)
def test_member_stat_type_and_sample_metadata(
    panel, metric_name, inference_name, expected_stat_type
):
    """Inference identity and the canonical tested-sample count stay aligned.

    The bootstrap's ``stat`` is the observed mean under an empirical p, not
    a t-ratio, so reporting ``"t"`` there would invite reading it against a
    t-distribution. ``n_periods`` is the sole public metadata alias for
    ``n_obs`` across every spread metric and inference member.
    """
    inference = {
        "NON_OVERLAPPING": NON_OVERLAPPING,
        "NEWEY_WEST": NEWEY_WEST,
        "STATIONARY_BOOTSTRAP": StationaryBootstrap(n_resamples=199, rng=3),
    }[inference_name]
    result = _run(metric_name, panel, inference)
    assert result.metadata["stat_type"] == expected_stat_type
    assert result.metadata["n_periods"] == result.n_obs


@pytest.mark.parametrize(
    "metric_name", ["quantile_spread", "quantile_spread_vw", "k_spread"]
)
def test_stationary_bootstrap_metadata_keys(panel, metric_name):
    """The bootstrap path surfaces the same three knobs ``ic`` surfaces."""
    result = _run(metric_name, panel, StationaryBootstrap(n_resamples=199, rng=7))
    assert result.metadata["method"] == STATIONARY_BOOTSTRAP.summary
    for key in ("n_resamples", "seed", "p_value_mc_se"):
        assert key in result.metadata
    assert result.metadata["n_resamples"] == 199
    assert result.metadata["seed"] == 7


@pytest.mark.parametrize(
    "metric_name", ["quantile_spread", "quantile_spread_vw", "k_spread"]
)
def test_stationary_bootstrap_reproducible(panel, metric_name):
    """Same seed, same empirical p — twice."""
    inference = StationaryBootstrap(n_resamples=199, rng=11)
    first = _run(metric_name, panel, inference)
    second = _run(metric_name, panel, inference)
    assert first.p_value == second.p_value
    assert first.value == second.value
    assert first.n_obs == second.n_obs
