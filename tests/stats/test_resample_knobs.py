"""The shared bootstrap resampling contract: ``n_resamples`` / ``seed``.

Every entry point that turns a resample count into a *reported inference*
takes the same two knobs under the same names, defaults to 999, and refuses
a count below ``BOOTSTRAP_RESAMPLES_FLOOR`` with the same exception. These
tests hold that contract across all four together, so a fifth entry point
cannot be added with its own spelling of the same knob.

``stationary_bootstrap_resamples`` is deliberately outside the floor: it
returns draws and claims no inference on them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._stats.bootstrap import BOOTSTRAP_RESAMPLES_FLOOR, _empirical_p
from factrix.inference import StationaryBootstrap
from factrix.metrics import ic, monotonicity
from factrix.slicing import slice_period_joint_test, slice_period_pairwise_test
from factrix.stats import bootstrap_mean_ci, stationary_bootstrap_resamples

from tests._slice_panel import build_disjoint_period_panel


def _ic_series(n: int = 80):
    from datetime import datetime, timedelta

    import polars as pl

    vals = np.random.default_rng(0).standard_normal(n) * 0.05
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, "ic": vals}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


def _mono_panel():
    import factrix as fx

    return fx.preprocess.compute_forward_return(
        fx.datasets.make_cs_panel(n_assets=40, n_dates=120, seed=3),
        forward_periods=1,
    )


def _regime_panel():
    return build_disjoint_period_panel(
        seed=1, spans={"bull": (60, 0.1), "bear": (60, 0.1)}, label_col="regime"
    )


def test_the_floor_is_the_smallest_davison_hinkley_grid_point() -> None:
    """199, not 200: p lives on the ``1/(B+1)`` grid and ``alpha*(B+1)``
    must be an integer (Davidson-MacKinnon), so 199 / 399 / 999 are the
    admissible counts and 199 is the smallest of them."""
    assert BOOTSTRAP_RESAMPLES_FLOOR == 199
    for alpha in (0.05, 0.10):
        assert (alpha * (BOOTSTRAP_RESAMPLES_FLOOR + 1)) % 1 == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("func", "param"),
    [
        (StationaryBootstrap, "n_resamples"),
        (monotonicity, "n_resamples"),
        (bootstrap_mean_ci, "n_resamples"),
        (stationary_bootstrap_resamples, "n_resamples"),
        (slice_period_pairwise_test, "n_resamples"),
        (slice_period_joint_test, "n_resamples"),
    ],
    ids=[
        "stationary_bootstrap",
        "monotonicity",
        "bootstrap_mean_ci",
        "stationary_bootstrap_resamples",
        "slice_pairwise",
        "slice_joint",
    ],
)
def test_resample_count_is_spelled_n_resamples_and_defaults_to_999(
    func: object, param: str
) -> None:
    sig = inspect.signature(func)  # type: ignore[arg-type]
    assert param in sig.parameters, f"{func} does not expose {param}"
    assert sig.parameters[param].default == 999
    assert sig.parameters["seed"].default is None


@pytest.mark.parametrize("bad", [0, 1, BOOTSTRAP_RESAMPLES_FLOOR - 1])
class TestFloorIsEnforcedAtEveryInferenceEntryPoint:
    def test_stationary_bootstrap(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            StationaryBootstrap(n_resamples=bad)

    def test_ic_through_the_inference_member(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            ic(_ic_series(), overlap_periods=1, inference=StationaryBootstrap(bad))

    def test_monotonicity(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            monotonicity(_mono_panel(), n_resamples=bad, seed=0)

    def test_bootstrap_mean_ci(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            bootstrap_mean_ci(np.arange(50.0), n_resamples=bad)

    def test_slice_period_pairwise_test(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            slice_period_pairwise_test(
                _regime_panel(),
                ic(),
                by="regime",
                factor_col="factor",
                n_resamples=bad,
            )

    def test_slice_period_joint_test(self, bad: int) -> None:
        with pytest.raises(UserInputError, match="at least 199 resamples"):
            slice_period_joint_test(
                _regime_panel(),
                ic(),
                by="regime",
                factor_col="factor",
                n_resamples=bad,
            )


def test_resample_draws_are_deliberately_not_floored() -> None:
    """``stationary_bootstrap_resamples`` returns draws, not an inference,
    so it stays outside the floor — only a non-positive count is refused."""
    out = stationary_bootstrap_resamples(np.arange(30.0), 5, seed=0)
    assert out.shape == (5, 30)


class TestIcSurfacesTheBootstrapKnobs:
    def test_configured_member_is_reproducible_and_reports_its_knobs(self) -> None:
        df = _ic_series()
        member = StationaryBootstrap(n_resamples=399, seed=7)
        a = ic(df, overlap_periods=1, inference=member)
        b = ic(df, overlap_periods=1, inference=member)
        assert a.p_value == b.p_value
        assert a.metadata["n_resamples"] == 399
        assert a.metadata["seed"] == 7
        assert a.metadata["p_value_mc_se"] == pytest.approx(
            float(np.sqrt(a.p_value * (1.0 - a.p_value) / 399)), rel=1e-6
        )

    def test_an_unseeded_run_reports_the_resolved_seed(self) -> None:
        result = ic(
            _ic_series(),
            overlap_periods=1,
            inference=StationaryBootstrap(n_resamples=199),
        )
        assert isinstance(result.metadata["seed"], int)
        assert result.metadata["seed"] >= 0

    def test_a_configured_member_is_still_allowlisted(self) -> None:
        """The allowlist is by method, not by configuration: a knob-carrying
        member must not be rejected for differing from the default value."""
        assert not np.isnan(
            ic(
                _ic_series(),
                overlap_periods=1,
                inference=StationaryBootstrap(n_resamples=199, seed=1),
            ).value
        )


class TestSlicePeriodTestsAreReproducible:
    def test_pairwise(self) -> None:
        panel = _regime_panel()
        kw = dict(by="regime", factor_col="factor", n_resamples=199, seed=3)
        a = slice_period_pairwise_test(panel, ic(), **kw)
        b = slice_period_pairwise_test(panel, ic(), **kw)
        assert a["p_raw"].to_list() == b["p_raw"].to_list()

    def test_joint(self) -> None:
        panel = _regime_panel()
        kw = dict(by="regime", factor_col="factor", n_resamples=199, seed=3)
        a = slice_period_joint_test(panel, ic(), **kw)
        b = slice_period_joint_test(panel, ic(), **kw)
        assert a["p_value"].to_list() == b["p_value"].to_list()

    def test_n_resamples_sets_the_p_grid(self) -> None:
        """``p_raw`` is a Davison-Hinkley p, so ``p*(B+1)`` is an integer."""
        panel = _regime_panel()
        out = slice_period_pairwise_test(
            panel, ic(), by="regime", factor_col="factor", n_resamples=199, seed=3
        )
        for p in out["p_raw"].to_list():
            assert p is not None
            assert (p * 200) == pytest.approx(round(p * 200))


class TestEmpiricalP:
    def test_davison_hinkley_smoothing(self) -> None:
        p, _se = _empirical_p(0, 999)
        assert p == pytest.approx(1 / 1000)
        p, _se = _empirical_p(999, 999)
        assert p == pytest.approx(1.0)

    def test_monte_carlo_se_matches_hardcoded_analytic_values(self) -> None:
        """Hardcoded so a wrong-but-self-consistent formula (say ``/(B+1)``)
        fails here rather than being re-derived from the value under test.
        ``sqrt(.05*.95/999) = 0.006895`` is the ~0.7pp the docstrings quote.
        """
        p, se = _empirical_p(49, 999)
        assert p == pytest.approx(0.05)
        assert se == pytest.approx(0.006895, abs=1e-6)

        p, se = _empirical_p(199, 399)
        assert p == pytest.approx(0.5)
        assert se == pytest.approx(0.025031, abs=1e-6)

    def test_se_shrinks_as_one_over_sqrt_b(self) -> None:
        _p, se_small = _empirical_p(9, 199)
        _p, se_large = _empirical_p(99, 1999)
        assert se_small / se_large == pytest.approx(np.sqrt(1999 / 199), rel=0.02)
