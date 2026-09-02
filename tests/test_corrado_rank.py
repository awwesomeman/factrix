"""Tests for factrix.metrics.corrado_rank."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics.corrado_rank import corrado_rank


def _panel(returns: np.ndarray, factor: np.ndarray) -> pl.DataFrame:
    n = len(returns)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A"] * n,
            "factor": factor,
            "forward_return": returns,
        }
    )


def _tail_returns(rng: np.random.Generator, sign: float, k: int) -> np.ndarray:
    """``k`` returns deep in the ``sign`` tail, spread rather than identical.

    The spread is load-bearing: the denominator is the time-series SD of the
    per-event-period mean rank, so ``k`` identical event returns give it zero
    dispersion and the metric (correctly) short-circuits as degenerate. Real
    event returns are never all the same value.
    """
    return sign * (5.0 + rng.uniform(0.0, 2.0, size=k))


def _directional_panel(sign: float, n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Baseline noise returns with a handful of events whose own return is
    shifted well into the tail in the direction ``sign * factor``, so the
    event ranks are genuinely extreme rather than incidentally so.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=n)
    factor = np.zeros(n)
    event_idx = np.arange(70, n, 30)
    factor[event_idx] = sign
    returns[event_idx] = _tail_returns(rng, sign, len(event_idx))
    return _panel(returns, factor)


class TestOneSidedPValue:
    def test_anti_predictive_factor_is_not_significant(self):
        """Events sit at the top of the return distribution (rank ≈ +0.5)
        while ``factor`` points the wrong way (-1), so the direction-signed
        rank is negative: z < 0, and a one-sided p should be large.
        """
        rng = np.random.default_rng(0)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(70, n, 30)
        # events are the largest returns...
        returns[event_idx] = _tail_returns(rng, 1.0, len(event_idx))
        factor[event_idx] = -1.0  # ...but the factor calls them down

        result = corrado_rank(_panel(returns, factor))

        assert result.stat < 0
        assert result.p_value > 0.5

    def test_predictive_factor_is_significant(self):
        """Mirror case: factor direction matches the extreme rank at each
        event, so z > 0 and the one-sided p should be small.
        """
        result = corrado_rank(_directional_panel(sign=1.0))

        assert result.stat > 0
        assert result.p_value < 0.05

    def test_p_value_is_one_sided_sf(self):
        """p should equal the upper-tail normal survival function of z,
        not the two-sided 2*sf(|z|)."""
        from scipy import stats as sp_stats

        rng = np.random.default_rng(1)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(70, n, 15)
        factor[event_idx] = rng.choice([-1.0, 1.0], size=len(event_idx))

        result = corrado_rank(_panel(returns, factor))

        assert result.p_value == sp_stats.norm.sf(result.stat)


class TestDegenerateVariance:
    def test_keeps_the_defined_mean_rank_when_only_the_test_is_undefined(self):
        n = 300
        factor = np.zeros(n)
        factor[np.arange(70, n, 20)] = 1.0
        result = corrado_rank(
            _panel(np.zeros(n), factor),
            overlap_periods=1,
            expected_warnings=tuple(code.value for code in WarningCode),
        )

        assert result.value == pytest.approx(0.0)
        assert result.stat is None
        assert result.p_value is None
        assert result.alternative is None
        assert result.metadata["signal_status"] == "degenerate_zero_variance"
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes


class TestNonFiniteReturns:
    """Ranks must be formed over finite returns only."""

    def _panel_with_hole(self, hole, n: int = 300, seed: int = 0) -> pl.DataFrame:
        """Directional panel whose *first non-event* return is replaced."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(70, n, 20)
        factor[event_idx] = 1.0
        returns[event_idx] = _tail_returns(rng, 1.0, len(event_idx))
        returns[0] = hole  # a non-event row
        return _panel(returns, factor)

    def test_null_return_does_not_nan_out_the_statistic(self):
        # Old behaviour: a null return produced a null rank, std(u_all)
        # became NaN, and _calc_t_stat silently returned z=0 -> p=0.5.
        result = corrado_rank(self._panel_with_hole(None))
        assert np.isfinite(result.stat)
        assert result.stat > 0
        assert result.p_value < 0.05

    def test_nan_return_is_not_ranked_as_the_largest_value(self):
        # polars ranks float NaN as the largest value, so an unmasked NaN
        # silently entered the sample as a top-rank observation and shifted
        # every other rank down by one.
        nan_result = corrado_rank(self._panel_with_hole(float("nan")))
        null_result = corrado_rank(self._panel_with_hole(None))
        assert nan_result.stat == pytest.approx(null_result.stat)
        assert nan_result.value == pytest.approx(null_result.value)
        # The non-finite cell is excluded from the ranked sample, which is the
        # abnormal-return series: it starts once the estimation mean exists
        # (min_samples=20 plus the overlap_periods lag of 5), so 300 - 24 cells
        # would remain and the hole removes one more.
        assert nan_result.metadata["n_pairs_total"] == 275
        assert (
            nan_result.metadata["n_pairs_total"]
            == null_result.metadata["n_pairs_total"]
        )

    @pytest.mark.parametrize("hole", [float("nan"), None])
    def test_non_finite_event_row_is_dropped_and_counted(self, hole):
        rng = np.random.default_rng(3)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(70, n, 20)
        factor[event_idx] = 1.0
        returns[event_idx] = _tail_returns(rng, 1.0, len(event_idx))
        returns[event_idx[0]] = hole  # poison one *event*

        result = corrado_rank(_panel(returns, factor))

        assert result.metadata["n_events_dropped_non_finite"] == 1
        assert result.metadata["n_events"] == len(event_idx) - 1
        assert result.n_obs == len(event_idx) - 1
        assert np.isfinite(result.stat)

    def test_nan_factor_survives_the_event_filter_and_is_dropped(self):
        # `NaN != 0` is True in polars, so a NaN factor reaches the event
        # sample and sign(NaN) would poison u_event.
        rng = np.random.default_rng(5)
        n = 300
        returns = rng.normal(size=n)
        factor = np.zeros(n)
        event_idx = np.arange(70, n, 20)
        factor[event_idx] = 1.0
        returns[event_idx] = _tail_returns(rng, 1.0, len(event_idx))
        factor[1] = float("nan")

        result = corrado_rank(_panel(returns, factor))

        assert result.metadata["n_events_dropped_non_finite"] == 1
        assert result.metadata["n_events"] == len(event_idx)
        assert np.isfinite(result.stat)

    def test_clean_panel_reports_zero_drops(self):
        result = corrado_rank(_directional_panel(sign=1.0))
        assert result.metadata["n_events_dropped_non_finite"] == 0
        assert result.metadata["n_events_dropped_no_estimation_window"] == 0
        # 300 rows less the 24 the estimation mean needs (min_samples=20 plus
        # the overlap_periods lag of 5).
        assert result.metadata["n_pairs_total"] == 276


class TestClusterRobustDenominator:
    """The denominator must absorb same-period event clustering.

    ``corrado_rank`` exists as the nonparametric fallback for exactly the
    regime where ``caar``'s t-test breaks down — clustered event periods. A
    pooled std over every ``(asset, date)`` rank cell ignored that
    clustering, so the metric was liberal in the one situation it was
    recommended for. The unit of inference is now the event DATE.
    """

    @staticmethod
    def _clustered_panel(events_per_period: int, n_dates: int = 200, seed: int = 0):
        """Panel where every event period carries ``events_per_period`` events.

        Same-period events share one common shock, so they are far from
        independent draws — the shape a pooled denominator misreads.
        """
        rng = np.random.default_rng(seed)
        n_assets = 20
        event_periods = set(range(10, n_dates, 5))
        rows = []
        for d in range(n_dates):
            shock = rng.normal() * 3.0 if d in event_periods else 0.0
            for a in range(n_assets):
                is_event = d in event_periods and a < events_per_period
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if is_event else 0.0,
                        "forward_return": float(
                            rng.normal() + (shock if is_event else 0.0)
                        ),
                    }
                )
        return pl.DataFrame(rows)

    def test_n_obs_counts_event_dates_not_events(self):
        result = corrado_rank(self._clustered_panel(events_per_period=4))
        # ``n_obs`` is the event-period count; the axis token is the event
        # battery's shared ``"events"`` (see SampleAxis).
        assert result.n_obs_axis == "events"
        assert result.n_obs == result.metadata["n_event_periods"]
        # Four events per period: the event count is 4x the date count, and
        # using it as the sample size is what inflated z.
        assert result.metadata["n_events"] == 4 * result.metadata["n_event_periods"]
        assert result.metadata["events_per_period_max"] == 4

    def test_clustering_does_not_inflate_the_statistic(self):
        """Piling more correlated events onto the same dates must not buy
        significance. Under the pooled denominator it did: N_events grew
        while the denominator stayed put, so z scaled with sqrt(N_events).
        """
        z_light = corrado_rank(self._clustered_panel(events_per_period=1)).stat
        z_heavy = corrado_rank(self._clustered_panel(events_per_period=8)).stat
        # Same dates, same common shocks — 8x the events adds no independent
        # information, so z must not scale up with the event count.
        assert abs(z_heavy) < abs(z_light) * np.sqrt(8.0)

    def test_sparse_event_dates_still_run_with_a_warning(self):
        """A quarterly-cadence factor must not be locked out.

        The floor is ``caar``'s ``MIN_EVENTS_HARD``, shared because both
        metrics test an event-period series. A private, stricter floor here
        would have short-circuited factors that ``caar`` reports on happily
        — corrado_rank serves every sparse cell (single-asset TIMESERIES,
        wide PANEL, COMMON broadcast), so its floor has to be the general
        one.
        """
        rng = np.random.default_rng(2)
        rows = []
        event_days = {30, 110, 200, 290, 380, 470}  # ~quarterly, 6 dates
        for d in range(540):
            for a in range(8):
                is_event = d in event_days
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if is_event else 0.0,
                        "forward_return": float(
                            rng.normal() + (2.0 if is_event else 0.0)
                        ),
                    }
                )
        with pytest.warns(UserWarning, match="6 event periods survive"):
            result = corrado_rank(pl.DataFrame(rows))
        assert result.n_obs == 6
        assert result.stat is not None
        assert WarningCode.FEW_EVENTS.value in result.warning_codes

    def test_common_scope_whole_cross_section_on_event_periods(self):
        """COMMON-scope factors fire on every asset at once.

        That is Corrado's original event-time layout — one full
        cross-section per event period — so the per-period collapse is the
        identity on the cross-sectional mean and the test must simply work.
        """
        rng = np.random.default_rng(4)
        rows = []
        event_days = set(range(30, 200, 4))
        for d in range(200):
            shock = 1.5 if d in event_days else 0.0
            for a in range(15):
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if d in event_days else 0.0,
                        "forward_return": float(rng.normal() + shock),
                    }
                )
        # Event periods are four calendar steps apart, so the horizon is four:
        # every event clears the non-overlap stride and the collapse is the
        # identity, as the docstring above describes.
        result = corrado_rank(pl.DataFrame(rows), overlap_periods=4)
        assert result.n_obs == len(event_days)
        assert result.metadata["events_per_period_mean"] == pytest.approx(15.0)
        assert result.stat > 0
        assert result.p_value < 0.05

    def test_too_few_event_dates_short_circuits(self):
        """Many events on a handful of dates cannot estimate a time-series SD."""
        rng = np.random.default_rng(0)
        rows = []
        for d in range(60):
            for a in range(20):
                is_event = d in (35, 40, 45) and a < 10
                rows.append(
                    {
                        "date": datetime(2020, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": 1.0 if is_event else 0.0,
                        "forward_return": float(rng.normal()),
                    }
                )
        result = corrado_rank(pl.DataFrame(rows))
        # 30 events, but only 3 dates behind the SD.
        assert result.metadata["reason"] == "insufficient_event_periods"
        assert result.metadata["n_events"] == 30
        assert result.n_obs == 3


class TestStatisticAgainstHandComputation:
    """The published formula, computed by hand on the sample the metric says
    it tests: ranks over each asset's full non-missing series,
    ``U = rank / (T + 1) - 0.5`` (Corrado 1989, with the Corrado & Zivney 1992
    missing-data denominator), sign-adjusted, averaged per event period over
    the non-overlap-sampled events, then ``z = mean / (sd / sqrt(D))``.
    """

    @staticmethod
    def _panel(event_ordinals: list[int], n: int, seed: int = 0) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0, 1.0, n)
        factor = np.zeros(n)
        for o in event_ordinals:
            factor[o] = 1.0
        return pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "factor": factor,
                "forward_return": returns,
            }
        )

    def test_z_matches_the_formula_on_the_sampled_events(self):
        import scipy.stats as sp_stats

        fp = 5
        # A clustered block (thinned to every 5th) plus a spaced tail (all kept).
        event_ordinals = list(range(20, 60)) + [70 + 5 * k for k in range(30)]
        n = 260
        panel = self._panel(event_ordinals, n)

        result = corrado_rank(panel, overlap_periods=fp)

        returns = panel["forward_return"].to_numpy()
        # The ranked quantity is the ABNORMAL return: R_t less the mean of the
        # estimation window ending overlap_periods rows earlier (60 rows,
        # min_samples 20). Rows without that mean have no abnormal return and
        # never enter the ranking.
        window, min_samples = 60, 20
        ar = np.full(len(returns), np.nan)
        for t in range(len(returns)):
            end = t - fp
            start = max(0, end - window + 1)
            if end >= 0 and (end - start + 1) >= min_samples:
                ar[t] = returns[t] - returns[start : end + 1].mean()
        finite = ~np.isnan(ar)
        # rank / (T + 1) - 0.5 over the finite abnormal returns (T = their count).
        u = np.full(len(ar), np.nan)
        order = ar[finite].argsort().argsort() + 1  # 1-based ranks, no ties here
        u[finite] = order / (finite.sum() + 1) - 0.5

        # Greedy non-overlap keep over the events that have an abnormal
        # return; the first survivor is always kept.
        kept, last = [], None
        for o in sorted(event_ordinals):
            if not finite[o]:
                continue
            if last is None or o - last >= fp:
                kept.append(o)
                last = o
        # One asset, so one event per period: the per-period mean is the event.
        u_bar = np.array([u[o] for o in kept])
        mean_u = float(np.mean(u_bar))
        z_ref = mean_u / (float(np.std(u_bar, ddof=1)) / np.sqrt(len(u_bar)))

        assert result.n_obs == len(kept)
        assert result.value == pytest.approx(mean_u)
        assert result.stat == pytest.approx(z_ref)
        assert result.p_value == pytest.approx(float(sp_stats.norm.sf(z_ref)))
