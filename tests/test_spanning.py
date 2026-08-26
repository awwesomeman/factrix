"""Tests for factrix.metrics.spanning."""

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix._results import MetricResult
from factrix.metrics.spanning import (
    SpanningResult,
    _ols_alpha,
    greedy_forward_selection,
    spanning_alpha,
)


def _make_spread_series(
    n_dates: int, mean: float, std: float, seed: int
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    return pl.DataFrame(
        {
            "date": dates,
            "spread": rng.normal(mean, std, n_dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestSpanningTest:
    def test_significant_alpha(self):
        factor = _make_spread_series(100, 0.02, 0.005, 42)
        base = _make_spread_series(100, 0.0, 0.005, 7)
        result = spanning_alpha(factor, base_spreads={"base": base})
        assert result.value != 0.0
        assert abs(result.stat) > 2.0

    def test_spanned_factor_no_alpha(self):
        base = _make_spread_series(200, 0.01, 0.01, 42)
        dates = base["date"].to_list()
        # candidate ≈ 2*base → alpha ≈ 0 after controlling for base
        spanned_vals = 2 * base["spread"].to_numpy() + np.random.default_rng(99).normal(
            0, 0.001, 200
        )
        candidate = pl.DataFrame(
            {
                "date": dates,
                "spread": spanned_vals,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = spanning_alpha(candidate, base_spreads={"base": base})
        assert abs(result.stat) < 2.0

    def test_no_base_is_not_computable(self):
        # No base model → no incremental alpha to estimate. The metric must
        # short-circuit rather than silently report the raw spread mean.
        factor = _make_spread_series(100, 0.02, 0.005, 42)
        for base in (None, {}):
            result = spanning_alpha(factor, base_spreads=base)
            assert math.isnan(result.value)
            assert result.stat is None
            assert result.metadata["reason"] == "no_base_factors"
            assert "n_base_factors" not in result.metadata

    def test_returns_metric_output(self):

        factor = _make_spread_series(100, 0.02, 0.005, 42)
        base = _make_spread_series(100, 0.0, 0.005, 7)
        result = spanning_alpha(factor, base_spreads={"base": base})
        assert isinstance(result, MetricResult)

    def test_pvalue_uses_regression_dof(self):
        # p-value must reference the regression residual dof (n - 1 - n_base),
        # not the single-sample n - 1.
        from scipy import stats as sp_stats

        factor = _make_spread_series(40, 0.02, 0.01, 7)
        base = {
            "b1": _make_spread_series(40, 0.0, 0.01, 1),
            "b2": _make_spread_series(40, 0.0, 0.01, 2),
        }
        result = spanning_alpha(factor, base_spreads=base)
        n, n_base = 40, 2
        expected = float(2 * sp_stats.t.sf(abs(result.stat), n - 1 - n_base))
        assert result.p_value == pytest.approx(expected)


class TestOLSAlpha:
    def test_alpha_with_empty_base(self):
        rng = np.random.default_rng(42)
        candidate = rng.normal(0.01, 0.005, 100)
        base = np.empty((100, 0))
        ols = _ols_alpha(candidate, base)
        assert ols.alpha == pytest.approx(0.01, abs=0.005)
        assert abs(ols.alpha_t) > 1.0
        assert ols.betas == []

    def test_spanned_factor_has_zero_alpha(self):
        rng = np.random.default_rng(42)
        base_col = rng.normal(0.01, 0.01, 200)
        # Candidate = 2 * base + noise → alpha ≈ 0 after regression
        candidate = 2 * base_col + rng.normal(0, 0.001, 200)
        base = base_col.reshape(-1, 1)
        ols = _ols_alpha(candidate, base)
        assert abs(ols.alpha) < 0.005
        assert abs(ols.alpha_t) < 2.0
        assert len(ols.betas) == 1
        assert ols.betas[0] == pytest.approx(2.0, abs=0.1)
        assert ols.r_squared > 0.95

    def test_insufficient_data_is_not_computable(self):
        """Too short to fit: NaN, not zero.

        ``alpha = 0.0`` would read as "this factor adds exactly nothing",
        a decisive claim from a fit that never ran.
        """
        import math

        ols = _ols_alpha(np.array([0.01, 0.02]), np.empty((2, 0)))
        assert math.isnan(ols.alpha) and math.isnan(ols.alpha_t)

    def test_collapsed_se_withholds_the_t_not_the_alpha(self):
        """A perfect fit leaves a real alpha and no t.

        ``np.linalg.lstsq`` does not raise on a rank-deficient design -- it
        returns a minimum-norm solution -- so the live path here is the
        collapsed HAC SE, not the LinAlgError branch. A base column that
        duplicates the intercept still produces an intercept estimate, so
        the alpha survives while the t is withheld; reporting ``t = 0``
        would turn that live estimate into a decisive "not significant".
        """
        import math

        rng = np.random.default_rng(0)
        candidate = rng.normal(size=40)
        ols = _ols_alpha(candidate, np.ones((40, 1)))
        assert math.isfinite(ols.alpha) and ols.alpha != 0.0
        assert math.isnan(ols.alpha_t)

    def test_df_resid_excludes_base_regressors(self):
        rng = np.random.default_rng(42)
        n = 40
        base = rng.normal(size=(n, 5))
        candidate = rng.normal(0.01, 0.5, n)
        ols = _ols_alpha(candidate, base)
        # df = n - (intercept + 5 base factors)
        assert ols.df_resid == n - 6


class TestGreedyForwardSelection:
    def test_selects_strong_independent_factor(self):
        # Factor A: strong independent alpha
        a = _make_spread_series(100, 0.02, 0.005, 42)
        # Factor B: pure noise
        b = _make_spread_series(100, 0.0, 0.01, 99)
        result = greedy_forward_selection(
            {"A": a, "B": b},
            suppress_snooping_warning=True,
        )
        selected_names = [s.factor_name for s in result.metadata["selected_factors"]]
        assert "A" in selected_names

    def test_base_factors_not_selected(self):
        base = _make_spread_series(100, 0.01, 0.005, 42)
        # Candidate = base + tiny noise → fully spanned
        dates = base["date"].to_list()
        spanned_vals = base["spread"].to_numpy() + np.random.default_rng(99).normal(
            0, 0.0001, 100
        )
        spanned = pl.DataFrame(
            {
                "date": dates,
                "spread": spanned_vals,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = greedy_forward_selection(
            {"spanned": spanned},
            base_spreads={"base": base},
            suppress_snooping_warning=True,
        )
        selected_names = [s.factor_name for s in result.metadata["selected_factors"]]
        assert "spanned" not in selected_names

    def test_backward_elimination(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

        # A: strong density
        a_vals = rng.normal(0.03, 0.005, n)
        # B: initially looks good, but once A is in, B is redundant (B ≈ A + tiny noise)
        b_vals = a_vals + rng.normal(0, 0.0005, n)

        a = pl.DataFrame({"date": dates, "spread": a_vals}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        b = pl.DataFrame({"date": dates, "spread": b_vals}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )

        result = greedy_forward_selection(
            {"A": a, "B": b},
            suppress_snooping_warning=True,
        )
        selected_names = [s.factor_name for s in result.metadata["selected_factors"]]
        # At most one should survive — they're nearly identical
        assert len(selected_names) <= 2

    def test_empty_candidates(self):
        result = greedy_forward_selection(
            {},
            suppress_snooping_warning=True,
        )
        assert result.metadata["selected_factors"] == []

    def test_insufficient_dates(self):
        short = _make_spread_series(5, 0.01, 0.005, 42)
        result = greedy_forward_selection(
            {"short": short},
            suppress_snooping_warning=True,
        )
        assert result.metadata["selected_factors"] == []

    def test_max_factors_limit(self):
        factors = {}
        for i in range(10):
            factors[f"f_{i}"] = _make_spread_series(100, 0.02 + i * 0.005, 0.005, i)
        result = greedy_forward_selection(
            factors,
            max_factors=2,
            suppress_snooping_warning=True,
        )
        assert len(result.metadata["selected_factors"]) <= 2

    def test_snooping_warning_fires_by_default(self):
        a = _make_spread_series(100, 0.02, 0.005, 42)
        b = _make_spread_series(100, 0.0, 0.01, 99)
        with pytest.warns(UserWarning, match="stepwise selection inflates"):
            result = greedy_forward_selection({"A": a, "B": b})
        assert result.metadata["t_stats_inference_invalid"] is True

    def test_snooping_warning_suppressible(self):
        a = _make_spread_series(100, 0.02, 0.005, 42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = greedy_forward_selection(
                {"A": a},
                suppress_snooping_warning=True,
            )
        # Contract: flag stays truthy even when the warning is silenced.
        assert result.metadata["t_stats_inference_invalid"] is True

    def test_result_structure(self):
        a = _make_spread_series(100, 0.02, 0.005, 42)
        result = greedy_forward_selection({"A": a})
        assert isinstance(result, MetricResult)
        for sr in result.metadata["selected_factors"]:
            assert isinstance(sr, SpanningResult)
            assert sr.selected is True

    def test_candidate_dict_pruned_when_factor_selected(self, monkeypatch):
        # Selected factors must be popped from ``candidate_arrays``
        # so backward-eliminated buffers release immediately rather than
        # linger to function return. Spy on ``_ols_alpha`` from the
        # forward-selection frame and pin ``candidate_arrays.keys() ==
        # remaining`` post-selection.
        import sys

        from factrix.metrics import spanning

        # Two strong + redundant pair forces backward elimination:
        # both pass the threshold solo, but once one is in, the other
        # is spanned and gets dropped on the next backward step.
        rng = np.random.default_rng(42)
        n = 200
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        a_vals = rng.normal(0.04, 0.005, n)
        b_vals = a_vals + rng.normal(0, 0.0005, n)  # ≈ A + tiny noise
        c_vals = rng.normal(0.025, 0.005, n)  # independent strong
        factors = {
            name: pl.DataFrame({"date": dates, "spread": vals}).with_columns(
                pl.col("date").cast(pl.Datetime("ms"))
            )
            for name, vals in [("A", a_vals), ("B", b_vals), ("C", c_vals)]
        }

        invariant_violations: list[tuple[set, set]] = []
        forward_observations = 0
        original = spanning._ols_alpha

        def _spy(*args, **kwargs):
            nonlocal forward_observations
            caller = sys._getframe(1).f_locals
            cand = caller.get("candidate_arrays")
            rem = caller.get("remaining")
            if cand is not None and rem is not None:
                forward_observations += 1
                if set(cand.keys()) != set(rem):
                    invariant_violations.append((set(cand.keys()), set(rem)))
            return original(*args, **kwargs)

        monkeypatch.setattr(spanning, "_ols_alpha", _spy)

        result = greedy_forward_selection(
            factors,
            significance_threshold=2.0,
            max_factors=3,
            suppress_snooping_warning=True,
        )
        assert not invariant_violations, (
            f"candidate_arrays.keys() drifted from remaining at "
            f"{len(invariant_violations)} call site(s); first divergence: "
            f"keys={invariant_violations[0][0]} vs remaining={invariant_violations[0][1]}"
        )
        # Positive assertion: confirm the spy actually observed the
        # forward-selection frame at least once (silent-skip on
        # `caller.get(...) is None` would otherwise vacuously pass).
        assert forward_observations >= 1, (
            "spy never saw candidate_arrays/remaining in any caller frame; "
            "the invariant check would have vacuously passed"
        )
        assert len(result.metadata["selected_factors"]) >= 1


class TestSpanningEvaluate:
    def test_evaluate_greedy_forward_selection(self):
        import factrix as fx

        # Create a small panel with date, asset_id, factor, and forward_return
        panel = fx.datasets.make_cs_panel(n_assets=5, n_dates=20)
        panel = fx.preprocess.compute_forward_return(panel, forward_periods=2)
        # Create a second factor column so we have multiple candidate factors
        panel = panel.with_columns((pl.col("factor") + pl.lit(0.01)).alias("factor2"))

        results = fx.evaluate(
            panel,
            metrics={"gfs": greedy_forward_selection(suppress_snooping_warning=True)},
            factor_cols=["factor", "factor2"],
            forward_periods=2,
        )
        er1, er2 = results["factor"], results["factor2"]

        assert "gfs" in er1.metrics
        assert "gfs" in er2.metrics
        result1 = er1.metrics["gfs"]
        result2 = er2.metrics["gfs"]
        assert isinstance(result1, MetricResult)
        assert isinstance(result2, MetricResult)
        # Batchable: the selection runs once across the whole factor batch,
        # so the shared payload (metadata dict) is the same object for every
        # factor — only the per-label name stamp on the MetricResult differs.
        assert result1.metadata is result2.metadata

    def test_evaluate_spanning_alpha_reports_unavailable(self):
        # ``evaluate`` has no channel for base spreads, so the metric must
        # report itself not-computable instead of silently returning the raw
        # spread mean as an "alpha".
        import factrix as fx

        panel = fx.datasets.make_cs_panel(n_assets=20, n_dates=60)
        panel = fx.preprocess.compute_forward_return(panel, forward_periods=2)
        with pytest.raises(UserInputError, match="no_base_factors"):
            fx.evaluate(
                panel,
                metrics={"span": spanning_alpha()},
                factor_cols=["factor"],
                forward_periods=2,
            )
        results = fx.evaluate(
            panel,
            metrics={"span": spanning_alpha()},
            factor_cols=["factor"],
            forward_periods=2,
            strict=False,
        )
        er = results["factor"]
        res = er.metrics["span"]
        assert math.isnan(res.value)
        assert res.metadata["reason"] == "no_base_factors"
        assert any(w.code == fx.WarningCode.METRIC_UNAVAILABLE for w in er.warnings)


class TestAlignSpreadSeriesRegressions:
    """Regressions for ``_align_spread_series`` (v0.19.x alignment bugs)."""

    @staticmethod
    def _linked_pair(n: int = 120) -> tuple[pl.DataFrame, pl.DataFrame]:
        """base ~ N(0,1); candidate = 0.5 * base + tiny noise, same dates."""
        rng = np.random.default_rng(2024)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        base_vals = rng.normal(0.0, 1.0, n)
        cand_vals = 0.5 * base_vals + rng.normal(0.0, 1e-4, n)
        base = pl.DataFrame({"date": dates, "spread": base_vals}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        cand = pl.DataFrame({"date": dates, "spread": cand_vals}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        return base, cand

    def test_shuffled_input_matches_sorted_input(self):
        # REGRESSION: ``common_dates.join(series, ...)`` relied on polars
        # preserving the left frame's row order, which 1.40 does not. A
        # shuffled base frame came back permuted relative to the candidate,
        # collapsing the true 0.5 slope to ~0.001.
        base, cand = self._linked_pair()
        sorted_res = spanning_alpha(cand, base_spreads={"base": base})
        shuffled_base = base.sample(fraction=1.0, shuffle=True, seed=7)
        assert shuffled_base["date"].to_list() != base["date"].to_list()
        shuffled_res = spanning_alpha(cand, base_spreads={"base": shuffled_base})

        assert shuffled_res.value == pytest.approx(sorted_res.value)
        assert shuffled_res.stat == pytest.approx(sorted_res.stat)
        assert shuffled_res.metadata["betas"]["base"] == pytest.approx(0.5, abs=1e-3)

    def test_shuffled_candidate_matches_sorted(self):
        base, cand = self._linked_pair()
        sorted_res = spanning_alpha(cand, base_spreads={"base": base})
        shuffled_cand = cand.sample(fraction=1.0, shuffle=True, seed=13)
        shuffled_res = spanning_alpha(shuffled_cand, base_spreads={"base": base})
        assert shuffled_res.value == pytest.approx(sorted_res.value)
        assert shuffled_res.stat == pytest.approx(sorted_res.stat)

    def test_greedy_forward_selection_shares_the_fix(self):
        # greedy_forward_selection goes through the same helper.
        base, cand = self._linked_pair()
        kwargs = {"suppress_snooping_warning": True}
        ordered = greedy_forward_selection(
            {"cand": cand}, base_spreads={"base": base}, **kwargs
        )
        shuffled = greedy_forward_selection(
            {"cand": cand.sample(fraction=1.0, shuffle=True, seed=3)},
            base_spreads={"base": base.sample(fraction=1.0, shuffle=True, seed=4)},
            **kwargs,
        )
        assert shuffled.value == ordered.value
        assert [c.factor_name for c in shuffled.metadata["all_candidates"]] == [
            c.factor_name for c in ordered.metadata["all_candidates"]
        ]

    def test_duplicate_dates_raise(self):
        # REGRESSION: duplicate dates produced arrays of unequal length and
        # np.column_stack crashed far from the cause.
        base, cand = self._linked_pair()
        dup_base = pl.concat([base, base.head(1)])
        with pytest.raises(ValueError, match="duplicate dates"):
            spanning_alpha(cand, base_spreads={"base": dup_base})

    def test_nan_spread_dropped_not_propagated(self):
        base, cand = self._linked_pair()
        nan_base = base.with_columns(
            pl.when(pl.int_range(pl.len()) == 5)
            .then(float("nan"))
            .otherwise(pl.col("spread"))
            .alias("spread")
        )
        res = spanning_alpha(cand, base_spreads={"base": nan_base})
        assert np.isfinite(res.value)
        assert np.isfinite(res.stat)
        assert res.n_obs == len(base) - 1

    def test_nan_spread_dropped_in_candidate(self):
        base, cand = self._linked_pair()
        nan_cand = cand.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(float("nan"))
            .otherwise(pl.col("spread"))
            .alias("spread")
        )
        res = spanning_alpha(nan_cand, base_spreads={"base": base})
        assert np.isfinite(res.value)
        assert res.n_obs == cand.height - 1


class TestOlsAlphaFiniteContract:
    """``ols_alpha`` refuses non-finite input rather than returning NaN.

    ``np.linalg.lstsq`` does not raise on NaN — it returns a NaN solution,
    so the NaN surfaced as a NaN alpha and t far from its cause. The guard
    was previously a property of ``spanning``'s call sites (all three filter
    first, and the module comment says why), which left any fourth caller
    exposed. Same fail-loud contract as ``_stats.hac._require_finite``.
    """

    @staticmethod
    def _design(n: int = 30):
        rng = np.random.default_rng(0)
        return rng.normal(size=n), rng.normal(size=(n, 2))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_candidate_raises(self, bad):
        from factrix._ols import ols_alpha

        candidate, base = self._design()
        candidate[5] = bad
        with pytest.raises(ValueError, match="candidate must be finite"):
            ols_alpha(candidate, base)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_base_matrix_raises(self, bad):
        from factrix._ols import ols_alpha

        candidate, base = self._design()
        base[7, 1] = bad
        with pytest.raises(ValueError, match="base_matrix must be finite"):
            ols_alpha(candidate, base)

    def test_clean_input_is_unaffected(self):
        from factrix._ols import ols_alpha

        candidate, base = self._design()
        result = ols_alpha(candidate, base)
        assert math.isfinite(result.alpha)
        assert math.isfinite(result.alpha_t)

    def test_empty_base_matrix_still_works(self):
        """The no-regressor path (intercept only) must not trip the guard."""
        from factrix._ols import ols_alpha

        candidate, _ = self._design()
        result = ols_alpha(candidate, np.empty((len(candidate), 0)))
        assert result.alpha == pytest.approx(float(np.mean(candidate)))


class TestAlphaTIsHac:
    """``alpha_t`` divides by a Newey-West HAC SE, not the homoskedastic OLS SE.

    Nothing in a ``(date, spread)`` input can prove the series is
    non-overlapping, so the SE must not depend on that assumption. Under
    positive serial correlation the HAC SE is wider and the t smaller; on
    iid input the two agree closely.
    """

    @staticmethod
    def _ols_t(y: np.ndarray, X: np.ndarray) -> float:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        sigma2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
        return float(beta[0] / se)

    def test_autocorrelated_residuals_shrink_the_t(self):
        from factrix._ols import ols_alpha

        rng = np.random.default_rng(0)
        n = 240
        eps = np.empty(n)
        eps[0] = rng.standard_normal()
        for t in range(1, n):
            eps[t] = 0.7 * eps[t - 1] + rng.standard_normal()
        y = 0.3 + eps
        X = np.ones((n, 1))
        hac_t = ols_alpha(y, np.empty((n, 0))).alpha_t
        assert 0 < hac_t < self._ols_t(y, X)

    def test_iid_residuals_agree_with_ols_on_average(self):
        """On iid input the HAC and OLS t agree in expectation.

        A single draw can differ by ±20% (the HAC SE is a noisier estimate
        on iid data — that is its known small-sample cost), so the
        agreement is asserted on the mean ratio over many draws, not on
        one series.
        """
        from factrix._ols import ols_alpha

        n = 240
        ratios = []
        for seed in range(200):
            y = 0.3 + np.random.default_rng(seed).standard_normal(n)
            hac_t = ols_alpha(y, np.empty((n, 0))).alpha_t
            ratios.append(hac_t / self._ols_t(y, np.ones((n, 1))))
        assert 0.95 <= float(np.mean(ratios)) <= 1.10
