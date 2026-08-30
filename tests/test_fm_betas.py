"""Tests for ``compute_fm_betas`` — vectorized per-date cross-sectional OLS slope.

Mirrors ``test_ic.py``: single-factor behaviour (schema, closed-form value,
small-date / zero-variance drops) plus the multi-factor batch contract (each
batch element equals the corresponding list-of-one call).
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics.fm_beta import (
    compute_fm_betas,
    fm_beta,
    fm_beta_sign_consistency,
    pooled_beta,
)


def _lstsq_betas(df: pl.DataFrame, factor_col: str = "factor") -> dict:
    """Reference per-date slope via the pre-vectorization lstsq path."""
    out = {}
    for dt in df["date"].unique().sort():
        chunk = df.filter(pl.col("date") == dt)
        y = chunk["forward_return"].to_numpy().astype(np.float64)
        x = chunk[factor_col].to_numpy().astype(np.float64)
        if len(y) < 3:
            continue
        beta, _, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(x)), x]), y, rcond=None
        )
        out[dt] = float(beta[1])
    return out


class TestComputeFMBetas:
    def test_returns_dict_keyed_by_factor(self, tiny_panel):
        result = compute_fm_betas(tiny_panel)
        assert isinstance(result, dict)
        assert set(result) == {"factor"}

    def test_output_schema(self, noisy_panel):
        df = compute_fm_betas(noisy_panel)["factor"]
        # ``_drop_stats`` is an internal diagnostic struct column appended by
        # the primitive; the public series columns are ``date, beta, n_assets``.
        assert df.columns == ["date", "beta", "n_assets", "_drop_stats"]
        assert df["date"].is_sorted()

    def test_closed_form_value(self, tiny_panel):
        # forward_return = 0.01 * factor exactly → per-date slope == 0.01.
        df = compute_fm_betas(tiny_panel)["factor"]
        assert df.height == 3
        assert df["beta"].to_numpy() == pytest.approx(0.01, abs=1e-12)

    def test_matches_lstsq_reference(self, noisy_panel):
        df = compute_fm_betas(noisy_panel)["factor"]
        ref = _lstsq_betas(noisy_panel)
        got = dict(zip(df["date"].to_list(), df["beta"].to_list(), strict=True))
        assert got.keys() == ref.keys()
        for dt, beta in got.items():
            assert beta == pytest.approx(ref[dt], abs=1e-12)

    def test_drops_dates_below_min_obs(self):
        # date d0 has 2 assets (below MIN_FM_ASSETS_HARD=3), d1 has 4.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "A",
                "factor": 1.0,
                "forward_return": 0.1,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "B",
                "factor": 2.0,
                "forward_return": 0.2,
            },
        ] + [
            {
                "date": datetime(2024, 1, 2),
                "asset_id": a,
                "factor": f,
                "forward_return": 0.05 * f,
            }
            for a, f in zip("ABCD", (1.0, 2.0, 3.0, 4.0), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["date"].to_list() == [datetime(2024, 1, 2)]
        assert df["n_assets"].to_list() == [4]

    def test_drops_zero_variance_dates(self):
        # date d0: factor constant → no identifiable slope, dropped.
        # date d1: ordinary spread → kept.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": a,
                "factor": 5.0,
                "forward_return": r,
            }
            for a, r in zip("ABC", (0.1, 0.2, 0.3), strict=True)
        ] + [
            {
                "date": datetime(2024, 1, 2),
                "asset_id": a,
                "factor": f,
                "forward_return": 0.05 * f,
            }
            for a, f in zip("ABC", (1.0, 2.0, 3.0), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["date"].to_list() == [datetime(2024, 1, 2)]
        assert df["beta"].is_finite().all()

    def test_null_return_uses_pairwise_complete_slope(self):
        # One asset has a null return: cov drops the pair, and var(factor)
        # must drop it too so the slope is the OLS fit on complete pairs
        # (forward_return = 0.1 * factor over the 3 complete rows → 0.1),
        # not a numerator/denominator-mismatched value.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "A",
                "factor": 1.0,
                "forward_return": 0.1,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "B",
                "factor": 2.0,
                "forward_return": 0.2,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "C",
                "factor": 3.0,
                "forward_return": 0.3,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "D",
                "factor": 4.0,
                "forward_return": None,
            },
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["beta"].to_numpy() == pytest.approx(0.1, abs=1e-12)

    def test_tiny_scale_variance_is_not_dropped(self):
        # Variance of a 1e-5-scale factor is ~1e-10; a fixed absolute epsilon
        # would wrongly discard this legitimately dispersed date. The slope is
        # scale-free: forward_return = 1e4 * factor.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": a,
                "factor": f,
                "forward_return": 1e4 * f,
            }
            for a, f in zip("ABC", (1e-5, 2e-5, 3e-5), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df.height == 1
        assert df["beta"][0] == pytest.approx(1e4, rel=1e-9)


class TestFMBetaConsumers:
    @staticmethod
    def _thin_beta_df(n_dates: int = 35, n_assets: int = 8) -> pl.DataFrame:
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(n_assets):
                factor = float(i + 1)
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": factor,
                        "forward_return": 0.01 * factor,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        return compute_fm_betas(panel)["factor"]

    def test_few_assets_warns_without_blocking(self):
        beta_df = self._thin_beta_df()
        with pytest.warns(UserWarning, match="MIN_FM_ASSETS_WARN"):
            result = fm_beta(beta_df)
        assert not np.isnan(result.value)
        assert WarningCode.FEW_ASSETS.value in result.warning_codes
        assert result.metadata["min_assets_per_period"] == 8
        assert result.metadata["warn_assets_per_period"] == 10

    def test_sign_consistency_surfaces_few_assets(self):
        beta_df = self._thin_beta_df()
        with pytest.warns(UserWarning, match="MIN_FM_ASSETS_WARN"):
            result = fm_beta_sign_consistency(beta_df)
        assert not np.isnan(result.value)
        assert WarningCode.FEW_ASSETS.value in result.warning_codes


class TestComputeFMBetasBatch:
    """Multi-factor path — each batch element equals the list-of-one call."""

    def test_multi_factor_matches_list_of_one(self, noisy_panel):
        rng = np.random.default_rng(7)
        panel = noisy_panel.with_columns(
            (pl.col("factor") * 1.5).alias("f1"),
            pl.Series("f2", rng.standard_normal(noisy_panel.height)),
        )
        cols = ["f1", "f2"]
        batch = compute_fm_betas(panel, factor_cols=cols)
        for col in cols:
            single = compute_fm_betas(panel, factor_cols=[col])[col]
            assert batch[col].equals(single), col

    def test_empty_factor_list_rejected(self, tiny_panel):
        with pytest.raises(
            ValueError, match="non-empty sequence of factor column names"
        ):
            compute_fm_betas(tiny_panel, factor_cols=[])


def _nan_panel(n_dates: int = 30, n_assets: int = 8, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for a in range(n_assets):
            f = float(rng.normal())
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{a}",
                    "factor": f,
                    "forward_return": 0.5 * f + float(rng.normal(0, 0.1)),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestNonFinitePassThrough:
    """REGRESSION: polars var/cov/drop_nulls do not skip float NaN."""

    def test_single_nan_return_does_not_emit_nan_beta(self):
        panel = _nan_panel()
        target = panel["date"][0]
        poisoned = panel.with_columns(
            pl.when((pl.col("date") == target) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        out = compute_fm_betas(poisoned)["factor"]
        betas = out["beta"].to_numpy()
        # Old code: var(x) > 0 stayed True while cov went NaN -> NaN beta that
        # survived drop_nulls("beta").
        assert np.isfinite(betas).all()
        # The NaN row is excluded from the effective n for that date.
        row = out.filter(pl.col("date") == target)
        assert row.height == 1
        assert row["n_assets"][0] == 7

    def test_single_nan_factor_does_not_emit_nan_beta(self):
        panel = _nan_panel()
        poisoned = panel.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        out = compute_fm_betas(poisoned)["factor"]
        assert np.isfinite(out["beta"].to_numpy()).all()

    def test_date_below_hard_floor_after_nan_drop_is_removed(self):
        panel = _nan_panel(n_dates=12, n_assets=3)
        target = panel["date"][0]
        poisoned = panel.with_columns(
            pl.when((pl.col("date") == target) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        out = compute_fm_betas(poisoned)["factor"]
        # 3 assets - 1 non-finite = 2 finite pairs < MIN_FM_ASSETS_HARD.
        assert target not in out["date"].to_list()
        stats = out["_drop_stats"][0]
        assert stats["dropped_periods"] >= 1

    def test_pooled_beta_survives_a_nan_cell(self):
        panel = _nan_panel()
        poisoned = panel.with_columns(
            pl.when(pl.int_range(pl.len()) == 11)
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        # Old code: drop_nulls kept the NaN -> lstsq slope NaN -> MetricResult
        # rejected the non-finite value with a ValueError.
        result = pooled_beta(poisoned)
        assert np.isfinite(result.value)
        assert result.metadata["dropped_pairs"] == 1
        assert result.n_obs == poisoned.height - 1

    def test_sign_consistency_ignores_nan_betas(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        beta_df = pl.DataFrame(
            {
                "date": dates,
                "beta": [1.0, 2.0, float("nan"), 3.0, 4.0],
                "n_assets": [10, 10, 10, 10, 10],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = fm_beta_sign_consistency(beta_df, expected_sign=1)
        # Old code: NaN > 0 is False -> counted as a wrong-sign period AND in
        # n_obs, giving 4/5 = 0.8 instead of 4/4 = 1.0.
        assert result.value == pytest.approx(1.0)
        assert result.n_obs == 4


class TestClusterMeatSegmentSum:
    """``_cluster_meat`` is a segment sum, not a per-cluster mask loop.

    The masked loop was ``O(G · N)`` and made ``pooled_beta`` roughly an
    order of magnitude slower than every other metric on a panel with one
    cluster per period.
    """

    @staticmethod
    def _naive(X, resid, clusters):
        unique = np.unique(clusters)
        k = X.shape[1]
        meat = np.zeros((k, k))
        for c in unique:
            mask = clusters == c
            score = X[mask].T @ resid[mask]
            meat += np.outer(score, score)
        return meat, len(unique)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_matches_the_naive_loop(self, k):
        from factrix.metrics.fm_beta import _cluster_meat

        rng = np.random.default_rng(7)
        n = 500
        X = rng.normal(size=(n, k))
        resid = rng.normal(size=n)
        clusters = rng.integers(0, 37, size=n)

        meat, g = _cluster_meat(X, resid, clusters)
        ref_meat, ref_g = self._naive(X, resid, clusters)
        assert g == ref_g
        assert np.allclose(meat, ref_meat)

    def test_matches_on_datetime_cluster_keys(self):
        from factrix.metrics.fm_beta import _cluster_meat

        rng = np.random.default_rng(11)
        n = 400
        dates = np.array(
            [
                np.datetime64(datetime(2024, 1, 1) + timedelta(days=int(i % 25)))
                for i in range(n)
            ]
        )
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        resid = rng.normal(size=n)

        meat, g = _cluster_meat(X, resid, dates)
        ref_meat, ref_g = self._naive(X, resid, dates)
        assert g == ref_g == 25
        assert np.allclose(meat, ref_meat)

    def test_single_cluster_is_the_full_outer_product(self):
        from factrix.metrics.fm_beta import _cluster_meat

        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 2))
        resid = rng.normal(size=20)
        meat, g = _cluster_meat(X, resid, np.zeros(20, dtype=int))
        score = X.T @ resid
        assert g == 1
        assert np.allclose(meat, np.outer(score, score))


class TestPooledClusteredSEAgainstHandComputation:
    r"""The clustered sandwich matches the textbook formula, coefficient included.

    $V = \frac{G}{G-1}\cdot\frac{N-1}{N-K}\,(X'X)^{-1}
         \bigl[\sum_g (X_g'e_g)(X_g'e_g)'\bigr](X'X)^{-1}$
    (Cameron-Miller 2015 / Stata's `vce(cluster)`), read against $t(G-1)$.
    Two-way is $V_A + V_B - V_{A\cap B}$ (Cameron-Gelbach-Miller 2011) with
    $\mathrm{df} = \min(G_A, G_B) - 1$ (Thompson 2011).
    """

    @staticmethod
    def _panel(n_dates=40, n_assets=15, seed=99):
        rng = np.random.default_rng(seed)
        rows = {"date": [], "asset_id": [], "factor": [], "forward_return": []}
        for d in range(n_dates):
            shock = rng.normal(0, 0.01)  # per-period common component
            for a in range(n_assets):
                f = rng.normal()
                rows["date"].append(datetime(2024, 1, 1) + timedelta(days=d))
                rows["asset_id"].append(f"a{a}")
                rows["factor"].append(f)
                rows["forward_return"].append(0.002 * f + shock + rng.normal(0, 0.01))
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_one_way_cluster_matches_the_textbook_sandwich(self):
        from factrix.metrics.fm_beta import pooled_beta
        from scipy import stats as sp_stats

        panel = self._panel()
        out = pooled_beta(panel)

        y = panel["forward_return"].to_numpy()
        x = panel["factor"].to_numpy()
        n, k = len(y), 2
        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        resid = y - X @ beta

        codes = panel["date"].to_numpy()
        uniq = np.unique(codes)
        meat = np.zeros((k, k))
        for c in uniq:
            m = codes == c
            score = X[m].T @ resid[m]
            meat += np.outer(score, score)
        g = len(uniq)

        xtx_inv = np.linalg.inv(X.T @ X)
        c_factor = (g / (g - 1)) * ((n - 1) / (n - k))
        V = c_factor * xtx_inv @ meat @ xtx_inv
        t_ref = beta[1] / np.sqrt(V[1, 1])
        p_ref = 2 * sp_stats.t.sf(abs(t_ref), g - 1)

        assert out.value == pytest.approx(float(beta[1]))
        assert out.stat == pytest.approx(float(t_ref))
        assert out.p_value == pytest.approx(float(p_ref))
        assert out.metadata["n_clusters"] == g

    def test_one_way_too_few_clusters_hides_the_slope(self):
        panel = self._panel(n_dates=2, n_assets=15)

        out = pooled_beta(panel)

        assert np.isnan(out.value)
        assert out.stat is None
        assert out.p_value == 1.0
        assert out.metadata["reason"] == "insufficient_clusters"
        assert out.metadata["n_clusters"] == 2
        assert WarningCode.METRIC_UNAVAILABLE.value in out.warning_codes

    def test_two_way_is_the_cgm_sum_and_thompson_df(self):
        from factrix.metrics.fm_beta import _cluster_meat, pooled_beta
        from scipy import stats as sp_stats

        panel = self._panel()
        out = pooled_beta(panel, two_way_cluster_col="asset_id")

        y = panel["forward_return"].to_numpy()
        x = panel["factor"].to_numpy()
        n, k = len(y), 2
        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        resid = y - X @ beta

        a = panel["date"].to_numpy()
        b = panel["asset_id"].to_numpy()
        _, ids_a = np.unique(a, return_inverse=True)
        _, ids_b = np.unique(b, return_inverse=True)
        inter = ids_a.astype(np.int64) * (int(ids_b.max()) + 1) + ids_b

        meat_a, g_a = _cluster_meat(X, resid, a)
        meat_b, g_b = _cluster_meat(X, resid, b)
        meat_i, g_i = _cluster_meat(X, resid, inter)

        combined = (
            (g_a / (g_a - 1)) * meat_a
            + (g_b / (g_b - 1)) * meat_b
            - (g_i / max(g_i - 1, 1)) * meat_i
        )
        xtx_inv = np.linalg.inv(X.T @ X)
        V = ((n - 1) / (n - k)) * xtx_inv @ combined @ xtx_inv
        df = min(g_a, g_b) - 1
        t_ref = beta[1] / np.sqrt(V[1, 1])

        assert out.stat == pytest.approx(float(t_ref))
        assert out.p_value == pytest.approx(float(2 * sp_stats.t.sf(abs(t_ref), df)))
        assert out.metadata["n_clusters_a"] == g_a
        assert out.metadata["n_clusters_b"] == g_b

    def test_two_way_too_few_clusters_hides_the_slope(self):
        panel = self._panel(n_dates=10, n_assets=2)

        out = pooled_beta(panel, two_way_cluster_col="asset_id")

        assert np.isnan(out.value)
        assert out.stat is None
        assert out.p_value == 1.0
        assert out.metadata["reason"] == "insufficient_clusters"
        assert out.metadata["n_clusters"] == 2
        assert out.metadata["n_clusters_a"] == 10
        assert out.metadata["n_clusters_b"] == 2
        assert WarningCode.METRIC_UNAVAILABLE.value in out.warning_codes


class TestShankenCorrection:
    """The EIV path: mandatory σ²_f, HAR df, and the degenerate regime."""

    @staticmethod
    def _beta_df(betas: np.ndarray) -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        return pl.DataFrame(
            {
                "date": [start + timedelta(days=i) for i in range(len(betas))],
                "beta": betas,
            }
        )

    @staticmethod
    def _draw(n: int, mean: float, rng_seed: int) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        return mean + rng.standard_normal(n) * 0.01

    @pytest.mark.parametrize("n_periods", [60, 120, 240])
    @pytest.mark.parametrize("overlap_periods", [None, 5])
    def test_correction_can_only_raise_the_p_value(self, n_periods, overlap_periods):
        """c >= 1 shrinks |t|; read against the SAME df, p can only grow.

        The bug this pins: the corrected t used to be read against ``n - 1``
        while the uncorrected p used the HAR effective df (~16 at T = 240),
        so the wider reference distribution more than undid the sqrt(c) SE
        inflation and the "conservative" correction returned a SMALLER p.
        """
        betas = self._draw(n_periods, 0.004, 11)
        beta_df = self._beta_df(betas)
        out = fm_beta(
            beta_df,
            overlap_periods=overlap_periods,
            is_estimated_factor=True,
            factor_return_var=0.0004,
        )
        assert out.metadata["shanken_c"] >= 1.0
        assert abs(out.stat) <= abs(out.metadata["stat_uncorrected"])
        assert out.p_value >= out.metadata["p_value_uncorrected"]

    def test_corrected_p_is_read_against_the_reported_hac_dof(self):
        from factrix._stats import _p_value_from_t

        betas = self._draw(120, 0.004, 3)
        out = fm_beta(
            self._beta_df(betas),
            overlap_periods=5,
            is_estimated_factor=True,
            factor_return_var=0.0004,
        )
        assert out.p_value == pytest.approx(
            _p_value_from_t(out.stat, out.n_obs, dof=out.metadata["hac_dof"])
        )

    def test_estimated_factor_without_a_variance_is_a_user_input_error(self):
        from factrix._errors import UserInputError

        betas = self._draw(120, 0.004, 5)
        with pytest.raises(UserInputError) as excinfo:
            fm_beta(self._beta_df(betas), is_estimated_factor=True)
        err = excinfo.value
        assert err.field == "factor_return_var"
        assert err.func_name == "fm_beta"
        # The proxy is refused because it is algebraically inert, and the
        # message has to say so — see the docstring's 1 + t^2/T identity.
        assert "1 + t" in str(err)

    def test_degenerate_factor_variance_raises_a_warning_code(self):
        betas = self._draw(120, 0.004, 7)
        with pytest.warns(UserWarning, match="Shanken"):
            out = fm_beta(
                self._beta_df(betas),
                is_estimated_factor=True,
                factor_return_var=0.0,
            )
        assert WarningCode.DEGENERATE_VARIANCE.value in out.warning_codes
        assert out.metadata["shanken_correction"] == "skipped_zero_factor_variance"
        # Skipped, so the uncorrected result stands and no Shanken key appears.
        assert "shanken_c" not in out.metadata
        uncorrected = fm_beta(self._beta_df(betas))
        assert out.p_value == pytest.approx(uncorrected.p_value)

    def test_expected_warnings_silences_the_degenerate_regime(self):
        betas = self._draw(120, 0.004, 7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = fm_beta(
                self._beta_df(betas),
                is_estimated_factor=True,
                factor_return_var=0.0,
                expected_warnings=(WarningCode.DEGENERATE_VARIANCE.value,),
            )
        assert WarningCode.DEGENERATE_VARIANCE.value in out.warning_codes

    def test_no_variance_source_key_is_reported(self):
        """Only one source survives, so the key carries no information."""
        betas = self._draw(120, 0.004, 9)
        out = fm_beta(
            self._beta_df(betas),
            is_estimated_factor=True,
            factor_return_var=0.0004,
        )
        assert "shanken_factor_return_var_source" not in out.metadata
        assert out.metadata["shanken_factor_return_var"] == pytest.approx(0.0004)
