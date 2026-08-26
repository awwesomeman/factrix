"""Tests for factrix.preprocess.normalize."""

import math
from datetime import datetime

import numpy as np
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.preprocess.normalize import cross_sectional_zscore, mad_winsorize


class TestMADWinsorize:
    def test_clips_outlier(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        assert result["factor"].max() < 100.0

    def test_noop_when_disabled(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=0)
        assert result["factor"].to_list() == df["factor"].to_list()

    def test_per_date(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 2)] * 3,
                "factor": [1.0, 2.0, 100.0, 10.0, 20.0, 1000.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        d1 = result.filter(pl.col("date") == datetime(2024, 1, 1))["factor"].max()
        d2 = result.filter(pl.col("date") == datetime(2024, 1, 2))["factor"].max()
        assert d1 < 100.0
        assert d2 < 1000.0


class TestCrossSectionalZScore:
    def test_zero_mad(self):
        """All same value → MAD=0 → fill_nan(0.0)."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        for v in result["factor_zscore"].to_list():
            assert v == 0.0

    def test_output_column(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        assert "factor_zscore" in result.columns

    def test_median_near_zero(self):
        """After z-score, median should be near 0."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 100,
                "factor": list(range(100)),
            }
        ).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),
            pl.col("factor").cast(pl.Float64),
        )
        result = cross_sectional_zscore(df)
        median = result["factor_zscore"].median()
        assert abs(median) < 0.1


class TestZeroMADFallback:
    """Regression: MAD == 0 (>50% ties) must not blow the scale up to inf."""

    def _bucketed(self) -> pl.DataFrame:
        # 4 ties + 1 outlier + 1 null → median 1.0, MAD 0.0, std(ddof=1) > 0.
        return pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 6,
                "factor": [1.0, 1.0, 1.0, 1.0, 2.0, None],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_zscore_is_finite_and_rank_preserving(self):
        z = cross_sectional_zscore(self._bucketed())["factor_zscore"].to_list()
        assert all(v is None or math.isfinite(v) for v in z)
        assert z[:4] == [0.0, 0.0, 0.0, 0.0]
        # std(ddof=1) of [1,1,1,1,2] = 0.4472 → (2 - 1) / 0.4472
        assert z[4] == pytest.approx(1.0 / 0.4472135954999579)

    def test_zscore_keeps_null_null(self):
        """`fill_null(0.0)` imputed missing factors to exactly the median."""
        z = cross_sectional_zscore(self._bucketed())["factor_zscore"].to_list()
        assert z[5] is None

    def test_zscore_nan_input_yields_null(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, float("nan"), 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        z = cross_sectional_zscore(df)["factor_zscore"].to_list()
        assert z[2] is None
        # The NaN must not poison the rest of the cross-section.
        assert all(math.isfinite(v) for i, v in enumerate(z) if i != 2)

    def test_zscore_constant_date_is_zero_not_nan(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 4,
                "factor": [3.0, 3.0, 3.0, None],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        z = cross_sectional_zscore(df)["factor_zscore"].to_list()
        assert z == [0.0, 0.0, 0.0, None]

    def test_winsorize_does_not_collapse_bucketed_factor(self):
        """MAD == 0 clipped every value to the median, destroying the factor."""
        out = mad_winsorize(self._bucketed(), n_mad=3.0)["factor"].to_list()
        assert out == [1.0, 1.0, 1.0, 1.0, 2.0, None]

    def test_winsorize_std_fallback_still_clips_far_outlier(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 12,
                "factor": [1.0] * 10 + [2.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        out = mad_winsorize(df, n_mad=3.0)["factor"].to_list()
        assert max(out) < 100.0


def _one_date(values, name="factor"):
    return pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * len(values),
            "asset_id": [str(i) for i in range(len(values))],
            name: values,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestSmallSampleMADScaling:
    """Croux-Rousseeuw (1992) finite-sample correction on the MAD scale."""

    def test_scale_carries_b_n_at_n_five(self):
        # median([1,2,3,4,5]) = 3; MAD = median([2,1,0,1,2]) = 1.
        # scale = b_5 * 1.4826 * 1 = 1.206 * 1.4826 = 1.78802...
        z = cross_sectional_zscore(_one_date([1.0, 2.0, 3.0, 4.0, 5.0]))
        scale = 1.206 * 1.4826
        assert z["factor_zscore"].to_list() == pytest.approx(
            [(v - 3.0) / scale for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
        )

    def test_scale_uses_asymptotic_expansion_above_the_table(self):
        values = [float(v) for v in range(1, 12)]  # n = 11, median 6, MAD 3
        z = cross_sectional_zscore(_one_date(values))
        scale = (11 / (11 - 0.8)) * 1.4826 * 3.0
        assert z["factor_zscore"][-1] == pytest.approx((11.0 - 6.0) / scale)

    @pytest.mark.parametrize(
        ("n_assets", "uncorrected"), [(5, 0.821), (10, 0.913), (20, 0.959)]
    )
    def test_scale_is_unbiased_for_sigma_at_every_cross_section_size(
        self, n_assets, uncorrected
    ):
        """The bias is N-dependent, so thin dates were handed larger z-scores.

        Monte-Carlo on a standard-normal factor (true sigma = 1): the bare
        1.4826 constant under-estimates sigma by 18% at n=5, 9% at n=10 and 4%
        at n=20, while ``b_n * 1.4826 * MAD`` is unbiased at all three. In an
        unbalanced panel that difference is exactly what makes anything pooling
        or weighting by z over-weight the thinnest cross-sections.
        """
        rng = np.random.default_rng(7)
        n_dates = 4000
        x = rng.standard_normal((n_dates, n_assets))
        df = pl.DataFrame(
            {
                "date": np.repeat(np.arange(n_dates), n_assets),
                "asset_id": np.tile(np.arange(n_assets), n_dates),
                "factor": x.ravel(),
            }
        )
        z = cross_sectional_zscore(df).sort("date", "asset_id")["factor_zscore"]
        # scale = value / z, recovered from any non-centre name.
        centred = (x - np.median(x, axis=1, keepdims=True)).ravel()
        off_centre = np.abs(centred) > 1e-9
        mean_scale = float(np.mean(centred[off_centre] / z.to_numpy()[off_centre]))
        assert mean_scale == pytest.approx(1.0, abs=0.02)
        # ... and materially better than the uncorrected asymptotic constant.
        assert abs(mean_scale - 1.0) < abs(uncorrected - 1.0)

    def test_n_mad_is_a_nominal_sigma_clip_at_small_n(self):
        """3 MAD units must not be an effective 2.46 sigma band at n=5."""
        # spread [-2,-1,0,1,2]: MAD = 1, corrected scale 1.78802, band +-5.364.
        out = mad_winsorize(_one_date([-2.0, -1.0, 0.0, 1.0, 6.0]), n_mad=3.0)
        assert out["factor"].to_list()[-1] == pytest.approx(3.0 * 1.206 * 1.4826)


class TestCenterChoice:
    def test_median_centred_output_is_not_mean_zero(self):
        """Documented consequence: w ~ z carries a net long leg on a skewed factor."""
        sparse = [0.0] * 18 + [1.0, 1.0]
        with pytest.warns(UserWarning, match="sparse_winsorize_skipped|zero_mad"):
            z = cross_sectional_zscore(_one_date(sparse))["factor_zscore"]
        assert float(z.mean()) > 0.3

    def test_mean_centre_is_exactly_mean_zero(self):
        sparse = [0.0] * 18 + [1.0, 1.0]
        with pytest.warns(UserWarning, match="zero_mad_std_fallback"):
            z = cross_sectional_zscore(_one_date(sparse), center="mean")[
                "factor_zscore"
            ]
        assert float(z.mean()) == pytest.approx(0.0, abs=1e-12)

    def test_rejects_unknown_centre(self):
        with pytest.raises(UserInputError):
            cross_sectional_zscore(_one_date([1.0, 2.0, 3.0]), center="mode")


class TestScaleMethodSwitchIsAnnounced:
    def test_std_fallback_emits_the_code(self):
        with pytest.warns(UserWarning, match="zero_mad_std_fallback"):
            cross_sectional_zscore(_one_date([1.0, 1.0, 1.0, 1.0, 2.0]))

    def test_thin_cross_section_yields_null_not_a_fabricated_zero(self):
        with pytest.warns(UserWarning, match="insufficient_scale_assets"):
            z = cross_sectional_zscore(_one_date([7.0]))["factor_zscore"].to_list()
        assert z == [None]

    def test_two_asset_date_no_longer_returns_a_constant(self):
        with pytest.warns(UserWarning, match="insufficient_scale_assets"):
            z = cross_sectional_zscore(_one_date([1.0, 9.0]))["factor_zscore"].to_list()
        assert z == [None, None]

    def test_thin_date_is_left_unwinsorized(self):
        with pytest.warns(UserWarning, match="insufficient_scale_assets"):
            out = mad_winsorize(_one_date([1.0, 500.0]), n_mad=3.0)
        assert out["factor"].to_list() == [1.0, 500.0]


class TestSparseWinsorize:
    def test_sparse_trigger_magnitude_is_preserved(self):
        """1 trigger / 50 assets used to lose 58% of its magnitude to 3 x std."""
        with pytest.warns(UserWarning, match="sparse_winsorize_skipped"):
            out = mad_winsorize(_one_date([0.0] * 49 + [1.0]), n_mad=3.0)
        assert out["factor"].to_list()[-1] == 1.0

    def test_bucketed_non_sparse_factor_still_uses_the_std_fallback(self):
        with pytest.warns(UserWarning, match="zero_mad_std_fallback"):
            out = mad_winsorize(_one_date([1.0] * 10 + [2.0, 100.0]), n_mad=3.0)
        assert max(out["factor"].to_list()) < 100.0


class TestNonFiniteHandling:
    def test_infinity_becomes_null_not_a_fabricated_finite_value(self):
        with pytest.warns(UserWarning, match="non_finite_input_dropped"):
            out = mad_winsorize(
                _one_date([1.0, 2.0, 3.0, 4.0, float("inf")]), n_mad=3.0
            )
        assert out["factor"].to_list() == [1.0, 2.0, 3.0, 4.0, None]

    def test_nan_becomes_null_in_winsorize_too(self):
        with pytest.warns(UserWarning, match="non_finite_input_dropped"):
            out = mad_winsorize(
                _one_date([1.0, 2.0, 3.0, 4.0, float("nan")]), n_mad=3.0
            )
        assert out["factor"].to_list()[-1] is None


class TestNMadValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "3", None, True])
    def test_rejects_non_finite_or_non_numeric(self, bad):
        with pytest.raises(UserInputError):
            mad_winsorize(_one_date([1.0, 2.0, 3.0]), n_mad=bad)

    def test_zero_still_disables(self):
        df = _one_date([1.0, 2.0, 300.0])
        assert mad_winsorize(df, n_mad=0.0)["factor"].to_list() == [1.0, 2.0, 300.0]


class TestZScoreOutputColumnName:
    def test_column_name_derives_from_factor_col(self):
        df = _one_date([1.0, 2.0, 3.0, 4.0, 5.0], name="value_raw")
        out = cross_sectional_zscore(df, factor_col="value_raw")
        assert "value_raw_zscore" in out.columns
        assert "factor_zscore" not in out.columns

    def test_two_factors_do_not_collide(self):
        df = _one_date([1.0, 2.0, 3.0, 4.0, 5.0]).with_columns(
            pl.col("factor").alias("momentum")
        )
        out = cross_sectional_zscore(cross_sectional_zscore(df), factor_col="momentum")
        assert {"factor_zscore", "momentum_zscore"} <= set(out.columns)
