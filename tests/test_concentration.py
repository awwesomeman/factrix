"""Tests for factrix.metrics.concentration."""

from datetime import datetime, timedelta

import polars as pl
from factrix.metrics.concentration import top_concentration


class TestQ1Concentration:
    def test_uniform_factor(self):
        """All Q1 stocks have same |factor| → HHI = 1/n_top → eff_n = n_top."""
        n_dates, n_assets = 10, 20
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(n_assets):
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),  # ranks 1..20
                        "forward_return": 0.01,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, forward_periods=1, q_top=0.2)
        # Top 20% = 4 stocks, all with similar |factor| → eff_n near 4
        assert result.value > 2.0  # reasonably diversified

    def test_single_dominant(self):
        """One stock has extreme factor → eff_n near 1."""
        n_dates = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(10):
                f = 100.0 if i == 9 else 1.0  # asset_9 dominates
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": f,
                        "forward_return": 0.01,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, forward_periods=1, q_top=0.2)
        assert result.value < 2.0  # highly concentrated

    def test_alpha_contribution_sees_return_concentration(self):
        """Uniform factor + one outlier return → alpha-weighted HHI flags it."""
        n_dates = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            # Top bucket = assets 8, 9 (top 20% of 10 with uniform factor
            # values — ranks are broken by tie handling).
            for i in range(10):
                ret = 0.10 if i == 9 else 0.001
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),
                        "forward_return": ret,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        density = top_concentration(
            df,
            forward_periods=1,
            q_top=0.2,
            weight_by="abs_factor",
        )
        alpha = top_concentration(
            df,
            forward_periods=1,
            q_top=0.2,
            weight_by="alpha_contribution",
        )
        # Signal says top bucket is balanced (two near-equal |factor|).
        # Alpha says bucket is driven by one outlier → far more concentrated.
        assert alpha.value < density.value
        assert alpha.metadata["weight_by"] == "alpha_contribution"
        assert density.metadata["weight_by"] == "abs_factor"

    def test_alpha_contribution_missing_return_column(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "asset_id": [f"A{i}" for i in range(5)],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, weight_by="alpha_contribution")
        assert result.metadata.get("reason") == "no_return_column"


def _panel(rows):
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestTopBucketMembership:
    """The bucket is a strict count over the finite cross-section."""

    @staticmethod
    def _uniform_panel(n_assets, n_dates=10, factor_of=None, return_of=None):
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(n_assets):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": (float(i + 1) if factor_of is None else factor_of(i)),
                        "forward_return": (0.01 if return_of is None else return_of(i)),
                    }
                )
        return _panel(rows)

    def test_count_cutoff_is_floor_n_times_q(self):
        """n=10, q=0.2 selects 2 names, not 3 (percent-rank cutoff was
        inclusive at the boundary and took one name too many)."""
        result = top_concentration(
            self._uniform_panel(10), forward_periods=1, q_top=0.2
        )
        assert result.metadata["mean_n_top"] == 2.0

    def test_count_cutoff_does_not_drift_with_n(self):
        """n=100, q=0.2 selects 20, not 21 — the old off-by-one grew with n."""
        result = top_concentration(
            self._uniform_panel(100), forward_periods=1, q_top=0.2
        )
        assert result.metadata["mean_n_top"] == 20.0

    def test_bucket_never_empty(self):
        """floor(n*q) == 0 still selects the single top name."""
        result = top_concentration(self._uniform_panel(4), forward_periods=1, q_top=0.2)
        assert result.metadata["mean_n_top"] == 1.0

    def test_null_factors_do_not_shrink_the_bucket(self):
        """5 valid + 5 null names, q=0.2 → 1 name (floor(5*0.2)).

        The old denominator was ``pl.len()`` (10), so the percent-rank of the
        best valid name was 5/10 = 0.5 < 0.8 and the bucket came out EMPTY.
        """
        panel = self._uniform_panel(
            10, factor_of=lambda i: float(i + 1) if i < 5 else None
        )
        result = top_concentration(panel, forward_periods=1, q_top=0.2)
        assert result.metadata["mean_n_top"] == 1.0
        assert result.n_obs == 10

    def test_nan_factor_is_never_selected(self):
        """NaN sorts as the largest value under a descending rank."""
        panel = self._uniform_panel(
            10, factor_of=lambda i: float("nan") if i == 0 else float(i)
        )
        result = top_concentration(panel, forward_periods=1, q_top=0.2)
        # 9 finite names -> floor(9*0.2) = 1, and it must be asset A9 (factor
        # 9.0), not the NaN one.
        assert result.metadata["mean_n_top"] == 1.0
        # A single name in the bucket -> HHI 1 -> eff_n 1.
        assert result.value == 1.0


class TestAlphaContributionWeights:
    def test_null_return_leaves_both_hhi_and_n_top(self):
        """A selected name with no realised return is not a member.

        Old behaviour: its weight was null so it left the HHI numerator, but
        ``n_top = pl.len()`` still counted it — the eff_n / n_top ratio came
        out biased low (spurious "concentration").
        """
        rows = []
        for d in range(10):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(10):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),
                        # Top bucket is A8, A9; A9 has no return.
                        "forward_return": None if i == 9 else 0.01,
                    }
                )
        result = top_concentration(
            _panel(rows),
            forward_periods=1,
            q_top=0.2,
            weight_by="alpha_contribution",
        )
        # Only A8 carries a weight -> bucket of one -> eff_n == n_top == 1,
        # ratio 1.0 (perfectly "diversified" over the one name observed),
        # not eff_n=1 / n_top=2 = 0.5.
        assert result.metadata["mean_n_top"] == 1.0
        assert result.metadata["ratio_eff_to_total"] == 1.0
        assert result.metadata["n_top_members_selected"] == 20
        assert result.metadata["n_top_members_dropped"] == 10

    def test_all_weights_missing_short_circuits(self):
        rows = []
        for d in range(10):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(10):
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),
                        "forward_return": None,
                    }
                )
        result = top_concentration(
            _panel(rows),
            forward_periods=1,
            q_top=0.2,
            weight_by="alpha_contribution",
        )
        assert result.metadata["reason"] == "insufficient_top_bucket_periods"
        assert result.n_obs == 0


class TestOneSignedFactorWarning:
    """``abs_factor`` weights assume a zero-centred factor.

    The HHI of |f| is not location-invariant, so a factor that never
    changes sign gives a concentration reading that is an artefact of its
    level. Advisory: the metric still runs.
    """

    @staticmethod
    def _panel(shift: float, n_assets: int = 30):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(0)
        rows = []
        for d in range(40):
            f = rng.standard_normal(n_assets) + shift
            for a in range(n_assets):
                rows.append(
                    {
                        "date": datetime(2024, 1, 1) + timedelta(days=d),
                        "asset_id": f"A{a}",
                        "factor": float(f[a]),
                        "forward_return": float(rng.normal(0, 0.01)),
                    }
                )
        return pl.DataFrame(rows)

    def test_all_negative_factor_warns(self):
        import pytest
        from factrix._codes import WarningCode
        from factrix.metrics.concentration import top_concentration

        with pytest.warns(UserWarning, match="never changes sign"):
            result = top_concentration(self._panel(shift=-10.0), forward_periods=1)
        assert WarningCode.ONE_SIGNED_FACTOR.value in result.warning_codes
        assert result.value is not None  # advisory only — metric still ran

    def test_counts_land_in_metadata_and_the_message_is_constant(self):
        """Counts belong in metadata so the warning text can de-duplicate.

        Python keys its duplicate filter on (text, category, module, lineno).
        Interpolating the sign counts made the text vary with the panel, so a
        ``by_slice`` sweep or a multi-factor ``evaluate`` — where each call
        sees a different number of values — emitted one warning per call
        instead of one per session. Two panels of *different sizes* are what
        exercises that; two identical calls de-duplicate either way.
        """
        import warnings as _warnings

        from factrix._codes import WarningCode
        from factrix.metrics.concentration import top_concentration

        small = self._panel(shift=-10.0, n_assets=30)
        large = self._panel(shift=-10.0, n_assets=45)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("default")
            first = top_concentration(small, forward_periods=1)
            second = top_concentration(large, forward_periods=1)

        one_signed = [w for w in caught if "never changes sign" in str(w.message)]
        assert len(one_signed) == 1, (
            "panels of different sizes must still produce one de-duplicated "
            f"warning; got {[str(w.message) for w in one_signed]}"
        )
        assert not any(ch.isdigit() for ch in str(one_signed[0].message))

        assert first.metadata["n_negative_factor_values"] == 30 * 40
        assert second.metadata["n_negative_factor_values"] == 45 * 40
        for result in (first, second):
            assert WarningCode.ONE_SIGNED_FACTOR.value in result.warning_codes
            assert result.metadata["n_positive_factor_values"] == 0

    def test_counts_absent_when_the_factor_is_two_sided(self):
        """The keys are conditional — they describe why the code fired."""
        from factrix.metrics.concentration import top_concentration

        result = top_concentration(self._panel(shift=0.0), forward_periods=1)
        assert "n_positive_factor_values" not in result.metadata
        assert "n_negative_factor_values" not in result.metadata

    def test_centred_factor_does_not_warn(self):
        from factrix._codes import WarningCode
        from factrix.metrics.concentration import top_concentration

        result = top_concentration(self._panel(shift=0.0), forward_periods=1)
        assert WarningCode.ONE_SIGNED_FACTOR.value not in result.warning_codes

    def test_alpha_contribution_is_exempt(self):
        """The realised-contribution weight does not read |factor| as
        strength, so a one-signed factor is fine there."""
        from factrix._codes import WarningCode
        from factrix.metrics.concentration import top_concentration

        result = top_concentration(
            self._panel(shift=-10.0), forward_periods=1, weight_by="alpha_contribution"
        )
        assert WarningCode.ONE_SIGNED_FACTOR.value not in result.warning_codes
