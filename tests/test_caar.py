"""Tests for factrix.metrics.caar — CAAR, BMP, event_hit_rate, event_ic."""

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._stats import _calc_t_stat, _p_value_from_t
from factrix.metrics.caar import (
    bmp_test,
    caar,
    compute_caar,
)
from factrix.metrics.event_quality import event_hit_rate, event_ic


def _event_calendar_panel(
    event_ordinals: list[int],
    returns: list[float],
    n_calendar: int,
) -> pl.DataFrame:
    """Single-asset panel on a contiguous calendar of ``n_calendar`` days.

    Events (factor=1, forward_return=given) sit at ``event_ordinals``; all
    other days are non-events (factor=0). The non-event days populate the
    full calendar so ``compute_caar``'s ``date_ordinal`` reflects true
    calendar position, not event index.
    """
    base = datetime(2020, 1, 1)
    ret_by_ord = dict(zip(event_ordinals, returns, strict=True))
    rows = []
    for i in range(n_calendar):
        d = base + timedelta(days=i)
        is_event = i in ret_by_ord
        rows.append(
            {
                "date": d,
                "asset_id": "A",
                "factor": 1.0 if is_event else 0.0,
                "forward_return": ret_by_ord.get(i, 0.0),
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _greedy_keep(ordinals: list[int], forward_periods: int) -> list[int]:
    """Reference: greedily keep ordinals >= forward_periods calendar apart."""
    kept: list[int] = []
    last: int | None = None
    for o in ordinals:
        if last is None or o - last >= forward_periods:
            kept.append(o)
            last = o
    return kept


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    signal_strength: float = 0.01,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic event density data.

    Each day, each asset has ``event_prob`` chance of triggering an event
    (factor = +1 or -1). Post-event forward_return = signal_strength *
    sign(factor) + noise.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        for a in assets:
            is_event = rng.random() < event_prob
            if is_event:
                direction = rng.choice([-1.0, 1.0])
                ret = signal_strength * direction + rng.normal(0, 0.02)
            else:
                direction = 0.0
                ret = rng.normal(0, 0.02)

            rows.append(
                {
                    "date": d,
                    "asset_id": a,
                    "factor": direction,
                    "forward_return": ret,
                    "price": 100 + rng.normal(0, 5),
                }
            )

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_signal() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.03)


@pytest.fixture
def noise_signal() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.0, seed=99)


@pytest.fixture
def single_asset_signal() -> pl.DataFrame:
    return _make_event_signal(
        n_assets=1, n_dates=1000, event_prob=0.05, signal_strength=0.03, seed=77
    )


# ---------------------------------------------------------------------------
# compute_caar
# ---------------------------------------------------------------------------


class TestComputeCaar:
    def test_returns_date_caar_columns(self, strong_signal):
        result = compute_caar(strong_signal)
        assert "date" in result.columns
        assert "caar" in result.columns
        assert len(result) > 0

    def test_filters_non_events(self, strong_signal):
        result = compute_caar(strong_signal)
        n_event_periods = strong_signal.filter(pl.col("factor") != 0)["date"].n_unique()
        assert len(result) == n_event_periods

    def test_strong_signal_positive_mean(self, strong_signal):
        result = compute_caar(strong_signal)
        assert result["caar"].mean() > 0

    def test_noise_mean_near_zero(self, noise_signal):
        result = compute_caar(noise_signal)
        assert abs(result["caar"].mean()) < 0.01

    def test_empty_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [0.0],
                "forward_return": [0.01],
            }
        )
        result = compute_caar(df)
        assert len(result) == 0

    def test_n_events_column_counts_per_date_events(self, strong_signal):
        result = compute_caar(strong_signal)
        assert "n_events" in result.columns
        events = strong_signal.filter(pl.col("factor") != 0)
        expected = events.group_by("date").len().sort("date")
        got = result.sort("date")
        assert got["n_events"].to_list() == expected["len"].to_list()
        assert result["n_events"].sum() == events.height

    def test_n_events_reflects_clustering(self):
        d1 = datetime(2020, 1, 1)
        d2 = datetime(2020, 1, 2)
        df = pl.DataFrame(
            {
                "date": [d1, d1, d1, d1, d1, d2],
                "asset_id": ["a", "b", "c", "d", "e", "f"],
                "factor": [1.0] * 6,
                "forward_return": [0.01, 0.02, -0.01, 0.0, 0.03, -0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_caar(df).sort("date")
        assert result["n_events"].to_list() == [5, 1]


# ---------------------------------------------------------------------------
# compute_caar — input-form behaviour matrix
# ---------------------------------------------------------------------------


def _two_event_panel(
    factor_a: float,
    factor_b: float,
    ret_a: float,
    ret_b: float,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "asset_id": ["A", "B"],
            "factor": [factor_a, factor_b],
            "forward_return": [ret_a, ret_b],
        }
    )


class TestComputeCaarInputForms:
    def test_occurrence_only_zero_one(self):
        df = _two_event_panel(1.0, 1.0, 0.02, -0.01)
        result = compute_caar(df)
        assert result["caar"].to_list() == pytest.approx([0.02, -0.01])

    def test_signed_ternary(self):
        df = _two_event_panel(-1.0, 1.0, -0.02, 0.03)
        result = compute_caar(df)
        assert result["caar"].to_list() == pytest.approx([0.02, 0.03])

    def test_magnitude_weighted_zero_R(self):
        df = _two_event_panel(2.5, -3.0, 0.01, 0.02)
        result = compute_caar(df)
        assert result["caar"].to_list() == pytest.approx([0.025, -0.06])

    def test_magnitude_preserved_not_dropped(self):
        # Pins the post-change behaviour — sign-coerced math would yield
        # 0.01, magnitude-preserving math yields 0.025.
        df = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1)],
                "asset_id": ["A"],
                "factor": [2.5],
                "forward_return": [0.01],
            }
        )
        result = compute_caar(df)
        assert result["caar"][0] == pytest.approx(0.025)
        assert result["caar"][0] != pytest.approx(0.01)

    def test_within_date_cs_average_weighted(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "asset_id": ["A", "B"],
                "factor": [2.0, -1.0],
                "forward_return": [0.03, 0.04],
            }
        )
        result = compute_caar(df)
        assert len(result) == 1
        assert result["caar"][0] == pytest.approx(0.01)

    def test_warns_on_mixed_sign_non_ternary(self):
        df = _two_event_panel(2.5, -3.0, 0.01, 0.02)
        with pytest.warns(UserWarning, match="magnitude-weighted CAAR"):
            compute_caar(df)

    def test_no_warn_on_clean_ternary(self):
        df = _two_event_panel(-1.0, 1.0, -0.02, 0.03)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            compute_caar(df)  # must not raise

    def test_no_warn_on_all_non_negative(self):
        df = _two_event_panel(2.5, 3.0, 0.01, 0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            compute_caar(df)

    def test_no_warn_on_indicator_zero_one(self):
        df = _two_event_panel(1.0, 1.0, 0.02, -0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            compute_caar(df)

    def test_caller_can_opt_into_ternary_via_sign(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2020, 1, 1)],
                "asset_id": ["A"],
                "factor": [2.5],
                "forward_return": [0.01],
            }
        )
        coerced = df.with_columns(pl.col("factor").sign())
        result = compute_caar(coerced)
        assert result["caar"][0] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# caar (significance test)
# ---------------------------------------------------------------------------


class TestCaar:
    def test_strong_signal_significant(self, strong_signal):
        caar_df = compute_caar(strong_signal)
        result = caar(caar_df)
        assert result.value > 0
        assert abs(result.stat) > 2.0
        assert result.p_value < 0.05

    def test_noise_not_significant(self, noise_signal):
        caar_df = compute_caar(noise_signal)
        result = caar(caar_df)
        assert abs(result.stat) < 2.0

    def test_insufficient_data(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "caar": pl.Series([], dtype=pl.Float64),
            }
        )
        result = caar(df)
        assert math.isnan(result.value)
        assert result.stat is None

    def test_total_events_in_metadata(self, strong_signal):
        # caar reports the underlying event count (across-asset, pre-collapse)
        # next to n_event_periods (the number of periods with an event).
        caar_df = compute_caar(strong_signal)
        result = caar(caar_df)
        n_events_panel = strong_signal.filter(pl.col("factor") != 0).height
        assert result.metadata["total_events"] == n_events_panel
        # Multi-asset clustering: far more events than event dates.
        assert result.metadata["total_events"] > result.metadata["n_event_periods"]

    def test_total_events_falls_back_without_n_events_column(self):
        # Hand-built caar_df bypassing compute_caar has no n_events column;
        # total_events degrades to the event-date count rather than raising.
        rng = np.random.default_rng(0)
        n = 40
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame(
            {"date": dates, "caar": rng.normal(0.01, 0.02, n)}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = caar(df, forward_periods=1)
        assert result.metadata["total_events"] == result.metadata["n_event_periods"]


class TestCaarEventSpacedSampling:
    """caar() non-overlap subsampling on the calendar-irregular event series."""

    def test_pvalue_matches_handrolled_calendar_reference(self):
        # 90 events spanning a clustered block (gap 1, gets thinned) and a
        # sparse tail (gap 5, all kept). n=90 == MIN_EVENTS_WARN*fp, so no
        # FEW_EVENTS warning fires — assert the path is clean.
        fp = 3
        clustered = list(range(50))
        sparse = [60 + 5 * k for k in range(40)]
        ordinals = clustered + sparse
        n_cal = ordinals[-1] + 1
        rng = np.random.default_rng(7)
        returns = list(rng.normal(0.01, 0.02, len(ordinals)))

        panel = _event_calendar_panel(ordinals, returns, n_cal)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            caar_df = compute_caar(panel)
            result = caar(caar_df, forward_periods=fp)

        kept = _greedy_keep(ordinals, fp)
        ret_by_ord = dict(zip(ordinals, returns, strict=True))
        kept_vals = np.array([ret_by_ord[o] for o in kept])
        t_ref = _calc_t_stat(
            float(kept_vals.mean()), float(kept_vals.std(ddof=1)), len(kept_vals)
        )
        p_ref = _p_value_from_t(t_ref, len(kept_vals))

        assert result.metadata["n_event_periods_sampled"] == len(kept)
        assert result.stat == pytest.approx(t_ref)
        assert result.p_value == pytest.approx(p_ref)

    def test_sparse_events_not_downsampled(self):
        # Every event already separated by > forward_periods → all kept,
        # unlike index-stride sampling which would thin independent events.
        ordinals = [10 * k for k in range(25)]  # gap 10, fp 3
        rng = np.random.default_rng(1)
        returns = list(rng.normal(0.0, 0.02, len(ordinals)))
        panel = _event_calendar_panel(ordinals, returns, ordinals[-1] + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = caar(compute_caar(panel), forward_periods=3)
        assert result.metadata["n_event_periods_sampled"] == len(ordinals)

    def test_clustered_events_downsampled_to_calendar_gap(self):
        # 40 consecutive-day events (gap 1) thinned to >= fp apart.
        fp = 5
        ordinals = list(range(40))
        rng = np.random.default_rng(2)
        returns = list(rng.normal(0.0, 0.02, len(ordinals)))
        panel = _event_calendar_panel(ordinals, returns, len(ordinals))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = caar(compute_caar(panel), forward_periods=fp)
        assert result.metadata["n_event_periods_sampled"] == len(_greedy_keep(ordinals, fp))
        assert result.metadata["n_event_periods_sampled"] == 8  # 0,5,10,...,35

    def test_dense_regime_equals_index_stride(self):
        # Contiguous daily events: calendar gap == index gap, so n_sampled
        # matches every-fp-th-row sampling (sanity that the fix is a no-op
        # on the regime where the old behaviour was already correct).
        fp = 3
        ordinals = list(range(60))
        rng = np.random.default_rng(4)
        returns = list(rng.normal(0.01, 0.02, len(ordinals)))
        panel = _event_calendar_panel(ordinals, returns, len(ordinals))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = caar(compute_caar(panel), forward_periods=fp)
        # 60 dates, every 3rd → 20 kept
        assert result.metadata["n_event_periods_sampled"] == 20


# ---------------------------------------------------------------------------
# bmp_test
# ---------------------------------------------------------------------------


class TestBmpTest:
    def test_strong_signal_significant(self, strong_signal):
        result = bmp_test(strong_signal)
        assert abs(result.stat) > 2.0
        assert result.metadata["stat_type"] == "z"

    def test_noise_not_significant(self, noise_signal):
        result = bmp_test(noise_signal)
        assert abs(result.stat) < 2.0

    def test_no_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [0.0],
                "forward_return": [0.01],
                "price": [100.0],
            }
        )
        result = bmp_test(df)
        assert math.isnan(result.value)

    def test_uses_price_for_vol(self, strong_signal):
        result = bmp_test(strong_signal)
        assert result.metadata.get("n_events", 0) > 0

    def test_kolari_pynnonen_shrinks_z_when_clustered(self, strong_signal):
        """K-P adjustment must not expand |z|; when ρ>0 it strictly shrinks."""
        raw = bmp_test(strong_signal, kolari_pynnonen_adjust=False)
        adj = bmp_test(strong_signal, kolari_pynnonen_adjust=True)
        assert adj.metadata.get("kolari_pynnonen_applied") is True
        r = adj.metadata["kolari_pynnonen_r"]
        n_eff = adj.metadata["kolari_pynnonen_n_eff"]
        assert 0.0 <= r <= 1.0
        assert n_eff >= 1.0
        # Scaling factor ≤ 1 always (r ∈ [0,1]), so |z_adj| ≤ |z_raw|.
        assert abs(adj.stat) <= abs(raw.stat) + 1e-9
        assert adj.metadata["stat_uncorrected"] == pytest.approx(raw.stat)

    def test_prediction_error_variance_default_off(self, strong_signal):
        result = bmp_test(strong_signal)
        assert result.metadata["include_prediction_error_variance"] is False

    def test_prediction_error_variance_scales_value_and_std(self, strong_signal):
        """Strict denominator √(1+1/T) shrinks each SAR uniformly; z is invariant."""
        T = 60
        ratio = 1.0 / math.sqrt(1.0 + 1.0 / T)
        raw = bmp_test(strong_signal, estimation_window=T)
        strict = bmp_test(
            strong_signal,
            estimation_window=T,
            include_prediction_error_variance=True,
        )
        assert strict.metadata["include_prediction_error_variance"] is True
        # Uniform SAR rescaling → mean and std scale by the same ratio,
        # leaving z = mean / (std / √N) invariant. This is the expected
        # property of BMP under mean-adjusted residuals + a single
        # estimation_window; the flag exists to document the strict
        # standardiser, not to alter inference in this regime.
        assert strict.stat == pytest.approx(raw.stat, rel=1e-6)
        assert strict.value == pytest.approx(raw.value * ratio, rel=1e-6)
        assert strict.metadata["std_sar"] == pytest.approx(
            raw.metadata["std_sar"] * ratio, rel=1e-6
        )

    def test_pe_variance_composes_with_kolari_pynnonen(self, strong_signal):
        """Both flags on: PE scales SAR uniformly; KP shrinks z. No interference."""
        T = 60
        ratio = 1.0 / math.sqrt(1.0 + 1.0 / T)
        base = bmp_test(strong_signal, estimation_window=T)
        pe = bmp_test(
            strong_signal,
            estimation_window=T,
            include_prediction_error_variance=True,
        )
        both = bmp_test(
            strong_signal,
            estimation_window=T,
            include_prediction_error_variance=True,
            kolari_pynnonen_adjust=True,
        )
        # PE alone leaves z untouched (uniform SAR rescaling).
        assert pe.stat == pytest.approx(base.stat, rel=1e-6)
        # PE+KP composition: value tracks PE rescaling; z is the KP-shrunk
        # version of the (PE-invariant) BMP z, recoverable from
        # stat_uncorrected.
        assert both.metadata["include_prediction_error_variance"] is True
        assert both.value == pytest.approx(base.value * ratio, rel=1e-6)
        if both.metadata.get("kolari_pynnonen_applied"):
            assert abs(both.stat) <= abs(both.metadata["stat_uncorrected"]) + 1e-9
            assert both.metadata["stat_uncorrected"] == pytest.approx(
                base.stat, rel=1e-6
            )

    def test_kolari_pynnonen_skipped_without_clusters(self):
        """No multi-event dates → r̂ undefined → correction bypassed."""
        from datetime import datetime, timedelta

        rng = np.random.default_rng(0)
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(120)]
        rows = []
        # One event per date on a single asset: no same-date clustering.
        price = 100.0
        for i, d in enumerate(dates):
            ret = float(0.001 + 0.02 * rng.standard_normal())
            price *= 1.0 + ret
            rows.append(
                {
                    "date": d,
                    "asset_id": "A",
                    "factor": 1.0 if i >= 80 else 0.0,
                    "forward_return": ret,
                    "price": price,
                }
            )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = bmp_test(df, kolari_pynnonen_adjust=True)
        assert result.metadata["kolari_pynnonen_applied"] is False
        assert result.metadata["kolari_pynnonen_r_source"] == ("no_multi_event_dates")


# ---------------------------------------------------------------------------
# event_hit_rate
# ---------------------------------------------------------------------------


class TestEventHitRate:
    def test_strong_signal_high_hit_rate(self, strong_signal):
        result = event_hit_rate(strong_signal)
        assert result.value > 0.5
        assert abs(result.stat) > 2.0

    def test_noise_hit_rate_near_half(self, noise_signal):
        result = event_hit_rate(noise_signal)
        assert abs(result.value - 0.5) < 0.1

    def test_single_asset(self, single_asset_signal):
        result = event_hit_rate(single_asset_signal)
        assert result.metadata["n_events"] > 0
        assert 0.0 <= result.value <= 1.0

    def test_no_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [0.0],
                "forward_return": [0.01],
            }
        )
        result = event_hit_rate(df)
        assert math.isnan(result.value)


# ---------------------------------------------------------------------------
# event_ic
# ---------------------------------------------------------------------------


def _make_continuous_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic continuous event density: stronger |density| → larger return."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        for a in assets:
            is_event = rng.random() < event_prob
            if is_event:
                magnitude = rng.uniform(0.5, 5.0)
                direction = rng.choice([-1.0, 1.0])
                density = direction * magnitude
                # Stronger density → larger directional return
                ret = 0.005 * magnitude * direction + rng.normal(0, 0.02)
            else:
                density = 0.0
                ret = rng.normal(0, 0.02)

            rows.append(
                {
                    "date": d,
                    "asset_id": a,
                    "factor": density,
                    "forward_return": ret,
                }
            )

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


class TestEventIc:
    def test_continuous_signal_positive_ic(self):
        df = _make_continuous_signal()
        result = event_ic(df)
        assert result.value > 0
        assert (
            result.metadata["method"]
            == "Spearman rank correlation (|density| vs signed_car)"
        )

    def test_discrete_signal_skipped(self, strong_signal):
        """All ±1 values → |factor| constant → IC = 0 (no variance)."""
        result = event_ic(strong_signal)
        assert math.isnan(result.value)
        assert result.stat is None

    def test_insufficient_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [2.5],
                "forward_return": [0.01],
            }
        )
        result = event_ic(df)
        assert math.isnan(result.value)

    def test_standalone_import(self):
        from factrix.metrics import event_ic as eic

        assert callable(eic)
