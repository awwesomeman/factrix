"""``expected_warnings`` covers every metric-layer ``UserWarning`` echo.

The sibling ``test_expected_warnings`` module pins the contract on the
soft-floor helpers (``few_assets`` and the drop-rate codes). This module
pins the same contract on the remaining metric-layer echoes — the ones
that used to call ``warnings.warn`` without reading the declaration, so a
declared sweep still had its stderr flooded.

One pair of cases per site: a declared run under
``simplefilter("error", UserWarning)`` (the echo must not fire, the
structured code must still be on ``MetricResult.warning_codes``, and its
:class:`~factrix.Warning` record must be marked ``expected=True``), and an
undeclared run that must still echo.
"""

from __future__ import annotations

import datetime as dt
import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics import (
    bmp_z,
    directional_pair_accuracy,
    fm_beta,
    ic,
    ic_ir,
    pooled_beta,
    quantile_spread,
    quantile_spread_vw,
)
from factrix.metrics.concentration import top_concentration

from tests._slice_panel import build_autocorrelated_ic_panel

# ---------------------------------------------------------------------------
# Panels — one per triggered code, kept minimal and deterministic.
# ---------------------------------------------------------------------------


def _persistent_beta_panel() -> pl.DataFrame:
    """Per-period FM beta series with lag-1 autocorrelation above the screen."""
    return build_autocorrelated_ic_panel(
        n_dates=240, seed=0, signal={"x": 0.0}, label_col="lbl", phi=0.9
    )


def _short_panel() -> pl.DataFrame:
    """Too few periods for a Driscoll-Kraay cross-sectional HAC."""
    raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=25, seed=3)
    return fx.preprocess.compute_forward_return(raw, forward_periods=1)


def _tied_panel() -> pl.DataFrame:
    """Low-cardinality factor: the per-period IC rests mostly on mid-ranks."""
    raw = fx.datasets.make_cs_panel(n_assets=30, n_dates=200, seed=5)
    tied = raw.with_columns(
        (pl.col("factor").rank("dense") % 3).cast(pl.Float64).alias("factor")
    )
    return fx.preprocess.compute_forward_return(tied, forward_periods=1)


def _thin_bucket_panel() -> pl.DataFrame:
    """Six assets cut into three buckets — two names per bucket."""
    raw = fx.datasets.make_cs_panel(n_assets=6, n_dates=220, seed=11)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
    return panel.with_columns(pl.lit(1.0).alias("market_cap"))


def _one_signed_panel() -> pl.DataFrame:
    """A factor that never crosses zero, so |factor| is not a density weight."""
    raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80, seed=7)
    pos = raw.with_columns((pl.col("factor").abs() + 10.0).alias("factor"))
    return fx.preprocess.compute_forward_return(pos, forward_periods=1)


def _few_pairs_panel() -> pl.DataFrame:
    """Three assets over four periods — twelve comparable ordering pairs."""
    rows = []
    for d in range(4):
        for a, (f, r) in enumerate([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]):
            rows.append(
                {
                    "date": dt.datetime(2024, 1, 1) + dt.timedelta(days=d),
                    "asset_id": f"A{a}",
                    "factor": f + 0.1 * d,
                    "forward_return": r * (1 if d % 2 else -1),
                }
            )
    return pl.DataFrame(rows)


def _no_price_event_panel() -> pl.DataFrame:
    """Sparse event panel with no ``price`` column for the BMP standardiser."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for d in range(400):
        for a in range(12):
            rows.append(
                {
                    "date": dt.datetime(2024, 1, 1) + dt.timedelta(days=d),
                    "asset_id": f"A{a}",
                    "factor": 1.0 if (d % 40 == 0 and a % 2 == 0) else 0.0,
                    "forward_return": float(rng.normal(0, 0.02)),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Contract harness
# ---------------------------------------------------------------------------


def _records(result, code: WarningCode) -> list:
    return [w for w in result.warnings if w.code is code]


def _assert_declared_is_quiet_but_recorded(
    panel: pl.DataFrame,
    metrics: dict,
    code: WarningCode,
    metric_key: str,
    *,
    also_declare: tuple[str, ...] = (),
    **evaluate_kwargs,
) -> None:
    """Declared: no echo, code still attached, record marked expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = fx.evaluate(
            panel,
            metrics=metrics,
            factor_cols=["factor"],
            expected_warnings=(code.value, *also_declare),
            **evaluate_kwargs,
        )
    res = results["factor"]
    assert code.value in res.metrics[metric_key].warning_codes
    records = _records(res, code)
    assert records, "the structured record must never be dropped"
    assert all(w.expected for w in records)
    assert not any(w.code is code for w in res.unexpected_warnings)


def _assert_undeclared_still_echoes(
    panel: pl.DataFrame,
    metrics: dict,
    match: str,
    **evaluate_kwargs,
) -> None:
    """Undeclared: behaviour is bit-for-bit what it always was."""
    with pytest.warns(UserWarning, match=match):
        fx.evaluate(panel, metrics=metrics, factor_cols=["factor"], **evaluate_kwargs)


class TestFmBetaSerialCorrelation:
    """``fm_beta`` — SERIAL_CORRELATION_DETECTED on a persistent beta series."""

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _persistent_beta_panel(),
            {"fm": fm_beta()},
            WarningCode.SERIAL_CORRELATION_DETECTED,
            "fm",
            forward_periods=1,
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _persistent_beta_panel(),
            {"fm": fm_beta()},
            "lag-1 autocorrelation",
            forward_periods=1,
        )


class TestPooledBetaShortPeriods:
    """``pooled_beta`` DK path — UNRELIABLE_SE_SHORT_PERIODS."""

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _short_panel(),
            {"pb": pooled_beta(driscoll_kraay=True)},
            WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
            "pb",
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _short_panel(),
            {"pb": pooled_beta(driscoll_kraay=True)},
            "Driscoll-Kraay SE on n_periods",
        )


class TestBmpReturnVolFallback:
    """``bmp_z`` — BMP_RETURN_VOL_FALLBACK when no ``price`` column exists."""

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _no_price_event_panel(),
            {"bmp": bmp_z()},
            WarningCode.BMP_RETURN_VOL_FALLBACK,
            "bmp",
            # The same short event sample also trips FEW_EVENTS; declare it so
            # the error filter isolates the echo under test.
            also_declare=(WarningCode.FEW_EVENTS.value,),
            forward_periods=5,
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _no_price_event_panel(),
            {"bmp": bmp_z()},
            "no 'price' column",
            forward_periods=5,
        )


class TestTopConcentrationOneSignedFactor:
    """``top_concentration`` — ONE_SIGNED_FACTOR under ``weight_by='abs_factor'``."""

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _one_signed_panel(),
            {"tc": top_concentration(weight_by="abs_factor")},
            WarningCode.ONE_SIGNED_FACTOR,
            "tc",
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _one_signed_panel(),
            {"tc": top_concentration(weight_by="abs_factor")},
            "never changes sign",
        )


class TestDirectionalPairAccuracyFewPairs:
    """``directional_pair_accuracy`` — FEW_ORDERING_PAIRS on a thin pair sample."""

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _few_pairs_panel(),
            {"dpa": directional_pair_accuracy()},
            WarningCode.FEW_ORDERING_PAIRS,
            "dpa",
            forward_periods=1,
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _few_pairs_panel(),
            {"dpa": directional_pair_accuracy()},
            "MIN_PAIR_ACCURACY_PAIRS_WARN",
            forward_periods=1,
        )


class TestThinQuantileGroups:
    """``quantile_spread`` / ``quantile_spread_vw`` — THIN_QUANTILE_GROUPS.

    The echo lives on the shared bucketing helper, reached once through the
    spread primitive and once through the value-weighted metric's inline
    bucketing, so both consumers are pinned here.
    """

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _thin_bucket_panel(),
            {"qs": quantile_spread(n_groups=3), "vw": quantile_spread_vw(n_groups=3)},
            WarningCode.THIN_QUANTILE_GROUPS,
            "qs",
            # A six-asset panel is a few-asset panel too; declare it so the
            # error filter isolates the echo under test.
            also_declare=(WarningCode.FEW_ASSETS.value,),
        )

    def test_declared_is_quiet_but_recorded_on_the_value_weighted_path(self):
        _assert_declared_is_quiet_but_recorded(
            _thin_bucket_panel(),
            {"vw": quantile_spread_vw(n_groups=3)},
            WarningCode.THIN_QUANTILE_GROUPS,
            "vw",
            also_declare=(WarningCode.FEW_ASSETS.value,),
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _thin_bucket_panel(),
            {"qs": quantile_spread(n_groups=3)},
            "assets per group",
        )

    def test_undeclared_still_echoes_on_the_value_weighted_path(self):
        _assert_undeclared_still_echoes(
            _thin_bucket_panel(),
            {"vw": quantile_spread_vw(n_groups=3)},
            "assets per group",
        )


class TestHighTieRatio:
    """``ic`` / ``ic_ir`` — the new HIGH_TIE_RATIO code.

    The tie-ratio advisory had no structured twin at all, so a declared
    sweep had no way to quiet it and result-only inspection never saw it.
    """

    def test_declared_is_quiet_but_recorded(self):
        _assert_declared_is_quiet_but_recorded(
            _tied_panel(),
            {"ic": ic()},
            WarningCode.HIGH_TIE_RATIO,
            "ic",
        )

    def test_declared_is_quiet_but_recorded_on_ic_ir(self):
        _assert_declared_is_quiet_but_recorded(
            _tied_panel(),
            {"icir": ic_ir()},
            WarningCode.HIGH_TIE_RATIO,
            "icir",
        )

    def test_undeclared_still_echoes(self):
        _assert_undeclared_still_echoes(
            _tied_panel(),
            {"ic": ic()},
            "median tie_ratio",
        )

    def test_code_is_attached_to_the_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = fx.evaluate(
                _tied_panel(),
                metrics={"ic": ic(), "icir": ic_ir()},
                factor_cols=["factor"],
            )
        res = results["factor"]
        assert WarningCode.HIGH_TIE_RATIO.value in res.metrics["ic"].warning_codes
        assert WarningCode.HIGH_TIE_RATIO.value in res.metrics["icir"].warning_codes
        assert all(not w.expected for w in _records(res, WarningCode.HIGH_TIE_RATIO))

    def test_a_continuous_factor_does_not_trip_it(self):
        raw = fx.datasets.make_cs_panel(n_assets=30, n_dates=200, seed=5)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=["factor"])
        codes = results["factor"].metrics["ic"].warning_codes
        assert WarningCode.HIGH_TIE_RATIO.value not in codes
