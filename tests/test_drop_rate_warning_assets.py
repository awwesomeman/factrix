"""Aggregate drop-rate warning — assets axis (``compute_ts_betas``).

``compute_ts_betas`` silently drops assets at its per-asset filter (insufficient
history, zero factor variation, or no complete pairs). It records the canonical
drop-stat schema on the assets axis (``n_assets_*`` / ``dropped_assets``); its
three cross-asset consumers (``ts_beta`` / ``mean_r_squared`` /
``ts_beta_sign_consistency``) copy the schema into ``MetricResult.metadata`` and
emit one aggregate ``UserWarning`` (plus a ``WarningCode.EXCESSIVE_ASSET_DROPS``)
when ``drop_rate`` clears the threshold. This is the assets-axis sibling of the
periods-axis path in ``test_drop_rate_warning.py``.
"""

from __future__ import annotations

import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics._helpers import (
    DROP_RATE_WARN_THRESHOLD,
    _drop_stat_keys,
    _read_drop_stats,
)
from factrix.metrics.ts_beta import (
    compute_ts_betas,
    mean_r_squared,
    ts_beta,
    ts_beta_sign_consistency,
)

_ASSET_KEYS = _drop_stat_keys("assets")
_CONSUMERS = (ts_beta, mean_r_squared, ts_beta_sign_consistency)


def _sparse_factor_panel(
    *, n_assets: int = 40, survivors: int = 6, n_dates: int = 50, seed: int = 0
) -> pl.DataFrame:
    """Panel where only ``survivors`` assets carry a time-varying factor.

    The remaining assets get a constant-zero factor → zero time-variation → null
    slope → dropped by ``compute_ts_betas``. ``survivors`` clears the consumers'
    ``min_assets`` floors while the drop rate (``1 - survivors/n_assets``) sits
    well above ``DROP_RATE_WARN_THRESHOLD``.
    """
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=seed)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=1)
    keep = panel["asset_id"].unique().sort().gather(range(survivors))
    return panel.with_columns(
        pl.when(pl.col("asset_id").is_in(keep.implode()))
        .then(pl.col("factor"))
        .otherwise(0.0)
        .alias("factor")
    )


def _full_panel(
    *, n_assets: int = 40, n_dates: int = 50, seed: int = 0
) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=seed)
    return fx.preprocess.compute_forward_return(raw, forward_periods=1)


class TestSchema:
    def test_keys_are_canonical(self):
        assert _ASSET_KEYS == (
            "n_assets_in",
            "n_assets_out",
            "dropped_assets",
            "drop_rate",
            "drop_reason",
        )

    def test_compute_ts_betas_attaches_drop_stats(self):
        stats = _read_drop_stats(compute_ts_betas(_sparse_factor_panel())["factor"])
        assert stats is not None
        assert set(stats) == set(_ASSET_KEYS)
        # 40 assets in, 6 survive → 34 dropped for zero factor variation.
        assert stats["n_assets_in"] == 40
        assert stats["n_assets_out"] == 6
        assert stats["dropped_assets"] == 34
        assert stats["drop_rate"] == pytest.approx(34 / 40)
        assert "MIN_TS_PERIODS" in stats["drop_reason"]

    def test_full_panel_reports_zero_drop(self):
        stats = _read_drop_stats(compute_ts_betas(_full_panel())["factor"])
        assert stats is not None
        assert stats["dropped_assets"] == 0
        assert stats["drop_rate"] == 0.0

    def test_hand_built_frame_has_no_stats(self):
        bare = compute_ts_betas(_full_panel())["factor"].select("asset_id", "beta")
        assert _read_drop_stats(bare) is None


class TestConsumerWarning:
    @pytest.mark.parametrize("metric_fn", _CONSUMERS)
    def test_high_drop_rate_warns_and_codes(self, metric_fn):
        betas_df = compute_ts_betas(_sparse_factor_panel())["factor"]
        with pytest.warns(UserWarning, match="of assets dropped"):
            result = metric_fn(betas_df)
        assert WarningCode.EXCESSIVE_ASSET_DROPS.value in result.warning_codes
        assert result.metadata["drop_rate"] > DROP_RATE_WARN_THRESHOLD
        # Full five-key assets schema lands in metadata for inspection.
        assert set(_ASSET_KEYS) <= set(result.metadata)

    @pytest.mark.parametrize("metric_fn", _CONSUMERS)
    def test_low_drop_rate_no_warn_but_keys_present(self, metric_fn):
        betas_df = compute_ts_betas(_full_panel())["factor"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = metric_fn(betas_df)
        assert WarningCode.EXCESSIVE_ASSET_DROPS.value not in result.warning_codes
        assert result.metadata["drop_rate"] == 0.0
        assert set(_ASSET_KEYS) <= set(result.metadata)

    def test_full_drop_defers_to_short_circuit(self):
        # Every asset's factor is zeroed → all slopes null → empty betas frame →
        # the consumer short-circuits and must NOT emit a drop-rate warning.
        panel = _sparse_factor_panel(survivors=0)
        betas_df = compute_ts_betas(panel)["factor"]
        assert betas_df.height == 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = ts_beta(betas_df)
        assert result.value != result.value  # NaN short-circuit
        assert "reason" in result.metadata
        assert WarningCode.EXCESSIVE_ASSET_DROPS.value not in result.warning_codes


class TestEvaluateBoundary:
    def test_evaluate_records_drop_warning(self):
        panel = _sparse_factor_panel()
        with pytest.warns(UserWarning, match="of assets dropped"):
            results = fx.evaluate(
                panel, metrics={"m": ts_beta()}, factor_cols=["factor"]
            )
        er = results["factor"]
        codes = [w.code for w in er.warnings]
        assert WarningCode.EXCESSIVE_ASSET_DROPS in codes
        drop_warn = next(
            w for w in er.warnings if w.code == WarningCode.EXCESSIVE_ASSET_DROPS
        )
        assert drop_warn.source == "m"

    def test_direct_and_evaluate_metadata_parity(self):
        panel = _sparse_factor_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = ts_beta(compute_ts_betas(panel)["factor"])
            results = fx.evaluate(
                panel, metrics={"m": ts_beta()}, factor_cols=["factor"]
            )
        via_eval = results["factor"].metrics["m"]
        for key in _ASSET_KEYS:
            assert direct.metadata[key] == via_eval.metadata[key]
