"""Factor profile computation.

After all gates pass, ``compute_profile`` runs the full suite of
metrics to produce raw MetricOutput values — no scores, no mapping.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from factorlib.config import BaseConfig, CrossSectionalConfig, EventConfig, MacroCommonConfig, MacroPanelConfig
from factorlib.evaluation._protocol import Artifacts, FactorProfile
from factorlib._types import MetricOutput
from factorlib.metrics.ic import ic as ic_metric, ic_ir as ic_ir_metric
from factorlib.metrics.quantile import quantile_spread
from factorlib.metrics.monotonicity import monotonicity
from factorlib.metrics.concentration import q1_concentration
from factorlib.metrics.hit_rate import hit_rate
from factorlib.metrics.trend import ic_trend
from factorlib.metrics.oos import multi_split_oos_decay
from factorlib.metrics.tradability import turnover, breakeven_cost, net_spread


def compute_profile(artifacts: Artifacts) -> FactorProfile:
    """Compute the full factor profile from pre-built artifacts."""
    match artifacts.config:
        case CrossSectionalConfig():
            return _cs_profile(artifacts, artifacts.config)
        case EventConfig():
            return _event_signal_profile(artifacts, artifacts.config)
        case MacroPanelConfig():
            return _macro_panel_profile(artifacts, artifacts.config)
        case MacroCommonConfig():
            return _macro_common_profile(artifacts, artifacts.config)
        case _:
            ft = type(artifacts.config).factor_type
            raise NotImplementedError(
                f"compute_profile not yet implemented for {ft}"
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _oos_decay_metric(values_df: pl.DataFrame) -> MetricOutput:
    oos = multi_split_oos_decay(values_df)
    return MetricOutput(
        name="oos_decay",
        value=oos.decay_ratio,
        metadata={
            "sign_flipped": oos.sign_flipped,
            "status": oos.status,
            "per_split": [
                {"is_ratio": s.is_ratio, "mean_is": s.mean_is,
                 "mean_oos": s.mean_oos, "decay_ratio": s.decay_ratio}
                for s in oos.per_split
            ],
        },
    )


def _beta_trend_metric(values_df: pl.DataFrame) -> MetricOutput:
    trn = ic_trend(values_df)
    return MetricOutput(
        name="beta_trend", value=trn.value,
        stat=trn.stat, significance=trn.significance,
        metadata=trn.metadata,
    )


# ---------------------------------------------------------------------------
# Event signal profile
# ---------------------------------------------------------------------------

def _event_signal_profile(
    artifacts: Artifacts, config: EventConfig,
) -> FactorProfile:
    # WHY: lazy import — avoid loading event modules when only CS is used
    from factorlib.metrics.caar import caar as caar_metric, bmp_test, event_hit_rate, event_ic
    from factorlib.metrics.mfe_mae import mfe_mae_summary, profit_factor, event_skewness
    from factorlib.metrics.clustering import clustering_diagnostic

    caar_series = artifacts.get("caar_series")
    caar_values = artifacts.get("caar_values")

    ret_col = (
        "abnormal_return" if "abnormal_return" in artifacts.prepared.columns
        else "forward_return"
    )

    caar_m = caar_metric(caar_series, forward_periods=config.forward_periods)
    bmp_m = bmp_test(
        artifacts.prepared, return_col=ret_col,
        forward_periods=config.forward_periods,
    )
    hit_m = event_hit_rate(artifacts.prepared, return_col=ret_col)
    oos = _oos_decay_metric(caar_values)
    caar_trn = replace(_beta_trend_metric(caar_values), name="caar_trend")

    pf = profit_factor(artifacts.prepared, return_col=ret_col)
    skew = event_skewness(artifacts.prepared, return_col=ret_col)

    metrics: list[MetricOutput] = [caar_m, bmp_m, hit_m, oos, caar_trn, pf, skew]

    if "mfe_mae" in artifacts.intermediates:
        mfe_summary = mfe_mae_summary(artifacts.get("mfe_mae"))
        if mfe_summary is not None:
            metrics.append(mfe_summary)

    # Signal magnitude IC: only when signal values have magnitude variance
    events = artifacts.prepared.filter(pl.col("factor") != 0)
    if events["factor"].abs().n_unique() > 1:
        metrics.append(event_ic(artifacts.prepared, return_col=ret_col))

    n_assets = artifacts.prepared["asset_id"].n_unique()
    if n_assets > 1:
        clust = clustering_diagnostic(
            artifacts.prepared, cluster_window=config.cluster_window,
        )
        metrics.append(clust)

    return FactorProfile(metrics=metrics)


# ---------------------------------------------------------------------------
# Cross-sectional profile
# ---------------------------------------------------------------------------

def _cs_profile(
    artifacts: Artifacts, config: CrossSectionalConfig,
) -> FactorProfile:
    fp = config.forward_periods
    ic_series = artifacts.get("ic_series")
    ic_values = artifacts.get("ic_values")
    spread_series = artifacts.get("spread_series")

    ic_sig = ic_metric(ic_series, forward_periods=fp)
    ic_ir = ic_ir_metric(ic_series)
    hit = hit_rate(ic_values, forward_periods=fp)
    ic_trn = ic_trend(ic_values)
    mono = monotonicity(
        artifacts.prepared, forward_periods=fp, n_groups=config.n_groups,
    )
    oos = _oos_decay_metric(ic_values)

    spread = quantile_spread(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=config.n_groups,
        _precomputed_series=spread_series,
    )
    turn = turnover(artifacts.prepared)
    be = breakeven_cost(spread.value, turn.value)
    ns = net_spread(
        spread.value, turn.value, config.estimated_cost_bps,
    )
    conc = q1_concentration(
        artifacts.prepared, forward_periods=fp, q_top=config.q_top,
    )

    return FactorProfile(
        metrics=[ic_sig, ic_ir, hit, ic_trn, mono, oos,
                 spread, turn, be, ns, conc],
    )


# ---------------------------------------------------------------------------
# Macro panel profile
# ---------------------------------------------------------------------------

def _macro_panel_profile(
    artifacts: Artifacts, config: MacroPanelConfig,
) -> FactorProfile:
    # WHY: lazy import — avoid loading FM/TS modules when only CS is used
    from factorlib.metrics.fama_macbeth import (
        fama_macbeth, pooled_ols, beta_sign_consistency,
    )

    fp = config.forward_periods
    beta_series = artifacts.get("beta_series")
    beta_values = artifacts.get("beta_values")
    spread_series = artifacts.get("spread_series")

    fm = fama_macbeth(beta_series)
    pooled = pooled_ols(artifacts.prepared)
    sign_cons = beta_sign_consistency(beta_series)
    oos = _oos_decay_metric(beta_values)
    beta_trn = _beta_trend_metric(beta_values)
    # WHY: uses simple t-test (not NW) — non-overlapping sampling already
    # eliminates return overlap autocorrelation; NW is reserved for FM β
    # where the β series itself has time-series persistence.
    spread = quantile_spread(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=config.n_groups,
        _precomputed_series=spread_series,
    )
    turn = turnover(artifacts.prepared)
    be = breakeven_cost(spread.value, turn.value)
    ns = net_spread(spread.value, turn.value, config.estimated_cost_bps)

    return FactorProfile(
        metrics=[fm, pooled, sign_cons, oos, beta_trn, spread, turn, be, ns],
    )


# ---------------------------------------------------------------------------
# Macro common profile
# ---------------------------------------------------------------------------

def _macro_common_profile(
    artifacts: Artifacts, config: MacroCommonConfig,
) -> FactorProfile:
    # WHY: lazy import — avoid loading FM/TS modules when only CS is used
    from factorlib.metrics.ts_beta import (
        ts_beta, mean_r_squared, ts_beta_sign_consistency,
    )

    ts_betas_df = artifacts.get("beta_series")
    beta_values = artifacts.get("beta_values")
    n_assets = len(ts_betas_df)

    # N-awareness: cross-sectional t-test degenerates at N=1
    # (can't compute std from 1 observation). Report per-asset stats directly.
    if n_assets == 1:
        beta_metric = MetricOutput(
            name="ts_beta",
            value=float(ts_betas_df["beta"][0]),
            stat=float(ts_betas_df["t_stat"][0]),
            metadata={
                "n_assets": 1,
                "method": "single-asset TS regression (no cross-asset aggregation)",
            },
        )
    else:
        beta_metric = ts_beta(ts_betas_df)

    r2 = mean_r_squared(ts_betas_df)
    sign_cons = ts_beta_sign_consistency(ts_betas_df)
    oos = _oos_decay_metric(beta_values)
    beta_trn = _beta_trend_metric(beta_values)

    return FactorProfile(
        metrics=[beta_metric, r2, sign_cons, oos, beta_trn],
    )
