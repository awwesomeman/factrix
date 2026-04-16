"""Factor profile computation.

After all gates pass, ``compute_profile`` runs the full suite of
metrics to produce raw MetricOutput values — no scores, no mapping.
"""

from __future__ import annotations

import polars as pl

from factorlib.config import BaseConfig, CrossSectionalConfig, MacroCommonConfig, MacroPanelConfig
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
    from factorlib.metrics.fama_macbeth import (
        fama_macbeth, pooled_ols, beta_sign_consistency, long_short_tercile,
    )

    beta_series = artifacts.get("beta_series")
    beta_values = artifacts.get("beta_values")
    tercile_series = artifacts.get("tercile_series")

    fm = fama_macbeth(beta_series)
    pooled = pooled_ols(artifacts.prepared)
    sign_cons = beta_sign_consistency(beta_series)
    oos = _oos_decay_metric(beta_values)
    beta_trn = _beta_trend_metric(beta_values)
    ls = long_short_tercile(tercile_series)
    turn = turnover(artifacts.prepared)

    return FactorProfile(
        metrics=[fm, pooled, sign_cons, oos, beta_trn, ls, turn],
    )


# ---------------------------------------------------------------------------
# Macro common profile
# ---------------------------------------------------------------------------

def _macro_common_profile(
    artifacts: Artifacts, config: MacroCommonConfig,
) -> FactorProfile:
    from factorlib.metrics.ts_beta import (
        ts_beta, mean_r_squared, ts_beta_sign_consistency,
    )

    ts_betas_df = artifacts.get("ts_betas")
    beta_values = artifacts.get("beta_values")

    beta_metric = ts_beta(ts_betas_df)
    r2 = mean_r_squared(ts_betas_df)
    sign_cons = ts_beta_sign_consistency(ts_betas_df)
    oos = _oos_decay_metric(beta_values)
    beta_trn = _beta_trend_metric(beta_values)

    return FactorProfile(
        metrics=[beta_metric, r2, sign_cons, oos, beta_trn],
    )
