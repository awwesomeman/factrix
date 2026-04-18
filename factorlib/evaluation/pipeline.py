"""Artifact construction for the profile-era evaluation flow.

``build_artifacts`` pre-computes the per-type intermediates shared by
every Profile classmethod. The top-level orchestration lives in
``factorlib._api.evaluate`` / ``evaluate_batch``; this module is the
lower-level plumbing.
"""

from __future__ import annotations

import polars as pl

from factorlib.config import (
    BaseConfig,
    CrossSectionalConfig,
    EventConfig,
    MacroCommonConfig,
    MacroPanelConfig,
)
from factorlib.evaluation._protocol import Artifacts
from factorlib.metrics.ic import compute_ic
from factorlib.metrics.quantile import compute_spread_series


def build_artifacts(df: pl.DataFrame, config: BaseConfig) -> Artifacts:
    """Pre-compute shared intermediate results."""
    match config:
        case CrossSectionalConfig():
            return _build_cs_artifacts(df, config)
        case EventConfig():
            return _build_event_artifacts(df, config)
        case MacroPanelConfig():
            return _build_macro_panel_artifacts(df, config)
        case MacroCommonConfig():
            return _build_macro_common_artifacts(df, config)
        case _:
            ft = type(config).factor_type
            raise NotImplementedError(
                f"build_artifacts not yet implemented for {ft}"
            )


_REQUIRED_COLUMNS = {"date", "asset_id", "factor", "forward_return"}

_SCHEMA_HINT = (
    "Expected DataFrame schema:\n"
    "  date           Datetime[ms]   — 交易日期\n"
    "  asset_id       String         — 資產代碼\n"
    "  factor         Float64        — 因子值\n"
    "  forward_return Float64        — N 期前瞻報酬"
)


def _validate_columns(df: pl.DataFrame, factor_type: str) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{factor_type} requires columns {_REQUIRED_COLUMNS}. "
            f"Missing: {missing}.\n\n{_SCHEMA_HINT}\n\n"
            f"Hint: call fl.preprocess(df) first, or pass preprocess=True "
            f"to fl.evaluate()."
        )


def _build_event_artifacts(
    df: pl.DataFrame, config: EventConfig,
) -> Artifacts:
    """Build event signal artifacts: CAAR series and optional MFE/MAE."""
    _validate_columns(df, "event_signal")

    from factorlib.metrics.caar import compute_caar
    from factorlib.metrics.mfe_mae import compute_mfe_mae

    # WHY: use abnormal_return (market-adjusted) when available;
    # fall back to forward_return for non-preprocessed data.
    ret_col = (
        "abnormal_return" if "abnormal_return" in df.columns
        else "forward_return"
    )

    caar_series = compute_caar(df, return_col=ret_col)
    caar_values = caar_series.rename({"caar": "value"})

    mfe_mae_df = compute_mfe_mae(df, window=config.event_window_post)

    intermediates: dict[str, pl.DataFrame] = {
        "caar_series": caar_series,
        "caar_values": caar_values,
    }
    if not mfe_mae_df.is_empty():
        intermediates["mfe_mae"] = mfe_mae_df

    return Artifacts(
        prepared=df,
        config=config,
        intermediates=intermediates,
    )


def _build_cs_artifacts(
    df: pl.DataFrame, config: CrossSectionalConfig,
) -> Artifacts:
    _validate_columns(df, "cross_sectional")

    ortho_info: pl.DataFrame | None = None
    if config.orthogonalize is not None:
        df, ortho_info = _apply_orthogonalize(df, config)

    ic_series = compute_ic(df)
    ic_values = ic_series.rename({"ic": "value"})
    spread_series = compute_spread_series(
        df, config.forward_periods, config.n_groups,
    )
    intermediates: dict[str, pl.DataFrame] = {
        "ic_series": ic_series,
        "ic_values": ic_values,
        "spread_series": spread_series,
    }
    if ortho_info is not None:
        intermediates["ortho_stats"] = ortho_info
    return Artifacts(
        prepared=df,
        config=config,
        intermediates=intermediates,
    )


def _apply_orthogonalize(
    df: pl.DataFrame, config: CrossSectionalConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace df['factor'] with the per-date residual against config.orthogonalize.

    Residuals are MAD-winsorized then re-z-scored to preserve the same
    Step 4-5 invariant the rest of the pipeline enforces ("factor is
    MAD-clipped and unit-variance per-date") — OLS residuals can be
    fat-tailed, so skipping winsorize would let outliers drag IC.

    Coverage below ``config.orthogonalize_min_coverage`` fails loud —
    half-orthogonalized data was the silent-bug path we refactored
    away in Phase 1 T1.1. The standalone helper's own ``factor_pre_ortho``
    column is dropped here; pipeline callers get a clean schema, and
    one-off comparisons should use ``orthogonalize_factor`` directly.

    Returns ``(df_with_residualized_factor, stats_df)`` where stats_df
    is a 1-row DataFrame carrying ``r2_mean`` / ``n_base`` / ``coverage``
    for the Profile layer to surface.
    """
    from factorlib.preprocess.normalize import (
        cross_sectional_zscore,
        mad_winsorize,
    )
    from factorlib.preprocess.orthogonalize import orthogonalize_factor

    result = orthogonalize_factor(
        df,
        config.orthogonalize,
        factor_col="factor",
        base_cols=config.orthogonalize_cols,
    )

    if result.coverage < config.orthogonalize_min_coverage:
        raise ValueError(
            f"orthogonalize: coverage {result.coverage:.3f} is below "
            f"orthogonalize_min_coverage={config.orthogonalize_min_coverage:.3f}. "
            f"Either lower the threshold explicitly (accepting partial "
            f"residualization), extend base_factors to cover more rows, or "
            f"trim the factor panel to rows with base-factor coverage."
        )

    residualized = result.df.drop("factor_pre_ortho")
    residualized = mad_winsorize(
        residualized, factor_col="factor", n_mad=config.mad_n,
    )
    residualized = cross_sectional_zscore(residualized, factor_col="factor")
    residualized = residualized.with_columns(
        pl.col("factor_zscore").alias("factor")
    ).drop("factor_zscore")

    info_df = pl.DataFrame({
        "r2_mean": [float(result.mean_r_squared)],
        "n_base": [int(result.n_base)],
        "coverage": [float(result.coverage)],
    })
    return residualized, info_df


def _build_macro_panel_artifacts(
    df: pl.DataFrame, config: MacroPanelConfig,
) -> Artifacts:
    _validate_columns(df, "macro_panel")

    from factorlib.metrics.fama_macbeth import compute_fm_betas

    beta_series = compute_fm_betas(df)
    beta_values = beta_series.rename({"beta": "value"})
    spread_series = compute_spread_series(
        df, config.forward_periods, config.n_groups,
    )

    return Artifacts(
        prepared=df,
        config=config,
        intermediates={
            "beta_series": beta_series,
            "beta_values": beta_values,
            "spread_series": spread_series,
        },
    )


def _build_macro_common_artifacts(
    df: pl.DataFrame, config: MacroCommonConfig,
) -> Artifacts:
    _validate_columns(df, "macro_common")

    from factorlib.metrics.ts_beta import compute_ts_betas, compute_rolling_mean_beta

    ts_betas = compute_ts_betas(df)
    rolling = compute_rolling_mean_beta(df, window=config.ts_window)

    return Artifacts(
        prepared=df,
        config=config,
        intermediates={
            "beta_series": ts_betas,
            "beta_values": rolling,
        },
    )
