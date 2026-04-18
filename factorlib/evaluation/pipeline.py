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

    ic_series = compute_ic(df)
    ic_values = ic_series.rename({"ic": "value"})
    spread_series = compute_spread_series(
        df, config.forward_periods, config.n_groups,
    )
    return Artifacts(
        prepared=df,
        config=config,
        intermediates={
            "ic_series": ic_series,
            "ic_values": ic_values,
            "spread_series": spread_series,
        },
    )


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
