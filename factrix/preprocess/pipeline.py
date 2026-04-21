"""Preprocessing orchestration.

Each step is independently importable from ``returns`` and ``normalize``.

Expects canonical column names (date, asset_id, price, factor).
Use ``adapt()`` to rename before calling.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

from factrix.preprocess.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)
from factrix.preprocess.normalize import (
    cross_sectional_zscore,
    mad_winsorize,
)

from factrix._types import UNSET

if TYPE_CHECKING:
    from factrix.config import BaseConfig


# Marker column stamped onto prepared panels so evaluate / factor can
# detect silent preprocess-time config drift. Captures factor_type plus
# all preprocess-time fields actually baked into prepared for that type.
#
# Deliberately excluded: ``ortho`` (CS, applied at build_artifacts time)
# and evaluate-time knobs (``n_groups``, ``tie_policy``, ``regime_labels``,
# ``multi_horizon_periods``, ``spanning_base_spreads``). Excluding these
# keeps the sweep pattern cheap — a single ``prepared`` can drive many
# evaluate calls with different downstream knobs. See
# ``docs/plan_direction.md`` for the trade-off rationale on ortho.
PREPROCESS_SIG_MARKER = "_fl_preprocess_sig"


def preprocess_sig(cfg: BaseConfig) -> dict[str, Any]:
    """Canonical form of the preprocess-time fields consumed by ``cfg``.

    Always includes ``factor_type`` so a prepared panel pinned to one
    factor type cannot be silently re-evaluated under a different type's
    config (which would pass column-presence checks but compute the
    wrong canonical test). JSON-friendly (lists for tuples, no
    polars/numpy types) so it can be embedded verbatim via ``pl.lit``
    and diffed at the evaluate gate.
    """
    from factrix.config import (
        CrossSectionalConfig,
        EventConfig,
        MacroCommonConfig,
        MacroPanelConfig,
    )

    sig: dict[str, Any] = {
        "factor_type": type(cfg).factor_type.value,
        "forward_periods": int(cfg.forward_periods),
    }
    match cfg:
        case CrossSectionalConfig():
            sig["mad_n"] = float(cfg.mad_n)
            sig["return_clip_pct"] = [
                float(cfg.return_clip_pct[0]),
                float(cfg.return_clip_pct[1]),
            ]
        case MacroPanelConfig():
            sig["demean_cross_section"] = bool(cfg.demean_cross_section)
        case EventConfig() | MacroCommonConfig():
            pass
        case _:
            ft = getattr(type(cfg), "factor_type", type(cfg).__name__)
            raise NotImplementedError(f"preprocess_sig not yet implemented for {ft}")
    return sig


def _sig_marker_expr(cfg: BaseConfig) -> pl.Expr:
    # pl.Categorical dedupes the single literal into one dict entry — keeps
    # per-row cost at ~4 bytes (index), matching the prior Int32 marker. A
    # plain pl.Utf8 literal stores the JSON string verbatim per row, which
    # is ~17× larger on a wide prepared panel.
    return (
        pl.lit(json.dumps(preprocess_sig(cfg), sort_keys=True))
        .cast(pl.Categorical)
        .alias(PREPROCESS_SIG_MARKER)
    )


def preprocess(
    df: pl.DataFrame,
    *,
    config: BaseConfig | None = None,
) -> pl.DataFrame:
    """Preprocess factor data based on config type.

    Dispatches to the appropriate type-specific preprocessor. ``config``
    is required — silently defaulting to ``CrossSectionalConfig`` would
    let event / macro panels be CS-preprocessed without any downstream
    marker to detect the mismatch.
    """
    from factrix.config import (
        CrossSectionalConfig, EventConfig, MacroCommonConfig, MacroPanelConfig,
    )

    if config is None:
        raise TypeError(
            "fl.preprocess requires an explicit config= argument. Pick one of:\n"
            "    fl.CrossSectionalConfig(forward_periods=5)   # 連續因子，寬截面\n"
            "    fl.EventConfig(forward_periods=5)            # 稀疏事件\n"
            "    fl.MacroPanelConfig(forward_periods=5)       # 連續因子，小 panel\n"
            "    fl.MacroCommonConfig(forward_periods=5)      # 共用時序\n"
            "then pass the SAME cfg to fl.evaluate / fl.factor so the "
            "preprocess-sig gate can catch config drift."
        )

    match config:
        case CrossSectionalConfig():
            return preprocess_cs_factor(df, config=config)
        case EventConfig():
            return preprocess_event_signal(df, config=config)
        case MacroPanelConfig():
            return preprocess_macro_panel(df, config=config)
        case MacroCommonConfig():
            return preprocess_macro_common(df, config=config)
        case _:
            ft = type(config).factor_type
            raise NotImplementedError(
                f"preprocess not yet implemented for {ft}"
            )


def preprocess_cs_factor(
    df: pl.DataFrame,
    *,
    config: CrossSectionalConfig | None = None,
    forward_periods: int = UNSET,  # type: ignore[assignment]
    return_clip_pct: tuple[float, float] = UNSET,  # type: ignore[assignment]
    mad_n: float = UNSET,  # type: ignore[assignment]
) -> pl.DataFrame:
    """Run the full Step 1-5 preprocessing pipeline for cross-sectional factors.

    Steps:
        1. Forward return (``(price[t+N] / price[t] - 1) / N``, per-period).
        2. Forward return percentile winsorization.
        3. Abnormal return (cross-sectional de-mean).
        4. Factor MAD winsorization.
        5. Cross-sectional MAD z-score.

    Args:
        df: Data with canonical columns ``date``, ``asset_id``, ``price``,
            ``factor``. Use ``adapt()`` to rename if needed.
        config: Pipeline config; when provided, ``forward_periods`` /
            ``return_clip_pct`` / ``mad_n`` are read from it. Passing both
            ``config`` and any of these kwargs raises ``TypeError`` — pick
            one so the effective value is unambiguous and matches the
            downstream tools bound to the same ``config``.
        forward_periods: Number of periods for forward return (default 5).
        return_clip_pct: (lower, upper) quantile bounds for return clipping.
        mad_n: Number of MAD units for factor winsorization (0 to disable).

    Returns:
        DataFrame with columns:
        ``date, asset_id, factor_raw, factor, forward_return, abnormal_return, price``.
    """
    from factrix.config import CrossSectionalConfig
    if config is not None:
        overrides = {
            k: v for k, v in (
                ("forward_periods", forward_periods),
                ("return_clip_pct", return_clip_pct),
                ("mad_n", mad_n),
            ) if v is not UNSET
        }
        if overrides:
            raise TypeError(
                "preprocess_cs_factor: cannot pass both config= and "
                f"{list(overrides)} — the config carries these. Pick one."
            )
        forward_periods = config.forward_periods
        return_clip_pct = config.return_clip_pct
        mad_n = config.mad_n
        sig_cfg = config
    else:
        if forward_periods is UNSET:
            forward_periods = 5
        if return_clip_pct is UNSET:
            return_clip_pct = (0.01, 0.99)
        if mad_n is UNSET:
            mad_n = 3.0
        # Reconstruct a config so the ad-hoc-kwargs path stamps the same
        # preprocess-sig marker as the config= path.
        sig_cfg = CrossSectionalConfig(
            forward_periods=forward_periods,
            return_clip_pct=return_clip_pct,
            mad_n=mad_n,
        )

    # Fail fast on structural mismatch (N=1, or staggered per-date N < 2)
    # before the z-score step silently produces NaN→0 output.
    from factrix._validators import validate_n_assets
    validate_n_assets(df, "cross_sectional")

    out = compute_forward_return(df, forward_periods)
    out = winsorize_forward_return(out, lower=return_clip_pct[0], upper=return_clip_pct[1])
    out = compute_abnormal_return(out)

    # WHY: 保留原始因子值供後續 Profile 分析（如 top_concentration 使用原始分佈）
    out = out.with_columns(pl.col("factor").alias("factor_raw"))

    out = mad_winsorize(out, n_mad=mad_n)
    out = cross_sectional_zscore(out)

    return out.select(
        pl.col("date"),
        pl.col("asset_id"),
        pl.col("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
        pl.col("abnormal_return"),
        pl.col("price"),
        _sig_marker_expr(sig_cfg),
    )


def preprocess_macro_panel(
    df: pl.DataFrame,
    *,
    config: MacroPanelConfig,
) -> pl.DataFrame:
    """Preprocess macro panel data.

    Steps:
        1. Forward return.
        2. Return percentile winsorization.
        3. Optional: cross-section demean signal.
        4. Factor z-score (per-date, no MAD — unstable at small N).
    """
    # Fail fast: FM stage-1 OLS needs ≥ 3 assets per date; catch both
    # N=1 and staggered-schedule cases here rather than at build_artifacts.
    from factrix._validators import validate_n_assets
    validate_n_assets(df, "macro_panel")

    out = compute_forward_return(df, config.forward_periods)
    out = winsorize_forward_return(out)

    out = out.with_columns(pl.col("factor").alias("factor_raw"))

    if config is not None and config.demean_cross_section:
        out = out.with_columns(
            (pl.col("factor") - pl.col("factor").mean().over("date"))
            .alias("factor")
        )

    out = cross_sectional_zscore(out)

    cols = ["date", "asset_id", "factor_raw",
            pl.col("factor_zscore").alias("factor"), "forward_return"]
    if "price" in out.columns:
        cols.append("price")
    cols.append(_sig_marker_expr(config))

    return out.select(cols)


def preprocess_event_signal(
    df: pl.DataFrame,
    *,
    config: EventConfig,
) -> pl.DataFrame:
    """Preprocess event signal data.

    Steps:
        1. Forward return (reuse ``compute_forward_return``).
        2. Return winsorize (per-date percentile clip).
        3. Abnormal return (cross-sectional de-mean, multi-asset only).

    Does NOT apply: MAD winsorize, z-score — factor values are already
    discrete {-1, 0, +1}, normalization would destroy the signal.
    Preserves ``factor`` as-is (no factor_raw/factor_zscore split).

    Args:
        df: Data with ``date, asset_id, price, factor``.

    Returns:
        DataFrame with columns:
        ``date, asset_id, factor, forward_return, abnormal_return, price``.
        ``abnormal_return`` equals ``forward_return`` when N=1
        (no cross-section to de-mean against).
    """
    out = compute_forward_return(df, config.forward_periods)
    out = winsorize_forward_return(out)

    n_assets = out["asset_id"].n_unique()
    if n_assets > 1:
        out = compute_abnormal_return(out)
    else:
        # N=1: no cross-section to de-mean; AR = raw return
        out = out.with_columns(
            pl.col("forward_return").alias("abnormal_return")
        )

    cols = ["date", "asset_id", "factor", "forward_return", "abnormal_return"]
    if "price" in out.columns:
        cols.append("price")
    cols.append(_sig_marker_expr(config))

    return out.select(cols)


def preprocess_macro_common(
    df: pl.DataFrame,
    *,
    config: MacroCommonConfig,
) -> pl.DataFrame:
    """Preprocess macro common factor data.

    Steps:
        1. Forward return.
        2. Return percentile winsorization.
        3. Time-series z-score of factor (not cross-sectional,
           since the factor value is the same for all assets).
    """
    out = compute_forward_return(df, config.forward_periods)
    out = winsorize_forward_return(out)

    out = out.with_columns(pl.col("factor").alias("factor_raw"))

    # WHY: time-series z-score, not cross-sectional — common factor is
    # the same for all assets at each date, so cross-sectional std = 0.
    factor_mean = out["factor"].mean()
    factor_std = out["factor"].std()
    if factor_std is not None and factor_std > 1e-9:
        out = out.with_columns(
            ((pl.col("factor") - factor_mean) / factor_std).alias("factor")
        )

    cols = ["date", "asset_id", "factor_raw", "factor", "forward_return"]
    if "price" in out.columns:
        cols.append("price")
    cols.append(_sig_marker_expr(config))

    return out.select(cols)
