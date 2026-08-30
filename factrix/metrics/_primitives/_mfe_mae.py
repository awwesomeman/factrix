from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import cell
from factrix._stats.constants import DEFAULT_MIN_ESTIMATION_PERIODS
from factrix._types import EPSILON
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _densify_on_period_grid,
    _ragged_event_grid_message,
)

__all__ = ["DEFAULT_MIN_ESTIMATION_PERIODS", "compute_mfe_mae"]


def _empty_mfe_mae_schema(date_dtype: pl.DataType) -> dict[str, pl.DataType]:
    """Output schema with ``date`` dtype mirroring the caller's panel.

    Users with Datetime('us') or TZ-aware inputs get a joinable result.
    """
    return {
        "date": date_dtype,
        "asset_id": pl.String(),
        "mfe": pl.Float64(),
        "mae": pl.Float64(),
        "mfe_z": pl.Float64(),
        "mae_z": pl.Float64(),
        "est_sigma": pl.Float64(),
        "bars_to_mfe": pl.Int32(),
        "bars_to_mae": pl.Int32(),
        "ragged_period_grid_note": pl.String(),
    }


def _validate_compute_mfe_mae(m: MetricBase) -> None:
    """Knob bounds for ``compute_mfe_mae``, applied at construction."""
    from factrix._errors import UserInputError

    value = m.min_estimation_periods  # type: ignore[attr-defined]
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise UserInputError(
            func_name="compute_mfe_mae",
            field="min_estimation_periods",
            value=value,
            expected=(
                "an integer >= 2. It is the pre-event window the volatility "
                "normalizer is estimated on, and a sample standard deviation "
                "(ddof=1) needs at least two observations."
            ),
            docs_path="api/metrics/mfe_mae",
        )


@metric(
    cell=cell(
        None, FactorDensity.SPARSE, DataStructure.PANEL, raw="(*, SPARSE, PANEL)"
    ),
    aggregation=Aggregation.EVENT_TIME,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    validate=_validate_compute_mfe_mae,
)
def compute_mfe_mae(
    data: pl.DataFrame,
    *,
    window: int = 20,
    estimation_window: int = 60,
    min_estimation_periods: int = DEFAULT_MIN_ESTIMATION_PERIODS,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    r"""Per-event Maximum Favorable/Adverse Excursion.

    For each event ($\text{factor} \neq 0$), examines the ``window``
    subsequent periods to find the peak gain (MFE) and peak loss (MAE)
    relative to event entry price, adjusted for density direction.

    Period grid:
        ``window`` and ``estimation_window`` are counts of periods on the
        panel's own distinct-date grid, not counts of the asset's rows. Each
        event asset is laid onto the full grid before the excursion walk, so
        on a ragged panel — an asset missing periods the other names have —
        the excursion spans exactly ``window`` grid periods and the missing
        ones count as missing observations inside it, rather than the walk
        stepping over a hole and reaching further out. A dense panel is
        unaffected. Any price that is not finite is treated the same way, as
        a period with no observation. ``bars_to_mfe`` / ``bars_to_mae`` are
        therefore offsets in grid periods. The raggedness is reported to the
        caller as the ``ragged_period_grid_note`` column, which
        :func:`~factrix.metrics.mfe_mae.mfe_mae` records as
        :attr:`~factrix._codes.WarningCode.RAGGED_PERIOD_GRID`.

    Entry convention:
        Entry is the **event bar's own close**, ``prices[idx]``; the
        excursion path is bars ``idx+1 .. idx+window``. This differs from
        :func:`~factrix.metrics._primitives._event_returns.compute_event_returns`
        and the rest of the event primitives, which *enter* at ``idx+1``
        (the bar after the signal) and so exclude the event-bar gap from
        the measured move. The two conventions answer different questions
        and are both kept deliberately:

        - Excursion analysis asks "how far did the trade go against me
          before it worked?" — a stop-loss / heat question. The trade is
          assumed filled at the signal bar's close, which is the price the
          signal was actually observed at, so the overnight gap into
          ``idx+1`` is part of the adverse excursion a real position would
          have suffered. Excluding it would understate MAE exactly where
          MAE is used (stop placement).
        - The return-profile primitives ask "what is the tradable forward
          return?" and therefore skip the event bar to stay free of
          look-ahead in the *return* series.

        Practical consequence: MFE/MAE here are measured over
        ``window + 1`` bars of price movement (the event-bar close to each
        subsequent close), not ``window`` bars of post-entry return, and
        they are *not* directly comparable to
        ``compute_event_returns(offsets=[window])``.

    Floor convention (Sweeney):
        ``mfe = max(0, max(signed_returns))`` and
        ``mae = min(0, min(signed_returns))``, so MFE is non-negative and
        MAE non-positive by construction, following the original
        Sweeney (1996) / Tharp definitions: MFE
        is the *best* the trade ever was and MAE the *worst*, each relative
        to the entry — a trade that never traded above entry has zero
        favorable excursion, not a "negative favorable excursion". The
        earlier unfloored ``max``/``min`` over the realised path let a
        monotonically losing trade report a *positive* MAE (its least-bad
        bar) and a monotonically winning trade a *negative* MFE — on the
        demo event panel roughly 17% of events carried a positive MAE.
        Those sign-inverted values corrupt every downstream aggregate:
        ``|MAE|`` quantiles, the MFE/MAE ratio, and any stop-distance read.
        ``bars_to_mfe`` / ``bars_to_mae`` are ``0`` when the floor binds —
        the excursion is attained at entry, before any path bar — and
        otherwise stay 1-based offsets into the post-event window.
    """
    date_dtype = data.schema["date"]
    empty_schema = _empty_mfe_mae_schema(date_dtype)

    if price_col not in data.columns:
        return pl.DataFrame(schema=empty_schema)

    sorted_df = data.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=empty_schema)

    # One partition pass over the event-bearing assets instead of an
    # ``asset_id == a`` filter per asset (which re-scanned the whole panel N
    # times). Restrict to event assets first so non-event assets are not
    # materialised.
    event_assets = set(events["asset_id"].unique().to_list())
    # Windows are counted on the panel's period grid, not on the asset's own
    # rows: lay every event asset onto the full grid first, so a period an
    # asset is missing becomes a null price that is *excluded* from the
    # excursion rather than a period that is stepped over (which stretched a
    # ``window``-period excursion across more grid periods on a ragged name).
    # The grid is the whole panel's, so non-event assets still define it.
    dense, _ = _densify_on_period_grid(sorted_df)
    asset_groups: dict[str, tuple[dict, np.ndarray]] = {}
    for key, asset_data in (
        dense.filter(pl.col("asset_id").is_in(list(event_assets)))
        .partition_by("asset_id", as_dict=True, maintain_order=True)
        .items()
    ):
        asset_id = key[0]
        date_to_idx = {d: i for i, d in enumerate(asset_data["date"].to_list())}
        prices = asset_data[price_col].cast(pl.Float64).to_numpy(allow_copy=True)
        asset_groups[asset_id] = (date_to_idx, prices)

    rows: list[dict] = []
    for row in events.iter_rows(named=True):
        asset_id = row["asset_id"]
        event_date = row["date"]
        direction = 1.0 if row[factor_col] > 0 else -1.0

        date_to_idx, prices = asset_groups[asset_id]
        idx = date_to_idx.get(event_date)
        if idx is None:
            continue

        entry_price = prices[idx]
        if not np.isfinite(entry_price) or entry_price < EPSILON:
            continue

        # ``window`` grid periods after the event bar, whether or not this
        # asset trades on all of them.
        end_idx = min(idx + window + 1, len(prices))
        if idx + 1 >= end_idx:
            continue

        future_prices = prices[idx + 1 : end_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            signed_returns = direction * (future_prices / entry_price - 1)
        # A period the asset is missing contributes no excursion; it neither
        # poisons the extremes with NaN nor pulls a later period into the
        # window to replace itself.
        if not np.isfinite(signed_returns).any():
            continue
        signed_returns = np.where(np.isfinite(signed_returns), signed_returns, np.nan)

        # Sweeney floor: excursions are measured against the entry itself,
        # so a trade that never traded above (below) entry has zero
        # favorable (adverse) excursion. bars_to_* = 0 marks "attained at
        # entry", i.e. the floor bound and no path bar is responsible.
        raw_mfe = float(np.nanmax(signed_returns))
        raw_mae = float(np.nanmin(signed_returns))
        mfe = max(0.0, raw_mfe)
        mae = min(0.0, raw_mae)
        bars_to_mfe = int(np.nanargmax(signed_returns)) + 1 if raw_mfe > 0.0 else 0
        bars_to_mae = int(np.nanargmin(signed_returns)) + 1 if raw_mae < 0.0 else 0

        # The estimation window is the ``estimation_window`` grid periods
        # before the event; periods the asset is missing count as missing
        # observations inside it rather than reaching further back.
        est_start = max(0, idx - estimation_window)
        est_prices = prices[est_start:idx]
        est_sigma = float("nan")
        if len(est_prices) > min_estimation_periods:
            prior = est_prices[:-1]
            here = est_prices[1:]
            with np.errstate(invalid="ignore"):
                safe = (prior > EPSILON) & np.isfinite(here)
            if safe.sum() >= min_estimation_periods:
                period_rets = (here[safe] / prior[safe]) - 1.0
                if len(period_rets) >= 2:
                    est_sigma = float(np.std(period_rets, ddof=1))
        window_scale = (
            est_sigma * np.sqrt(window)
            if est_sigma > 0 and np.isfinite(est_sigma)
            else float("nan")
        )
        if np.isfinite(window_scale) and window_scale > EPSILON:
            mfe_z = float(mfe / window_scale)
            mae_z = float(mae / window_scale)
        else:
            mfe_z = float("nan")
            mae_z = float("nan")

        rows.append(
            {
                "date": event_date,
                "asset_id": asset_id,
                "mfe": mfe,
                "mae": mae,
                "mfe_z": mfe_z,
                "mae_z": mae_z,
                "est_sigma": est_sigma,
                "bars_to_mfe": bars_to_mfe,
                "bars_to_mae": bars_to_mae,
            }
        )

    if not rows:
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(date_dtype),
        pl.col("bars_to_mfe").cast(pl.Int32),
        pl.col("bars_to_mae").cast(pl.Int32),
        # Raggedness is a property of the panel, which only this node sees;
        # ``mfe_mae`` reads the note off the frame and records the code there.
        # Empty rather than null on a dense panel, so a caller's ``drop_nulls``
        # over the per-event table does not empty it.
        pl.lit(_ragged_event_grid_message(data) or "", dtype=pl.String).alias(
            "ragged_period_grid_note"
        ),
    )
