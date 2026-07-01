"""Input-type gateway for public entry points.

factrix is polars-native: ``fx.evaluate`` accepts
``pl.DataFrame`` (canonical) or ``pl.LazyFrame`` (collected immediately
at the boundary — no projection or predicate pushdown applied by
factrix, so call ``.select(...)`` / ``.filter(...)`` upstream for
memory efficiency on large sources).

``pd.DataFrame`` is **not** accepted on these entry points by design.
pandas users have two clean paths:

- ``factrix.adapt(data, ...)`` — converts and renames columns in one
  step; the natural entry point when pandas column names are not
  already canonical.
- ``pl.from_pandas(data)`` — when columns are already canonical and
  only the type conversion is needed.

This keeps the type contract honest (factrix's internal pipeline is
polars throughout) and avoids hiding the pd → pl copy inside every
``evaluate()`` call.
"""

from __future__ import annotations

import polars as pl

type DataInput = pl.DataFrame | pl.LazyFrame
"""Accepted panel input type for every data-consuming entry point.

Either an eager ``pl.DataFrame`` or a ``pl.LazyFrame`` carrying the panel
schema (see [Data schema](../api/data-schema.md)). A ``LazyFrame`` is
collected internally, so passing one is purely an ergonomic convenience —
the validation and dispatch contract is identical.
"""

# Canonical input-data schema — single source of truth shared by evaluate's
# baseline validation and the DAG executor's per-factor projection.
# Required columns every panel must carry; optional columns are passed
# through to metrics when present (e.g. ``price`` for event-study metrics)
# but never required.
_BASELINE_COLUMNS: tuple[str, ...] = ("date", "asset_id", "forward_return")
_OPTIONAL_COLUMNS: tuple[str, ...] = ("price",)

# Reserved column carrying the panel's single overlap horizon (the
# ``forward_periods`` used to build ``forward_return``). ``compute_forward_return``
# stamps it once; ``evaluate`` reads it and strips it before dispatch, so it
# never reaches a metric, a projection, or ``EvaluationResult.to_frame``. A
# constant int column is the one carrier that survives the ordinary polars
# transforms a panel goes through between construction and evaluation
# (``with_columns`` winsorize / abnormal-return, ``partition_by`` in ``by_slice``,
# user ``.with_columns(sector)`` / joins) — DataFrame-level metadata does not.
_FORWARD_PERIODS_COL: str = "_forward_periods"


def _stamp_forward_periods(data: pl.DataFrame, forward_periods: int) -> pl.DataFrame:
    """Stamp the panel's single overlap horizon as a reserved constant column."""
    return data.with_columns(
        pl.lit(forward_periods, dtype=pl.Int32).alias(_FORWARD_PERIODS_COL)
    )


def _read_forward_periods_stamp(data: pl.DataFrame) -> int | None:
    """Read the stamped overlap horizon, or ``None`` when the panel carries none."""
    if _FORWARD_PERIODS_COL not in data.columns or data.height == 0:
        return None
    return int(data[_FORWARD_PERIODS_COL][0])


def _is_pandas_dataframe(obj: object) -> bool:
    """Detect ``pd.DataFrame`` without importing pandas (optional dep)."""
    return type(obj).__module__.split(".", 1)[0] == "pandas"


def _coerce_data(data: DataInput) -> pl.DataFrame:
    """Coerce ``DataInput`` to eager ``pl.DataFrame``.

    ``pl.LazyFrame`` is collected immediately. ``pd.DataFrame`` is
    rejected with a ``TypeError`` that points to the documented
    conversion paths.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if _is_pandas_dataframe(data):
        raise TypeError(
            "data must be pl.DataFrame or pl.LazyFrame; got pandas DataFrame. "
            "factrix is polars-native — convert with `pl.from_pandas(data)`, "
            "or use `factrix.adapt(data, ...)` if column renaming is needed."
        )
    raise TypeError(
        f"data must be pl.DataFrame or pl.LazyFrame; got {type(data).__name__}."
    )
