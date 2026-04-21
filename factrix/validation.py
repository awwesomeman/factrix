"""Data validation — Pandera schema enforcement.

Per-type schemas ensure data meets the requirements of the corresponding
evaluation pipeline before expensive computation. The ``date`` column is
checked separately (``_check_date_dtype``) because pandera-polars matches
``pl.Datetime(time_unit, time_zone)`` exactly — we want to accept any
time_unit and any timezone (including naive) so high-precision or TZ-aware
panels validate without contortions.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from factrix._types import FactorType


_SCHEMAS: dict[FactorType, pa.DataFrameSchema] = {
    FactorType.CROSS_SECTIONAL: pa.DataFrameSchema(
        {
            "asset_id": pa.Column(pl.String, nullable=False),
            "factor_raw": pa.Column(pl.Float64, nullable=True),
            "factor": pa.Column(pl.Float64, nullable=False),
            "forward_return": pa.Column(pl.Float64, nullable=False),
            "abnormal_return": pa.Column(pl.Float64, nullable=False),
        }
    ),
    FactorType.MACRO_PANEL: pa.DataFrameSchema(
        {
            "asset_id": pa.Column(pl.String, nullable=False),
            "factor": pa.Column(pl.Float64, nullable=False),
            "forward_return": pa.Column(pl.Float64, nullable=False),
        }
    ),
}
# WHY: event_signal uses the same minimal schema as macro_panel —
# factor is {-1, 0, +1} (Float64), forward_return is standard.
_SCHEMAS[FactorType.EVENT_SIGNAL] = _SCHEMAS[FactorType.MACRO_PANEL]
# WHY: macro_common uses the same minimal schema as macro_panel
_SCHEMAS[FactorType.MACRO_COMMON] = _SCHEMAS[FactorType.MACRO_PANEL]

FACTOR_SCHEMA = _SCHEMAS[FactorType.CROSS_SECTIONAL]


def _check_date_dtype(df: pl.DataFrame) -> None:
    """Require the ``date`` column to be ``pl.Date`` or any ``pl.Datetime``.

    Accepts any ``time_unit`` / ``time_zone`` combination. TZ consistency
    across joined DataFrames (main panel / regime_labels /
    spanning_base_spreads) is checked separately at join points.
    """
    if "date" not in df.columns:
        raise ValueError("date column missing")
    dtype = df.schema["date"]
    if isinstance(dtype, (pl.Date, pl.Datetime)):
        return
    raise ValueError(
        f"date column must be pl.Date or pl.Datetime, got {dtype}. "
        f"Call fl.adapt(...) to normalize, or cast manually with "
        f"`df.with_columns(pl.col('date').cast(pl.Datetime('ms')))`."
    )


def validate_factor_data(
    df: pl.DataFrame,
    factor_type: FactorType = FactorType.CROSS_SECTIONAL,
) -> pl.DataFrame:
    """Validate DataFrame against the schema for the given factor type.

    Args:
        df: Data to validate.
        factor_type: Which schema to use. Defaults to cross_sectional.

    Raises:
        ValueError: If NaN or Inf values are found in numeric columns, or
            if the ``date`` column is not ``pl.Date`` / ``pl.Datetime``.
        pandera.errors.SchemaError: If schema validation fails on other
            columns.
    """
    schema = _SCHEMAS.get(factor_type)
    if schema is None:
        raise NotImplementedError(
            f"No validation schema defined for {factor_type}. "
            f"Available: {list(_SCHEMAS.keys())}"
        )

    _check_date_dtype(df)
    validated = schema.validate(df)

    numeric_cols = [
        c for c in ("factor", "forward_return", "abnormal_return")
        if c in validated.columns
    ]
    for col in numeric_cols:
        nan_count = validated.select(pl.col(col).is_nan().sum()).item()
        if nan_count > 0:
            raise ValueError(f"{col} contains {nan_count} NaN values")

    if "factor" in validated.columns:
        inf_count = validated.select(pl.col("factor").is_infinite().sum()).item()
        if inf_count > 0:
            raise ValueError(f"factor contains {inf_count} Inf values")

    return validated
