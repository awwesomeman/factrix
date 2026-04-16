"""Data validation — Pandera schema enforcement.

Per-type schemas ensure data meets the requirements of the corresponding
evaluation pipeline before expensive computation.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from factorlib._types import FactorType


_SCHEMAS: dict[FactorType, pa.DataFrameSchema] = {
    FactorType.CROSS_SECTIONAL: pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Datetime("ms"), nullable=False),
            "asset_id": pa.Column(pl.String, nullable=False),
            "factor_raw": pa.Column(pl.Float64, nullable=True),
            "factor": pa.Column(pl.Float64, nullable=False),
            "forward_return": pa.Column(pl.Float64, nullable=False),
            "abnormal_return": pa.Column(pl.Float64, nullable=False),
        }
    ),
    FactorType.MACRO_PANEL: pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Datetime("ms"), nullable=False),
            "asset_id": pa.Column(pl.String, nullable=False),
            "factor": pa.Column(pl.Float64, nullable=False),
            "forward_return": pa.Column(pl.Float64, nullable=False),
        }
    ),
}

FACTOR_SCHEMA = _SCHEMAS[FactorType.CROSS_SECTIONAL]


def validate_factor_data(
    df: pl.DataFrame,
    factor_type: FactorType = FactorType.CROSS_SECTIONAL,
) -> pl.DataFrame:
    """Validate DataFrame against the schema for the given factor type.

    Args:
        df: Data to validate.
        factor_type: Which schema to use. Defaults to cross_sectional.

    Raises:
        ValueError: If NaN or Inf values are found in numeric columns.
        pandera.errors.SchemaError: If schema validation fails.
    """
    schema = _SCHEMAS.get(factor_type)
    if schema is None:
        raise NotImplementedError(
            f"No validation schema defined for {factor_type}. "
            f"Available: {list(_SCHEMAS.keys())}"
        )

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
