"""
Layer 2: Data Validation — Pandera schema enforcement.
Catches malformed data before expensive scoring computation.
"""

import pandera.polars as pa
import polars as pl


FactorSchema = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Datetime("ms"), nullable=False),
        "asset_id": pa.Column(pl.String, nullable=False),
        "factor": pa.Column(pl.Float64, nullable=False),
        "forward_return": pa.Column(pl.Float64, nullable=False),
    }
)


def validate_factor_data(df: pl.DataFrame) -> pl.DataFrame:
    """Validate DataFrame against FactorSchema, plus NaN/Inf checks."""
    validated = FactorSchema.validate(df)

    nan_count = validated.select(pl.col("factor").is_nan().sum()).item()
    if nan_count > 0:
        raise ValueError(f"factor contains {nan_count} NaN values")

    inf_count = validated.select(pl.col("factor").is_infinite().sum()).item()
    if inf_count > 0:
        raise ValueError(f"factor contains {inf_count} Inf values")

    return validated
