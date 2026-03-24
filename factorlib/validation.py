"""
Layer 2: Data Validation — Pandera schema enforcement.
Catches malformed data before expensive scoring computation.
"""

import pandera.polars as pa
import polars as pl


FACTOR_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Datetime("ms"), nullable=False),
        "asset_id": pa.Column(pl.String, nullable=False),
        "factor_raw": pa.Column(pl.Float64, nullable=True),
        "factor": pa.Column(pl.Float64, nullable=False),
        "forward_return": pa.Column(pl.Float64, nullable=False),
        "abnormal_return": pa.Column(pl.Float64, nullable=False),
    }
)


def validate_factor_data(df: pl.DataFrame) -> pl.DataFrame:
    """Validate DataFrame against FACTOR_SCHEMA, plus NaN/Inf checks.

    WHY: Pandera nullable=False only catches null, not NaN. NaN propagates through
    .mean()/.std() silently, so we must explicitly guard here.
    """
    validated = FACTOR_SCHEMA.validate(df)

    for col in ("factor", "forward_return", "abnormal_return"):
        nan_count = validated.select(pl.col(col).is_nan().sum()).item()
        if nan_count > 0:
            raise ValueError(f"{col} contains {nan_count} NaN values")

    inf_count = validated.select(pl.col("factor").is_infinite().sum()).item()
    if inf_count > 0:
        raise ValueError(f"factor contains {inf_count} Inf values")

    return validated
