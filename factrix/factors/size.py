"""Size factor generator."""

import polars as pl


def generate_size(
    df: pl.DataFrame,
    market_cap_col: str = "market_cap",
) -> pl.DataFrame:
    """Size 因子：log(market_cap)。

    大市值股票 factor 值高、小市值低。做空大市值做多小市值
    即 Fama & French (1993) SMB 的反向。使用者可依需求取負。

    Args:
        df: Raw data with ``date``, ``asset_id``, and ``market_cap_col``.
        market_cap_col: Column containing market capitalization.

    Returns:
        Input DataFrame with ``factor`` column appended (= ln(market_cap)).
    """
    return (
        df.sort(["asset_id", "date"])
        .filter(
            pl.col(market_cap_col).is_not_null()
            & (pl.col(market_cap_col) > 0)
        )
        .with_columns(
            pl.col(market_cap_col).log().alias("factor")
        )
    )
