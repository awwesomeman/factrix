"""Input-type gateway for public entry points.

factrix is polars-native: ``fx.evaluate`` and ``fx.run_metrics`` accept
``pl.DataFrame`` (canonical) or ``pl.LazyFrame`` (collected immediately
at the boundary — no projection or predicate pushdown applied by
factrix, so call ``.select(...)`` / ``.filter(...)`` upstream for
memory efficiency on large sources).

``pd.DataFrame`` is **not** accepted on these entry points by design.
pandas users have two clean paths:

- ``factrix.adapt(df, ...)`` — converts and renames columns in one
  step; the natural entry point when pandas column names are not
  already canonical.
- ``pl.from_pandas(df)`` — when columns are already canonical and
  only the type conversion is needed.

This keeps the type contract honest (factrix's internal pipeline is
polars throughout) and avoids hiding the pd → pl copy inside every
``evaluate()`` call.
"""

from __future__ import annotations

import polars as pl

type PanelInput = pl.DataFrame | pl.LazyFrame


def _is_pandas_dataframe(obj: object) -> bool:
    """Detect ``pd.DataFrame`` without importing pandas (optional dep)."""
    return type(obj).__module__.split(".", 1)[0] == "pandas"


def _coerce_panel(panel: PanelInput) -> pl.DataFrame:
    """Coerce ``PanelInput`` to eager ``pl.DataFrame``.

    ``pl.LazyFrame`` is collected immediately. ``pd.DataFrame`` is
    rejected with a ``TypeError`` that points to the documented
    conversion paths.
    """
    if isinstance(panel, pl.DataFrame):
        return panel
    if isinstance(panel, pl.LazyFrame):
        return panel.collect()
    if _is_pandas_dataframe(panel):
        raise TypeError(
            "panel must be pl.DataFrame or pl.LazyFrame; got pandas DataFrame. "
            "factrix is polars-native — convert with `pl.from_pandas(df)`, "
            "or use `factrix.adapt(df, ...)` if column renaming is needed."
        )
    raise TypeError(
        f"panel must be pl.DataFrame or pl.LazyFrame; got {type(panel).__name__}."
    )
