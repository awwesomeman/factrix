"""Shared input validators reused by preprocess and evaluation layers.

``validate_n_assets`` catches data-shape errors that would otherwise
produce silent NaN or empty output downstream — specifically the cases
where ``cross_sectional`` / ``macro_panel`` canonical tests are
structurally incompatible with the panel supplied:

  (a) global ``n_unique(asset_id) == 1`` — single-asset panel fed to a
      cross-sectional test. Per-date z-score / rank correlation
      collapse, metric short-circuits to ``insufficient_*`` with a
      message about periods rather than the real cause (wrong type).
  (b) staggered schedule: global N ≥ 2 but ``max per-date n_unique``
      < the minimum the canonical test needs. Every date gets dropped
      by the metric-level filter, producing an empty IC / λ series
      with no clear signal to the user.

``event_signal`` and ``macro_common`` tolerate N=1 natively (both have
explicit fallback paths in Profile construction) and are not validated
here — pass their ``factor_type`` in and the function returns without
raising.

Lives in ``factorlib._validators`` (not in ``preprocess`` or
``evaluation``) so both layers can import without creating a circular
dependency. Both layers call this: preprocess fails fast before the
z-score step wastes a pass; evaluation keeps it as a backstop for
callers that skip ``fl.preprocess``.
"""

from __future__ import annotations

import polars as pl


# Minimum "max per-date N" for each factor_type. Values below this at
# the per-date level would make the canonical test mathematically
# undefined on every date.
#
#   cross_sectional: rank correlation needs ≥ 2 distinct assets per date;
#                    MAD z-score needs ≥ 2 for non-zero scale.
#   macro_panel:     Fama-MacBeth stage-1 OLS with factor + intercept
#                    needs ≥ 3 observations per date (2 cols + ≥ 1 dof).
_PER_DATE_MIN: dict[str, int] = {
    "cross_sectional": 2,
    "macro_panel": 3,
}


def validate_n_assets(df: pl.DataFrame, factor_type: str) -> None:
    """Raise ``ValueError`` when a panel cannot support the canonical
    test for ``factor_type``.

    Args:
        df: Panel with at minimum ``date`` and ``asset_id`` columns.
        factor_type: String FactorType value. Only ``cross_sectional``
            and ``macro_panel`` are validated; others return without
            raising.
    """
    if factor_type not in _PER_DATE_MIN:
        return

    n_assets = df["asset_id"].n_unique()

    # Case (a): global single-asset panel.
    if n_assets <= 1:
        if factor_type == "cross_sectional":
            raise ValueError(
                f"cross_sectional expects a multi-asset panel (N≥2; canonical "
                f"test requires cross-sectional variation); got N={n_assets}. "
                f"For a single-asset time series, use:\n"
                f"    fl.MacroCommonConfig(...)  # common time-series factor, "
                f"auto-falls back to per-asset OLS t-test at N=1\n"
                f"factorlib currently has no first-class canonical test for "
                f"'continuous factor × single asset' (see "
                f"docs/plan_direction.md §7 待定決策)."
            )
        # macro_panel
        raise ValueError(
            f"macro_panel expects a small cross-section (N≥2; Fama-MacBeth "
            f"stage-1 OLS requires at least 2 assets per date); got "
            f"N={n_assets}. For a single-asset time series, use "
            f"fl.MacroCommonConfig(...) — it has a dedicated N=1 fallback."
        )

    # Case (b): staggered schedule — max per-date N below the canonical
    # test's structural minimum, so EVERY date would be dropped.
    min_req = _PER_DATE_MIN[factor_type]
    max_per_date = int(
        df.group_by("date")
        .agg(pl.col("asset_id").n_unique().alias("_n"))
        .get_column("_n")
        .max()
        or 0
    )
    if max_per_date < min_req:
        raise ValueError(
            f"{factor_type}: global N={n_assets} but max per-date "
            f"n_unique(asset_id)={max_per_date} (< {min_req} required by "
            f"the canonical test's per-date structure). Every date would "
            f"short-circuit at the metric layer.\n"
            f"  Fix options (pick based on what your data actually is):\n"
            f"    - Rolling / staggered universe (legitimate panel with "
            f"membership turnover): reindex each date onto a common "
            f"calendar of assets, or filter to the intersection of asset "
            f"histories that co-exist long enough for the canonical test.\n"
            f"    - Data quality issue (most dates accidentally missing "
            f"rows): fix upstream, then retry.\n"
            f"    - Genuine single-asset-per-date time series (each date "
            f"really has one observation): use fl.MacroCommonConfig(...) "
            f"which has a dedicated single-asset fallback."
        )
