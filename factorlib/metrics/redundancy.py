"""Pairwise redundancy matrix for a ProfileSet.

Two methods answer slightly different questions:

``factor_rank`` — do two factors rank the cross-section similarly?
    Per date, rank assets by each factor, take Spearman correlation
    between the two rank vectors, then average across dates. This is
    the closest to "these factors would pick the same stocks". Only
    available when prepared panels are present (keep_artifacts=True).

``value_series`` — do two factors' *performance* move together?
    Align each factor's value-of-interest time series (IC, CAAR, β)
    by date, compute Spearman correlation. Survives compact mode
    because it only needs intermediates, not the full panel.

References:
    Green, Hand & Zhang (2017), "The characteristics that provide
    independent information about average U.S. monthly stock returns."
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from factorlib._types import FactorType

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts
    from factorlib.evaluation.profile_set import ProfileSet


# Map each factor type to the name of its "performance series" in
# Artifacts.intermediates. All four happen to use the same "value" column
# naming after rename() in build_artifacts.
_VALUE_SERIES_KEY: dict[FactorType, str] = {
    FactorType.CROSS_SECTIONAL: "ic_values",
    FactorType.EVENT_SIGNAL: "caar_values",
    FactorType.MACRO_PANEL: "beta_values",
    FactorType.MACRO_COMMON: "beta_values",
}


def redundancy_matrix(
    profiles: "ProfileSet",
    method: Literal["factor_rank", "value_series"] = "factor_rank",
    *,
    artifacts: dict[str, "Artifacts"] | None = None,
) -> pl.DataFrame:
    """Pairwise |Spearman ρ| matrix across factors in ``profiles``.

    Args:
        profiles: A ProfileSet produced by ``evaluate_batch``. Must be
            non-empty; single-type is guaranteed by ProfileSet itself.
        method: See module docstring.
        artifacts: Mapping ``factor_name -> Artifacts``; required
            because profile dataclasses do not carry intermediate
            DataFrames. Keys must be a superset of the profile names.

    Returns:
        DataFrame with a ``factor`` column and one column per factor
        holding the pairwise |ρ|. Diagonal is 1.0; symmetric.

    Behaviour in compact mode:
        ``factor_rank`` auto-downgrades to ``value_series`` with a
        UserWarning if any Artifacts has been compacted (prepared
        dropped). Caller can force an error by setting
        method='factor_rank' and ensuring keep_artifacts=True upstream.
    """
    from factorlib.evaluation._protocol import _CompactedPrepared

    if artifacts is None:
        raise ValueError(
            "redundancy_matrix requires artifacts= (mapping of "
            "factor_name -> Artifacts). Profile dataclasses do not hold "
            "intermediate series; pass the artifacts dict returned by "
            "evaluate_batch(..., keep_artifacts=True)."
        )

    names = [p.factor_name for p in profiles.iter_profiles()]
    if not names:
        raise ValueError("redundancy_matrix: ProfileSet is empty.")
    missing = [n for n in names if n not in artifacts]
    if missing:
        raise KeyError(
            f"redundancy_matrix: artifacts dict missing factor(s) {missing}. "
            f"Provided: {sorted(artifacts.keys())}."
        )

    # Auto-downgrade factor_rank -> value_series if any artifacts are compact
    if method == "factor_rank":
        compact_names = [
            n for n in names
            if isinstance(artifacts[n].prepared, _CompactedPrepared)
        ]
        if compact_names:
            warnings.warn(
                f"redundancy_matrix(method='factor_rank'): auto-downgrading to "
                f"'value_series' because these factors' artifacts are compact: "
                f"{compact_names}. Re-run with keep_artifacts=True to use "
                f"factor_rank.",
                UserWarning,
                stacklevel=2,
            )
            method = "value_series"

    if method == "value_series":
        return _value_series_matrix(names, artifacts, profiles.profile_cls)
    if method == "factor_rank":
        return _factor_rank_matrix(names, artifacts)
    raise ValueError(
        f"Unknown method {method!r}. Valid: 'factor_rank' | 'value_series'."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _value_series_matrix(
    names: list[str],
    artifacts: dict[str, "Artifacts"],
    profile_cls: type,
) -> pl.DataFrame:
    """Outer-join each factor's value series on date, then pairwise Spearman."""
    # All profiles in a ProfileSet share the same class → same factor type
    # → same intermediates key.
    sample_config = artifacts[names[0]].config
    ft = type(sample_config).factor_type
    key = _VALUE_SERIES_KEY[ft]

    # Collect per-factor (date, value) renamed to (date, <name>).
    aligned: pl.DataFrame | None = None
    for n in names:
        series = artifacts[n].get(key).rename({"value": n})
        if aligned is None:
            aligned = series
        else:
            aligned = aligned.join(series, on="date", how="full", coalesce=True)

    assert aligned is not None  # non-empty by earlier check
    matrix = _pairwise_abs_spearman(aligned, names)
    return _matrix_to_df(matrix, names)


def _factor_rank_matrix(
    names: list[str],
    artifacts: dict[str, "Artifacts"],
) -> pl.DataFrame:
    """Per-date cross-sectional rank correlation, averaged across dates.

    Requires each factor's prepared panel to be present (not compact).
    Aligns on (date, asset_id): for each pair, take per-date Spearman
    on factor values, then mean across dates.
    """
    # Grab prepared per factor, keep only date / asset / factor.
    per_factor: dict[str, pl.DataFrame] = {}
    for n in names:
        prep = artifacts[n].prepared
        per_factor[n] = prep.select("date", "asset_id", "factor").rename(
            {"factor": n}
        )

    size = len(names)
    matrix = np.eye(size, dtype=float)
    for i in range(size):
        for j in range(i + 1, size):
            ni, nj = names[i], names[j]
            joined = per_factor[ni].join(
                per_factor[nj], on=["date", "asset_id"], how="inner",
            )
            if joined.is_empty():
                rho = 0.0
            else:
                # Per-date Spearman; average across dates, take abs.
                per_date = (
                    joined.with_columns(
                        pl.col(ni).rank().over("date").alias("_ri"),
                        pl.col(nj).rank().over("date").alias("_rj"),
                    )
                    .group_by("date")
                    .agg(pl.corr("_ri", "_rj").alias("rho"))
                )
                rho_vals = per_date["rho"].drop_nulls().to_numpy()
                rho = float(np.mean(rho_vals)) if len(rho_vals) else 0.0
            matrix[i, j] = abs(rho)
            matrix[j, i] = abs(rho)
    return _matrix_to_df(matrix, names)


def _pairwise_abs_spearman(
    aligned: pl.DataFrame,
    names: list[str],
) -> np.ndarray:
    size = len(names)
    matrix = np.eye(size, dtype=float)
    # Rank once per column, then compute pairwise correlation on ranks
    # which is equivalent to Spearman.
    ranked = aligned.with_columns([pl.col(n).rank().alias(f"__r_{n}") for n in names])
    for i in range(size):
        for j in range(i + 1, size):
            ri = ranked[f"__r_{names[i]}"].drop_nulls()
            rj = ranked[f"__r_{names[j]}"].drop_nulls()
            # Align by dropping rows with nulls in EITHER via inner-join-like
            both = ranked.select(f"__r_{names[i]}", f"__r_{names[j]}").drop_nulls()
            if both.height < 2:
                rho = 0.0
            else:
                rho = float(
                    both.select(
                        pl.corr(f"__r_{names[i]}", f"__r_{names[j]}")
                    ).item()
                    or 0.0
                )
            matrix[i, j] = abs(rho)
            matrix[j, i] = abs(rho)
    return matrix


def _matrix_to_df(matrix: np.ndarray, names: list[str]) -> pl.DataFrame:
    data = {"factor": names}
    for j, n in enumerate(names):
        data[n] = matrix[:, j].tolist()
    return pl.DataFrame(data)
