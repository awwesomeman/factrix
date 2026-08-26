"""Pin the polars / numpy / scipy engine defaults factrix's guards rely on.

``_finite_expr`` and the ingestion gate are correct *because* of these engine
behaviours. If a dependency upgrade changes one of them, this file fails
loudly instead of a metric quietly returning a different number.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.stats


def test_polars_is_not_null_lets_nan_through() -> None:
    s = pl.Series([1.0, float("nan"), None])
    assert s.is_not_null().to_list() == [True, True, False]
    assert s.is_finite().to_list() == [True, False, None]


def test_polars_rank_places_nan_last_and_null_as_null() -> None:
    ranks = pl.Series([1.0, float("nan"), 2.0, None]).rank("average").to_list()
    assert ranks == [1.0, 3.0, 2.0, None]


def test_polars_spearman_corr_ranks_nan_silently() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    rho = df.select(pl.corr("a", "b", method="spearman")).item()
    assert rho is not None and math.isfinite(rho)


def test_polars_all_null_sum_is_zero_but_mean_is_null() -> None:
    s = pl.Series([None, None], dtype=pl.Float64)
    assert s.sum() == 0.0
    assert s.mean() is None


def test_polars_std_default_ddof_is_one_numpy_is_zero() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pl.Series(vals).std() == np.std(vals, ddof=1)
    assert pl.Series(vals).std() != np.std(vals)


def test_polars_quantile_default_is_nearest_not_linear() -> None:
    vals = [1.0, 2.0, 3.0, 4.0]
    assert pl.Series(vals).quantile(0.3) == 2.0
    assert pl.Series(vals).quantile(0.3, interpolation="linear") == np.quantile(
        vals, 0.3
    )


def test_polars_to_numpy_maps_null_to_nan_and_widens_int() -> None:
    out = pl.Series([1.0, None]).to_numpy()
    assert np.isnan(out[1])
    assert pl.Series([1, None], dtype=pl.Int64).to_numpy().dtype == np.float64


def test_polars_int64_overflow_wraps_silently() -> None:
    assert (pl.Series([2**62], dtype=pl.Int64) * 4).to_list() == [0]


def test_scipy_rankdata_nan_poisons_whole_array() -> None:
    assert np.isnan(scipy.stats.rankdata([1.0, float("nan"), 2.0])).all()
