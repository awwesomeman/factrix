"""factrix.datasets.make_multi_factor_panel — schema + correlation knob."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from factrix.datasets import DATASET_SPEC_VERSION, make_multi_factor_panel


def test_dataset_spec_version_pinned():
    assert DATASET_SPEC_VERSION == "1"


def test_schema_and_factor_columns():
    df = make_multi_factor_panel(n_factors=3, n_assets=8, n_dates=40, seed=0)
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    assert len(factor_cols) == 3
    assert df.schema["date"] == pl.Datetime("ms")
    assert df.schema["asset_id"] == pl.String
    assert df.schema["price"] == pl.Float64
    for c in factor_cols:
        assert df.schema[c] == pl.Float64
    assert df.height == 8 * 40


def test_seed_is_deterministic():
    a = make_multi_factor_panel(n_factors=2, n_assets=6, n_dates=30, seed=5)
    b = make_multi_factor_panel(n_factors=2, n_assets=6, n_dates=30, seed=5)
    assert a.equals(b)


def test_factor_correlation_raises_outside_range():
    with pytest.raises(ValueError, match="factor_correlation"):
        make_multi_factor_panel(n_factors=2, n_assets=4, n_dates=30, factor_correlation=1.0)
    with pytest.raises(ValueError, match="factor_correlation"):
        make_multi_factor_panel(n_factors=2, n_assets=4, n_dates=30, factor_correlation=-0.1)


def test_factor_correlation_increases_pairwise_corr():
    """Higher factor_correlation -> higher mean pairwise factor correlation.

    Coarse sanity check, not a precise calibration test.
    """

    def mean_pair_corr(c: float) -> float:
        df = make_multi_factor_panel(
            n_factors=5,
            n_assets=40,
            n_dates=80,
            ic_target=0.0,  # remove the fr-shared component
            factor_correlation=c,
            seed=11,
        )
        cols = [c for c in df.columns if c.startswith("factor_")]
        mat = df.select(cols).to_numpy()
        # Drop the tail dates with zero IC component (they're noise either way).
        corr = np.corrcoef(mat, rowvar=False)
        n = corr.shape[0]
        off_diag = corr[np.triu_indices(n, k=1)]
        return float(off_diag.mean())

    low = mean_pair_corr(0.0)
    high = mean_pair_corr(0.7)
    assert high > low + 0.3
