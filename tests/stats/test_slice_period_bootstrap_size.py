"""Size of the date-disjoint pairwise bootstrap under overlapping returns."""

from __future__ import annotations

import numpy as np
import pytest
from factrix.slicing.period_inference import _pairwise_contrasts

N_PERIODS = 240
N_RESAMPLES = 199
N_REPS = 120


def _overlapping_null(rng: np.random.Generator, horizon: int) -> np.ndarray:
    innovations = rng.standard_normal(N_PERIODS + horizon - 1)
    return np.convolve(innovations, np.ones(horizon), mode="valid") / np.sqrt(horizon)


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_pairwise_bootstrap_size_with_overlap(horizon: int) -> None:
    """Known MA(h-1) dependence stays near the nominal five-percent size."""
    rejected = 0
    for rep in range(N_REPS):
        data_rng = np.random.default_rng(1_011_000 + 10_000 * horizon + rep)
        series = [
            _overlapping_null(data_rng, horizon),
            _overlapping_null(data_rng, horizon),
        ]
        p_raw = _pairwise_contrasts(
            series,
            [N_PERIODS, N_PERIODS],
            [(0, 1)],
            method="bootstrap",
            overlap_periods=horizon,
            n_resamples=N_RESAMPLES,
            rng=np.random.default_rng(2_026_000 + 10_000 * horizon + rep),
        )[0][2]
        rejected += p_raw < 0.05

    assert rejected / N_REPS <= 0.12
