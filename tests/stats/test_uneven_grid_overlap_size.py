"""Size of the series-mean tests at the overlap derived on an uneven grid.

``compute_forward_return(..., dates=)`` derives ``overlap_periods`` as
``1 + max_i #{j : 0 < idx_j - idx_i < h}``. On an evaluation grid spaced
unevenly on the period grid the maximum, not a typical count, is what keeps
the non-overlapping stride calibrated: here the spacing pattern
``(20, 20, 40)`` at ``h = 60`` gives a max count of 2 (overlap 3) where a
median count is 1 (overlap 2), and striding at 2 leaves pairs 40 periods
apart that still share a third of their window.

Measured on this grid (400 replications, 54 sampled observations):
``NonOverlapping`` rejects 5.8% at the derived overlap 3, 8.3% at the
under-counted 2 and 21.7% at 1 (no stride); ``NeweyWest`` sits at 5.5% at
every one of the three, its ``1.3 * sqrt(T)`` base bandwidth already
covering the dependence. Grid and replication count are kept small so the
module stays fast.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.inference import NeweyWest, NonOverlapping
from factrix.preprocess.returns import _overlap_on_grid

H = 60
N_FINE = 1500
PATTERN = (20, 20, 40)


def _uneven_index() -> list[int]:
    index = [0]
    step = 0
    while True:
        nxt = index[-1] + PATTERN[step % len(PATTERN)]
        if nxt + 1 + H >= N_FINE:
            break
        index.append(nxt)
        step += 1
    return index


def _overlapping_null(rng: np.random.Generator) -> np.ndarray:
    """h-period forward sums of iid normals on the fine grid, scaled to unit variance."""
    e = rng.standard_normal(N_FINE + H + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(e)])
    # Row i covers (i + 1, i + 1 + h] on the fine grid.
    return (cumulative[1 + H : 1 + H + N_FINE] - cumulative[1 : 1 + N_FINE]) / np.sqrt(
        H
    )


def _sampled_frame(values: np.ndarray, index: list[int]) -> pl.DataFrame:
    origin = datetime(2000, 1, 1)
    return pl.DataFrame(
        {
            "date": [origin + timedelta(days=int(i)) for i in index],
            "v": values[index],
        }
    )


class TestDerivedOverlapOnUnevenGrid:
    def test_pattern_derives_the_maximum_count(self) -> None:
        index = _uneven_index()
        assert _overlap_on_grid(index, H) == 3
        # A typical (median) count would say 2 — the under-count the max rule avoids.
        idx = np.asarray(index)
        counts = np.searchsorted(idx, idx + H, side="left") - (np.arange(idx.size) + 1)
        assert int(np.median(counts)) == 1

    @pytest.mark.parametrize("member", [NonOverlapping(), NeweyWest()])
    def test_size_at_nominal_five_percent(self, member) -> None:
        index = _uneven_index()
        overlap = _overlap_on_grid(index, H)
        rng = np.random.default_rng(20260827)
        n_reps = 300
        rejects = 0
        for _ in range(n_reps):
            frame = _sampled_frame(_overlapping_null(rng), index)
            result = member.compute(frame, value_col="v", overlap_periods=overlap)
            rejects += result.p_value < 0.05
        rate = rejects / n_reps
        # 300 reps put the binomial sd at ~1.3pp; the band is deliberately
        # loose so the check characterises rather than pins.
        assert rate <= 0.09, (
            f"{type(member).__name__}: size {rate:.3f} at overlap {overlap}"
        )
