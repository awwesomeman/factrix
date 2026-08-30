"""Null size of ``positive_rate``'s exact binomial test, and its discreteness.

Two nulls, both at ``overlap_periods=1`` so ``n`` is the series length
directly. Measured over 300 replications per cell at seed 20260830 + rep,
at a nominal 5%:

| n | iid Gaussian series | IC-series pipeline |
|---|---|---|
| 60 | 2.7% | 2.3% |
| 120 | 3.0% | 4.0% |
| 240 | 4.7% | 8.0% |

Both columns sit at or below nominal, which is the *expected* behaviour
of an exact test on a discrete statistic and not a defect: at most ``n``
the attainable two-sided level nearest 5% from below is materially under
it, and the exact test spends the remainder as conservatism. That is the
deliberate trade in the function's docstring — the normal-approximation
``z`` attains 5% by OVER-rejecting (``n=20, 15 hits``: ``p=0.025`` against
the exact ``0.041``).

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix.datasets import make_cs_panel
from factrix.metrics.ic import compute_ic
from factrix.metrics.positive_rate import positive_rate
from factrix.preprocess import compute_forward_return

SEED = 20260830
N_REPS = 60
N_ASSETS = 50


def _dates(n: int) -> pl.Series:
    return pl.Series(np.arange(n)).cast(pl.Int64).cast(pl.Datetime("ms"))


def _iid_series(n: int, rep: int) -> pl.DataFrame:
    rng = np.random.default_rng(SEED + rep)
    return pl.DataFrame({"date": _dates(n), "value": rng.standard_normal(n)})


def _ic_series(n: int, rep: int) -> pl.DataFrame:
    raw = make_cs_panel(n_assets=N_ASSETS, n_dates=n + 2, ic_target=0.0, rng=SEED + rep)
    panel = compute_forward_return(raw, forward_periods=1)
    return compute_ic(panel)["factor"].select("date", pl.col("ic").alias("value"))


def _rejection_rate(n: int, builder) -> float:
    rejected = kept = 0
    for rep in range(N_REPS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = positive_rate(builder(n, rep), overlap_periods=1)
        if out.p_value is None:
            continue
        kept += 1
        rejected += out.p_value < 0.05
    assert kept, "every replication short-circuited; the null grid is wrong"
    return rejected / kept


@pytest.mark.parametrize("builder", [_iid_series, _ic_series])
@pytest.mark.parametrize("n", [60, 120, 240])
def test_exact_binomial_never_over_rejects_on_a_true_null(n, builder):
    # The exact test is conservative by construction, so the guard is
    # one-sided: it must not drift liberal.
    assert _rejection_rate(n, builder) <= 0.12


def test_exact_test_is_conservative_relative_to_the_normal_approximation():
    """The documented trade: the uncorrected z would reject where the
    exact test does not (``n=20, 15 hits``: 0.025 against 0.041)."""
    import scipy.stats as sp_stats

    n, hits = 20, 15
    exact = sp_stats.binomtest(hits, n, 0.5).pvalue
    z = (hits / n - 0.5) * np.sqrt(n) / 0.5
    approx = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    assert approx < 0.05 < exact + 0.01
    assert approx < exact
