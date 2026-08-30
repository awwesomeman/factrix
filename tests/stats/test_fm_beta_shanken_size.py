"""Null size of ``fm_beta``'s Shanken path, corrected against uncorrected.

The panel is ``make_cs_panel(ic_target=0.0)``, so the FM premium is truly
zero; ``factor_return_var`` is the variance of the panel's own realised
long-short (top-minus-bottom quintile) forward return, which is what a
caller with a traded spread would supply. Measured over 300 replications
per cell at seed 20260830 + rep, at a nominal 5%:

| T | h | uncorrected | Shanken |
|---|---|---|---|
| 120 | 1 | 7.3% | 7.3% |
| 120 | 5 | 3.3% | 3.3% |
| 240 | 1 | 4.3% | 4.0% |
| 240 | 5 | 6.0% | 6.0% |

Under the null the multiplier ``c = 1 + mean(beta)^2 / sigma^2_f`` sits at
essentially 1 — mean(beta) is zero by construction — so the correction is
close to inert and the row inherits the scalar series-mean size of the HAR
path. That is the point of the table: the correction costs nothing under
the null and only shrinks ``t`` when a premium is actually present.

The same script measured the pre-fix code — the corrected ``t`` read
against ``n - 1`` instead of the HAR effective df — returning a p-value
BELOW the uncorrected one in 300/300 draws in every cell.

Replication count here is cut to keep the module fast, so the bounds
characterise rather than pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._stats import _p_value_from_t
from factrix.datasets import make_cs_panel
from factrix.metrics.fm_beta import compute_fm_betas, fm_beta
from factrix.preprocess import compute_forward_return

SEED = 20260830
N_REPS = 60
N_ASSETS = 50


def _long_short_variance(panel: pl.DataFrame) -> float:
    """Variance of the realised top-minus-bottom-quintile forward return."""
    labels = [str(i) for i in range(5)]
    bucketed = panel.with_columns(
        pl.col("factor").qcut(5, labels=labels).over("date").alias("bucket")
    )
    per_period = (
        bucketed.group_by(["date", "bucket"])
        .agg(pl.col("forward_return").mean().alias("leg"))
        .pivot(on="bucket", index="date", values="leg")
        .drop_nulls()
    )
    spread = (per_period["4"] - per_period["0"]).to_numpy()
    return float(np.var(spread, ddof=1))


def _rejection_rates(n_periods: int, horizon: int) -> tuple[float, float, float]:
    """(uncorrected size, Shanken size, fraction where the pre-fix p is smaller)."""
    rejected_uncorrected = rejected_shanken = prefix_smaller = kept = 0
    for rep in range(N_REPS):
        raw = make_cs_panel(
            n_assets=N_ASSETS,
            n_dates=n_periods + horizon + 1,
            ic_target=0.0,
            rng=SEED + rep,
        )
        panel = compute_forward_return(raw, forward_periods=horizon)
        beta_df = compute_fm_betas(panel)["factor"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fm_beta(
                beta_df,
                overlap_periods=horizon,
                is_estimated_factor=True,
                factor_return_var=_long_short_variance(panel),
            )
        p_uncorrected = out.metadata.get("p_value_uncorrected")
        if out.p_value is None or p_uncorrected is None:
            continue
        kept += 1
        rejected_uncorrected += p_uncorrected < 0.05
        rejected_shanken += out.p_value < 0.05
        # The pre-fix path: same corrected t, read against n - 1.
        prefix_smaller += (
            _p_value_from_t(out.stat, out.metadata["n_periods"]) < p_uncorrected
        )
    return (
        rejected_uncorrected / kept,
        rejected_shanken / kept,
        prefix_smaller / kept,
    )


@pytest.mark.parametrize(
    ("n_periods", "horizon"), [(120, 1), (120, 5), (240, 1), (240, 5)]
)
def test_shanken_path_is_sized_like_the_uncorrected_one(n_periods, horizon):
    size_uncorrected, size_shanken, prefix_smaller = _rejection_rates(
        n_periods, horizon
    )
    assert size_uncorrected <= 0.12
    assert size_shanken <= 0.12
    # Under a true null the multiplier is ~1, so the correction neither
    # buys nor costs size; it must never be ANTI-conservative.
    assert size_shanken <= size_uncorrected + 1e-12
    # And the pre-fix df bug is reproducible on every draw in every cell.
    assert prefix_smaller == 1.0
