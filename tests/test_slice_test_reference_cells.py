"""End-to-end reference cells doubling as docs examples for the
``slice_pairwise_test`` / ``slice_joint_test`` verb pair.

Each cell runs a realistic data-first pipeline (raw panel + metric
instance + ``by`` + ``factor_col``) and asserts the headline result, so
the file serves both as a smoke check and as a copy-paste recipe for
users."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from factrix import slice_joint_test, slice_pairwise_test
from factrix.metrics import ic


def test_pairwise_universe_cross_sectional() -> None:
    """Date-shared universe slice → analytic WaldNWCluster pairwise contrast."""
    rng = np.random.default_rng(11)
    base = dt.date(2024, 1, 1)
    n_dates = 200
    dates = [base + dt.timedelta(days=i) for i in range(n_dates)]

    # Two universes share dates; signal stronger in 'large_cap'.
    rows = []
    for d in dates:
        for u, mu in [("large_cap", 0.10), ("small_cap", 0.02)]:
            for a in range(20):
                factor = rng.normal()
                forward_return = mu * factor + rng.normal(scale=0.02)
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"{u}_{a:03d}",
                        "factor": factor,
                        "forward_return": forward_return,
                        "universe": u,
                    }
                )
    panel = pl.DataFrame(rows)

    out = slice_pairwise_test(panel, ic(), by="universe", factor_col="factor")
    assert out.height == 1
    assert out["p_adj"][0] < 0.05


def test_joint_sector_wald_nw_cluster() -> None:
    """Date-shared sector slice → WaldNWCluster omnibus.

    Joint NW HAC over the (T, K) per-date metric panel requires the K
    slices to share dates; time-disjoint slices (e.g. regimes with
    non-overlapping windows) yield zero aligned rows and raise. Supply a
    date-shared classification (sector / market cap tier / universe).
    """
    rng = np.random.default_rng(22)
    base = dt.date(2024, 1, 1)
    rows = []
    for d in range(150):
        for s, mu in [("tech", 0.10), ("finance", 0.0), ("consumer", -0.05)]:
            for a in range(20):
                factor = rng.normal()
                forward_return = mu * factor + rng.normal(scale=0.02)
                rows.append(
                    {
                        "date": base + dt.timedelta(days=d),
                        "asset_id": f"{s}_{a:03d}",
                        "factor": factor,
                        "forward_return": forward_return,
                        "sector": s,
                    }
                )
    panel = pl.DataFrame(rows)

    out = slice_joint_test(panel, ic(), by="sector", factor_col="factor")
    assert out.height == 1
    assert out["df"][0] == 2
    assert out["p_value"][0] < 0.05
