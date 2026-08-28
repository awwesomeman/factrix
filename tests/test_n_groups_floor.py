"""One ``n_groups`` floor for every quantile-bucketing metric (#878).

``notional_turnover`` used to demand ``n_groups >= 3`` while
``quantile_spread`` accepted — and priced — the two-group top-half /
bottom-half book, so the matched spread / turnover pair a six-name
allocation study needs could not be built. Meanwhile ``n_groups=1`` sailed
through every spread metric as a spread of exactly zero. The bound now
lives once, in :data:`factrix._types.N_GROUPS_FLOOR`, and is enforced by the
shared group-assignment kernels, so these tests pin the *set* of consumers
rather than one function each.
"""

from __future__ import annotations

from datetime import date, timedelta

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix._types import DEFAULT_FORWARD_PERIODS, N_GROUPS_FLOOR
from factrix.metrics import (
    breakeven_cost,
    common_quantile_spread,
    compute_spread_series,
    monotonicity,
    net_spread,
    notional_turnover,
    quantile_spread,
    quantile_spread_vw,
)
from factrix.preprocess import compute_forward_return


def _panel(n_assets: int = 8, n_dates: int = 60) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, seed=0)
    return compute_forward_return(raw, forward_periods=DEFAULT_FORWARD_PERIODS)


def _weighted(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(pl.lit(1.0).alias("market_cap"))


# (label, callable(panel, n_groups) -> MetricResult-ish). Every public
# consumer of the bucketing kernels plus the one metric with its own cut.
_CONSUMERS = [
    ("quantile_spread", lambda p, g: quantile_spread(p, n_groups=g)["factor"]),
    # Single-factor by construction (one weight column), so it hands back the
    # MetricResult itself rather than a per-factor dict.
    (
        "quantile_spread_vw",
        lambda p, g: quantile_spread_vw(
            _weighted(p), n_groups=g, weight_col="market_cap"
        ),
    ),
    (
        "compute_spread_series",
        lambda p, g: compute_spread_series(p, n_groups=g)["factor"],
    ),
    ("monotonicity", lambda p, g: monotonicity(p, n_groups=g)["factor"]),
    (
        "common_quantile_spread",
        lambda p, g: common_quantile_spread(
            p.filter(pl.col("asset_id") == p["asset_id"][0]), n_groups=g
        ),
    ),
    (
        "notional_turnover",
        lambda p, g: notional_turnover(p, n_groups=g, rebalance_lag=1),
    ),
]


@pytest.mark.parametrize("n_groups", [0, 1])
@pytest.mark.parametrize(("label", "run"), _CONSUMERS, ids=[c[0] for c in _CONSUMERS])
def test_every_bucketing_metric_rejects_the_same_floor(label, run, n_groups):
    with pytest.raises(ValueError, match="n_groups"):
        run(_panel(), n_groups)


@pytest.mark.parametrize(("label", "run"), _CONSUMERS, ids=[c[0] for c in _CONSUMERS])
def test_every_bucketing_metric_accepts_the_two_group_book(label, run):
    out = run(_panel(), N_GROUPS_FLOOR)
    # A spread series is a frame; everything else is a MetricResult that ran.
    if isinstance(out, pl.DataFrame):
        assert out.height > 0
    else:
        assert out.metadata.get("reason") is None
        assert out.metadata["n_groups"] == N_GROUPS_FLOOR


class TestTwoGroupNotionalTurnover:
    """#878 — six names, top 3 versus bottom 3."""

    @staticmethod
    def _six_country_panel() -> pl.DataFrame:
        """Eight dates; on odd dates ``C2`` and ``C3`` swap halves.

        Top half on even dates is ``{C3, C4, C5}``, on odd dates ``{C2, C4,
        C5}`` — one of three names replaced in each leg at every transition,
        so the per-leg replaced fraction is exactly ``1/3``.
        """
        rows = []
        for t in range(8):
            for a in range(6):
                factor = float(a)
                if t % 2 == 1 and a in (2, 3):
                    factor = 5.0 - a  # C2 -> 3.0, C3 -> 2.0
                rows.append(
                    {
                        "date": date(2024, 1, 1) + timedelta(days=t),
                        "asset_id": f"C{a}",
                        "factor": factor,
                    }
                )
        return pl.DataFrame(rows)

    def test_two_group_membership_churn(self):
        out = notional_turnover(
            self._six_country_panel(),
            n_groups=2,
            rebalance_lag=1,
            overlap_periods=4,
        )
        assert out.value == pytest.approx(1.0 / 3.0)
        assert out.n_obs == 7
        assert out.metadata["n_groups"] == 2
        assert out.metadata["overlap_periods"] == 4
        assert out.metadata["rebalance_lag"] == 1
        assert out.metadata["mean_tail_size"] == pytest.approx(3.0)

    def test_pairs_with_the_two_group_spread(self):
        panel = _panel(n_assets=6)
        spread = quantile_spread(panel, n_groups=2)["factor"]
        turnover = notional_turnover(panel, n_groups=2)
        assert spread.metadata["n_groups"] == turnover.metadata["n_groups"] == 2

        be = breakeven_cost(spread, turnover=turnover, holding_periods=5)
        net = net_spread(spread, turnover=turnover, holding_periods=5)
        assert be.metadata["pairing_checked"] is True
        assert net.metadata["pairing_checked"] is True
        assert be.metadata["n_groups"] == 2

        # The bucketing check still bites when the books differ.
        with pytest.raises(UserInputError, match="n_groups"):
            breakeven_cost(spread, turnover=notional_turnover(panel, n_groups=3))
