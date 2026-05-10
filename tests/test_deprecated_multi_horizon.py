"""Deprecation surface for multi_horizon_* metrics (#186).

Covers the user-facing contract: DeprecationWarning fires on call,
``list_metrics`` no longer surfaces the deprecated names, and
``run_metrics`` auto-discover (already covered by #147) keeps skipping
them via ``_AUTO_DISCOVER_EXCLUDED``.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._metric_index import (
    _AUTO_DISCOVER_EXCLUDED,
    _DEPRECATED,
    user_facing_rows,
)
from factrix.metrics.event_horizon import multi_horizon_hit_rate
from factrix.metrics.ic import multi_horizon_ic


def _make_panel(
    n_assets: int = 10,
    n_dates: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for a in range(n_assets):
        price = 100.0
        for d in dates:
            ret = rng.normal(0, 0.01)
            price *= 1 + ret
            rows.append(
                {
                    "date": d,
                    "asset_id": f"asset_{a}",
                    "factor": rng.normal(),
                    "price": price,
                }
            )
    return pl.DataFrame(rows)


class TestDeprecationSurface:
    def test_deprecated_set_contents(self):
        assert frozenset({"multi_horizon_ic", "multi_horizon_hit_rate"}) == _DEPRECATED

    def test_list_metrics_excludes_deprecated(self):
        names = {row.name for row in user_facing_rows()}
        assert "multi_horizon_ic" not in names
        assert "multi_horizon_hit_rate" not in names

    def test_run_metrics_auto_discover_excludes_deprecated(self):
        assert "multi_horizon_ic" in _AUTO_DISCOVER_EXCLUDED
        assert "multi_horizon_hit_rate" in _AUTO_DISCOVER_EXCLUDED

    def test_multi_horizon_ic_emits_deprecation_warning(self):
        df = _make_panel(seed=1)
        with pytest.warns(DeprecationWarning, match=r"#186"):
            multi_horizon_ic(df, periods=[1, 5])

    def test_multi_horizon_hit_rate_emits_deprecation_warning(self):
        df = _make_panel(seed=2)
        with pytest.warns(DeprecationWarning, match=r"#186"):
            multi_horizon_hit_rate(df, horizons=[1, 5])


class TestStillCallable:
    """Deprecated metrics remain importable and runnable for one cycle."""

    def test_imports(self):
        from factrix.metrics import multi_horizon_hit_rate, multi_horizon_ic

        assert callable(multi_horizon_ic)
        assert callable(multi_horizon_hit_rate)

    def test_multi_horizon_ic_still_returns_metric_output(self):
        df = _make_panel(seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = multi_horizon_ic(df, periods=[1, 5])
        assert result.name == "multi_horizon_ic"
