"""``_evaluate`` IC stage-1 cross-factor share (#426).

Mirrors ``tests/test_run_metrics_iter.py::test_ic_stage1_runs_once_across_factors``
for the ``evaluate`` path: ``compute_ic`` must run **once** across an
N-factor batch (the win the ``_ICContPanelProcedure.compute_batch``
override exists for), and the per-factor profiles must equal what the
default per-factor loop produces.
"""

from __future__ import annotations

import factrix.metrics as metrics_pkg
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._evaluate import _evaluate

from tests._run_metrics_helpers import factor_cols, make_multi_panel


@pytest.fixture
def multi_panel() -> pl.DataFrame:
    return make_multi_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


def test_compute_ic_runs_once_across_factors(
    multi_panel: pl.DataFrame,
    cfg: AnalysisConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = metrics_pkg.compute_ic
    calls: list[tuple[str, ...]] = []

    def _spy(panel: pl.DataFrame, *, factor_cols: list[str], **kwargs: object):
        calls.append(tuple(factor_cols))
        return original(panel, factor_cols=factor_cols, **kwargs)

    monkeypatch.setattr(metrics_pkg, "compute_ic", _spy)

    cols = factor_cols(multi_panel)
    _evaluate(multi_panel, cfg, factor_cols=cols)

    assert len(calls) == 1, f"compute_ic ran {len(calls)} times, expected 1"
    assert calls[0] == tuple(cols)


def test_batch_profiles_equal_per_factor_loop(
    multi_panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    cols = factor_cols(multi_panel)

    batched = _evaluate(multi_panel, cfg, factor_cols=cols)
    per_factor = {
        col: _evaluate(multi_panel, cfg, factor_cols=[col])[col] for col in cols
    }

    assert set(batched) == set(per_factor)
    for col in cols:
        b = batched[col]
        p = per_factor[col]
        assert b.factor_id == p.factor_id == col
        assert b.n_obs == p.n_obs
        assert b.n_pairs == p.n_pairs
        assert b.n_periods == p.n_periods
        assert b.n_assets == p.n_assets
        assert b.primary_p == pytest.approx(p.primary_p, nan_ok=True)
        assert b.primary_stat == pytest.approx(p.primary_stat, nan_ok=True)
