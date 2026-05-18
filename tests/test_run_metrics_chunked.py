"""``run_metrics_chunked`` — memory-bounded factor-batching iterator (#417).

Equivalence to ``run_metrics`` (chunk-merge = full batch result) is the
contract. Chunk size only changes peak working set, not output.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import UserInputError
from factrix._run_metrics import (
    _auto_chunk_size,
    run_metrics,
    run_metrics_chunked,
)


def _build_multi_factor_panel(
    *,
    n_factors: int = 6,
    n_dates: int = 40,
    n_assets: int = 20,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        factors = [
            0.3 * fwd + 0.7 * rng.standard_normal(n_assets) for _ in range(n_factors)
        ]
        for j in range(n_assets):
            row: dict[str, object] = {
                "date": d,
                "asset_id": f"A{j:03d}",
                "forward_return": float(fwd[j]),
            }
            for k in range(n_factors):
                row[f"factor_{k:04d}"] = float(factors[k][j])
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture
def multi_panel() -> pl.DataFrame:
    return _build_multi_factor_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


def _merge(chunks):
    out: dict = {}
    for chunk in chunks:
        out.update(chunk)
    return out


def _bundle_equals(a, b) -> bool:
    """Compare two MetricsBundle instances by identity + per-metric output."""
    if a.identity != b.identity:
        return False
    if set(a.metrics) != set(b.metrics):
        return False
    for name in a.metrics:
        ma, mb = a.metrics[name], b.metrics[name]
        # MetricOutput is a frozen dataclass with numpy / float fields;
        # repr equality is sufficient for these synthetic panels (no NaN).
        if repr(ma) != repr(mb):
            return False
    return True


class TestEquivalence:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 6, 100])
    def test_merged_chunks_match_full_batch(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        chunk_size: int,
    ) -> None:
        factor_cols = [c for c in multi_panel.columns if c.startswith("factor_")]
        full = run_metrics(multi_panel, cfg, factor_cols=factor_cols)
        merged = _merge(
            run_metrics_chunked(
                multi_panel,
                cfg,
                factor_cols=factor_cols,
                chunk_size=chunk_size,
            )
        )
        assert set(merged) == set(full)
        for fid in full:
            assert _bundle_equals(merged[fid], full[fid]), fid

    def test_yield_order_follows_factor_cols(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        factor_cols = [c for c in multi_panel.columns if c.startswith("factor_")]
        emitted: list[str] = []
        for chunk in run_metrics_chunked(
            multi_panel,
            cfg,
            factor_cols=factor_cols,
            chunk_size=2,
        ):
            emitted.extend(chunk)
        assert emitted == factor_cols


class TestChunkSizeValidation:
    def test_rejects_empty_factor_cols(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        with pytest.raises(UserInputError):
            list(run_metrics_chunked(multi_panel, cfg, factor_cols=[]))

    def test_rejects_duplicate_factor_cols(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        with pytest.raises(UserInputError):
            list(
                run_metrics_chunked(
                    multi_panel,
                    cfg,
                    factor_cols=["factor_0000", "factor_0000"],
                )
            )

    def test_rejects_zero_chunk_size(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                run_metrics_chunked(
                    multi_panel,
                    cfg,
                    factor_cols=["factor_0000"],
                    chunk_size=0,
                )
            )

    def test_rejects_negative_chunk_size(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                run_metrics_chunked(
                    multi_panel,
                    cfg,
                    factor_cols=["factor_0000"],
                    chunk_size=-1,
                )
            )


class TestAutoChunkSize:
    def test_scales_with_available_memory(
        self,
        multi_panel: pl.DataFrame,
    ) -> None:
        # 1 GB available, panel.height=800 rows → per_factor ~25 KB → ~
        # huge chunk; clamps to n_factors.
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024 * 1024 * 1024
            cs = _auto_chunk_size(multi_panel, n_factors=6)
        assert cs == 6  # available RAM dwarfs panel → take full batch

    def test_shrinks_under_memory_pressure(
        self,
        multi_panel: pl.DataFrame,
    ) -> None:
        # Tight available memory shrinks chunk well below n_factors;
        # the exact value depends on the per-factor bytes estimate so
        # the assertion is qualitative (smaller, not exact).
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024 * 1024
            cs_tight = _auto_chunk_size(multi_panel, n_factors=1000)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024 * 1024 * 1024
            cs_loose = _auto_chunk_size(multi_panel, n_factors=1000)
        assert cs_tight < cs_loose
        assert cs_tight < 1000
        assert cs_tight >= 1  # floor

    def test_floor_at_one_when_budget_below_per_factor_estimate(self) -> None:
        # Single very wide panel + tiny RAM: budget would compute < 1
        # without the floor; assert the floor protects us.
        panel = _build_multi_factor_panel(n_dates=100, n_assets=100)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 8  # 8 bytes
            cs = _auto_chunk_size(panel, n_factors=50)
        assert cs == 1

    def test_uses_auto_when_chunk_size_is_none(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        factor_cols = [c for c in multi_panel.columns if c.startswith("factor_")]
        chunks = list(
            run_metrics_chunked(multi_panel, cfg, factor_cols=factor_cols),
        )
        assert sum(len(c) for c in chunks) == len(factor_cols)


class TestLazyFramePath:
    def test_lazy_input_per_chunk_projection(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
    ) -> None:
        # LazyFrame path must produce the same output as eager.
        factor_cols = [c for c in multi_panel.columns if c.startswith("factor_")]
        eager_full = run_metrics(multi_panel, cfg, factor_cols=factor_cols)
        lazy_chunks = _merge(
            run_metrics_chunked(
                multi_panel.lazy(),
                cfg,
                factor_cols=factor_cols,
                chunk_size=2,
            )
        )
        assert set(lazy_chunks) == set(eager_full)
        for fid in eager_full:
            assert _bundle_equals(lazy_chunks[fid], eager_full[fid])


class TestBaseColsOverride:
    def test_custom_base_cols_passed_through(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # Panel with an extra column that would normally be dropped by
        # the default base_cols projection; override base_cols to keep
        # it for primitives that need it. Validates the parameter is
        # actually wired (output matches whether or not the extra col
        # is preserved through select).
        panel = _build_multi_factor_panel(n_factors=3)
        panel = panel.with_columns(pl.lit(1.0).alias("weight"))
        factor_cols = [c for c in panel.columns if c.startswith("factor_")]
        chunks = _merge(
            run_metrics_chunked(
                panel,
                cfg,
                factor_cols=factor_cols,
                chunk_size=2,
                base_cols=("date", "asset_id", "forward_return", "weight"),
            )
        )
        assert set(chunks) == set(factor_cols)
