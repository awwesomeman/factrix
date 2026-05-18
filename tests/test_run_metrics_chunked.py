"""``run_metrics_chunked`` — memory-bounded factor-batching iterator (#417).

Equivalence to ``run_metrics`` (chunk-merge = full batch result) is the
contract. Chunk size only changes peak working set, not output.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._chunk_size import _AUTO_CHUNK_OVERHEAD_FACTOR, _AUTO_CHUNK_RSS_DIVISOR
from factrix._errors import UserInputError
from factrix._run_metrics import _auto_chunk_size, run_metrics, run_metrics_chunked
from factrix.datasets import make_multi_factor_panel

from tests._run_metrics_helpers import bundle_equals, factor_cols, make_multi_panel


@pytest.fixture
def multi_panel() -> pl.DataFrame:
    return make_multi_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


def _merge(chunks):
    out: dict = {}
    for chunk in chunks:
        out.update(chunk)
    return out


class TestEquivalence:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 6, 100])
    def test_merged_chunks_match_full_batch(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig, chunk_size: int
    ) -> None:
        cols = factor_cols(multi_panel)
        full = run_metrics(multi_panel, cfg, factor_cols=cols)
        merged = _merge(
            run_metrics_chunked(
                multi_panel, cfg, factor_cols=cols, chunk_size=chunk_size
            )
        )
        assert set(merged) == set(full)
        for fid in full:
            assert bundle_equals(merged[fid], full[fid]), fid

    def test_yield_order_follows_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        emitted: list[str] = []
        for chunk in run_metrics_chunked(
            multi_panel, cfg, factor_cols=cols, chunk_size=2
        ):
            emitted.extend(chunk)
        assert emitted == cols


class TestValidation:
    def test_rejects_empty_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            list(run_metrics_chunked(multi_panel, cfg, factor_cols=[]))

    def test_rejects_duplicate_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            list(
                run_metrics_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000", "factor_0000"]
                )
            )

    def test_rejects_zero_chunk_size(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                run_metrics_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000"], chunk_size=0
                )
            )

    def test_rejects_negative_chunk_size(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                run_metrics_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000"], chunk_size=-1
                )
            )

    def test_rejects_missing_factor_column_eagerly(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # Mid-iteration polars ColumnNotFoundError would be a worse UX
        # than the project's eager UserInputError on schema mismatch.
        with pytest.raises(UserInputError, match="missing"):
            list(
                run_metrics_chunked(
                    multi_panel,
                    cfg,
                    factor_cols=["factor_0000", "factor_does_not_exist"],
                )
            )

    def test_rejects_missing_base_column_eagerly(self, cfg: AnalysisConfig) -> None:
        raw = make_multi_factor_panel(n_factors=2, n_dates=50, n_assets=10, seed=0)
        # raw has date / asset_id / factor_* / price but no forward_return.
        with pytest.raises(UserInputError, match="missing"):
            list(
                run_metrics_chunked(raw, cfg, factor_cols=["factor_0000"], chunk_size=1)
            )

    def test_rejects_non_dataframe_panel(self, cfg: AnalysisConfig) -> None:
        with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
            list(
                run_metrics_chunked(
                    "not a frame",  # type: ignore[arg-type]
                    cfg,
                    factor_cols=["factor_0000"],
                    chunk_size=1,
                )
            )


class TestAutoChunkSize:
    def test_formula_pinned(self) -> None:
        # Pins the heuristic so a quiet refactor of the formula is
        # caught at test time.
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024 * 1024 * 1024  # 1 GiB
            cs = _auto_chunk_size(n_rows=1000, n_factors=200)
        per_factor = 1000 * 8 * _AUTO_CHUNK_OVERHEAD_FACTOR  # 32_000
        budget = (1024 * 1024 * 1024) // _AUTO_CHUNK_RSS_DIVISOR  # 268_435_456
        expected = max(1, min(200, budget // per_factor))
        assert cs == expected

    def test_clamps_to_n_factors(self) -> None:
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 100 * 1024**3  # generous
            cs = _auto_chunk_size(n_rows=800, n_factors=6)
        assert cs == 6

    def test_floor_at_one_when_budget_below_per_factor(self) -> None:
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 8  # 8 bytes
            cs = _auto_chunk_size(n_rows=1_000_000, n_factors=50)
        assert cs == 1

    def test_uses_auto_when_chunk_size_is_none(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        chunks = list(run_metrics_chunked(multi_panel, cfg, factor_cols=cols))
        assert sum(len(c) for c in chunks) == len(cols)

    def test_missing_psutil_raises_actionable_error(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # Auto-sizing requires psutil; the error must steer the caller
        # at either installing it or passing chunk_size explicitly.
        with (
            patch.dict(sys.modules, {"psutil": None}),
            pytest.raises(UserInputError, match="psutil"),
        ):
            list(
                run_metrics_chunked(
                    multi_panel, cfg, factor_cols=factor_cols(multi_panel)
                )
            )


class TestLazyFramePath:
    def test_lazy_input_per_chunk_projection(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        eager_full = run_metrics(multi_panel, cfg, factor_cols=cols)
        lazy_chunks = _merge(
            run_metrics_chunked(multi_panel.lazy(), cfg, factor_cols=cols, chunk_size=2)
        )
        assert set(lazy_chunks) == set(eager_full)
        for fid in eager_full:
            assert bundle_equals(lazy_chunks[fid], eager_full[fid])


class TestBaseColsOverride:
    def test_custom_base_cols_passed_through(self, cfg: AnalysisConfig) -> None:
        panel = make_multi_panel(n_factors=3)
        panel = panel.with_columns(pl.lit(1.0).alias("weight"))
        cols = factor_cols(panel)
        chunks = _merge(
            run_metrics_chunked(
                panel,
                cfg,
                factor_cols=cols,
                chunk_size=2,
                base_cols=("date", "asset_id", "forward_return", "weight"),
            )
        )
        assert set(chunks) == set(cols)
