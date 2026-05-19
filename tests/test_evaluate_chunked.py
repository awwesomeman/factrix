"""``evaluate_chunked`` — memory-bounded factor-batching iterator (#427).

Equivalence to ``evaluate`` (chunk-merge = full batch result) is the
contract. Chunk size only changes peak working set, not output.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import polars as pl
import pytest
from factrix import evaluate, evaluate_chunked
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import UserInputError
from factrix.datasets import make_multi_factor_panel

from tests._run_metrics_helpers import factor_cols, make_multi_panel

pytestmark = pytest.mark.skip(
    reason=(
        "Tests target the legacy fx.evaluate(panel, cfg)->FactorProfile path "
        "deleted in #445; full test deletion / rewrite is queued for "
        "#448 (public-surface retire) and #449 (internal-module retire)."
    )
)


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


def _profile_equal(a, b) -> bool:
    return (
        a.factor_id == b.factor_id
        and a.n_obs == b.n_obs
        and a.n_pairs == b.n_pairs
        and a.n_periods == b.n_periods
        and a.n_assets == b.n_assets
        and a.primary_p == pytest.approx(b.primary_p, nan_ok=True)
        and a.primary_stat == pytest.approx(b.primary_stat, nan_ok=True)
    )


class TestEquivalence:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 6, 100])
    def test_merged_chunks_match_full_batch(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig, chunk_size: int
    ) -> None:
        cols = factor_cols(multi_panel)
        full = evaluate(multi_panel, cfg, factor_cols=cols)
        merged = _merge(
            evaluate_chunked(multi_panel, cfg, factor_cols=cols, chunk_size=chunk_size)
        )
        assert set(merged) == set(full)
        for fid in full:
            assert _profile_equal(merged[fid], full[fid]), fid

    def test_yield_order_follows_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        emitted: list[str] = []
        for chunk in evaluate_chunked(multi_panel, cfg, factor_cols=cols, chunk_size=2):
            emitted.extend(chunk)
        assert emitted == cols


class TestValidation:
    def test_rejects_empty_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            list(evaluate_chunked(multi_panel, cfg, factor_cols=[]))

    def test_rejects_duplicate_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            list(
                evaluate_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000", "factor_0000"]
                )
            )

    def test_rejects_zero_chunk_size(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                evaluate_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000"], chunk_size=0
                )
            )

    def test_rejects_negative_chunk_size(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(
                evaluate_chunked(
                    multi_panel, cfg, factor_cols=["factor_0000"], chunk_size=-1
                )
            )

    def test_rejects_missing_factor_column_eagerly(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError, match="missing"):
            list(
                evaluate_chunked(
                    multi_panel,
                    cfg,
                    factor_cols=["factor_0000", "factor_does_not_exist"],
                )
            )

    def test_rejects_missing_base_column_eagerly(self, cfg: AnalysisConfig) -> None:
        raw = make_multi_factor_panel(n_factors=2, n_dates=50, n_assets=10, seed=0)
        with pytest.raises(UserInputError, match="missing"):
            list(evaluate_chunked(raw, cfg, factor_cols=["factor_0000"], chunk_size=1))

    def test_rejects_non_dataframe_panel(self, cfg: AnalysisConfig) -> None:
        with pytest.raises(TypeError, match=r"pl\.DataFrame or pl\.LazyFrame"):
            list(
                evaluate_chunked(
                    "not a frame",  # type: ignore[arg-type]
                    cfg,
                    factor_cols=["factor_0000"],
                    chunk_size=1,
                )
            )


class TestAutoChunkSize:
    def test_uses_auto_when_chunk_size_is_none(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        chunks = list(evaluate_chunked(multi_panel, cfg, factor_cols=cols))
        assert sum(len(c) for c in chunks) == len(cols)

    def test_missing_psutil_raises_actionable_error(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with (
            patch.dict(sys.modules, {"psutil": None}),
            pytest.raises(UserInputError, match="psutil"),
        ):
            list(
                evaluate_chunked(multi_panel, cfg, factor_cols=factor_cols(multi_panel))
            )


class TestLazyFramePath:
    def test_lazy_input_per_chunk_projection(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        eager_full = evaluate(multi_panel, cfg, factor_cols=cols)
        lazy_chunks = _merge(
            evaluate_chunked(multi_panel.lazy(), cfg, factor_cols=cols, chunk_size=2)
        )
        assert set(lazy_chunks) == set(eager_full)
        for fid in eager_full:
            assert _profile_equal(lazy_chunks[fid], eager_full[fid])


class TestBaseColsOverride:
    def test_custom_base_cols_passed_through(self, cfg: AnalysisConfig) -> None:
        panel = make_multi_panel(n_factors=3)
        panel = panel.with_columns(pl.lit(1.0).alias("weight"))
        cols = factor_cols(panel)
        chunks = _merge(
            evaluate_chunked(
                panel,
                cfg,
                factor_cols=cols,
                chunk_size=2,
                base_cols=("date", "asset_id", "forward_return", "weight"),
            )
        )
        assert set(chunks) == set(cols)
