"""``run_metrics_iter`` — per-factor streaming yield (#419).

Bundle-by-bundle equivalence to ``run_metrics`` (yield order matches
``factor_cols``; each yielded bundle equals the dict entry from the
eager API) is the contract. Streaming semantics (first yield lands
before the dispatcher has finished every factor's per-factor work) is
the value proposition.
"""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import UserInputError
from factrix._run_metrics import _project_factor, run_metrics, run_metrics_iter

from tests._run_metrics_helpers import bundle_equals, factor_cols, make_multi_panel


@pytest.fixture
def multi_panel() -> pl.DataFrame:
    return make_multi_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


class TestEquivalence:
    def test_iter_matches_eager_bundle_for_bundle(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        eager = run_metrics(multi_panel, cfg, factor_cols=cols)
        streamed = dict(run_metrics_iter(multi_panel, cfg, factor_cols=cols))
        assert set(streamed) == set(eager)
        for fid in eager:
            assert bundle_equals(streamed[fid], eager[fid]), fid

    def test_yield_order_follows_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        emitted = [
            fid for fid, _ in run_metrics_iter(multi_panel, cfg, factor_cols=cols)
        ]
        assert emitted == cols

    def test_metrics_subset_passes_through(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        streamed = dict(
            run_metrics_iter(multi_panel, cfg, factor_cols=cols, metrics=["ic"])
        )
        assert set(streamed) == set(cols)
        for bundle in streamed.values():
            assert set(bundle.metrics) == {"ic"}


class TestValidation:
    """Pin that input validation runs eagerly, before the first yield."""

    def test_rejects_empty_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            run_metrics_iter(multi_panel, cfg, factor_cols=[])

    def test_rejects_duplicate_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            run_metrics_iter(
                multi_panel, cfg, factor_cols=["factor_0000", "factor_0000"]
            )

    def test_rejects_missing_factor_column_eagerly(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # Without eager validation a polars ColumnNotFoundError would
        # surface only at first yield — worse UX than UserInputError.
        with pytest.raises(UserInputError):
            run_metrics_iter(
                multi_panel,
                cfg,
                factor_cols=["factor_0000", "factor_does_not_exist"],
            )


class TestStreamingSemantics:
    """Pin the streaming value-add: first yield lands after batch
    primitives + only the first factor's per-factor consumers, not all
    N factors' work.
    """

    def test_first_yield_only_projects_first_factor(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # _project_factor is called once per factor by _PerFactorProtocol.bind,
        # so its call count tracks how many factors the dispatcher has reached.
        cols = factor_cols(multi_panel)
        projected_for: list[str] = []

        def _spy(panel: pl.DataFrame, col: str) -> pl.DataFrame:
            projected_for.append(col)
            return _project_factor(panel, col)

        monkeypatch.setattr("factrix._run_metrics._project_factor", _spy)

        gen = run_metrics_iter(multi_panel, cfg, factor_cols=cols)
        first_fid, _ = next(gen)
        assert first_fid == cols[0]
        assert projected_for == [cols[0]], (
            f"expected only {cols[0]!r} projected after first yield; "
            f"got {projected_for!r} — dispatcher is no longer streaming"
        )

        list(gen)
        assert projected_for == cols

    def test_break_after_first_yield_does_not_run_remaining_factors(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cols = factor_cols(multi_panel)
        projected_for: list[str] = []

        def _spy(panel: pl.DataFrame, col: str) -> pl.DataFrame:
            projected_for.append(col)
            return _project_factor(panel, col)

        monkeypatch.setattr("factrix._run_metrics._project_factor", _spy)

        for fid, _bundle in run_metrics_iter(multi_panel, cfg, factor_cols=cols):
            assert fid == cols[0]
            break

        assert projected_for == [cols[0]]

    def test_iter_is_a_generator(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # A list-returning wrapper would defeat streaming silently;
        # pin that the return is a true iterator.
        result = run_metrics_iter(
            multi_panel, cfg, factor_cols=factor_cols(multi_panel)
        )
        assert isinstance(result, Iterator)


class TestBatchPrimitiveAmortisation:
    """Pin that batch primitives and IC stage-1 still run once across
    the whole batch (the streaming refactor must not regress the
    cross-factor sharing that ``run_metrics`` v1 introduced).
    """

    def test_ic_stage1_runs_once_across_factors(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import factrix.metrics as metrics_pkg

        original = metrics_pkg.compute_ic
        calls: list[tuple[str, ...]] = []

        def _spy(panel: pl.DataFrame, *, factor_cols: list[str]):
            calls.append(tuple(factor_cols))
            return original(panel, factor_cols=factor_cols)

        monkeypatch.setattr(metrics_pkg, "compute_ic", _spy)

        cols = factor_cols(multi_panel)
        list(run_metrics_iter(multi_panel, cfg, factor_cols=cols))

        assert len(calls) == 1, f"compute_ic ran {len(calls)} times, expected 1"
        assert calls[0] == tuple(cols)
