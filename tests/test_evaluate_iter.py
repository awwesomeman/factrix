"""``evaluate_iter`` — per-factor streaming yield (#428).

Profile-by-profile equivalence to ``evaluate`` (yield order matches
``factor_cols``; each yielded profile equals the dict entry from the
eager API) is the contract. Streaming semantics (first yield lands
before the dispatcher has finished every factor's inference) is the
value proposition. Mirrors ``tests/test_run_metrics_iter.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import factrix.metrics as metrics_pkg
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import ModeAxisError, UserInputError
from factrix._evaluate import _evaluate, evaluate_iter

from tests._run_metrics_helpers import factor_cols, make_multi_panel


@pytest.fixture
def multi_panel() -> pl.DataFrame:
    return make_multi_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


class TestEquivalence:
    def test_iter_matches_eager_profile_for_profile(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        eager = _evaluate(multi_panel, cfg, factor_cols=cols)
        streamed = dict(evaluate_iter(multi_panel, cfg, factor_cols=cols))
        assert set(streamed) == set(eager)
        for fid in eager:
            s, e = streamed[fid], eager[fid]
            assert s.factor_id == e.factor_id == fid
            assert s.n_obs == e.n_obs
            assert s.n_pairs == e.n_pairs
            assert s.n_periods == e.n_periods
            assert s.n_assets == e.n_assets
            assert s.primary_p == pytest.approx(e.primary_p, nan_ok=True)
            assert s.primary_stat == pytest.approx(e.primary_stat, nan_ok=True)
            assert s.info_notes == e.info_notes
            assert s.warnings == e.warnings

    def test_yield_order_follows_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        emitted = [fid for fid, _ in evaluate_iter(multi_panel, cfg, factor_cols=cols)]
        assert emitted == cols


class TestValidation:
    """Pin that input validation runs eagerly, before the first yield."""

    def test_rejects_empty_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            evaluate_iter(multi_panel, cfg, factor_cols=[])

    def test_rejects_duplicate_factor_cols(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        with pytest.raises(UserInputError):
            evaluate_iter(multi_panel, cfg, factor_cols=["factor_0000", "factor_0000"])

    def test_rejects_missing_factor_column_eagerly(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # Without eager validation a polars ColumnNotFoundError would
        # surface only at first yield — worse UX than UserInputError.
        with pytest.raises(UserInputError):
            evaluate_iter(
                multi_panel,
                cfg,
                factor_cols=["factor_0000", "factor_does_not_exist"],
            )

    def test_rejects_unregistered_cell_eagerly(self, multi_panel: pl.DataFrame) -> None:
        # (INDIVIDUAL, CONTINUOUS, *) has no TIMESERIES procedure;
        # restrict the panel to one asset to trigger the path.
        ts = multi_panel.filter(pl.col("asset_id") == multi_panel["asset_id"][0])
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        with pytest.raises(ModeAxisError):
            evaluate_iter(ts, cfg, factor_cols=factor_cols(ts))


class TestStreamingSemantics:
    """Pin the streaming value-add: first yield lands after stage-1 +
    only the first factor's per-factor work, not all N factors'.
    """

    def test_iter_is_a_generator(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        # A list-returning wrapper would defeat streaming silently;
        # pin that the return is a true iterator.
        result = evaluate_iter(multi_panel, cfg, factor_cols=factor_cols(multi_panel))
        assert isinstance(result, Iterator)

    def test_first_yield_only_runs_first_factor_inference(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # _hac_inference is the per-factor inference step called inside
        # the IC procedure's bind_batch closure — one call per factor.
        # If streaming works, only the first factor's call has fired
        # after next(gen).
        import factrix._procedures as procedures_mod

        original = procedures_mod._hac_inference
        call_count = 0

        def _spy(config, series):
            nonlocal call_count
            call_count += 1
            return original(config, series)

        monkeypatch.setattr(procedures_mod, "_hac_inference", _spy)

        cols = factor_cols(multi_panel)
        gen = evaluate_iter(multi_panel, cfg, factor_cols=cols)
        first_fid, _ = next(gen)
        assert first_fid == cols[0]
        assert call_count == 1, (
            f"expected exactly 1 _hac_inference call after first yield, "
            f"got {call_count} — dispatcher is no longer streaming"
        )

        list(gen)
        assert call_count == len(cols)

    def test_break_after_first_yield_skips_remaining_inference(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import factrix._procedures as procedures_mod

        original = procedures_mod._hac_inference
        call_count = 0

        def _spy(config, series):
            nonlocal call_count
            call_count += 1
            return original(config, series)

        monkeypatch.setattr(procedures_mod, "_hac_inference", _spy)

        cols = factor_cols(multi_panel)
        for fid, _profile in evaluate_iter(multi_panel, cfg, factor_cols=cols):
            assert fid == cols[0]
            break

        assert call_count == 1


class TestBatchPrimitiveAmortisation:
    """Pin that IC stage-1 still runs once across the whole batch —
    streaming refactor must not regress the cross-factor share that
    #426 introduced.
    """

    def test_ic_stage1_runs_once_across_factors(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = metrics_pkg.compute_ic
        calls: list[tuple[str, ...]] = []

        def _spy(panel: pl.DataFrame, *, factor_cols, **kwargs):
            calls.append(tuple(factor_cols))
            return original(panel, factor_cols=factor_cols, **kwargs)

        monkeypatch.setattr(metrics_pkg, "compute_ic", _spy)

        cols = factor_cols(multi_panel)
        list(evaluate_iter(multi_panel, cfg, factor_cols=cols))

        assert len(calls) == 1, f"compute_ic ran {len(calls)} times, expected 1"
        assert calls[0] == tuple(cols)

    def test_ic_stage1_runs_before_first_yield(
        self,
        multi_panel: pl.DataFrame,
        cfg: AnalysisConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Streaming contract: any cross-factor work the procedure does
        # in bind_batch must land *before* the generator yields its
        # first profile, so callers see one factor's latency, not N.
        original = metrics_pkg.compute_ic
        calls_before_first_yield = 0
        first_yielded = False

        def _spy(panel: pl.DataFrame, *, factor_cols, **kwargs):
            nonlocal calls_before_first_yield
            if not first_yielded:
                calls_before_first_yield += 1
            return original(panel, factor_cols=factor_cols, **kwargs)

        monkeypatch.setattr(metrics_pkg, "compute_ic", _spy)

        cols = factor_cols(multi_panel)
        gen = evaluate_iter(multi_panel, cfg, factor_cols=cols)
        next(gen)
        first_yielded = True
        assert calls_before_first_yield == 1
        list(gen)


class TestPublicWrapper:
    """`_evaluate` is now `dict(evaluate_iter(...))` — pin the wrapper."""

    def test_evaluate_equals_dict_of_iter(
        self, multi_panel: pl.DataFrame, cfg: AnalysisConfig
    ) -> None:
        cols = factor_cols(multi_panel)
        wrapped = _evaluate(multi_panel, cfg, factor_cols=cols)
        streamed = dict(evaluate_iter(multi_panel, cfg, factor_cols=cols))
        assert set(wrapped) == set(streamed)
        for fid in wrapped:
            assert wrapped[fid].factor_id == streamed[fid].factor_id == fid
