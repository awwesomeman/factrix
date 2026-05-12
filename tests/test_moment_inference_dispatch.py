"""Wiring tests for ``cfg.moment_estimator`` + ``_moment_inference`` (#191).

Covers the dispatch infrastructure without exercising a cell-level
procedure (the multi-horizon panel cell lands in a follow-up). Verifies:

- ``AnalysisConfig.moment_estimator`` field + applicability gate
- ``to_dict`` / ``from_dict`` round-trip
- ``_moment_inference`` helper stitches ``GMMResult`` into the
  procedure layer's ``(stats, metadata)`` contract
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._codes import StatCode
from factrix._errors import IncompatibleAxisError
from factrix._procedures import _moment_inference
from factrix.stats import GMM, NeweyWest


class TestCfgField:
    def test_default_is_none(self) -> None:
        cfg = AnalysisConfig.individual_continuous()
        assert cfg.moment_estimator is None

    def test_set_via_factory(self) -> None:
        cfg = AnalysisConfig.individual_continuous(moment_estimator=GMM())
        assert isinstance(cfg.moment_estimator, GMM)

    def test_independent_of_hac_estimator(self) -> None:
        cfg = AnalysisConfig.individual_continuous(
            estimator=NeweyWest(), moment_estimator=GMM()
        )
        assert isinstance(cfg.estimator, NeweyWest)
        assert isinstance(cfg.moment_estimator, GMM)


class TestApplicabilityGate:
    def test_rejects_inapplicable_cell(self) -> None:
        # GMM.applicable_to is False for (COMMON, *) — gate must fire.
        with pytest.raises(IncompatibleAxisError, match="not applicable to"):
            AnalysisConfig.common_continuous(moment_estimator=GMM())

    def test_none_skips_gate(self) -> None:
        # COMMON cell would reject GMM, but None bypasses the gate entirely.
        cfg = AnalysisConfig.common_continuous(moment_estimator=None)
        assert cfg.moment_estimator is None


class TestRoundTrip:
    def test_to_dict_records_moment_estimator_name(self) -> None:
        cfg = AnalysisConfig.individual_continuous(moment_estimator=GMM())
        d = cfg.to_dict()
        assert d["moment_estimator"] == "GMM"

    def test_to_dict_records_none(self) -> None:
        cfg = AnalysisConfig.individual_continuous()
        d = cfg.to_dict()
        assert d["moment_estimator"] is None

    def test_from_dict_rehydrates_gmm(self) -> None:
        cfg = AnalysisConfig.individual_continuous(moment_estimator=GMM())
        restored = AnalysisConfig.from_dict(cfg.to_dict())
        assert isinstance(restored.moment_estimator, GMM)
        assert restored == cfg

    def test_from_dict_backward_compatible_missing_key(self) -> None:
        # Pre-#191 serialized configs have no `moment_estimator` key.
        d = {
            "scope": "individual",
            "signal": "continuous",
            "metric": "ic",
            "forward_periods": 5,
            "estimator": "NeweyWest",
        }
        cfg = AnalysisConfig.from_dict(d)
        assert cfg.moment_estimator is None


class TestMomentInferenceHelper:
    def test_returns_gmm_result_and_stats(self) -> None:
        cfg = AnalysisConfig.individual_continuous(moment_estimator=GMM())
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((300, 4))
        result, stats, _ = _moment_inference(cfg, moments)

        assert result.df == 4
        assert stats[StatCode.J_GMM] == result.j_stat
        assert stats[StatCode.P_GMM] == result.overid_p

    def test_metadata_keyed_by_both_codes(self) -> None:
        cfg = AnalysisConfig.individual_continuous(
            forward_periods=5, moment_estimator=GMM()
        )
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((300, 3))
        _, _, metadata = _moment_inference(cfg, moments)

        # Both keys present, both carrying the same payload.
        assert set(metadata.keys()) == {StatCode.J_GMM, StatCode.P_GMM}
        assert metadata[StatCode.J_GMM]["n_moments"] == 3
        assert metadata[StatCode.P_GMM]["n_moments"] == 3
        assert metadata[StatCode.J_GMM]["df"] == 3
        # Inner dicts are independent — mutating one does not affect the other.
        metadata[StatCode.J_GMM]["new_key"] = "x"
        assert "new_key" not in metadata[StatCode.P_GMM]

    def test_metadata_includes_solver_extras(self) -> None:
        cfg = AnalysisConfig.individual_continuous(moment_estimator=GMM())
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((300, 4))
        _, _, metadata = _moment_inference(cfg, moments)
        assert "weight_matrix_iter" in metadata[StatCode.J_GMM]
        assert "weight_singular" in metadata[StatCode.J_GMM]

    def test_forward_periods_threaded_through(self) -> None:
        # cfg.forward_periods=20 should flow into the GMM compute's
        # long-run-covariance bandwidth floor.
        cfg_short = AnalysisConfig.individual_continuous(
            forward_periods=1, moment_estimator=GMM()
        )
        cfg_long = AnalysisConfig.individual_continuous(
            forward_periods=20, moment_estimator=GMM()
        )
        rng = np.random.default_rng(7)
        raw = rng.standard_normal(400)
        kernel = np.ones(10)
        series = np.convolve(raw, kernel, mode="valid")
        moments = np.column_stack(
            [series, series + 0.1 * rng.standard_normal(len(series))]
        )
        result_short, _, _ = _moment_inference(cfg_short, moments)
        result_long, _, _ = _moment_inference(cfg_long, moments)
        # Wider bandwidth → larger Ŝ → smaller J for the same g_bar.
        assert result_long.j_stat < result_short.j_stat
