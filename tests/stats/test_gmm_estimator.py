"""Tests for ``GMM`` MomentEstimator instance (#191).

Verifies:
- Protocol identity (`Estimator` + `MomentEstimator`).
- ``compute`` is bit-equal to the underlying ``_two_step_gmm_j_stat``
  primitive on the same inputs (no behavior drift through the
  Estimator dispatch layer).
- ``GMMResult`` carries the right metadata + WarningCode keys.
- Registry surfacing via ``_ESTIMATOR_REGISTRY`` + ``get_estimator``.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._axis import FactorScope, FactorSignal, Metric
from factrix._codes import StatCode, WarningCode
from factrix._stats import _two_step_gmm_j_stat
from factrix.stats import (
    _ESTIMATOR_REGISTRY,
    GMM,
    Estimator,
    MomentEstimator,
    get_estimator,
)
from scipy import stats as sp_stats


class TestProtocolIdentity:
    def test_satisfies_estimator_and_moment(self) -> None:
        gmm = GMM()
        assert isinstance(gmm, Estimator)
        assert isinstance(gmm, MomentEstimator)

    def test_frozen_dataclass(self) -> None:
        gmm = GMM()
        with pytest.raises(AttributeError):
            gmm.max_iter = 5  # type: ignore[misc]


class TestSelectionContract:
    def test_name_is_class_name(self) -> None:
        assert GMM().name == "GMM"

    def test_description_one_line(self) -> None:
        desc = GMM().description
        assert "GMM" in desc
        assert "\n" not in desc

    def test_emits_p_gmm(self) -> None:
        assert (
            GMM().emits_for(FactorScope.INDIVIDUAL, FactorSignal.CONTINUOUS, Metric.IC)
            is StatCode.P_GMM
        )

    def test_applicable_only_individual_continuous(self) -> None:
        gmm = GMM()
        assert gmm.applicable_to(FactorScope.INDIVIDUAL, FactorSignal.CONTINUOUS)
        assert not gmm.applicable_to(FactorScope.COMMON, FactorSignal.CONTINUOUS)
        assert not gmm.applicable_to(FactorScope.INDIVIDUAL, FactorSignal.SPARSE)


class TestComputeFidelity:
    def test_compute_matches_underlying_primitive(self) -> None:
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((300, 4))
        result = GMM().compute(moments, forward_periods=5)

        j_ref, df_ref, n_iter_ref, singular_ref = _two_step_gmm_j_stat(
            moments, forward_periods=5, max_iter=2
        )
        assert result.j_stat == j_ref
        assert result.df == df_ref
        assert result.n_moments == 4
        assert result.n_params == 0
        assert result.metadata["weight_matrix_iter"] == n_iter_ref
        assert result.metadata["weight_singular"] is singular_ref

    def test_overid_p_matches_chi_square_sf(self) -> None:
        rng = np.random.default_rng(1)
        moments = rng.standard_normal((300, 3))
        result = GMM().compute(moments, forward_periods=1)
        expected = float(sp_stats.chi2.sf(result.j_stat, df=result.df))
        assert result.overid_p == expected

    def test_max_iter_passes_through(self) -> None:
        rng = np.random.default_rng(2)
        moments = rng.standard_normal((200, 2))
        result = GMM(max_iter=1).compute(moments, forward_periods=1)
        assert result.metadata["weight_matrix_iter"] == 1


class TestWarnings:
    def test_short_sample_emits_warning(self) -> None:
        rng = np.random.default_rng(3)
        moments = rng.standard_normal((10, 2))  # < MIN_PERIODS_WARN (30)
        result = GMM().compute(moments, forward_periods=1)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_singular_weight_emits_warning(self) -> None:
        rng = np.random.default_rng(4)
        base = rng.standard_normal((100, 1))
        moments = np.hstack([base, 2.0 * base])  # rank-deficient → pinv
        result = GMM().compute(moments, forward_periods=1)
        assert WarningCode.SINGULAR_WEIGHT_MATRIX in result.warnings
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS not in result.warnings
        assert result.metadata["weight_singular"] is True

    def test_clean_sample_no_warnings(self) -> None:
        rng = np.random.default_rng(5)
        moments = rng.standard_normal((300, 3))
        result = GMM().compute(moments, forward_periods=1)
        assert result.warnings == frozenset()


class TestRegistry:
    def test_in_estimator_registry(self) -> None:
        assert any(isinstance(e, GMM) for e in _ESTIMATOR_REGISTRY)

    def test_get_estimator_returns_gmm_instance(self) -> None:
        est = get_estimator("GMM")
        assert isinstance(est, GMM)


class TestStatCodePair:
    def test_j_gmm_is_test_statistic(self) -> None:
        assert StatCode.J_GMM.value == "j_gmm"
        assert not StatCode.J_GMM.is_p_value

    def test_p_gmm_is_p_value(self) -> None:
        assert StatCode.P_GMM.value == "p_gmm"
        assert StatCode.P_GMM.is_p_value

    def test_descriptions_present(self) -> None:
        assert "Hansen" in StatCode.J_GMM.description
        assert "Hansen" in StatCode.P_GMM.description
