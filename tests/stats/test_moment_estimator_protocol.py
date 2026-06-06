"""``MomentEstimator`` sub-protocol + ``GMMResult`` surface tests.

Verifies the runtime-checkable protocol shape and dataclass contract
so cell procedures can dispatch via ``isinstance(obj, MomentEstimator)``
and stitch ``GMMResult`` into ``FactorProfile`` without an
``isinstance`` ladder.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._axis import FactorDensity, FactorScope
from factrix._codes import StatCode, WarningCode
from factrix.stats import (
    Estimator,
    GMMResult,
    MomentEstimator,
)


class _StubGMM:
    """Minimal MomentEstimator-conforming class for protocol checks."""

    @property
    def name(self) -> str:
        return "StubGMM"

    @property
    def description(self) -> str:
        return "Test stub for MomentEstimator."

    @property
    def min_periods(self) -> int:
        return 30

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        return scope is FactorScope.INDIVIDUAL and density is FactorDensity.DENSE

    def emits_for(self, scope: FactorScope, density: FactorDensity) -> StatCode:
        return StatCode.P_GMM

    def compute(self, moments: np.ndarray, *, forward_periods: int) -> GMMResult:
        return GMMResult(
            j_stat=0.0,
            df=moments.shape[1],
            overid_p=1.0,
            n_moments=moments.shape[1],
            n_params=0,
            metadata={"forward_periods": forward_periods},
            warnings=frozenset(),
        )


class _MissingCompute:
    @property
    def name(self) -> str:
        return "x"

    @property
    def description(self) -> str:
        return "x"

    @property
    def min_periods(self) -> int:
        return 1

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        return True

    def emits_for(self, scope: FactorScope, density: FactorDensity) -> StatCode:
        return StatCode.P_GMM


class TestProtocolIdentity:
    def test_stub_satisfies_base_and_moment(self) -> None:
        stub = _StubGMM()
        assert isinstance(stub, Estimator)
        assert isinstance(stub, MomentEstimator)

    def test_missing_compute_rejected(self) -> None:
        assert not isinstance(_MissingCompute(), MomentEstimator)


class TestComputeDispatch:
    def test_compute_returns_gmm_result(self) -> None:
        moments = np.zeros((100, 4))
        result = _StubGMM().compute(moments, forward_periods=5)
        assert isinstance(result, GMMResult)
        assert result.df == 4
        assert result.n_moments == 4
        assert result.n_params == 0
        assert result.metadata["forward_periods"] == 5


class TestGMMResultContract:
    def test_frozen(self) -> None:
        result = GMMResult(
            j_stat=1.0,
            df=2,
            overid_p=0.6,
            n_moments=2,
            n_params=0,
            metadata={},
            warnings=frozenset(),
        )
        with pytest.raises(AttributeError):
            result.j_stat = 2.0  # type: ignore[misc]

    def test_warnings_frozenset(self) -> None:
        result = GMMResult(
            j_stat=1.0,
            df=2,
            overid_p=0.6,
            n_moments=2,
            n_params=0,
            metadata={"weight_matrix_iter": 2},
            warnings=frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS}),
        )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_metadata_carries_solver_extras(self) -> None:
        result = GMMResult(
            j_stat=1.0,
            df=2,
            overid_p=0.6,
            n_moments=2,
            n_params=0,
            metadata={"weight_matrix_iter": 2, "kernel": "bartlett"},
            warnings=frozenset(),
        )
        assert result.metadata["weight_matrix_iter"] == 2
        assert result.metadata["kernel"] == "bartlett"
