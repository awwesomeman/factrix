"""Tests for ``AnalysisConfig.estimator`` field + serialization (#163).

Covers the study-level estimator switch:
- default ``NeweyWest()`` matches v0.11 behavior
- explicit ``HansenHodrick()`` on (INDIVIDUAL, CONTINUOUS) cells
- applicability gate raises early at ``__post_init__``
- ``to_dict`` / ``from_dict`` round-trip via name-string serialization
- legacy dicts without ``estimator`` key default to ``NeweyWest()``
- ``get_estimator(name)`` registry lookup + ``UnknownEstimatorError``
"""

from __future__ import annotations

import pytest
from factrix import AnalysisConfig, IncompatibleAxisError, UnknownEstimatorError
from factrix.stats import (
    BlockBootstrap,
    HansenHodrick,
    NeweyWest,
    WaldNWCluster,
    get_estimator,
)


class TestDefaultEstimator:
    """Every factory method defaults to ``NeweyWest()`` for v0.11 parity."""

    @pytest.mark.parametrize(
        "factory",
        [
            AnalysisConfig.individual_continuous,
            AnalysisConfig.individual_sparse,
            AnalysisConfig.common_continuous,
            AnalysisConfig.common_sparse,
        ],
    )
    def test_default_is_newey_west(self, factory) -> None:
        cfg = factory()
        assert cfg.estimator == NeweyWest()


class TestExplicitEstimator:
    """Factory ``estimator=`` kwarg forwards to the validated field."""

    def test_individual_continuous_with_hh(self) -> None:
        cfg = AnalysisConfig.individual_continuous(estimator=HansenHodrick())
        assert cfg.estimator == HansenHodrick()


class TestApplicabilityGate:
    """``__post_init__`` rejects non-applicable / non-HAC estimators."""

    def test_hh_on_common_continuous_raises(self) -> None:
        with pytest.raises(IncompatibleAxisError, match="not applicable"):
            AnalysisConfig.common_continuous(estimator=HansenHodrick())

    def test_hh_on_individual_sparse_raises(self) -> None:
        with pytest.raises(IncompatibleAxisError, match="not applicable"):
            AnalysisConfig.individual_sparse(estimator=HansenHodrick())

    def test_slice_estimator_rejected_as_non_hac(self) -> None:
        with pytest.raises(IncompatibleAxisError, match="not an HAC estimator"):
            AnalysisConfig.individual_continuous(estimator=WaldNWCluster())  # type: ignore[arg-type]

    def test_block_bootstrap_rejected_as_non_hac(self) -> None:
        with pytest.raises(IncompatibleAxisError, match="not an HAC estimator"):
            AnalysisConfig.individual_continuous(estimator=BlockBootstrap())  # type: ignore[arg-type]

    def test_error_message_lists_applicable(self) -> None:
        with pytest.raises(IncompatibleAxisError) as exc_info:
            AnalysisConfig.common_continuous(estimator=HansenHodrick())
        assert "NeweyWest" in str(exc_info.value)


class TestSerialization:
    """``to_dict`` / ``from_dict`` round-trip the estimator by name."""

    def test_to_dict_carries_estimator_name(self) -> None:
        cfg = AnalysisConfig.individual_continuous(estimator=HansenHodrick())
        assert cfg.to_dict()["estimator"] == "HansenHodrick"

    def test_roundtrip_preserves_equality(self) -> None:
        cfg = AnalysisConfig.individual_continuous(estimator=HansenHodrick())
        assert AnalysisConfig.from_dict(cfg.to_dict()) == cfg

    def test_roundtrip_default_estimator(self) -> None:
        cfg = AnalysisConfig.individual_continuous()
        assert AnalysisConfig.from_dict(cfg.to_dict()) == cfg

    def test_legacy_dict_without_estimator_defaults_nw(self) -> None:
        legacy = {
            "scope": "individual",
            "signal": "continuous",
            "metric": "ic",
            "forward_periods": 5,
        }
        cfg = AnalysisConfig.from_dict(legacy)
        assert cfg.estimator == NeweyWest()

    def test_from_dict_unknown_estimator_raises(self) -> None:
        d = {
            "scope": "individual",
            "signal": "continuous",
            "metric": "ic",
            "forward_periods": 5,
            "estimator": "Bogus",
        }
        with pytest.raises(UnknownEstimatorError, match="unknown estimator 'Bogus'"):
            AnalysisConfig.from_dict(d)


class TestGetEstimator:
    """``factrix.stats.get_estimator`` registry lookup."""

    def test_lookup_newey_west(self) -> None:
        assert get_estimator("NeweyWest") == NeweyWest()

    def test_lookup_hansen_hodrick(self) -> None:
        assert get_estimator("HansenHodrick") == HansenHodrick()

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(UnknownEstimatorError) as exc_info:
            get_estimator("DoesNotExist")
        # Message lists every registered name for discoverability.
        msg = str(exc_info.value)
        assert "NeweyWest" in msg
        assert "HansenHodrick" in msg
        assert "WaldNWCluster" in msg
