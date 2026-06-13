"""Estimator protocol surface tests.

Verify the runtime-checkable identity protocol shape so that the
slice-test path can rely on `isinstance(obj, Estimator)` and the two
required members (`name` / `description`).
"""

from __future__ import annotations

from factrix.stats import Estimator


class _Stub:
    """Minimal Estimator-conforming class for protocol surface checks."""

    @property
    def name(self) -> str:
        return "Stub"

    @property
    def description(self) -> str:
        return "Test stub."


class _MissingDescription:
    @property
    def name(self) -> str:
        return "x"


def test_runtime_checkable_accepts_conforming_stub() -> None:
    assert isinstance(_Stub(), Estimator)


def test_runtime_checkable_rejects_missing_member() -> None:
    assert not isinstance(_MissingDescription(), Estimator)


def test_member_calls_route_through_protocol() -> None:
    e: Estimator = _Stub()
    assert e.name == "Stub"
    assert e.description == "Test stub."
