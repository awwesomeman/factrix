"""Smoke tests for the shared logger infrastructure."""

from __future__ import annotations

import logging

from factrix._logging import (
    get_evaluation_logger,
    get_metrics_logger,
)


def test_loggers_have_expected_names() -> None:
    assert get_evaluation_logger().name == "factrix.evaluation"
    assert get_metrics_logger().name == "factrix.metrics"


def test_loggers_silent_by_default() -> None:
    """Library import must not emit records unless user configures handlers.

    The NullHandler makes ``logger.warning(...)`` a no-op when the root
    logger has no handlers either. We verify by asserting at least one
    NullHandler is attached to each logger.
    """
    for logger in (get_evaluation_logger(), get_metrics_logger()):
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers), (
            f"{logger.name} must have a NullHandler attached"
        )


def test_loggers_independently_configurable() -> None:
    """Users can raise one logger's level without touching the other."""
    ev = get_evaluation_logger()
    mt = get_metrics_logger()
    prior_ev, prior_mt = ev.level, mt.level
    try:
        mt.setLevel(logging.WARNING)
        ev.setLevel(logging.DEBUG)

        assert ev.isEnabledFor(logging.DEBUG)
        assert mt.isEnabledFor(logging.WARNING)
        assert not mt.isEnabledFor(logging.DEBUG)
    finally:
        ev.setLevel(prior_ev)
        mt.setLevel(prior_mt)
