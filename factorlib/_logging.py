"""Shared loggers for factorlib.

Two logger namespaces, by layer of responsibility:

- ``factorlib.evaluation`` — orchestration / decision layer. Emits one-line
  INFO per explicit user decision (e.g. ``multiple_testing_correct`` call,
  ``verdict()`` returning ``PASS_WITH_WARNINGS``), and WARNING when a
  diagnose rule fires but the user has not switched ``p_source``.
- ``factorlib.metrics`` — per-metric correction layer. Emits DEBUG on every
  public metric call (sampling interval, NW lags, n_before/n_after), and
  WARNING when a correction produces a degenerate sample (e.g. non-overlap
  shrinks below ``min_required * 1.5``, or NW lag ≥ T / 5).

Callers can silence either layer independently, e.g.::

    import logging
    logging.getLogger("factorlib.metrics").setLevel(logging.WARNING)

Both loggers attach a ``NullHandler`` so library import is silent by
default — applications opt in with ``logging.basicConfig`` or by adding
their own handlers.
"""

from __future__ import annotations

import logging

_EVALUATION_LOGGER_NAME = "factorlib.evaluation"
_METRICS_LOGGER_NAME = "factorlib.metrics"


def get_evaluation_logger() -> logging.Logger:
    """Orchestration-layer logger (BHY, verdict, diagnose warnings)."""
    return logging.getLogger(_EVALUATION_LOGGER_NAME)


def get_metrics_logger() -> logging.Logger:
    """Per-metric-correction layer logger (sampling, HAC, ADF, ...)."""
    return logging.getLogger(_METRICS_LOGGER_NAME)


for _name in (_EVALUATION_LOGGER_NAME, _METRICS_LOGGER_NAME):
    _logger = logging.getLogger(_name)
    if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
        _logger.addHandler(logging.NullHandler())
