"""Shared fixtures for bench tests.

`tiny` preset scales trip factrix's "median assets per group" /
small-sample warnings. The harness intentionally runs under-spec
sizes for CI smoke; these warnings are not the work under test.
"""

from __future__ import annotations

import warnings

import pytest


@pytest.fixture(autouse=True)
def _silence_sample_threshold_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield
