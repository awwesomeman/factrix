"""Public ``evaluate`` shim — friendly error on missing config (#72)."""

from __future__ import annotations

import datetime as dt

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._errors import ConfigError, FactrixError, MissingConfigError


def _build_panel(n_dates: int = 60, n_assets: int = 5, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for i in range(n_dates):
        d = start + dt.timedelta(days=i)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(rng.standard_normal()),
                    "forward_return": float(rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows)


def test_evaluate_without_config_raises_missing_config_error():
    panel = _build_panel()
    with pytest.raises(MissingConfigError):
        fx.evaluate(panel)


def test_missing_config_error_message_points_to_suggest_config_and_docs():
    panel = _build_panel()
    with pytest.raises(MissingConfigError) as exc_info:
        fx.evaluate(panel)
    msg = str(exc_info.value)
    assert "suggest_config" in msg
    assert "https://awwesomeman.github.io/factrix/getting-started/" in msg


def test_missing_config_error_is_catchable_as_config_and_factrix_error():
    panel = _build_panel()
    with pytest.raises(ConfigError):
        fx.evaluate(panel)
    with pytest.raises(FactrixError):
        fx.evaluate(panel)


def test_evaluate_with_config_routes_to_private_dispatcher():
    panel = _build_panel()
    cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
    profile = fx.evaluate(panel, cfg)
    assert isinstance(profile, fx.FactorProfile)


def test_missing_config_error_exported_from_top_level():
    assert fx.MissingConfigError is MissingConfigError
