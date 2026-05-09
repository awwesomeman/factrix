"""identity / context schema on FactorProfile (#160)."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix import AnalysisConfig, Metric, evaluate
from factrix._axis import Mode
from factrix._profile import FactorProfile


def _panel(
    *,
    factor_col: str = "momentum_12_1",
    n_dates: int = 60,
    n_assets: int = 20,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for i in range(n_dates):
        d = start + dt.timedelta(days=i)
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = 0.3 * fwd + 0.7 * noise
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    factor_col: float(factor[j]),
                    "forward_return": float(fwd[j]),
                }
            )
    return pl.DataFrame(rows)


def _cfg(forward_periods: int = 5) -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(
        metric=Metric.IC, forward_periods=forward_periods
    )


def test_identity_default_when_constructed_directly() -> None:
    p = FactorProfile(
        config=_cfg(),
        mode=Mode.PANEL,
        primary_p=0.04,
        n_obs=60,
        n_assets=20,
    )
    assert p.identity == ("factor", 0)
    assert dict(p.context) == {}
    assert p.factor_id == "factor"
    assert p.forward_periods == 0


def test_evaluate_stamps_identity_from_factor_col() -> None:
    panel = _panel()
    cfg = _cfg(forward_periods=5)
    profile = evaluate(panel, cfg, factor_col="momentum_12_1")
    assert profile.identity == ("momentum_12_1", 5)
    assert profile.factor_id == "momentum_12_1"
    assert profile.forward_periods == 5


def test_evaluate_default_factor_col_stamps_canonical_name() -> None:
    panel = _panel(factor_col="factor")
    profile = evaluate(panel, _cfg())
    assert profile.factor_id == "factor"


def test_context_empty_by_default() -> None:
    profile = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    assert dict(profile.context) == {}


def test_diagnose_includes_identity_and_context() -> None:
    profile = evaluate(_panel(), _cfg(forward_periods=7), factor_col="momentum_12_1")
    d = profile.diagnose()
    assert d["identity"] == {
        "factor_id": "momentum_12_1",
        "forward_periods": 7,
    }
    assert d["context"] == {}


def test_replace_preserves_identity_when_only_other_fields_change() -> None:
    p1 = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    p2 = dataclasses.replace(p1, primary_p=0.99)
    assert p2.identity == p1.identity
    assert dict(p2.context) == dict(p1.context)


def test_distinct_factor_cols_yield_distinct_identity() -> None:
    panel = _panel().with_columns(
        pl.lit(0.0).alias("noise"),
    )
    cfg = _cfg()
    p1 = evaluate(panel, cfg, factor_col="momentum_12_1")
    p2 = evaluate(panel, cfg, factor_col="noise")
    assert p1.identity != p2.identity
    assert p1.factor_id == "momentum_12_1"
    assert p2.factor_id == "noise"


@pytest.mark.parametrize("forward_periods", [1, 5, 21])
def test_identity_carries_config_forward_periods(forward_periods: int) -> None:
    profile = evaluate(
        _panel(n_dates=120),
        _cfg(forward_periods=forward_periods),
        factor_col="momentum_12_1",
    )
    assert profile.identity == ("momentum_12_1", forward_periods)


def test_context_can_be_extended_via_replace() -> None:
    p1 = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    p2 = dataclasses.replace(p1, context={"universe_id": "us_large_cap"})
    assert p2.context["universe_id"] == "us_large_cap"
    assert p2.identity == p1.identity


def test_repr_omits_empty_context_and_warnings() -> None:
    profile = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    text = repr(profile)
    assert text.startswith("FactorProfile(")
    assert "factor_id='momentum_12_1'" in text
    assert "primary_p=" in text
    assert "context=" not in text
    assert "warnings=" not in text


def test_repr_includes_context_when_set() -> None:
    p = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    p2 = dataclasses.replace(p, context={"universe_id": "us_large_cap"})
    text = repr(p2)
    assert "context=" in text
    assert "universe_id" in text


def test_repr_html_renders_identity_table() -> None:
    profile = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    html = profile._repr_html_()
    assert "<table" in html
    assert "</table>" in html
    assert "factor_id" in html
    assert "momentum_12_1" in html
    assert "primary_p" in html


def test_repr_html_lists_context_rows_when_set() -> None:
    p = evaluate(_panel(), _cfg(), factor_col="momentum_12_1")
    p2 = dataclasses.replace(
        p, context={"universe_id": "us_large", "regime_id": "low_vol"}
    )
    html = p2._repr_html_()
    assert "context.universe_id" in html
    assert "us_large" in html
    assert "context.regime_id" in html
