"""Regression checks for issue #726 routing guidance."""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_sparse_event_docs_cover_single_asset_and_magnitude_contracts() -> None:
    text = _read("docs/api/metrics/individual-sparse.md")
    assert "At `n_assets == 1`, sparse factors still use the event-density path" in text
    assert "`{-R, 0, +R}` event magnitude" in text
    assert "signed_car = forward_return * sign(factor)" in text
    assert "Always-in-market `{-1, +1}` signals are not sparse event signals" in text


def test_metrics_index_keeps_strategy_metrics_out_of_evaluate() -> None:
    text = _read("docs/api/metrics/index.md")
    assert "What stays out of `evaluate()`" in text
    assert "strategy return metrics" in text
    assert "execution-cost metrics" in text
    assert "threshold sweeps" in text
    assert "walk-forward optimization" in text
