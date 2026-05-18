"""Marker-driven dispatch protocol (#418).

The dispatcher must route by explicit ``@batch_primitive`` /
``@ic_consumer`` markers, not by accident of parameter naming
or central frozenset membership. These regression tests pin the
two failure modes the markers prevent:

1. A metric whose signature happens to include ``factor_cols`` but
   is **not** marked ``@batch_primitive`` must take the per-factor
   path — adding such a parameter on a non-batch metric should not
   silently promote it to the batch path.
2. ``@batch_primitive`` applied to a function whose signature lacks
   ``factor_cols`` must fail at decoration time — the marker and the
   signature must agree, no silent disagreement.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._run_metrics import run_metrics
from factrix._types import MetricOutput
from factrix.metrics._dispatch import (
    batch_primitive,
    ic_consumer,
    is_batch_primitive,
    is_ic_consumer,
)


def _panel(n_dates: int = 60, n_assets: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        factor = 0.3 * fwd + 0.7 * rng.standard_normal(n_assets)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[j]),
                    "forward_return": float(fwd[j]),
                }
            )
    return pl.DataFrame(rows)


class TestMarkerSignatureAgreement:
    def test_batch_primitive_rejects_signature_without_factor_cols(self) -> None:
        # Applying @batch_primitive to a function whose signature does
        # not accept factor_cols must fail loud at decoration time.
        with pytest.raises(TypeError, match="factor_cols"):

            @batch_primitive
            def bogus(df: pl.DataFrame, forward_periods: int = 5) -> MetricOutput: ...

    def test_batch_primitive_accepts_signature_with_factor_cols(self) -> None:
        @batch_primitive
        def good(df: pl.DataFrame, *, factor_cols: list[str]) -> dict:
            return {c: 1 for c in factor_cols}

        assert is_batch_primitive(good)

    def test_ic_consumer_marker_is_unconditional(self) -> None:
        # ic_consumer does not constrain the signature — IC consumers
        # take whatever shape suits the underlying inference call.
        @ic_consumer
        def custom(ic_df: pl.DataFrame, **kwargs) -> MetricOutput: ...

        assert is_ic_consumer(custom)


class TestDispatcherHonoursMarkers:
    def test_unmarked_factor_cols_param_routes_per_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a fake metric whose signature accepts factor_cols but
        # has NO @batch_primitive marker. Dispatcher must take the
        # per-factor projected path (one call per factor), not the
        # batch path. Without explicit markers, signature-only
        # detection would silently route it as batch and the test
        # would only catch it via perf regression.
        import factrix.metrics as metrics_pkg

        counter = {"n": 0, "args_per_call": []}

        def fake(
            df: pl.DataFrame, *, factor_cols: list[str] | None = None
        ) -> MetricOutput:
            counter["n"] += 1
            counter["args_per_call"].append(factor_cols)
            return MetricOutput(name="fake", value=float("nan"))

        # NOT decorated — assert that fact, then monkeypatch in.
        assert not is_batch_primitive(fake)
        monkeypatch.setattr(metrics_pkg, "fake_unmarked", fake, raising=False)

        # Need to also patch the candidate-name resolver so the
        # dispatcher accepts our fake name. Use the explicit
        # `metrics=[...]` path so auto-discovery doesn't drop it.
        from factrix import _run_metrics as rm

        panel = _panel()
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        monkeypatch.setattr(
            rm, "_candidate_metric_names", lambda *a, **kw: ["fake_unmarked"]
        )
        run_metrics(panel, cfg, factor_cols=["factor"], metrics=["fake_unmarked"])

        # One factor → per-factor path → one call.
        assert counter["n"] == 1
        # The dispatcher must NOT have passed factor_cols on the per-
        # factor path (the fake's factor_cols default kicks in).
        assert counter["args_per_call"] == [None]
