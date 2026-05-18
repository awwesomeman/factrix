"""``run_metrics`` batch interface (#402) — IC-cell stage-1 cache + panel-direct metrics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import RunMetricsError, UserInputError
from factrix._metric_index import _AUTO_DISCOVER_EXCLUDED, spec_by_name
from factrix._run_metrics import (
    MetricsBundle,
    run_metrics,
)
from factrix._types import MetricOutput

_IC_FAMILY = frozenset({"ic", "ic_newey_west", "ic_ir"})


def _build_panel(
    *,
    n_dates: int = 80,
    n_assets: int = 25,
    seed: int = 0,
    factor_strength: float = 0.2,
    extra_factor: str | None = None,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = factor_strength * fwd + (1.0 - factor_strength) * noise
        extra = rng.standard_normal(n_assets) if extra_factor else None
        price = rng.uniform(50.0, 150.0, n_assets)
        for j in range(n_assets):
            row: dict[str, object] = {
                "date": d,
                "asset_id": f"A{j:03d}",
                "factor": float(factor[j]),
                "forward_return": float(fwd[j]),
                "price": float(price[j]),
            }
            if extra_factor and extra is not None:
                row[extra_factor] = float(extra[j])
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture
def panel() -> pl.DataFrame:
    return _build_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


# ---------------------------------------------------------------------------
# Default auto-discover (single factor list-of-1 default)
# ---------------------------------------------------------------------------


def test_auto_discover_runs_ic_family_and_panel_direct(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg)["factor"]
    assert _IC_FAMILY.issubset(set(bundle.metrics))
    assert "monotonicity" in bundle.metrics
    assert "turnover" in bundle.metrics


def test_auto_discover_skipped_carries_excluded_reasons(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg)["factor"]
    for name, reason in bundle.skipped.items():
        assert reason == _AUTO_DISCOVER_EXCLUDED[name]


def test_auto_discover_skipped_logs_one_summary(
    panel: pl.DataFrame,
    cfg: AnalysisConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="factrix.run_metrics"):
        run_metrics(panel, cfg)
    summaries = [r for r in caplog.records if r.name == "factrix.run_metrics"]
    assert len(summaries) == 1
    assert "skipped" in summaries[0].getMessage()


# ---------------------------------------------------------------------------
# Explicit metrics= subset
# ---------------------------------------------------------------------------


def test_explicit_subset_runs_only_named(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg, metrics=["ic", "ic_ir"])["factor"]
    assert set(bundle.metrics) == {"ic", "ic_ir"}
    assert dict(bundle.skipped) == {}


def test_explicit_unknown_raises_user_input_error(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    with pytest.raises(UserInputError) as exc_info:
        run_metrics(panel, cfg, metrics=["icc"])
    msg = str(exc_info.value)
    assert "ic" in msg


def test_explicit_excluded_raises_with_reason(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    with pytest.raises(UserInputError) as exc_info:
        run_metrics(panel, cfg, metrics=["fama_macbeth"])
    msg = str(exc_info.value)
    assert "compute_fm_betas" in msg


# ---------------------------------------------------------------------------
# MetricsBundle access pattern
# ---------------------------------------------------------------------------


def test_bundle_dict_style_access(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg, metrics=["ic"])["factor"]
    assert isinstance(bundle["ic"], MetricOutput)
    assert "ic" in bundle
    assert list(bundle) == ["ic"]
    assert len(bundle) == 1


def test_bundle_identity_and_default_context(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg, factor_cols=["factor"], metrics=["ic"])["factor"]
    assert bundle.identity == ("factor", cfg.forward_periods)
    assert dict(bundle.context) == {}


def test_bundle_is_unhashable(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg, metrics=["ic"])["factor"]
    with pytest.raises(TypeError):
        hash(bundle)


def test_bundle_repr_html_smoke(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg)["factor"]
    html = bundle._repr_html_()
    assert "MetricsBundle" in html
    assert "ic" in html
    assert "skipped" in html  # the bundle has skipped entries by default


# ---------------------------------------------------------------------------
# to_frame schema
# ---------------------------------------------------------------------------

_TO_FRAME_SCHEMA = {
    "factor_id": pl.String,
    "forward_periods": pl.Int64,
    "metric": pl.String,
    "value": pl.Float64,
    "stat": pl.Float64,
    "significance": pl.String,
    "p_value": pl.Float64,
    "short_circuit_reason": pl.String,
}


def test_to_frame_schema_is_stable(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg, metrics=["ic"])["factor"]
    frame = bundle.to_frame()
    assert dict(frame.schema) == _TO_FRAME_SCHEMA
    assert frame.height == 1


def test_to_frame_empty_bundle_keeps_schema() -> None:
    empty = MetricsBundle(identity=("f", 5))
    frame = empty.to_frame()
    assert dict(frame.schema) == _TO_FRAME_SCHEMA
    assert frame.height == 0


def test_to_frame_short_circuit_metric_appears_with_reason() -> None:
    sc_output = MetricOutput(
        name="ic",
        value=float("nan"),
        stat=None,
        significance="",
        metadata={"reason": "insufficient_ic_periods", "p_value": 1.0},
    )
    bundle = MetricsBundle(identity=("f", 5), metrics={"ic": sc_output})
    frame = bundle.to_frame()
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["short_circuit_reason"] == "insufficient_ic_periods"
    assert np.isnan(row["value"])


# ---------------------------------------------------------------------------
# Stage-1 cache
# ---------------------------------------------------------------------------


def test_stage1_compute_ic_called_once_for_ic_family(
    panel: pl.DataFrame, cfg: AnalysisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factrix.metrics as metrics_pkg

    counter = {"n": 0}
    real_compute_ic = metrics_pkg.compute_ic

    def counting_compute_ic(*args: object, **kwargs: object) -> object:
        counter["n"] += 1
        return real_compute_ic(*args, **kwargs)

    monkeypatch.setattr(metrics_pkg, "compute_ic", counting_compute_ic)
    run_metrics(
        panel,
        cfg,
        metrics=["ic", "ic_newey_west", "ic_ir"],
    )
    assert counter["n"] == 1


def test_stage1_compute_ic_called_once_across_factors(
    cfg: AnalysisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-factor batch: IC stage-1 runs ONE polars query across N factors."""
    import factrix.metrics as metrics_pkg

    panel = _build_panel(extra_factor="momentum")
    counter = {"n": 0}
    real_compute_ic = metrics_pkg.compute_ic

    def counting_compute_ic(*args: object, **kwargs: object) -> object:
        counter["n"] += 1
        return real_compute_ic(*args, **kwargs)

    monkeypatch.setattr(metrics_pkg, "compute_ic", counting_compute_ic)
    run_metrics(
        panel,
        cfg,
        factor_cols=["factor", "momentum"],
        metrics=["ic", "ic_newey_west", "ic_ir"],
    )
    assert counter["n"] == 1


# ---------------------------------------------------------------------------
# Error strategy A / B / C
# ---------------------------------------------------------------------------


def test_class_a_short_circuit_keeps_other_metrics(
    cfg: AnalysisConfig,
) -> None:
    # Tiny panel with too few periods for IC's non-overlap test.
    tiny = _build_panel(n_dates=5, n_assets=12)
    bundle = run_metrics(tiny, cfg, metrics=["ic", "monotonicity"])["factor"]
    ic_out = bundle["ic"]
    assert ic_out.metadata.get("reason", "").startswith(("insufficient", "no_"))
    assert "monotonicity" in bundle.metrics


def test_class_c_unexpected_exception_wrapped(
    panel: pl.DataFrame,
    cfg: AnalysisConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise KeyError("unexpected")

    monkeypatch.setattr("factrix.metrics.monotonicity", boom)
    with pytest.raises(RunMetricsError) as exc_info:
        run_metrics(panel, cfg, metrics=["monotonicity"])
    assert exc_info.value.metric_name == "monotonicity"
    assert exc_info.value.cell == "individual/continuous"
    assert exc_info.value.stage == "consumer"
    assert isinstance(exc_info.value.__cause__, KeyError)


# ---------------------------------------------------------------------------
# factor_cols dispatch
# ---------------------------------------------------------------------------


def test_factor_cols_user_named_stamps_identity(
    cfg: AnalysisConfig,
) -> None:
    panel = _build_panel().rename({"factor": "momentum_12_1"})
    bundle = run_metrics(panel, cfg, factor_cols=["momentum_12_1"], metrics=["ic"])[
        "momentum_12_1"
    ]
    assert bundle.identity == ("momentum_12_1", cfg.forward_periods)


def test_factor_cols_missing_raises_user_input_error(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    with pytest.raises(UserInputError):
        run_metrics(panel, cfg, factor_cols=["nonexistent"], metrics=["ic"])


def test_missing_forward_return_raises_actionable_error(
    cfg: AnalysisConfig,
) -> None:
    panel_no_fwd = _build_panel().drop("forward_return")
    with pytest.raises(UserInputError) as exc_info:
        run_metrics(panel_no_fwd, cfg, metrics=["ic"])
    assert "compute_forward_return" in str(exc_info.value)


def test_empty_metrics_list_rejected(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    with pytest.raises(UserInputError) as exc_info:
        run_metrics(panel, cfg, metrics=[])
    assert "non-empty" in str(exc_info.value)


def test_empty_factor_cols_rejected(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    with pytest.raises(UserInputError):
        run_metrics(panel, cfg, factor_cols=[], metrics=["ic"])


def test_duplicate_factor_cols_rejected(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    with pytest.raises(UserInputError):
        run_metrics(panel, cfg, factor_cols=["factor", "factor"], metrics=["ic"])


# ---------------------------------------------------------------------------
# Multi-factor equivalence (fail-loud regression guard)
# ---------------------------------------------------------------------------


def _assert_output_equal(
    a: MetricOutput, b: MetricOutput, *, where: tuple[str, ...]
) -> None:
    if np.isnan(a.value):
        assert np.isnan(b.value), where
    else:
        assert a.value == pytest.approx(b.value), where
    if a.stat is None:
        assert b.stat is None, where
    else:
        assert b.stat is not None, where
        assert a.stat == pytest.approx(b.stat), where
    assert a.significance == b.significance, where
    assert a.metadata.get("p_value") == pytest.approx(b.metadata.get("p_value")), where


def test_batch_of_one_matches_default(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle_default = run_metrics(panel, cfg, metrics=["ic", "monotonicity"])["factor"]
    bundle_listed = run_metrics(
        panel, cfg, factor_cols=["factor"], metrics=["ic", "monotonicity"]
    )["factor"]
    assert bundle_default.identity == bundle_listed.identity
    for name in bundle_default.metrics:
        _assert_output_equal(
            bundle_default[name], bundle_listed[name], where=("default-vs-listed", name)
        )


def test_batch_of_n_matches_list_of_one_per_factor(
    cfg: AnalysisConfig,
) -> None:
    panel = _build_panel(extra_factor="momentum")
    batch = run_metrics(
        panel,
        cfg,
        factor_cols=["factor", "momentum"],
        metrics=["ic", "monotonicity"],
    )
    for c in ("factor", "momentum"):
        solo = run_metrics(panel, cfg, factor_cols=[c], metrics=["ic", "monotonicity"])[
            c
        ]
        for name in ("ic", "monotonicity"):
            _assert_output_equal(batch[c][name], solo[name], where=(c, name))


# ---------------------------------------------------------------------------
# Marker-driven batch dispatch (#418)
# ---------------------------------------------------------------------------


def test_known_batchable_specs() -> None:
    """The post-#401 batch primitives declare batchable=True on the spec."""
    specs = spec_by_name()
    assert specs["quantile_spread"].batchable
    assert specs["monotonicity"].batchable


def test_non_batch_metric_not_batchable() -> None:
    """Non-batch panel-direct metrics declare batchable=False (default)."""
    specs = spec_by_name()
    assert not specs["top_concentration"].batchable
    assert not specs["turnover"].batchable


def test_ic_family_requires_compute_ic() -> None:
    """The IC consumer family declares requires={"ic_df": compute_ic}."""
    import factrix.metrics as metrics_pkg

    specs = spec_by_name()
    for name in ("ic", "ic_newey_west", "ic_ir"):
        assert specs[name].requires.get("ic_df") is metrics_pkg.compute_ic


def test_batch_primitive_dispatched_once_across_factors(
    cfg: AnalysisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics marked @batch_primitive get one batched call across factors."""
    import functools

    import factrix.metrics as metrics_pkg

    panel = _build_panel(extra_factor="momentum")
    counter = {"n": 0}
    real = metrics_pkg.quantile_spread

    @functools.wraps(real)
    def counting(*args: object, **kwargs: object) -> object:
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics_pkg, "quantile_spread", counting)
    run_metrics(
        panel,
        cfg,
        factor_cols=["factor", "momentum"],
        metrics=["quantile_spread"],
    )
    assert counter["n"] == 1


def test_non_batch_metric_dispatched_per_factor(
    cfg: AnalysisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics without @batch_primitive get one call per factor on a projected panel."""
    import functools

    import factrix.metrics as metrics_pkg

    panel = _build_panel(extra_factor="momentum")
    counter = {"n": 0}
    real = metrics_pkg.turnover

    @functools.wraps(real)
    def counting(*args: object, **kwargs: object) -> object:
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics_pkg, "turnover", counting)
    run_metrics(
        panel,
        cfg,
        factor_cols=["factor", "momentum"],
        metrics=["turnover"],
    )
    assert counter["n"] == 2
