"""``run_metrics`` v1 — IC-cell stage-1 cache + panel-direct metrics (#147)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._errors import RunMetricsError, UserInputError
from factrix._metric_index import _AUTO_DISCOVER_EXCLUDED
from factrix._run_metrics import (
    _IC_CONSUMERS,
    MetricsBundle,
    run_metrics,
)
from factrix._types import MetricOutput


def _build_panel(
    *,
    n_dates: int = 80,
    n_assets: int = 25,
    seed: int = 0,
    factor_strength: float = 0.2,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = factor_strength * fwd + (1.0 - factor_strength) * noise
        price = rng.uniform(50.0, 150.0, n_assets)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[j]),
                    "forward_return": float(fwd[j]),
                    "price": float(price[j]),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def panel() -> pl.DataFrame:
    return _build_panel()


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_continuous(metric=Metric.IC)


# ---------------------------------------------------------------------------
# Default auto-discover
# ---------------------------------------------------------------------------


def test_auto_discover_runs_ic_family_and_panel_direct(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg)
    assert _IC_CONSUMERS.issubset(set(bundle.metrics))
    assert "monotonicity" in bundle.metrics
    assert "turnover" in bundle.metrics


def test_auto_discover_skipped_carries_excluded_reasons(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg)
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
    bundle = run_metrics(panel, cfg, metrics=["ic", "ic_ir"])
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
    bundle = run_metrics(panel, cfg, metrics=["ic"])
    assert isinstance(bundle["ic"], MetricOutput)
    assert "ic" in bundle
    assert list(bundle) == ["ic"]
    assert len(bundle) == 1


def test_bundle_identity_and_default_context(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    bundle = run_metrics(panel, cfg, factor_col="factor", metrics=["ic"])
    assert bundle.identity == ("factor", cfg.forward_periods)
    assert dict(bundle.context) == {}


def test_bundle_is_unhashable(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg, metrics=["ic"])
    with pytest.raises(TypeError):
        hash(bundle)


def test_bundle_repr_html_smoke(panel: pl.DataFrame, cfg: AnalysisConfig) -> None:
    bundle = run_metrics(panel, cfg)
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
    bundle = run_metrics(panel, cfg, metrics=["ic"])
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

    def counting_compute_ic(*args: object, **kwargs: object) -> pl.DataFrame:
        counter["n"] += 1
        return real_compute_ic(*args, **kwargs)

    monkeypatch.setattr(metrics_pkg, "compute_ic", counting_compute_ic)
    run_metrics(
        panel,
        cfg,
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
    bundle = run_metrics(tiny, cfg, metrics=["ic", "monotonicity"])
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
# factor_col rename path
# ---------------------------------------------------------------------------


def test_factor_col_renames_and_stamps_identity(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    renamed = panel.rename({"factor": "momentum_12_1"})
    bundle = run_metrics(renamed, cfg, factor_col="momentum_12_1", metrics=["ic"])
    assert bundle.identity == ("momentum_12_1", cfg.forward_periods)


def test_factor_col_missing_raises_user_input_error(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    with pytest.raises(UserInputError):
        run_metrics(panel, cfg, factor_col="nonexistent", metrics=["ic"])


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


def test_factor_col_collision_raises_user_input_error(
    panel: pl.DataFrame, cfg: AnalysisConfig
) -> None:
    # both 'factor' (already present) and a new candidate column
    panel_with_alias = panel.with_columns(pl.col("factor").alias("alt"))
    with pytest.raises(UserInputError):
        run_metrics(panel_with_alias, cfg, factor_col="alt", metrics=["ic"])
