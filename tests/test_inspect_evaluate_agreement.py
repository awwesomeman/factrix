"""Pre-flight verdict vs run-time outcome must agree, across panel shapes.

``inspect_data`` sells a per-metric usability verdict; ``evaluate`` then
either produces a value or refuses. The two read the same
:class:`~factrix._metric_index.SampleThreshold` and pre-flightable content
gates, so a disagreement is a bug in one of them — and a silent one, because
a caller who pre-filters on ``inspect_data(...).usable`` never sees the metric
fail.

The sweep is deliberately shape-driven: small universes (N as low as 5) are
the regime where the equity-calibrated defaults break, and short windows
crossed with an overlap horizon are where the effective (post-stride) sample
diverges from the raw date count.

``inspect_data`` pre-flights each metric at its **default configuration** (it
receives specs, not instances), so the sweep constructs default instances too;
a metric re-configured for a small universe (``monotonicity(n_groups=3)``) is
covered by its own module's tests.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import factrix as fx
import polars as pl
import pytest
from factrix import _detect_factor_cell
from factrix._metric_index import public_specs
from factrix.metrics._registry import REGISTRY
from factrix.preprocess import compute_forward_return

# Shapes: N spans the allocation regime (5) through a normal equity
# cross-section (40); T x h crosses a short window with an overlap horizon so
# the post-stride sample is exercised on both sides of its floor.
_N_ASSETS = (5, 8, 12, 20, 40)
_N_PERIODS = (60, 120, 240)
_HORIZONS = (1, 5)

_FactorShape = Literal["individual", "common_time_varying", "common_zero_variance"]
_FACTOR_SHAPES: tuple[_FactorShape, ...] = (
    "individual",
    "common_time_varying",
    "common_zero_variance",
)

# Metrics whose default constructor needs a positional argument (scalar
# helpers such as breakeven_cost / net_spread take an upstream value, not a
# panel) — there is no default instance to pre-flight.
_NO_DEFAULT_INSTANCE = frozenset({"breakeven_cost", "net_spread"})

# ``_dag._project_factor`` passes only the baseline + _OPTIONAL_COLUMNS schema
# through to a metric, so a weight column never reaches quantile_spread_vw
# under ``evaluate`` and it short-circuits on ``no_weight_column`` at every
# shape. That is a projection gap, not a sample-shape disagreement, and it is
# already excluded by the ``no_*`` rule below; named here so the exclusion is
# not mistaken for an untested metric.
_PROJECTION_GAP = frozenset({"quantile_spread_vw"})


def _panel(
    n_assets: int,
    n_periods: int,
    forward_periods: int,
    *,
    factor_shape: _FactorShape = "individual",
) -> pl.DataFrame:
    """Dense panel spanning individual and both common-factor regressions.

    ``market_cap`` is present so a weight-consuming metric fails (or not) on
    sample shape rather than on a missing column — the sweep tests the shape
    gate. The time-varying common control keeps stage-1 factor variation; the
    zero-variance shape forces every per-asset regression to drop, which is
    the producer/pre-flight disagreement the COMMON sweep must detect.
    """
    raw = fx.datasets.make_cs_panel(
        n_assets=n_assets, n_dates=n_periods, rng=17
    ).with_columns(pl.lit(1.0e9).alias("market_cap"))
    if factor_shape == "common_time_varying":
        raw = raw.with_columns(pl.col("factor").first().over("date"))
    elif factor_shape == "common_zero_variance":
        raw = raw.with_columns(pl.lit(1.0).alias("factor"))
    return compute_forward_return(raw, forward_periods=forward_periods)


def _shapes() -> list[tuple[int, int, int]]:
    return [(n, t, h) for n in _N_ASSETS for t in _N_PERIODS for h in _HORIZONS]


def _metric_names(_factor_shape: _FactorShape) -> list[str]:
    """Return the metrics belonging to this sample-shape invariant.

    Public ``role=METRIC`` specs are what ``inspect_data`` verdicts and
    ``evaluate`` accepts directly (PIPELINE producers are pulled via
    ``requires=`` and cannot be evaluated on their own). Every factor shape
    uses that full inventory: the zero-variance COMMON cells cover both the
    producer-survivor sample gate and the content contracts aligned in #1115
    and #1116.
    """
    return sorted({spec.name for _, spec in public_specs()} - _NO_DEFAULT_INSTANCE)


def _run(panel: pl.DataFrame, name: str, *, strict: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fx.evaluate(
            panel,
            metrics={name: REGISTRY[name]()},
            factor_cols=["factor"],
            strict=strict,
        )["factor"].metrics[name]


def _verdicts(panel: pl.DataFrame) -> dict[str, bool]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inspection = fx.inspect_data(panel, factor_cols=["factor"])
    return {m.name: m.usable for m in inspection.metrics}


def _is_out_of_scope(reason: object) -> bool:
    """True for a short-circuit outside what ``SampleThreshold`` models.

    A ``no_*`` reason (missing input column or config) is a schema gate and a
    ``not_applicable*`` reason is the type-routing verdict; neither belongs to
    the shared sample/content contract exercised here.
    """
    return isinstance(reason, str) and reason.startswith(("no_", "not_applicable"))


@pytest.mark.parametrize("n_assets,n_periods,forward_periods", _shapes())
@pytest.mark.parametrize("factor_shape", _FACTOR_SHAPES)
def test_inspect_verdict_matches_evaluate_outcome(
    n_assets: int,
    n_periods: int,
    forward_periods: int,
    factor_shape: _FactorShape,
) -> None:
    panel = _panel(n_assets, n_periods, forward_periods, factor_shape=factor_shape)
    usable = _verdicts(panel)

    disagreements: list[str] = []
    for name in _metric_names(factor_shape):
        if name not in usable:
            continue  # not a public METRIC spec (pipeline producer)
        try:
            out = _run(panel, name, strict=False)
        except fx.IncompatibleAxisError:
            continue  # cell mismatch, not a sample-shape verdict
        if _is_out_of_scope(out.metadata.get("reason")):
            continue
        ran = out.is_applicable and not math.isnan(out.value)
        if usable[name] != ran:
            disagreements.append(
                f"{name}: inspect_usable={usable[name]} ran={ran} "
                f"reason={out.metadata.get('reason')!r}"
            )
    assert not disagreements, (
        f"shape={factor_shape} "
        f"N={n_assets} T={n_periods} h={forward_periods}: " + "; ".join(disagreements)
    )


@pytest.mark.parametrize("n_assets,n_periods,forward_periods", _shapes())
@pytest.mark.parametrize("factor_shape", _FACTOR_SHAPES)
def test_strict_raises_exactly_when_inspect_says_unusable(
    n_assets: int,
    n_periods: int,
    forward_periods: int,
    factor_shape: _FactorShape,
) -> None:
    """``strict=True`` must refuse precisely the shapes pre-flight calls
    unusable, and the refusal must be the documented exception type carrying a
    legal axis token."""
    panel = _panel(n_assets, n_periods, forward_periods, factor_shape=factor_shape)
    usable = _verdicts(panel)
    legal_axes = {"periods", "assets", "events", "pairs", "asset_pairs"}

    disagreements: list[str] = []
    for name in _metric_names(factor_shape):
        if name not in usable:
            continue
        raised: str | None = None
        try:
            _run(panel, name, strict=True)
        except fx.InsufficientSampleError as exc:
            raised = "insufficient"
            assert exc.axis in legal_axes, (
                f"{name}: axis={exc.axis!r} outside SampleAxis"
            )
            assert exc.actual is None or exc.actual >= 0, f"{name}: actual={exc.actual}"
            assert exc.shortfalls, f"{name}: no shortfalls recorded"
        except fx.UserInputError:
            raised = "user_input"  # schema / missing input column, out of scope
        except fx.IncompatibleAxisError:
            continue
        if raised == "insufficient" and usable[name]:
            disagreements.append(f"{name}: pre-flight usable but strict refused")
        if raised is None and not usable[name]:
            disagreements.append(f"{name}: pre-flight unusable but strict ran")
    assert not disagreements, (
        f"shape={factor_shape} "
        f"N={n_assets} T={n_periods} h={forward_periods}: " + "; ".join(disagreements)
    )


def test_common_beta_preflight_uses_assets_surviving_the_producer() -> None:
    """The discovery bridge must not advertise a zero-variance beta sample."""
    raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120, rng=17)
    panel = compute_forward_return(
        raw.with_columns(pl.col("factor").mean().over("date").alias("macro")),
        forward_periods=5,
    )

    info = fx.inspect_data(panel, factor_cols=["macro"])
    beta_consumers = {
        "common_beta",
        "common_beta_profile",
        "common_beta_r_squared",
        "common_beta_sign_consistency",
    }

    assert info.properties.scope is fx.FactorScope.COMMON
    assert beta_consumers <= set(info.unusable.names)
    assert all(
        any("n_assets=0" in blocker for blocker in verdict.blockers)
        for verdict in info.unusable
        if verdict.name in beta_consumers
    )
    out = fx.evaluate(
        panel,
        metrics=info.usable.to_metrics_dict(),
        factor_cols=["macro"],
        forward_periods=5,
    )
    assert beta_consumers.isdisjoint(out["macro"].metrics)


def test_common_beta_preflight_keeps_time_varying_broadcast_factor() -> None:
    """Control: COMMON broadcasting alone keeps regressions usable.

    This passes before the #1074 fix by design; it pins the valid neighbouring
    regime while the zero-variance sweep cells provide regression force.
    """
    panel = _panel(20, 120, 5, factor_shape="common_time_varying")
    info = fx.inspect_data(panel, factor_cols=["factor"])
    beta_consumers = {
        "common_beta_profile",
        "common_beta_r_squared",
        "common_beta_sign_consistency",
    }

    assert beta_consumers <= set(info.usable.names)
    out = fx.evaluate(
        panel,
        metrics={name: REGISTRY[name]() for name in beta_consumers},
        factor_cols=["factor"],
        forward_periods=5,
    )
    assert all(out["factor"].metrics[name].is_applicable for name in beta_consumers)


def test_common_beta_preflight_warns_on_surviving_asset_count() -> None:
    """Warn tiers use producer survivors, not the wider raw universe."""
    raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=120, rng=17)
    retained = raw["asset_id"].unique().sort().head(5).to_list()
    panel = compute_forward_return(
        raw.with_columns(
            pl.col("factor").first().over("date").alias("macro")
        ).with_columns(
            pl.when(pl.col("asset_id").is_in(retained))
            .then(pl.col("macro"))
            .otherwise(None)
            .alias("macro")
        ),
        forward_periods=5,
    )

    info = fx.inspect_data(panel, factor_cols=["macro"])
    verdict = next(m for m in info.metrics if m.name == "common_beta")

    assert verdict.usable
    assert verdict in info.degraded
    assert [warning.code for warning in verdict.warnings] == [fx.WarningCode.FEW_ASSETS]


@pytest.mark.parametrize("n_assets,n_periods,forward_periods", _shapes())
def test_no_silent_nan_and_p_value_is_never_a_sentinel(
    n_assets: int, n_periods: int, forward_periods: int
) -> None:
    """Two invariants a screening library must not break.

    1. A finite value carries an honest ``p_value`` — in ``[0, 1]``, or ``None``
       for a descriptive metric. ``p_value=1.0`` is the short-circuit sentinel,
       so it may only ever appear alongside a NaN placeholder.
    2. A NaN placeholder is never silent: it carries a ``reason``,
       ``is_applicable=False``, and ``METRIC_UNAVAILABLE`` on its
       ``warning_codes``, so a caller scanning warnings cannot read a metric
       that never ran as a clean result.
    """
    panel = _panel(n_assets, n_periods, forward_periods)
    for name in _metric_names("individual"):
        try:
            out = _run(panel, name, strict=False)
        except fx.IncompatibleAxisError:
            continue  # metric's cell does not fit this panel at all
        where = f"{name} @ N={n_assets} T={n_periods} h={forward_periods}"
        if math.isnan(out.value):
            assert out.reason, f"{where}: silent NaN with no reason"
            assert not out.is_applicable, f"{where}: NaN marked applicable"
            assert fx.WarningCode.METRIC_UNAVAILABLE.value in out.warning_codes, (
                f"{where}: NaN placeholder carries no metric_unavailable code"
            )
        else:
            assert out.is_applicable, f"{where}: finite value marked inapplicable"
            if out.p_value is not None:
                assert 0.0 <= out.p_value <= 1.0, f"{where}: p={out.p_value}"
                assert out.reason is None, (
                    f"{where}: finite value carrying short-circuit reason "
                    f"{out.reason!r}"
                )


def _broadcast_panel_with_gap(
    n_assets: int = 20, n_periods: int = 120, forward_periods: int = 1
) -> pl.DataFrame:
    """Broadcast common factor with one asset's cell missing on one period.

    The gap is the #1055 shape: null is a missing observation, not a distinct
    factor value, so the panel is still common by period.
    """
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_periods, rng=17)
    one_per_period = raw.group_by("date").agg(pl.col("factor").first())
    data = raw.drop("factor").join(one_per_period, on="date").sort("date", "asset_id")
    gap_period = data["date"].min()
    gap_asset = data["asset_id"].min()
    data = data.with_columns(
        pl.when((pl.col("date") == gap_period) & (pl.col("asset_id") == gap_asset))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("factor"))
        .alias("factor")
    )
    return compute_forward_return(data, forward_periods=forward_periods)


def test_scope_routing_agrees_on_a_partly_missing_common_factor() -> None:
    """Pre-flight scope and the cell ``evaluate`` dispatches on are one verdict.

    Both read :func:`factrix._inspect._detect_scope`; this pins that they stay
    one detector *and* that the shared verdict is the correct one — a
    common-cell metric must actually run on a broadcast factor with a gap
    rather than be refused for a cell mismatch (#1055).
    """
    panel = _broadcast_panel_with_gap()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inspected = fx.inspect_data(panel, factor_cols=["factor"]).properties.scope
        dispatched, _, _ = _detect_factor_cell(panel, "factor")

    assert inspected is dispatched is fx.FactorScope.COMMON

    out = _run(panel, "common_beta", strict=False)
    assert out.is_applicable and not math.isnan(out.value)
