"""Cross-type invariant property tests — spec ↔ result ↔ applicability.

The three public type families that describe a metric must stay
mutually consistent, no matter which metric is added:

- **spec** — :class:`~factrix._metric_index.MetricSpec`, the single
  source of truth, projected from each ``@metric`` class's ClassVars
  via :meth:`MetricBase.spec`.
- **applicability** — :class:`~factrix._inspect.MetricApplicability`,
  the per-data pre-flight verdict :func:`factrix.inspect_data`
  attaches to every public spec.
- **result** — :class:`~factrix._results.MetricResult`, what
  :func:`factrix.evaluate` produces for an applicable metric.

Each test asserts a *property* that must hold for **every** registered
metric (or every produced result), not a single hand-picked case — so a
new metric that drifts one type out of sync with the other two trips a
failure here rather than silently shipping. The registry / public-spec
walk is the closed set the properties quantify over.

Companion to ``tests/test_docs_matrix.py`` (spec ↔ docs) and
``tests/test_inspect_data.py`` (applicability units); this file owns the
*cross-type* seam those two do not.
"""

from __future__ import annotations

import factrix as fx
import polars as pl
import pytest
from factrix._axis import OutputShape, SpecRole
from factrix._inspect import DataInspection, inspect_data
from factrix._metric_index import public_specs, spec_by_name
from factrix.metrics import (
    fm_beta,
    ic,
    ic_ir,
    monotonicity,
    quantile_spread,
    top_concentration,
    turnover,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._registry import REGISTRY

# ---------------------------------------------------------------------------
# Shared registry view + panel fixtures
# ---------------------------------------------------------------------------

# Frozen at import: the closed set every "for every metric" property
# quantifies over. ``id=`` keys parametrized failures by metric name.
_REGISTRY_CLASSES = sorted(REGISTRY.values(), key=lambda c: c.__name__)
_REGISTRY_IDS = [c.__name__ for c in _REGISTRY_CLASSES]

# A robust core of standard panel metrics for the result-side seam: each
# runs on a plain cross-sectional panel with no special upstream-data
# shape, mirroring ``tests/test_inspect_data.py``'s discovery-bridge test.
# Selection / spanning metrics that need a multi-factor pool or a
# post-pipeline return series are exercised by their own suites, not here.
_CORE_METRICS = {
    "ic": ic(),
    "ic_ir": ic_ir(),
    "fm_beta": fm_beta(),
    "monotonicity": monotonicity(),
    "quantile_spread": quantile_spread(),
    "turnover": turnover(),
    "top_concentration": top_concentration(),
}


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    """A standard INDIVIDUAL / DENSE / PANEL data with forward returns."""
    raw = fx.datasets.make_cs_panel(n_assets=25, n_dates=120)
    return fx.preprocess.compute_forward_return(raw, forward_periods=5)


@pytest.fixture(scope="module")
def timeseries_panel() -> pl.DataFrame:
    """A single-asset (TIMESERIES) data — flips the structure axis."""
    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=90)
    first = raw["asset_id"].unique()[0]
    return raw.filter(pl.col("asset_id") == first)


def _partition_ids(info: DataInspection) -> tuple[set[int], set[int], set[int]]:
    """Identity sets of the usable / degraded / unusable partitions."""
    return (
        {id(m) for m in info.usable},
        {id(m) for m in info.degraded},
        {id(m) for m in info.unusable},
    )


# ---------------------------------------------------------------------------
# 1. Spec is a faithful projection of the metric class (class ↔ spec)
# ---------------------------------------------------------------------------


class TestSpecMirrorsClass:
    """``MetricSpec`` must round-trip from the ``@metric`` class it describes."""

    @pytest.mark.parametrize("cls", _REGISTRY_CLASSES, ids=_REGISTRY_IDS)
    def test_spec_name_is_class_name(self, cls: type[MetricBase]) -> None:
        # The name is the join key across all three type families — it must
        # equal the class name so callers can go class → spec → result by name.
        assert cls.spec().name == cls.__name__

    @pytest.mark.parametrize("cls", _REGISTRY_CLASSES, ids=_REGISTRY_IDS)
    def test_spec_projects_classvars(self, cls: type[MetricBase]) -> None:
        spec = cls.spec()
        assert spec.cell == cls.cell
        assert spec.aggregation == cls.aggregation
        assert spec.input_shape == cls.input_shape
        assert spec.output_shape == cls.output_shape
        assert spec.role == cls.role
        assert spec.requires == cls.requires
        assert spec.batchable == cls.batchable
        if cls.sample_threshold_for is None:
            assert spec.sample_threshold == cls.sample_threshold
        else:
            # Dynamic floor: spec() resolves the hook against a default-built
            # instance, so the spec carries a concrete threshold rather than
            # the empty static placeholder.
            assert spec.sample_threshold == cls().sample_threshold_for()

    @pytest.mark.parametrize("cls", _REGISTRY_CLASSES, ids=_REGISTRY_IDS)
    def test_metric_role_implies_scalar_output(self, cls: type[MetricBase]) -> None:
        # The ``__post_init__`` guard, asserted as a property over the whole
        # registry: a user-facing METRIC always yields a SCALAR, so every
        # result key resolves to a scalar-valued spec.
        spec = cls.spec()
        if spec.role is SpecRole.METRIC:
            assert spec.output_shape is OutputShape.SCALAR

    @pytest.mark.parametrize("cls", _REGISTRY_CLASSES, ids=_REGISTRY_IDS)
    def test_spec_by_name_round_trips(self, cls: type[MetricBase]) -> None:
        # The registry-derived ``spec_by_name`` view must agree with the
        # class's own projection — one source of truth, two access paths.
        assert spec_by_name()[cls.__name__] == cls.spec()

    @pytest.mark.parametrize("cls", _REGISTRY_CLASSES, ids=_REGISTRY_IDS)
    def test_requires_producer_output_feeds_consumer_input(
        self, cls: type[MetricBase]
    ) -> None:
        # DAG wiring invariant: every upstream producer named in ``requires``
        # emits the shape the consumer declares it ingests, so the executor
        # can inject the producer's output at that kwarg without a reshape.
        sbn = spec_by_name()
        spec = cls.spec()
        for key, producer in spec.requires.items():
            producer_spec = sbn.get(producer.__name__)
            assert producer_spec is not None, (
                f"{cls.__name__}.requires[{key!r}] -> {producer.__name__} "
                f"has no registered spec"
            )
            assert producer_spec.output_shape.value == spec.input_shape.value, (
                f"{cls.__name__}.requires[{key!r}]: producer outputs "
                f"{producer_spec.output_shape.value!r} but consumer ingests "
                f"{spec.input_shape.value!r}"
            )


# ---------------------------------------------------------------------------
# 2. Applicability mirrors the spec it evaluates (spec ↔ applicability)
# ---------------------------------------------------------------------------


class TestApplicabilityMirrorsSpec:
    """Every ``inspect_data`` verdict must agree with the spec it carries."""

    def test_verdict_identity_fields_align(self, panel: pl.DataFrame) -> None:
        # name / spec / metric on a verdict are three views of one metric;
        # they must point at the same registered class.
        for m in inspect_data(panel).metrics:
            assert m.name == m.spec.name == m.metric.__name__
            assert m.spec == m.metric.spec()

    def test_only_metric_role_specs_get_a_verdict(self, panel: pl.DataFrame) -> None:
        # PIPELINE producers are pulled by the DAG, never surfaced as
        # user-facing applicability — so no verdict carries a non-METRIC spec.
        for m in inspect_data(panel).metrics:
            assert m.spec.role is SpecRole.METRIC

    def test_verdicts_cover_exactly_the_public_metric_specs(
        self, panel: pl.DataFrame
    ) -> None:
        verdict_names = {m.name for m in inspect_data(panel).metrics}
        public_names = {spec.name for _, spec in public_specs()}
        assert verdict_names == public_names

    @pytest.mark.parametrize("panel_name", ["panel", "timeseries_panel"])
    def test_partition_is_exclusive_and_exhaustive(
        self, panel_name: str, request: pytest.FixtureRequest
    ) -> None:
        # usable / degraded / unusable must tile ``metrics`` with no overlap
        # and no gaps — the property the three @property accessors promise,
        # checked on both a PANEL and a TIMESERIES data (varied axes).
        info: DataInspection = inspect_data(request.getfixturevalue(panel_name))
        usable, degraded, unusable = _partition_ids(info)
        everything = {id(m) for m in info.metrics}
        assert usable.isdisjoint(degraded)
        assert degraded.isdisjoint(unusable)
        assert usable.isdisjoint(unusable)
        assert usable | degraded | unusable == everything

    def test_usable_iff_no_blockers(self, panel: pl.DataFrame) -> None:
        for m in inspect_data(panel).metrics:
            assert m.usable == (not m.blockers)

    def test_partition_membership_matches_flags(self, panel: pl.DataFrame) -> None:
        info = inspect_data(panel)
        assert all(m.usable and not m.warnings for m in info.usable)
        assert all(m.usable and m.warnings for m in info.degraded)
        assert all(not m.usable for m in info.unusable)

    @pytest.mark.parametrize("panel_name", ["panel", "timeseries_panel"])
    def test_usable_specs_match_detected_cell(
        self, panel_name: str, request: pytest.FixtureRequest
    ) -> None:
        # A metric is only usable when its cell admits the detected axes —
        # the spec.cell ↔ detected-properties seam inspect_data enforces.
        info: DataInspection = inspect_data(request.getfixturevalue(panel_name))
        d = info.detected
        for m in info.usable:
            assert m.spec.cell.matches(d.scope, d.density, d.structure)


# ---------------------------------------------------------------------------
# 3. Result mirrors spec + applicability (applicability ↔ result ↔ spec)
# ---------------------------------------------------------------------------


class TestResultMirrorsSpecAndApplicability:
    """Every ``evaluate`` output must trace back to an applicable spec."""

    @pytest.fixture(scope="class")
    def evaluated(self, request: pytest.FixtureRequest):
        panel = request.getfixturevalue("panel")
        results = fx.evaluate(
            panel,
            metrics=_CORE_METRICS,
            factor_cols=["factor"],
            forward_periods=5,
        )
        return results["factor"]

    def test_outputs_are_all_metric_labels(self, evaluated) -> None:
        # All requested metric labels appear in outputs.
        assert set(evaluated.metrics) == set(_CORE_METRICS)

    def test_result_keys_resolve_to_metric_specs(self, evaluated) -> None:
        # Result dict keys are a join key back into the spec table; each must
        # resolve to a real, user-facing METRIC spec (never a PIPELINE node).
        sbn = spec_by_name()
        for key in evaluated.metrics:
            assert key in sbn
            assert sbn[key].role is SpecRole.METRIC

    def test_result_name_matches_its_key(self, evaluated) -> None:
        for key, out in evaluated.metrics.items():
            assert out.name == key

    def test_metric_outputs_are_scalar(self, evaluated) -> None:
        # role=METRIC ⟹ output_shape=SCALAR (TestSpecMirrorsClass), so every
        # produced value is a plain float and any p-value is a probability.
        for out in evaluated.metrics.values():
            assert isinstance(out.value, float)
            assert out.p_value is None or 0.0 <= out.p_value <= 1.0

    def test_evaluated_metrics_were_deemed_applicable(
        self, evaluated, panel: pl.DataFrame
    ) -> None:
        # The applicability → result link: nothing evaluate produced was
        # flagged unusable by inspect_data for the same data.
        info = inspect_data(panel)
        runnable = set(info.usable.names) | set(info.degraded.names)
        assert set(evaluated.metrics) <= runnable
