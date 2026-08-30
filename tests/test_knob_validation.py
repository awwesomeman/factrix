"""Metric knobs are validated at construction, at one site.

Every knob contract — a closed set annotated with a ``Literal``, an
``inference=`` allowlist, a numeric bound declared as ``@metric(validate=...)``
— is enforced by :meth:`MetricBase.__post_init__`, so a bad value raises where
the caller wrote it rather than several metrics into an ``evaluate``. Both call
forms reach the same site: ``metric(n_groups=1)`` is plain dataclass
instantiation and ``metric(data, n_groups=1)`` builds the instance first.

The tables below are the one place that enumerates the knobs, so a new closed
set or bound that skips the constructor shows up as a missing row rather than
as a silently wrong number.
"""

from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Any, get_args

import polars as pl
import pytest
from factrix import UserInputError
from factrix._errors import IncompatibleInferenceError
from factrix.datasets import make_cs_panel
from factrix.metrics._registry import REGISTRY
from factrix.metrics.concentration import top_concentration
from factrix.metrics.k_spread import k_spread
from factrix.metrics.monotonicity import monotonicity
from factrix.metrics.quantile import quantile_spread, quantile_spread_vw
from factrix.preprocess import compute_forward_return
from factrix.preprocess.normalize import Center, cross_sectional_zscore

BAD_TOKEN = "bogus"

#: Stand-in first argument for the direct-call form. Knob validation runs in
#: the constructor, before the body ever touches the data, so the parity checks
#: below never need a real panel.
_UNUSED_DATA = pl.DataFrame()


def _metrics_with_literal_fields() -> list[tuple[str, type, str, object]]:
    """Every registered metric x every ``Literal``-annotated knob it carries."""
    return [
        (f"{name}.{field}", cls, field, alias)
        for name, cls in sorted(REGISTRY.items())
        for field, alias in cls._literal_fields
    ]


LITERAL_KNOBS = _metrics_with_literal_fields()

# (metric name, knob, an out-of-bounds value) — one row per bound a
# ``@metric(validate=...)`` hook enforces. ``_HOOK_COVERAGE`` below pins the
# table against the registry so a new hook cannot land untested.
BOUNDED_KNOBS: list[tuple[str, dict[str, Any], str]] = [
    ("quantile_spread", {"n_groups": 1}, "n_groups"),
    ("quantile_spread", {"factor_cols": []}, "factor_cols"),
    ("quantile_spread_vw", {"n_groups": 1}, "n_groups"),
    ("monotonicity", {"n_groups": 1}, "n_groups"),
    ("monotonicity", {"n_resamples": 0}, "n_resamples"),
    ("monotonicity", {"factor_cols": ()}, "factor_cols"),
    ("top_concentration", {"q_top": 1.0}, "q_top"),
    ("k_spread", {"k": 0}, "k"),
    ("common_quantile_spread", {"n_groups": 1}, "n_groups"),
    ("rank_turnover", {"rebalance_lag": 0}, "rebalance_lag"),
    ("rank_turnover", {"quantile": 0.5}, "quantile"),
    ("notional_turnover", {"n_groups": 1}, "n_groups"),
    ("notional_turnover", {"rebalance_lag": 0}, "rebalance_lag"),
    # ``turnover`` is a required field on the two cost metrics, not a knob
    # under test; it is supplied so the constructor can be reached at all.
    ("breakeven_cost", {"turnover": 0.2, "holding_periods": 0}, "holding_periods"),
    ("net_spread", {"turnover": 0.2, "holding_periods": 0}, "holding_periods"),
    ("oos_decay", {"is_ratio": 1.0}, "is_ratio"),
    ("ic_trend", {"adf_threshold": 1.5}, "adf_threshold"),
    ("predictive_beta", {"adf_threshold": 0.0}, "adf_threshold"),
    ("common_beta_profile", {"neutral_epsilon": -1.0}, "neutral_epsilon"),
    (
        "pooled_beta",
        {"driscoll_kraay": True, "two_way_cluster_col": "asset_id"},
        "driscoll_kraay",
    ),
    ("compute_spread_series", {"n_groups": 1}, "n_groups"),
    ("compute_spread_series", {"factor_cols": []}, "factor_cols"),
    ("compute_group_returns", {"n_groups": 1}, "n_groups"),
    ("compute_fm_betas", {"factor_cols": []}, "factor_cols"),
    ("compute_common_betas", {"factor_cols": []}, "factor_cols"),
    ("compute_ic", {"factor_cols": []}, "factor_cols"),
    ("compute_mfe_mae", {"min_estimation_periods": 1}, "min_estimation_periods"),
]

_HOOK_COVERAGE = {name for name, _, _ in BOUNDED_KNOBS}


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return compute_forward_return(
        make_cs_panel(n_assets=60, n_dates=120, rng=0), forward_periods=5
    )


@pytest.fixture(scope="module")
def weighted_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(pl.lit(1.0).alias("weight"))


def _call(func: Any, data: pl.DataFrame, **knob: object) -> Any:
    return func(data, **knob)


class TestClosedSetKnobsAtConstruction:
    """A ``Literal`` annotation is the runtime contract, enforced by the
    constructor for every registered metric that carries one."""

    def test_the_sweep_is_not_empty(self) -> None:
        assert LITERAL_KNOBS, "no metric declares a Literal-annotated knob"

    @pytest.mark.parametrize(
        ("cls", "field", "alias"),
        [pytest.param(c, f, a, id=label) for label, c, f, a in LITERAL_KNOBS],
    )
    def test_bad_token_raises_at_construction(
        self, cls: type, field: str, alias: object
    ) -> None:
        with pytest.raises(UserInputError) as excinfo:
            cls(**{field: BAD_TOKEN})
        exc = excinfo.value
        assert exc.func_name == cls.__name__
        assert exc.field == field
        assert exc.value == BAD_TOKEN
        rendered = str(exc)
        for legal in get_args(alias):
            assert legal in rendered

    @pytest.mark.parametrize(
        ("cls", "field", "alias"),
        [pytest.param(c, f, a, id=label) for label, c, f, a in LITERAL_KNOBS],
    )
    def test_every_legal_token_constructs(
        self, cls: type, field: str, alias: object
    ) -> None:
        for legal in get_args(alias):
            cls(**{field: legal})

    @pytest.mark.parametrize(
        ("cls", "field"),
        [pytest.param(c, f, id=label) for label, c, f, _ in LITERAL_KNOBS],
    )
    def test_config_and_direct_call_raise_identically(
        self, cls: type, field: str
    ) -> None:
        with pytest.raises(UserInputError) as config:
            cls(**{field: BAD_TOKEN})
        with pytest.raises(UserInputError) as direct:
            cls(_UNUSED_DATA, **{field: BAD_TOKEN})
        assert type(config.value) is type(direct.value)
        assert config.value.field == direct.value.field
        assert str(config.value) == str(direct.value)


class TestBoundedKnobsAtConstruction:
    """Bounds a ``Literal`` cannot express, declared as ``@metric(validate=)``."""

    def test_every_hook_is_covered(self) -> None:
        """A metric declaring a validator without a row here is untested."""
        declared = {
            name for name, cls in REGISTRY.items() if cls._knob_validator is not None
        }
        assert declared == _HOOK_COVERAGE

    @pytest.mark.parametrize(
        ("name", "knobs", "field"),
        [pytest.param(n, k, f, id=f"{n}.{f}") for n, k, f in BOUNDED_KNOBS],
    )
    def test_out_of_bounds_raises_at_construction(
        self, name: str, knobs: dict[str, Any], field: str
    ) -> None:
        cls = REGISTRY[name]
        with pytest.raises(UserInputError) as excinfo:
            cls(**knobs)
        assert excinfo.value.func_name == name
        assert excinfo.value.field == field

    @pytest.mark.parametrize(
        ("name", "knobs", "field"),
        [pytest.param(n, k, f, id=f"{n}.{f}") for n, k, f in BOUNDED_KNOBS],
    )
    def test_config_and_direct_call_raise_identically(
        self, name: str, knobs: dict[str, Any], field: str
    ) -> None:
        cls = REGISTRY[name]
        with pytest.raises(UserInputError) as config:
            cls(**knobs)
        with pytest.raises(UserInputError) as direct:
            cls(_UNUSED_DATA, **knobs)
        assert type(config.value) is type(direct.value)
        assert config.value.field == direct.value.field
        assert str(config.value) == str(direct.value)


class TestInferenceAllowlistAtConstruction:
    """``inference=`` is vetted against the module allowlist at construction."""

    @pytest.mark.parametrize(
        "name",
        sorted(n for n, c in REGISTRY.items() if "inference" in c._param_names),
    )
    def test_unvetted_inference_raises_at_construction(self, name: str) -> None:
        cls = REGISTRY[name]
        with pytest.raises(IncompatibleInferenceError) as excinfo:
            cls(inference=BAD_TOKEN)
        assert excinfo.value.func_name == name

    @pytest.mark.parametrize(
        "name",
        sorted(n for n, c in REGISTRY.items() if "inference" in c._param_names),
    )
    def test_config_and_direct_call_raise_identically(self, name: str) -> None:
        cls = REGISTRY[name]
        with pytest.raises(IncompatibleInferenceError) as config:
            cls(inference=BAD_TOKEN)
        with pytest.raises(IncompatibleInferenceError) as direct:
            cls(_UNUSED_DATA, inference=BAD_TOKEN)
        assert str(config.value) == str(direct.value)


def test_no_metric_body_calls_a_knob_validator() -> None:
    """The validators are called from the constructor hook, nowhere else.

    A body-level call is the duplicate this contract replaced: it left a
    config object constructible with a knob the metric would later refuse.
    """
    validators = (
        "_validate_choice(",
        "_validate_n_groups(",
        "_validate_open_unit_interval(",
        "_validate_factor_cols(",
        "_validate_positive_count(",
        "_validate_adf_threshold(",
        "_check_applicable_inference(",
    )
    package = pathlib.Path("factrix/metrics")
    offenders: list[str] = []
    for source in sorted(package.rglob("*.py")):
        if source.name in {"_helpers.py", "_base.py"}:
            continue
        for lineno, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not any(v in stripped for v in validators):
                continue
            # A call inside a ``_validate_<metric>`` hook is the sanctioned
            # site; a call at any other indentation is a body-level duplicate.
            if line.startswith("    ") and _in_validator_hook(source, lineno):
                continue
            offenders.append(f"{source}:{lineno}: {stripped}")
    assert not offenders, "knob validators called outside the constructor hook:\n" + (
        "\n".join(offenders)
    )


def _in_validator_hook(source: pathlib.Path, lineno: int) -> bool:
    """True when line ``lineno`` sits inside a module-level ``_validate_*`` def."""
    lines = source.read_text(encoding="utf-8").splitlines()
    for i in range(lineno - 1, -1, -1):
        line = lines[i]
        if line.startswith("def ") or line.startswith("class "):
            return line.startswith("def _validate_")
    return False


def test_quantile_spread_vw_rejects_bad_tie_policy(
    weighted_panel: pl.DataFrame,
) -> None:
    with pytest.raises(UserInputError) as excinfo:
        quantile_spread_vw(weighted_panel, tie_policy=BAD_TOKEN)
    assert excinfo.value.field == "tie_policy"
    assert "ordinal" in str(excinfo.value)


@pytest.mark.parametrize("center", get_args(Center))
def test_cross_sectional_zscore_accepts_every_legal_center(
    panel: pl.DataFrame, center: str
) -> None:
    cross_sectional_zscore(panel, center=center)


def test_cross_sectional_zscore_rejects_bad_center(panel: pl.DataFrame) -> None:
    with pytest.raises(UserInputError) as excinfo:
        cross_sectional_zscore(panel, center=BAD_TOKEN)
    assert excinfo.value.field == "center"


def test_tie_policy_is_rejected_before_any_data_work() -> None:
    """A bad knob fails on a panel too short to compute anything on.

    The validator runs on the first statement of the body, so the caller
    hears about the typo instead of about the sample.
    """
    tiny = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)],
            "asset_id": ["A"],
            "factor": [1.0],
            "forward_return": [0.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    with pytest.raises(UserInputError) as excinfo:
        k_spread(tiny, tie_policy=BAD_TOKEN)
    assert excinfo.value.field == "tie_policy"


def test_weight_by_is_rejected_before_the_return_column_short_circuit(
    panel: pl.DataFrame,
) -> None:
    """The closed-set check precedes ``top_concentration``'s own early exit.

    ``weight_by="alpha_contribution"`` short-circuits when the return column
    is absent; a bogus token must not reach that branch and be reported as a
    data problem.
    """
    no_returns = panel.drop("forward_return")
    with pytest.raises(UserInputError) as excinfo:
        top_concentration(no_returns, weight_by=BAD_TOKEN)
    assert excinfo.value.field == "weight_by"


@pytest.mark.parametrize("q_top", [0.0, 1.0, 5.0, -1.0, float("nan")])
def test_q_top_outside_the_open_unit_interval_raises(
    panel: pl.DataFrame, q_top: float
) -> None:
    with pytest.raises(UserInputError) as excinfo:
        top_concentration(panel, q_top=q_top)
    assert excinfo.value.field == "q_top"
    assert "(0, 1)" in str(excinfo.value)


def test_q_top_rejects_a_non_number(panel: pl.DataFrame) -> None:
    with pytest.raises(UserInputError) as excinfo:
        top_concentration(panel, q_top="0.2")  # type: ignore[arg-type]
    assert excinfo.value.field == "q_top"


@pytest.mark.parametrize("q_top", [0.05, 0.2, 0.5, 0.99])
def test_q_top_inside_the_interval_is_accepted(
    panel: pl.DataFrame, q_top: float
) -> None:
    assert top_concentration(panel, q_top=q_top).value > 0.0


class TestValidInputUnchanged:
    """Pinned values: validation must not perturb any legal call."""

    def test_top_concentration(self, panel: pl.DataFrame) -> None:
        assert top_concentration(panel, weight_by="abs_factor").value == pytest.approx(
            top_concentration(panel).value
        )
        assert top_concentration(
            panel, weight_by="alpha_contribution"
        ).value != pytest.approx(top_concentration(panel).value)

    def test_quantile_spread(self, panel: pl.DataFrame) -> None:
        ordinal = quantile_spread(panel, tie_policy="ordinal")["factor"].value
        assert quantile_spread(panel)["factor"].value == pytest.approx(ordinal)

    def test_k_spread(self, panel: pl.DataFrame) -> None:
        ordinal = k_spread(panel, tie_policy="ordinal").value
        assert k_spread(panel).value == pytest.approx(ordinal)

    def test_monotonicity(self, panel: pl.DataFrame) -> None:
        pinned = monotonicity(panel, tie_policy="ordinal", rng=0)["factor"].value
        assert monotonicity(panel, rng=0)["factor"].value == pytest.approx(pinned)
