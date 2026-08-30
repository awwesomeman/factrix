"""Closed-set and bounded metric knobs are validated before any computation.

A knob annotated with a ``Literal`` alias is a contract, not a hint: a typo
must raise :class:`~factrix.UserInputError` naming the legal values, never
fall through to whichever branch happens to have no ``else`` and never leak
the underlying engine's own vocabulary. The bounded fractions (``q_top``)
are the same contract on an interval instead of a set.

The parametrisations below are the one place that enumerates the knobs, so a
new closed-set knob that skips its validator shows up as a missing row rather
than as a silently wrong number.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, get_args

import polars as pl
import pytest
from factrix import UserInputError
from factrix._types import ConcentrationWeight, TiePolicy
from factrix.datasets import make_cs_panel
from factrix.metrics.concentration import top_concentration
from factrix.metrics.k_spread import k_spread
from factrix.metrics.monotonicity import MRDirection, monotonicity
from factrix.metrics.quantile import quantile_spread, quantile_spread_vw
from factrix.preprocess import compute_forward_return
from factrix.preprocess.normalize import Center, cross_sectional_zscore

BAD_TOKEN = "bogus"


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


# (label, callable, knob name, Literal alias) — one row per closed-set knob.
CLOSED_SET_KNOBS = [
    (
        "top_concentration.weight_by",
        top_concentration,
        "weight_by",
        ConcentrationWeight,
    ),
    ("quantile_spread.tie_policy", quantile_spread, "tie_policy", TiePolicy),
    ("k_spread.tie_policy", k_spread, "tie_policy", TiePolicy),
    ("monotonicity.tie_policy", monotonicity, "tie_policy", TiePolicy),
    ("monotonicity.direction", monotonicity, "direction", MRDirection),
]


@pytest.mark.parametrize(
    ("func", "field", "alias"),
    [pytest.param(f, k, a, id=label) for label, f, k, a in CLOSED_SET_KNOBS],
)
def test_bad_closed_set_value_raises(
    panel: pl.DataFrame, func: Any, field: str, alias: object
) -> None:
    with pytest.raises(UserInputError) as excinfo:
        _call(func, panel, **{field: BAD_TOKEN})
    exc = excinfo.value
    assert exc.field == field
    assert exc.value == BAD_TOKEN
    # The legal set reaches the caller, whichever branch rendered it.
    rendered = str(exc)
    for legal in get_args(alias):
        assert legal in rendered


@pytest.mark.parametrize(
    ("func", "field", "alias"),
    [pytest.param(f, k, a, id=label) for label, f, k, a in CLOSED_SET_KNOBS],
)
def test_every_legal_value_is_accepted(
    panel: pl.DataFrame, func: Any, field: str, alias: object
) -> None:
    for legal in get_args(alias):
        _call(func, panel, **{field: legal})


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
