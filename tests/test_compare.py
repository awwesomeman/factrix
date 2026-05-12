"""``compare`` leaderboard renderer (#177)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric, Mode
from factrix._codes import StatCode
from factrix._compare import compare
from factrix._errors import UserInputError
from factrix._multi_factor import Survivors
from factrix._profile import FactorProfile
from factrix._run_metrics import MetricsBundle
from factrix._types import MetricOutput


def _profile(
    *,
    factor_id: str,
    forward_periods: int = 1,
    primary_p: float = 0.05,
    primary_stat: float | None = 2.0,
    primary_stat_name: StatCode = StatCode.T_NW,
    context: dict[str, object] | None = None,
) -> FactorProfile:
    cfg = AnalysisConfig.individual_continuous(
        metric=Metric.IC, forward_periods=forward_periods
    )
    return FactorProfile(
        config=cfg,
        mode=Mode.PANEL,
        primary_p=primary_p,
        primary_stat=primary_stat,
        primary_stat_name=primary_stat_name,
        n_obs=100,
        n_pairs=3000,
        n_periods=100,
        n_assets=30,
        factor_id=factor_id,
        context=context or {},
    )


def _bundle(
    *,
    factor_id: str,
    forward_periods: int = 1,
    metrics: dict[str, MetricOutput] | None = None,
    context: dict[str, object] | None = None,
) -> MetricsBundle:
    return MetricsBundle(
        identity=(factor_id, forward_periods),
        metrics=metrics or {},
        context=context or {},
    )


def _metric(value: float, *, name: str = "ic", n_obs: int = 100) -> MetricOutput:
    return MetricOutput(name=name, value=value, n_obs=n_obs)


# ---------------------------------------------------------------------------
# Profile branch
# ---------------------------------------------------------------------------


class TestProfileBranch:
    def test_columns_and_order(self) -> None:
        profiles = [
            _profile(
                factor_id="momentum", primary_p=0.01, context={"universe_id": "lc"}
            ),
            _profile(factor_id="value", primary_p=0.04, context={"universe_id": "lc"}),
        ]
        df = compare(profiles)
        assert df.columns == [
            "factor_id",
            "forward_periods",
            "universe_id",
            "primary_stat",
            "primary_stat_name",
            "primary_p",
        ]
        assert df["factor_id"].to_list() == ["momentum", "value"]
        assert df["primary_stat_name"].to_list() == ["t_nw", "t_nw"]

    def test_sort_by_primary_p(self) -> None:
        profiles = [
            _profile(factor_id="a", primary_p=0.10),
            _profile(factor_id="b", primary_p=0.01),
            _profile(factor_id="c", primary_p=0.05),
        ]
        df = compare(profiles, sort_by="primary_p")
        assert df["factor_id"].to_list() == ["b", "c", "a"]

    def test_heterogeneous_context_union_null_fill(self) -> None:
        profiles = [
            _profile(factor_id="a", context={"universe_id": "lc"}),
            _profile(factor_id="b", context={"regime_id": "calm"}),
        ]
        df = compare(profiles)
        assert df.columns[:4] == [
            "factor_id",
            "forward_periods",
            "universe_id",
            "regime_id",
        ]
        assert df["universe_id"].to_list() == ["lc", None]
        assert df["regime_id"].to_list() == [None, "calm"]

    def test_mixed_procedure_primary_stat_name(self) -> None:
        profiles = [
            _profile(
                factor_id="nw",
                primary_stat=2.5,
                primary_stat_name=StatCode.T_NW,
            ),
            _profile(
                factor_id="boot",
                primary_stat=None,
                primary_stat_name=StatCode.P_BOOT,
            ),
        ]
        df = compare(profiles)
        assert df["primary_stat_name"].to_list() == ["t_nw", "p_boot"]
        assert df["primary_stat"].to_list() == [2.5, None]


# ---------------------------------------------------------------------------
# Bundle branch
# ---------------------------------------------------------------------------


class TestBundleBranch:
    def test_metric_value_columns(self) -> None:
        bundles = [
            _bundle(
                factor_id="a",
                metrics={"ic": _metric(0.05), "ic_ir": _metric(1.5, name="ic_ir")},
            ),
            _bundle(
                factor_id="b",
                metrics={"ic": _metric(0.03), "ic_ir": _metric(0.8, name="ic_ir")},
            ),
        ]
        df = compare(bundles, sort_by="ic_ir")
        assert df.columns == ["factor_id", "forward_periods", "ic", "ic_ir"]
        assert df["factor_id"].to_list() == ["b", "a"]
        assert df["ic_ir"].to_list() == [0.8, 1.5]

    def test_metric_union_null_fill(self) -> None:
        bundles = [
            _bundle(factor_id="a", metrics={"ic": _metric(0.05)}),
            _bundle(
                factor_id="b", metrics={"hit_rate": _metric(0.55, name="hit_rate")}
            ),
        ]
        df = compare(bundles)
        assert df.columns == ["factor_id", "forward_periods", "ic", "hit_rate"]
        assert df["ic"].to_list() == [0.05, None]
        assert df["hit_rate"].to_list() == [None, 0.55]


# ---------------------------------------------------------------------------
# Survivors branch
# ---------------------------------------------------------------------------


class TestSurvivorsBranch:
    def test_adj_p_column_present(self) -> None:
        profiles = [
            _profile(factor_id="a", primary_p=0.01, context={"universe_id": "lc"}),
            _profile(factor_id="b", primary_p=0.02, context={"universe_id": "lc"}),
        ]
        survivors = Survivors(
            profiles=profiles,
            adj_p=np.array([0.008, 0.018], dtype=np.float64),
            q=0.05,
            expand_over=(),
            n_tests={(): 2},
        )
        df = compare(survivors, sort_by="adj_p")
        assert df.columns[-1] == "adj_p"
        assert df["adj_p"].to_list() == [0.008, 0.018]

    def test_expand_over_dimension_via_context(self) -> None:
        profiles = [
            _profile(factor_id="a", primary_p=0.01, context={"universe_id": "lc"}),
            _profile(factor_id="a", primary_p=0.02, context={"universe_id": "sc"}),
        ]
        survivors = Survivors(
            profiles=profiles,
            adj_p=np.array([0.011, 0.022], dtype=np.float64),
            q=0.05,
            expand_over=("universe_id",),
            n_tests={("lc",): 1, ("sc",): 1},
        )
        df = compare(survivors)
        assert "universe_id" in df.columns
        assert df["universe_id"].to_list() == ["lc", "sc"]

    def test_empty_survivors_raises(self) -> None:
        empty = Survivors(
            profiles=[],
            adj_p=np.zeros(0, dtype=np.float64),
            q=0.05,
            expand_over=(),
            n_tests={},
        )
        with pytest.raises(UserInputError) as exc:
            compare(empty)
        assert exc.value.field == "artifacts"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(UserInputError):
            compare([])

    def test_mixed_types_raise(self) -> None:
        with pytest.raises(UserInputError) as exc:
            compare([_profile(factor_id="a"), _bundle(factor_id="b")])  # type: ignore[list-item]
        assert "FactorProfile" in str(exc.value)
        assert "MetricsBundle" in str(exc.value)

    def test_unknown_artifact_type_raises(self) -> None:
        with pytest.raises(UserInputError):
            compare(["not an artifact"])  # type: ignore[list-item]

    def test_sort_by_unknown_column_raises_with_suggestion(self) -> None:
        profiles = [_profile(factor_id="a")]
        with pytest.raises(UserInputError) as exc:
            compare(profiles, sort_by="primary_pp")
        assert exc.value.field == "sort_by"
        assert "primary_p" in exc.value.suggestions

    def test_returns_polars_dataframe(self) -> None:
        df = compare([_profile(factor_id="a")])
        assert isinstance(df, pl.DataFrame)
