"""The shared ``seed`` contract across every entry point that resamples.

One knob, one type (``int | numpy.random.Generator | None``), one
resolution helper (``factrix._stats.bootstrap._resolve_rng``) and one
reporting rule:

- ``int`` — reproduces the run and is reported unchanged;
- ``None`` — resolved from system entropy and reported, so an unseeded
  run stays reproducible after the fact;
- ``Generator`` — used as-is and *advanced* by the call (two calls on one
  generator differ; two generators built from the same seed agree), and
  reported as ``None`` because only its owner can reproduce the draw.

Every entry point is exercised against the same five assertions so the
contract cannot drift per module.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix import slice_period_joint_test, slice_period_pairwise_test
from factrix._errors import UserInputError
from factrix.datasets import make_cs_panel
from factrix.inference import StationaryBootstrap
from factrix.metrics import ic, monotonicity
from factrix.stats import bootstrap_mean_ci, stationary_bootstrap_resamples

from tests._slice_panel import build_disjoint_period_panel

_B = 199
_DAY_ZERO = datetime(2024, 1, 1)


def _ic_series(n: int = 40) -> pl.DataFrame:
    """A weak-signal IC series: its bootstrap p sits off the ``1/(B+1)``
    floor, so a different draw shows up as a different p."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "date": [_DAY_ZERO + timedelta(days=i) for i in range(n)],
            "ic": rng.normal(0.02, 0.15, n),
        }
    )


def _slice_panel() -> pl.DataFrame:
    return build_disjoint_period_panel(
        seed=1,
        spans={"bull": (60, 0.1), "bear": (60, 0.1)},
        label_col="regime",
    )


def _monotonicity_panel(n_dates: int = 40, n_assets: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    dates = np.repeat(np.arange(n_dates), n_assets)
    factor = rng.normal(size=n_dates * n_assets)
    # Weak signal on purpose: a perfectly ordered panel pins the MR p at
    # the 1/(B+1) floor, where two different draws are indistinguishable.
    fwd = 0.02 * factor + rng.normal(scale=0.7, size=n_dates * n_assets)
    return pl.DataFrame(
        {
            "date": [_DAY_ZERO + timedelta(days=int(d)) for d in dates],
            "asset_id": [f"a{i}" for i in np.tile(np.arange(n_assets), n_dates)],
            "factor": factor,
            "forward_return": fwd,
        }
    )


# ---------------------------------------------------------------------------
# ic(inference=StationaryBootstrap(...))
# ---------------------------------------------------------------------------


def _ic_p(seed) -> tuple[float, object]:
    result = ic(
        _ic_series(),
        overlap_periods=1,
        inference=StationaryBootstrap(n_resamples=_B, seed=seed),
    )
    return result.p_value, result.metadata["seed"]


class TestStationaryBootstrapSeed:
    def test_same_int_reproduces(self) -> None:
        assert _ic_p(11) == _ic_p(11)
        assert _ic_p(11)[1] == 11

    def test_one_generator_advances(self) -> None:
        gen = np.random.default_rng(11)
        first, first_seed = _ic_p(gen)
        second, _ = _ic_p(gen)
        assert first_seed is None
        assert first != second

    def test_equal_generators_agree(self) -> None:
        assert _ic_p(np.random.default_rng(11)) == _ic_p(np.random.default_rng(11))

    def test_none_reports_a_seed_that_reproduces_the_run(self) -> None:
        p_unseeded, reported = _ic_p(None)
        assert isinstance(reported, int)
        assert _ic_p(reported)[0] == p_unseeded

    def test_bogus_type_rejected(self) -> None:
        with pytest.raises(UserInputError) as excinfo:
            _ic_p("11")
        assert excinfo.value.field == "seed"

    def test_degenerate_path_reports_the_resolved_seed(self) -> None:
        """A series too short to test still reports a usable seed, not -1."""
        from factrix._stats.bootstrap import _block_bootstrap_diff_p

        _, metadata = _block_bootstrap_diff_p(np.array([1.0]), n_resamples=_B, seed=3)
        assert metadata["seed"] == 3
        _, metadata = _block_bootstrap_diff_p(np.array([1.0]), n_resamples=_B)
        assert isinstance(metadata["seed"], int)
        assert metadata["seed"] >= 0
        _, metadata = _block_bootstrap_diff_p(
            np.array([1.0]), n_resamples=_B, seed=np.random.default_rng(3)
        )
        assert metadata["seed"] is None


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------


def _mono(seed) -> tuple[float, object]:
    result = monotonicity(
        _monotonicity_panel(), overlap_periods=1, n_groups=5, n_resamples=_B, seed=seed
    )["factor"]
    return result.p_value, result.metadata["seed"]


class TestMonotonicitySeed:
    def test_same_int_reproduces(self) -> None:
        assert _mono(11) == _mono(11)
        assert _mono(11)[1] == 11

    def test_one_generator_advances(self) -> None:
        gen = np.random.default_rng(11)
        first, first_seed = _mono(gen)
        second, _ = _mono(gen)
        assert first_seed is None
        assert first != second

    def test_equal_generators_agree(self) -> None:
        assert _mono(np.random.default_rng(11)) == _mono(np.random.default_rng(11))

    def test_none_reports_a_seed_that_reproduces_the_run(self) -> None:
        p_unseeded, reported = _mono(None)
        assert isinstance(reported, int)
        assert _mono(reported)[0] == p_unseeded

    def test_bogus_type_rejected(self) -> None:
        with pytest.raises(UserInputError) as excinfo:
            _mono(1.5)
        assert excinfo.value.field == "seed"


# ---------------------------------------------------------------------------
# slice_period_pairwise_test / slice_period_joint_test
# ---------------------------------------------------------------------------


def _pairwise(seed, method: str = "bootstrap"):
    out = slice_period_pairwise_test(
        _slice_panel(),
        ic(),
        by="regime",
        factor_col="factor",
        method=method,
        n_resamples=_B,
        seed=seed,
    )
    return list(out["p_raw"]), out["seed"][0]


def _joint(seed, method: str = "bootstrap"):
    out = slice_period_joint_test(
        _slice_panel(),
        ic(),
        by="regime",
        factor_col="factor",
        method=method,
        n_resamples=_B,
        seed=seed,
    )
    return out["p_value"][0], out["seed"][0]


@pytest.mark.parametrize("run", [_pairwise, _joint], ids=["pairwise", "joint"])
class TestSlicePeriodSeed:
    def test_same_int_reproduces(self, run) -> None:
        assert run(11) == run(11)
        assert run(11)[1] == 11

    def test_one_generator_advances(self, run) -> None:
        gen = np.random.default_rng(11)
        first, first_seed = run(gen)
        second, _ = run(gen)
        assert first_seed is None
        assert first != second

    def test_equal_generators_agree(self, run) -> None:
        assert run(np.random.default_rng(11)) == run(np.random.default_rng(11))

    def test_none_reports_a_seed_that_reproduces_the_run(self, run) -> None:
        unseeded, reported = run(None)
        assert isinstance(reported, int)
        assert run(reported)[0] == unseeded

    def test_analytic_reports_null(self, run) -> None:
        assert run(11, "analytic")[1] is None

    def test_bogus_type_rejected(self, run) -> None:
        with pytest.raises(UserInputError) as excinfo:
            run(object())
        assert excinfo.value.field == "seed"


# ---------------------------------------------------------------------------
# stationary_bootstrap_resamples / bootstrap_mean_ci (draws, no reporting)
# ---------------------------------------------------------------------------


class TestStatsBootstrapSeed:
    @staticmethod
    def _values() -> np.ndarray:
        return np.random.default_rng(0).normal(size=80)

    def test_same_int_reproduces(self) -> None:
        values = self._values()
        first = stationary_bootstrap_resamples(values, _B, seed=11)
        second = stationary_bootstrap_resamples(values, _B, seed=11)
        assert np.array_equal(first, second)

    def test_one_generator_advances(self) -> None:
        values = self._values()
        gen = np.random.default_rng(11)
        first = stationary_bootstrap_resamples(values, _B, seed=gen)
        second = stationary_bootstrap_resamples(values, _B, seed=gen)
        assert not np.array_equal(first, second)

    def test_equal_generators_agree(self) -> None:
        values = self._values()
        first = stationary_bootstrap_resamples(
            values, _B, seed=np.random.default_rng(11)
        )
        second = stationary_bootstrap_resamples(
            values, _B, seed=np.random.default_rng(11)
        )
        assert np.array_equal(first, second)

    def test_none_still_draws(self) -> None:
        assert stationary_bootstrap_resamples(self._values(), _B).shape == (_B, 80)

    def test_bogus_type_rejected(self) -> None:
        with pytest.raises(UserInputError) as excinfo:
            stationary_bootstrap_resamples(self._values(), _B, seed="11")
        assert excinfo.value.field == "seed"

    def test_bootstrap_mean_ci_takes_a_generator(self) -> None:
        values = self._values()
        first = bootstrap_mean_ci(
            values, n_resamples=_B, seed=np.random.default_rng(11)
        )
        second = bootstrap_mean_ci(
            values, n_resamples=_B, seed=np.random.default_rng(11)
        )
        assert first == second
        assert all(math.isfinite(v) for v in first)


# ---------------------------------------------------------------------------
# datasets.make_*
# ---------------------------------------------------------------------------


class TestDatasetsSeed:
    def test_same_int_reproduces(self) -> None:
        assert make_cs_panel(seed=11).equals(make_cs_panel(seed=11))

    def test_one_generator_advances(self) -> None:
        gen = np.random.default_rng(11)
        assert not make_cs_panel(seed=gen).equals(make_cs_panel(seed=gen))

    def test_equal_generators_agree(self) -> None:
        assert make_cs_panel(seed=np.random.default_rng(11)).equals(
            make_cs_panel(seed=np.random.default_rng(11))
        )

    def test_none_still_draws(self) -> None:
        assert make_cs_panel(seed=None).height > 0

    def test_bogus_type_rejected(self) -> None:
        with pytest.raises(UserInputError) as excinfo:
            make_cs_panel(seed="11")
        assert excinfo.value.field == "seed"
