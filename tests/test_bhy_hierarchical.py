"""``fx.multi_factor.bhy_hierarchical`` on the EvaluationResult contract."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import HierarchicalBhyResult, bhy_hierarchical

from .conftest import make_result, make_spec


def _grouped(p_by_factor: dict[str, float], group_value: str, primary):
    return [
        make_result(factor=factor, p=p, metric=primary, params={"family": group_value})
        for factor, p in p_by_factor.items()
    ]


def test_returns_dict_per_primary():
    make_spec("ic")
    results = _grouped({"mom_1": 0.001, "mom_2": 0.5}, "momentum", "ic") + _grouped(
        {"val_1": 0.001, "val_2": 0.5}, "value", "ic"
    )
    out = bhy_hierarchical(results, metrics=["ic"], group="family", q=0.05)
    assert isinstance(out, dict)
    assert set(out) == {"ic"}
    assert isinstance(out["ic"], HierarchicalBhyResult)


def test_n_tests_covers_all_input_groups():
    make_spec("ic")
    results = _grouped({"a": 0.5, "b": 0.5}, "g1", "ic") + _grouped(
        {"c": 0.5, "d": 0.5}, "g2", "ic"
    )
    out = bhy_hierarchical(results, metrics=["ic"], group="family", q=0.05)
    assert set(out["ic"].n_tests) == {("g1",), ("g2",)}


def test_single_group_raises():
    make_spec("ic")
    results = _grouped({"a": 0.001, "b": 0.001}, "only", "ic")
    with pytest.raises(UserInputError, match="at least 2"):
        bhy_hierarchical(results, metrics=["ic"], group="family")


def test_every_result_own_group_raises():
    make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.01, metric="ic", params={"family": f"g{i}"})
        for i in range(3)
    ]
    with pytest.raises(UserInputError, match="every result is its own group"):
        bhy_hierarchical(results, metrics=["ic"], group="family")


def test_singleton_group_warns():
    make_spec("ic")
    results = (
        _grouped({"a": 0.5}, "g1", "ic")
        + _grouped({"b": 0.5}, "g2", "ic")
        + _grouped({"c": 0.5, "d": 0.5}, "g3", "ic")
    )
    with pytest.warns(RuntimeWarning, match="single result"):
        bhy_hierarchical(results, metrics=["ic"], group="family", q=0.5)


def test_strong_group_survives_dead_group_does_not():
    make_spec("ic")
    results = _grouped({"hit_1": 1e-6, "hit_2": 1e-6}, "live", "ic") + _grouped(
        {"d1": 0.95, "d2": 0.95}, "dead", "ic"
    )
    out = bhy_hierarchical(results, metrics=["ic"], group="family", q=0.05)
    surviving = {r.factor for r in out["ic"].survivors}
    assert surviving == {"hit_1", "hit_2"}


def test_empty_input_raises():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        bhy_hierarchical([], metrics=["ic"], group="family", q=0.05)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1, float("nan"), True])
def test_q_must_be_open_unit_interval(q):
    make_spec("ic")
    results = _grouped({"a": 0.01, "b": 0.02}, "g1", "ic") + _grouped(
        {"c": 0.03, "d": 0.04}, "g2", "ic"
    )
    with pytest.raises(UserInputError, match="open interval"):
        bhy_hierarchical(
            results,
            metrics=["ic"],
            group="family",
            q=q,  # type: ignore[arg-type]
        )


def test_list_of_dict_keys_suggests_values():
    make_spec("ic")
    results = {
        "mom_1": make_result(
            factor="mom_1", p=0.01, metric="ic", params={"family": "momentum"}
        )
    }
    mistaken: list[str] = []
    mistaken.extend(results)

    with pytest.raises(UserInputError, match=r"list\(results\.values\(\)\)"):
        bhy_hierarchical(  # type: ignore[arg-type]
            mistaken, metrics=["ic"], group="family", q=0.05
        )


def test_primary_must_be_list_of_str():
    make_spec("ic")
    with pytest.raises(UserInputError, match="always a list"):
        bhy_hierarchical(
            [make_result(factor="f", p=0.01, metric="ic")],
            metrics="ic",  # type: ignore[arg-type]
            group="family",
        )


def test_missing_group_key_aggregates_all_offenders():
    make_spec("ic")
    results = [
        make_result(factor="mom_1", p=0.01, metric="ic", params={"family": "momentum"}),
        make_result(factor="mom_2", p=0.01, metric="ic", params={}),
    ]
    with pytest.raises(UserInputError) as excinfo:
        bhy_hierarchical(results, metrics=["ic"], group="family")
    assert "factor='mom_2' missing 'family'" in str(excinfo.value)
    assert excinfo.value.field == "group"


def test_group_equal_to_factor_raises_on_group_field():
    make_spec("ic")
    results = _grouped({"a": 0.01, "b": 0.02}, "g1", "ic") + _grouped(
        {"c": 0.03, "d": 0.04}, "g2", "ic"
    )
    with pytest.raises(UserInputError) as excinfo:
        bhy_hierarchical(results, metrics=["ic"], group="factor")
    assert excinfo.value.field == "group"
    assert "hypothesis identifier" in str(excinfo.value)


class TestHierarchicalFdrControl:
    """Realised FDR of the two-layer construction against its nominal q.

    ``bhy_hierarchical`` uses the nominal ``q`` at BOTH layers and takes the
    max against the outer adjusted p, rather than Yekutieli (2008)'s
    selection-adjusted inner level ``q · R / G``. That means overall FDR
    control is not inherited from the paper and has to be measured. These
    tests pin the measurement so a future change to either layer cannot
    quietly break it.

    The procedure is exercised through the same primitives
    ``_bhy_hierarchical_one`` composes (``simes_p`` / ``bhy_adjusted_p``),
    which keeps the simulation cheap enough for CI while testing the
    construction that matters.
    """

    @staticmethod
    def _adjusted(p_by_group: list[np.ndarray]) -> list[np.ndarray]:
        from factrix.stats.multiple_testing import bhy_adjusted_p, simes_p

        group_simes = np.array([simes_p(g) for g in p_by_group])
        inner = [bhy_adjusted_p(g) for g in p_by_group]
        outer = bhy_adjusted_p(group_simes)
        return [np.maximum(outer[i], inner[i]) for i in range(len(p_by_group))]

    @classmethod
    def _realised_fdr(
        cls,
        *,
        n_groups: int,
        size: int,
        live_groups: int,
        non_null_per_group: int,
        effect: float,
        q: float,
        rho: float = 0.0,
        reps: int = 400,
        seed: int = 0,
    ) -> float:
        from scipy import stats as sp_stats

        rng = np.random.default_rng(seed)
        fdps = []
        for _ in range(reps):
            p_by_group, truth = [], []
            for g in range(n_groups):
                is_nn = np.arange(size) < (non_null_per_group if g < live_groups else 0)
                if rho:
                    # One shared draw per group: equicorrelated members, the
                    # dependence BHY (unlike BH) is meant to survive.
                    z = np.sqrt(rho) * rng.normal() + np.sqrt(1 - rho) * rng.normal(
                        size=size
                    )
                else:
                    z = rng.normal(size=size)
                p_by_group.append(sp_stats.norm.sf(z + is_nn * effect))
                truth.append(is_nn)
            rejected = np.concatenate([a <= q for a in cls._adjusted(p_by_group)])
            is_nn_all = np.concatenate(truth)
            n_rej = int(rejected.sum())
            fdps.append(
                0.0 if n_rej == 0 else float((rejected & ~is_nn_all).sum()) / n_rej
            )
        return float(np.mean(fdps))

    @pytest.mark.parametrize("q", [0.05, 0.10, 0.20])
    def test_global_null_controls_fdr(self, q):
        """Nothing is real: every rejection is false, so FDR is the
        rejection rate itself."""
        fdr = self._realised_fdr(
            n_groups=10,
            size=5,
            live_groups=0,
            non_null_per_group=0,
            effect=0.0,
            q=q,
        )
        assert fdr <= q

    @pytest.mark.parametrize("q", [0.05, 0.10, 0.20])
    def test_selected_but_nearly_dead_group_controls_fdr(self, q):
        """The configuration selective inference is weakest on.

        One overwhelming non-null drags its group past the outer Simes
        screen; the other nine members of that group are null and the inner
        layer runs at the full nominal q. Without the max against the outer
        adjusted p this is where a missing q·R/G rescaling would show up.
        """
        fdr = self._realised_fdr(
            n_groups=10,
            size=10,
            live_groups=2,
            non_null_per_group=1,
            effect=6.0,
            q=q,
        )
        assert fdr <= q

    def test_within_group_dependence_controls_fdr(self):
        """Equicorrelated members — BHY's arbitrary-dependence guarantee is
        the reason the outer layer is BHY rather than BH."""
        fdr = self._realised_fdr(
            n_groups=10,
            size=10,
            live_groups=2,
            non_null_per_group=1,
            effect=6.0,
            q=0.10,
            rho=0.5,
        )
        assert fdr <= 0.10

    def test_not_dominated_by_the_selection_adjusted_variant(self):
        """The nominal-q-plus-max route must not be strictly worse.

        If it controlled FDR only by being uniformly more conservative than
        Yekutieli's q·R/G inner level, the honest fix would be to adopt that
        level instead. It is not: it detects at least as much.
        """
        from factrix.stats.multiple_testing import bhy_adjusted_p, simes_p
        from scipy import stats as sp_stats

        q, n_groups, size = 0.10, 20, 10
        rng = np.random.default_rng(3)
        power_max, power_scaled, reps = 0.0, 0.0, 200
        for _ in range(reps):
            p_by_group, truth = [], []
            for g in range(n_groups):
                is_nn = np.arange(size) < (5 if g < 4 else 0)
                p_by_group.append(sp_stats.norm.sf(rng.normal(size=size) + is_nn * 3.0))
                truth.append(is_nn)
            is_nn_all = np.concatenate(truth)

            rej_max = np.concatenate([a <= q for a in self._adjusted(p_by_group)])

            group_simes = np.array([simes_p(g) for g in p_by_group])
            selected = bhy_adjusted_p(group_simes) <= q
            n_sel = int(selected.sum())
            rej_scaled = np.concatenate(
                [
                    (bhy_adjusted_p(g) <= q * n_sel / n_groups)
                    if (selected[i] and n_sel)
                    else np.zeros(size, dtype=bool)
                    for i, g in enumerate(p_by_group)
                ]
            )
            power_max += (rej_max & is_nn_all).sum() / max(is_nn_all.sum(), 1)
            power_scaled += (rej_scaled & is_nn_all).sum() / max(is_nn_all.sum(), 1)
        assert power_max / reps >= power_scaled / reps
