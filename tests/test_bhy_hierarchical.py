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
    """Realised FDR of the selective two-layer construction against nominal q.

    The inner layer runs at Benjamini-Bogomolov's ``q · R / G``. An earlier
    nominal-q-at-both-layers construction was measured to exceed q on small
    groups of correlated members — the library's modal input — and that
    scan is pinned below as the regression guard. The procedure is
    exercised through the same primitives ``_bhy_hierarchical_one``
    composes (``simes_p`` / ``bhy_adjusted_p`` / the selection scale), which
    keeps the simulation cheap enough for CI while testing the construction
    that matters; ``test_small_correlated_groups_through_public_api`` runs
    one point through ``bhy_hierarchical`` itself so the helper cannot
    drift from the shipped code.
    """

    @staticmethod
    def _adjusted(p_by_group: list[np.ndarray], q: float) -> list[np.ndarray]:
        from factrix.stats.multiple_testing import bhy_adjusted_p, simes_p

        group_simes = np.array([simes_p(g) for g in p_by_group])
        inner = [bhy_adjusted_p(g) for g in p_by_group]
        outer = bhy_adjusted_p(group_simes)
        n_groups = len(p_by_group)
        n_selected = int(np.sum(outer <= q))
        scale = n_groups / n_selected if n_selected else np.inf
        return [
            np.maximum(outer[i], np.minimum(1.0, inner[i] * scale))
            for i in range(n_groups)
        ]

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
            rejected = np.concatenate([a <= q for a in cls._adjusted(p_by_group, q)])
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

    @pytest.mark.parametrize("n_groups,live", [(50, 1), (100, 1), (50, 2)])
    def test_tiny_selected_fraction_controls_fdr(self, n_groups, live):
        """``R / G`` down to 0.01 — where a missing q·R/G bites hardest.

        Yekutieli's inner level shrinks with the selected fraction precisely
        because a small ``R`` means the selection was aggressive. Using the
        nominal q instead is the divergence this procedure takes, so the
        smallest realistic ``R / G`` is the configuration that has to hold.
        The docstring quotes 0.01; this is what makes that claim checkable.
        """
        fdr = self._realised_fdr(
            n_groups=n_groups,
            size=20,
            live_groups=live,
            non_null_per_group=1,
            effect=7.0,
            q=0.10,
            reps=300,
        )
        assert fdr <= 0.10

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

    @pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
    def test_small_correlated_groups_control_fdr(self, rho):
        """The regime that broke the nominal-q construction.

        ``G = 30`` groups of 4, five live with one true effect each,
        equicorrelated members. Under nominal q at both layers realised FDR
        was 1.09x / 1.18x / 1.14x nominal at rho 0.5 / 0.7 / 0.9 — a family
        of four correlated factor variants is exactly this shape. The
        selective inner level must hold it under q.
        """
        fdr = self._realised_fdr(
            n_groups=30,
            size=4,
            live_groups=5,
            non_null_per_group=1,
            effect=3.5,
            q=0.10,
            rho=rho,
            reps=1500,
            seed=11,
        )
        assert fdr <= 0.10

    def test_small_correlated_groups_through_public_api(self):
        """One point of the scan through ``bhy_hierarchical`` on real inputs,
        so the simulation helper cannot silently diverge from shipped code."""
        from scipy import stats as sp_stats

        make_spec("ic")
        rng = np.random.default_rng(5)
        q, n_groups, size, live, effect, rho = 0.10, 30, 4, 5, 3.5, 0.7
        fdps = []
        for _ in range(150):
            results, truth = [], {}
            for g in range(n_groups):
                z = np.sqrt(rho) * rng.normal() + np.sqrt(1 - rho) * rng.normal(
                    size=size
                )
                for j in range(size):
                    is_nn = g < live and j == 0
                    p = float(sp_stats.norm.sf(z[j] + (effect if is_nn else 0.0)))
                    name = f"f{g}_{j}"
                    results.append(
                        make_result(factor=name, p=p, metric="ic", params={"family": g})
                    )
                    truth[name] = is_nn
            out = bhy_hierarchical(results, metrics=["ic"], group="family", q=q)["ic"]
            survivors = [r.factor for r in out.survivors]
            n_rej = len(survivors)
            fdps.append(
                0.0 if n_rej == 0 else sum(1 for f in survivors if not truth[f]) / n_rej
            )
        assert float(np.mean(fdps)) <= q


class TestSelectedGroupCount:
    """``n_selected_groups`` is ``R``, the term the inner level scales by."""

    def test_reproduces_the_inner_level_by_hand(self):
        """R is reported so a user can recompute the adjusted p-values.

        The inner level is ``q · R / G``. ``G`` is ``len(n_tests)`` and the
        raw p-values are on the entries, but ``R`` is not otherwise
        recoverable — without it the documented formula is unverifiable.
        """
        from factrix.stats.multiple_testing import bhy_adjusted_p, simes_p

        make_spec("ic")
        q = 0.10
        live = _grouped({"a1": 0.0001, "a2": 0.30}, "live", "ic")
        dead = _grouped({"b1": 0.60, "b2": 0.80}, "dead1", "ic") + _grouped(
            {"c1": 0.70, "c2": 0.90}, "dead2", "ic"
        )
        out = bhy_hierarchical(live + dead, metrics=["ic"], group="family", q=q)["ic"]

        groups = [[0.0001, 0.30], [0.60, 0.80], [0.70, 0.90]]
        outer = bhy_adjusted_p(np.array([simes_p(g) for g in groups]))
        assert out.n_selected_groups == int((outer <= q).sum())
        assert len(out.n_tests) == 3

        scale = 3 / out.n_selected_groups
        expected = [
            max(outer[gi], min(1.0, adj * scale))
            for gi, g in enumerate(groups)
            for adj in bhy_adjusted_p(np.array(g))
        ]
        assert out.adj_p_all == pytest.approx(expected)

    def test_zero_when_no_group_passes(self):
        make_spec("ic")
        results = _grouped({"a1": 0.7, "a2": 0.8}, "g1", "ic") + _grouped(
            {"b1": 0.75, "b2": 0.85}, "g2", "ic"
        )
        out = bhy_hierarchical(results, metrics=["ic"], group="family", q=0.05)["ic"]
        assert out.n_selected_groups == 0
        assert not out.survivors
        assert not np.isnan(out.adj_p_all).any()
