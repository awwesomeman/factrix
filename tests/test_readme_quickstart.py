"""Smoke test mirroring the README ``Typical usage`` quickstart.

Guards against silent drift between the README's import path and the
public surface — if ``compute_forward_return`` is moved or the
re-export is removed, this test fails before the README does.
"""

from __future__ import annotations

import warnings

import factrix as fx
from factrix.preprocess import compute_forward_return


def test_readme_quickstart_runs_end_to_end() -> None:
    raw = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024
    )
    panel = compute_forward_return(raw, forward_periods=5)

    from factrix.metrics import ic

    results = fx.evaluate(
        panel, metrics={"ic": ic()}, factor_cols=["factor"], forward_periods=5
    )

    assert len(results) == 1
    p = results["factor"].metrics["ic"].p_value
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_readme_multi_factor_bhy_screening() -> None:
    from factrix.metrics import ic

    raw_mf = fx.datasets.make_multi_factor_panel(
        n_factors=3, n_assets=100, n_dates=500, seed=2024
    )
    data_mf = compute_forward_return(raw_mf, forward_periods=5)

    factor_cols = ["factor_0000", "factor_0001", "factor_0002"]
    results = fx.evaluate(
        data_mf,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=factor_cols,
        forward_periods=5,
    )
    assert sorted(results) == factor_cols

    fdr_results = fx.multi_factor.bhy(list(results.values()), metrics=["ic"], q=0.05)
    bhy_ic = fdr_results["ic"]
    assert {r.factor for r in bhy_ic.survivors} <= set(factor_cols)


def test_readme_multi_horizon_sweep_is_warning_free() -> None:
    from factrix.metrics import ic

    raw_mf = fx.datasets.make_multi_factor_panel(
        n_factors=3, n_assets=100, n_dates=500, seed=2024
    )
    factor_cols = ["factor_0000", "factor_0001", "factor_0002"]
    horizons = [1, 5, 10]

    # The README pairs a multi-factor panel with the horizon sweep so every
    # expand_over bucket holds >1 result; a single-factor sweep would make each
    # horizon bucket n=1 and emit a "no FDR correction" RuntimeWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results_sweep = fx.evaluate_horizons(
            raw_mf,
            metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
            factor_cols=factor_cols,
            forward_periods=horizons,
        )
        assert len(results_sweep) == len(factor_cols) * len(horizons)

        fdr_results = fx.multi_factor.bhy(
            results_sweep,
            metrics=["ic"],
            expand_over=("forward_periods",),
            q=0.05,
        )

    bhy_ic = fdr_results["ic"]
    assert all(r.forward_periods in horizons for r in bhy_ic.survivors)


def test_compute_forward_return_is_reachable_from_namespace() -> None:
    assert fx.preprocess.compute_forward_return is compute_forward_return
