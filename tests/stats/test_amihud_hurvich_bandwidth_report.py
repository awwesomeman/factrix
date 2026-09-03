"""``predictive_beta`` must report the bandwidth its kernel actually used.

``_resolve_har_lags`` resolves against the full finite-pair count ``n``, but
the augmented Amihud-Hurvich design fits on ``m = n - h`` rows. At a long
horizon ``m`` can fall below the resolved bandwidth, and the Bartlett kernel
inside :func:`factrix._stats.ols._ols_nw_multivariate` clips to ``m - 1``.
The metadata has to follow the kernel, the way ``_ols_scalar_wald_hac`` and
``_driscoll_kraay_cov`` already report the bandwidth they ran at.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._stats import _MIN_PERIODS_PER_LAG
from factrix._stats.hac import _har_dof, _resolve_har_lags
from factrix._stats.ols import _amihud_hurvich_beta
from factrix.metrics.predictive_beta import predictive_beta


def _ts_panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    n = len(x)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "asset_id": ["A"] * n,
            "factor": x,
            "forward_return": y,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.mark.parametrize(("n", "overlap"), [(60, 42), (75, 63), (90, 63)])
def test_reported_har_lags_never_exceeds_the_fitted_design(
    n: int, overlap: int
) -> None:
    """A bandwidth wider than the design it ran on was never applied."""
    rng = np.random.default_rng(n * 100 + overlap)
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=overlap)

    assert not np.isnan(result.value), "guard the real-test path, not a short circuit"
    har_lags = result.metadata["har_lags"]
    n_periods = result.metadata["n_periods"]
    assert isinstance(har_lags, int)
    # The Bartlett kernel cannot use a lag it has no observation pair for.
    assert har_lags <= n_periods - 1


def test_fit_reports_the_bandwidth_the_kernel_clipped_to() -> None:
    """``lags_used`` is the applied bandwidth, not the requested one."""
    rng = np.random.default_rng(1041)
    n, h = 90, 63
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    requested = 30
    fit = _amihud_hurvich_beta(y, x, lags=requested, overlap_periods=h)

    assert fit.n_used > 0
    assert fit.lags_used == min(requested, fit.n_used - 1)
    assert fit.lags_used < requested


def test_ill_conditioned_warning_names_the_bandwidth_that_ran() -> None:
    """Warning text and ``metadata["har_lags"]`` must not contradict."""
    rng = np.random.default_rng(9063)
    n, h = 90, 63
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h)

    har_lags = result.metadata["har_lags"]
    bandwidth_msgs = [
        str(w.message)
        for w in caught
        if "hac_bandwidth_ill_conditioned" in str(w.message)
    ]
    assert bandwidth_msgs, "the ill-conditioned bandwidth screen should fire here"
    message = bandwidth_msgs[0]
    assert f"the kernel ran at L={har_lags}" in message


def test_short_horizon_reports_the_resolved_bandwidth_unchanged() -> None:
    """Where the design is long enough, the requested bandwidth is the used one."""
    rng = np.random.default_rng(7)
    n, h = 240, 5
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    fit = _amihud_hurvich_beta(y, x, lags=12, overlap_periods=h)

    assert fit.lags_used == 12

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h, newey_west_lags=12)
    assert result.metadata["har_lags"] == 12


def test_screen_reads_the_design_the_kernel_ran_on() -> None:
    """#1045: an explicit bandwidth can leave the design thin and every screen quiet.

    The ill-conditioned screen reads the resolved bandwidth against the finite
    pairs, and the effective-sample screen reads ``n_used // h``. Between them
    sits a band where the kernel forms its lag products on an augmented design
    that is thin by the library's own ``_MIN_PERIODS_PER_LAG`` rule and nothing
    says so: 4,252 of 757,735 reachable ``(n, h, newey_west_lags)`` cells.
    """
    rng = np.random.default_rng(65)
    n, h, requested = 65, 2, 13
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(
            _ts_panel(x, y), overlap_periods=h, newey_west_lags=requested
        )
    metadata = result.metadata
    messages = [str(item.message) for item in caught]

    # The cell itself: neither of the two existing screens can fire here, so
    # this warning is the only thing that can report the thin design.
    assert (
        metadata["n_periods_finite"] >= _MIN_PERIODS_PER_LAG * metadata["har_lags"]
    ), "the finite-pair screen must be satisfied, or this cell proves nothing"
    assert not [m for m in messages if "unreliable_se_short_periods" in m]
    assert metadata["n_periods"] < _MIN_PERIODS_PER_LAG * metadata["har_lags"]

    bandwidth = [m for m in messages if "hac_bandwidth_ill_conditioned" in m]
    assert bandwidth, "the design the kernel ran on is thin and must be reported"
    # One composite token: split ``in`` checks would stop verifying that the
    # constant is the divisor *of that key* on *that* row count.
    assert (
        f"the applied Bartlett bandwidth L={metadata['har_lags']} exceeds "
        f"n_periods / {_MIN_PERIODS_PER_LAG} on the {metadata['n_periods']} rows "
        "of the augmented design"
    ) in bandwidth[0]


def test_a_design_with_room_for_its_bandwidth_stays_quiet() -> None:
    """False-positive guard for the clause above: a healthy cell says nothing."""
    rng = np.random.default_rng(2405)
    n, h = 240, 5
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h)

    assert (
        result.metadata["n_periods"]
        >= _MIN_PERIODS_PER_LAG * (result.metadata["har_lags"])
    )
    assert not [
        str(item.message)
        for item in caught
        if "hac_bandwidth_ill_conditioned" in str(item.message)
    ]


def test_pinning_the_clip_moves_no_effective_degrees_of_freedom() -> None:
    """PINNING (#1046), not regression proof: nothing is broken today.

    ``_amihud_hurvich_beta`` feeds ``lags_used`` to :func:`_har_dof` rather than
    the requested bandwidth. Under today's calibration that changes no p-value
    anywhere, and two independent guards are why — measured, not argued from
    the formula:

    - a clip needs ``L > m - 1``, which only occurs at ``h / n`` around two
      thirds and up, and there the ``T / h - 1`` overlap cap is at most
      ``-0.375``. Being negative it wins the ``min()``, so the floor returns
      ``1.0`` on both sides;
    - drop the cap and ``1.5m/L - 1`` is still at most ``0.5`` requested and
      ``0.875`` applied, both under the same floor.

    So this pin only fires if the cap and the floor are **both** loosened;
    changing either alone leaves the other equalising the two sides, which was
    verified by mutating each in turn. That is the honest scope of the guard.
    """
    clipped = 0
    for n in range(10, 201):
        for h in range(2, n - 4):
            m = n - h
            if m < 5:
                continue
            seen: set[int] = set()
            for requested in [None, *range(n)]:
                resolved = _resolve_har_lags(n, requested, h)
                if resolved in seen:
                    continue
                seen.add(resolved)
                used = max(0, min(resolved, m - 1))
                if used == resolved:
                    continue
                clipped += 1
                assert _har_dof(m, used, h) == _har_dof(m, resolved, h), (
                    f"the clip moved the effective df at n={n}, h={h}, "
                    f"L={resolved} -> {used}"
                )
    assert clipped > 1000, "the enumeration must actually reach clipped cells"


def test_the_clip_tail_drops_where_the_applied_clause_already_said_it() -> None:
    """#1047 review: the tail must not repeat what the applied clause printed.

    The tail exists (#1038) so the text cannot contradict
    ``metadata["har_lags"]`` when only the resolved clause prints a bandwidth.
    Where the applied clause fires it prints ``har_lags`` itself, so the
    contradiction cannot arise and the tail would restate the design's row
    count and applied bandwidth a second time.
    """
    rng = np.random.default_rng(9060)
    n, h = 90, 60
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h)
    message = next(
        str(item.message)
        for item in caught
        if "hac_bandwidth_ill_conditioned" in str(item.message)
    )
    metadata = result.metadata

    # The cell: the bandwidth was clipped AND the applied clause fires, which
    # is the only combination the tail is dropped for.
    assert metadata["har_lags"] == metadata["n_periods"] - 1
    assert (
        f"the applied Bartlett bandwidth L={metadata['har_lags']} exceeds "
        f"n_periods / {_MIN_PERIODS_PER_LAG} on the {metadata['n_periods']} rows "
        "of the augmented design"
    ) in message
    assert "admits only" not in message
