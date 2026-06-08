"""Synthetic panels for examples, tests, and documentation.

Both generators emit **raw canonical-column panels** (``date, asset_id,
price, factor``). Callers attach ``forward_return`` (e.g. via
``factrix.preprocess.compute_forward_return``) before calling
``fx.evaluate(panel, cfg)``.

Usage:

    import factrix as fx
    from factrix.preprocess import compute_forward_return

    from factrix.metrics import ic

    raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=500)
    panel = compute_forward_return(raw, forward_periods=5)
    results = fx.evaluate(
        panel,
        metrics={"ic": ic()},
        factor_cols=["factor"],
        forward_periods=5,
    )

The dataset's ``signal_horizon`` (default 5) is a property of the
synthetic density, not a pipeline parameter. When
``forward_periods == signal_horizon`` the pipeline realizes the
nominal information coefficient (IC); other horizons realize a decayed density. No disk I/O, no
external data — everything is seeded RNG.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

_DEFAULT_START = "2020-01-02"

# Bumped whenever the default parameters or construction recipe of the
# synthetic generators in this module change in ways that would shift
# benchmark numbers. Recorded in JSONL `env.dataset_spec_version` so
# baselines from different recipes refuse silent comparison.
DATASET_SPEC_VERSION = "1"


def _daily_date_index(start_date: str, n_dates: int) -> pl.Series:
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(days=n_dates - 1)
    return pl.datetime_range(
        start=start,
        end=end,
        interval="1d",
        eager=True,
    ).cast(pl.Datetime("ms"))


def _asset_ids(n_assets: int) -> list[str]:
    width = max(4, len(str(n_assets - 1)))
    return [f"A{i:0{width}d}" for i in range(n_assets)]


def _zscore_cs(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Plain Gaussian z along the last axis — NOT the MAD-z used by
    # factrix/preprocess/normalize.py. Needed here so the Pearson
    # correlation identity Corr(ρ·z(y) + √(1-ρ²)·z(η), y) = ρ holds
    # exactly at target; swapping in MAD-z breaks the calibration.
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, ddof=0, keepdims=True) + eps
    )


def make_cs_panel(
    *,
    n_assets: int = 50,
    n_dates: int = 252,
    ic_target: float = 0.04,
    signal_horizon: int = 5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pl.DataFrame:
    """Synthetic cross-sectional panel with a calibrated target information coefficient (IC).

    Construction:
        1. Per-asset volatility ``σ_i ~ U[0.01, 0.03]``; daily arithmetic
           returns are ``ε_{t,i} ~ N(0, σ_i)``.
        2. Prices ``p[t,i] = 100 · cumprod(1 + ε)``.
        3. Signal-horizon forward return
           ``fr[t] = (p[t+1+H]/p[t+1] − 1) / H`` where ``H = signal_horizon``.
        4. Factor is a cross-sectional mixture of standardized forward
           return and iid noise::

               factor[t] = ρ · z(fr[t]) + √(1−ρ²) · z(η[t])

           where ``ρ = clip(ic_target, −0.99, 0.99)`` and ``z`` is plain
           Gaussian (not MAD) z-score so the identity
           ``Corr(factor, fr) = ρ`` holds exactly per date at horizon
           ``H``. Factorlib's ``ic_mean`` uses Spearman rank IC, which
           tracks Pearson ``ρ`` tightly at small ``|ρ|`` but is not
           identical — realized ``ic_mean`` at ``|ic_target| ≳ 0.2`` may
           diverge by a few bp.
        5. The last ``H+1`` dates have no defined forward return; factor
           values there are pure noise and will be dropped along with
           the null forward returns once ``compute_forward_return`` runs.

    Args:
        n_assets: Cross-sectional width.
        n_dates: Number of calendar dates (daily index, includes
            weekends — factrix doesn't prescribe a calendar).
        ic_target: Target per-date Pearson CS correlation between
            factor and forward return at ``signal_horizon``. Realized
            realized per-date IC after ``fx.evaluate`` will fall near this
            within a couple of standard errors — overlapping forward
            returns reduce effective independent dates by
            ``signal_horizon`` so s.e. ≈
            ``1 / √((n_dates / signal_horizon) · n_assets)``.
        signal_horizon: Horizon (in bars) at which the synthetic density
            lives — a property of the generated data, not a pipeline
            parameter. Pipelines measuring at
            ``forward_periods == signal_horizon``
            realize the nominal IC; different horizons realize a
            decayed IC (correct physics for a density with a natural
            time-scale, not a bug).
        seed: RNG seed.
        start_date: ISO date for the first row.

    Returns:
        Long DataFrame with ``date, asset_id, price, factor`` and
        ``date`` dtype ``pl.Datetime("ms")``. Attach ``forward_return``
        (e.g. via ``factrix.preprocess.compute_forward_return``)
        before passing to ``fx.evaluate``.

    Examples:
        >>> import factrix as fx
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> set(raw.columns) == {"date", "asset_id", "price", "factor"}
        True
        >>> raw["asset_id"].n_unique() == 20
        True

        Attach a forward return before evaluating:

        >>> from factrix.preprocess import compute_forward_return
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> "forward_return" in panel.columns
        True
    """
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2 for a cross-section")
    if n_dates < signal_horizon + 2:
        raise ValueError(
            f"n_dates must be >= signal_horizon + 2 (got n_dates={n_dates}, "
            f"signal_horizon={signal_horizon})"
        )

    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(0.01, 0.03, size=n_assets)
    daily_ret = rng.standard_normal((n_dates, n_assets)) * sigmas
    prices = 100.0 * np.cumprod(1.0 + daily_ret, axis=0)

    N = signal_horizon
    last_valid = n_dates - N - 1
    fr = (prices[1 + N : 1 + N + last_valid] / prices[1 : 1 + last_valid] - 1.0) / N

    rho = float(np.clip(ic_target, -0.99, 0.99))
    noise = rng.standard_normal((n_dates, n_assets))

    factor = noise.copy()
    factor[:last_valid] = rho * _zscore_cs(fr) + np.sqrt(1.0 - rho * rho) * _zscore_cs(
        noise[:last_valid]
    )

    return _to_long_panel(
        start_date=start_date,
        n_dates=n_dates,
        n_assets=n_assets,
        prices=prices,
        factor=factor,
    )


def make_event_panel(
    *,
    n_assets: int = 50,
    n_dates: int = 252,
    event_rate: float = 0.02,
    post_event_drift_bps: float = 10.0,
    signal_horizon: int = 5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pl.DataFrame:
    """Synthetic event-density panel — sparse ``{0, R}`` schema, emitted
    here as the canonical signed ternary ``factor ∈ {-1, 0, +1}``.

    Construction:
        1. Baseline returns as in ``make_cs_panel``.
        2. Independent ``Bernoulli(event_rate)`` per ``(t, i)``; sign
           ``±1`` with equal probability (this generator's chosen
           magnitude under the broader ``{0, R}`` sparse schema).
           Non-event cells get ``0``.
        3. Post-event drift: for each event with sign ``s``, add
           ``s · post_event_drift_bps / 1e4 / signal_horizon`` to the
           ``signal_horizon`` bars ``t+2 .. t+1+H`` of that asset — the
           exact window a pipeline measuring forward return at the same
           horizon will see. Drift magnitude is small (≈ bps-per-day)
           so the event density is discoverable but not trivial.
        4. Prices are cumulated after drift injection.

    Suitable for evaluating sparse metrics (e.g. event study ``caar``).

    Args:
        n_assets: Cross-sectional width.
        n_dates: Number of calendar dates.
        event_rate: Per-cell event probability (≈ expected events per
            asset per date).
        post_event_drift_bps: Total drift in basis points injected
            across the ``signal_horizon`` bars of the forward-return
            window (bars ``t+2 .. t+1+H``).
        signal_horizon: Horizon (in bars) over which post-event drift
            is distributed — a property of the generated data, not a
            pipeline parameter. Pipelines measuring at
            ``forward_periods == signal_horizon`` realize
            the nominal drift; different horizons realize a weakened or
            diluted density (correct physics for a density with a natural
            time-scale, not a bug).
        seed: RNG seed.
        start_date: ISO date for the first row.

    Returns:
        Long DataFrame with ``date, asset_id, price, factor``. Factor
        is ``Float64`` with values in ``{-1.0, 0.0, +1.0}``. Attach
        ``forward_return`` (e.g. via
        ``factrix.preprocess.compute_forward_return``) before
        passing to ``fx.evaluate``.

    Examples:
        >>> import factrix as fx
        >>> raw = fx.datasets.make_event_panel(n_assets=20, n_dates=120, event_rate=0.05)
        >>> set(raw["factor"].unique().to_list()) <= {-1.0, 0.0, 1.0}
        True
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    if n_dates < signal_horizon + 2:
        raise ValueError(
            f"n_dates must be >= signal_horizon + 2 (got n_dates={n_dates}, "
            f"signal_horizon={signal_horizon})"
        )
    if not 0.0 <= event_rate <= 1.0:
        raise ValueError(f"event_rate must be in [0, 1], got {event_rate}")

    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(0.01, 0.03, size=n_assets)
    daily_ret = rng.standard_normal((n_dates, n_assets)) * sigmas

    has_event = rng.random((n_dates, n_assets)) < event_rate
    signs = rng.choice([-1.0, 1.0], size=(n_dates, n_assets))
    factor = np.where(has_event, signs, 0.0)

    # compute_forward_return uses (p[t+1+H] / p[t+1] - 1)/H, whose realized
    # returns span bars t+2..t+1+H — NOT t+1..t+H. Drift has to be injected
    # on that exact window or the forward-return window partly misses it.
    drift_per_bar = post_event_drift_bps / 1e4 / signal_horizon
    event_idx = np.argwhere(has_event)
    for t, i in event_idx:
        end = min(t + 2 + signal_horizon, n_dates)
        daily_ret[t + 2 : end, i] += factor[t, i] * drift_per_bar

    prices = 100.0 * np.cumprod(1.0 + daily_ret, axis=0)

    return _to_long_panel(
        start_date=start_date,
        n_dates=n_dates,
        n_assets=n_assets,
        prices=prices,
        factor=factor,
    )


def make_multi_factor_panel(
    *,
    n_factors: int = 10,
    n_assets: int = 50,
    n_dates: int = 252,
    ic_target: float = 0.04,
    factor_correlation: float = 0.0,
    signal_horizon: int = 5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pl.DataFrame:
    """Synthetic multi-factor panel with calibrated IC and tunable
    cross-factor correlation.

    Each of ``n_factors`` factor columns is constructed as in
    ``make_cs_panel`` (``ρ · z(fr) + √(1−ρ²) · z(η_k)``), with the
    per-factor noise terms ``η_k`` sharing a common cross-sectional
    component so that pairwise factor-factor correlation is controllable:

        η_k = √c · ξ_common + √(1−c) · ξ_idio_k

    where ``c = factor_correlation``. The realized factor-factor
    cross-sectional correlation is approximately
    ``ρ² + (1 − ρ²) · c``.

    The last ``H + 1`` rows have no defined forward return; factor
    values there are pure noise (same construction convention as
    ``make_cs_panel``) and will be dropped along with the null forward
    returns once ``compute_forward_return`` runs.

    Args:
        n_factors: Number of factor columns to emit.
        n_assets: Cross-sectional width.
        n_dates: Number of calendar dates.
        ic_target: Target per-date Pearson CS correlation between each
            factor and the forward return at ``signal_horizon``.
        factor_correlation: Pairwise correlation between factor noise
            terms in ``[0, 1)``. ``0`` reproduces independent factors;
            higher values share more of a common cross-sectional driver.
        signal_horizon: Horizon (in bars) at which the synthetic density
            lives.
        seed: RNG seed.
        start_date: ISO date for the first row.

    Returns:
        Long DataFrame with ``date, asset_id, price, factor_0000,
        factor_0001, ...``. ``factor_*`` column count equals
        ``n_factors`` and column width is zero-padded to a stable
        sort order.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2 for a cross-section")
    if n_dates < signal_horizon + 2:
        raise ValueError(
            f"n_dates must be >= signal_horizon + 2 (got n_dates={n_dates}, "
            f"signal_horizon={signal_horizon})"
        )
    if not 0.0 <= factor_correlation < 1.0:
        raise ValueError(
            f"factor_correlation must be in [0, 1), got {factor_correlation}"
        )

    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(0.01, 0.03, size=n_assets)
    daily_ret = rng.standard_normal((n_dates, n_assets)) * sigmas
    prices = 100.0 * np.cumprod(1.0 + daily_ret, axis=0)

    H = signal_horizon
    last_valid = n_dates - H - 1
    fr = (prices[1 + H : 1 + H + last_valid] / prices[1 : 1 + last_valid] - 1.0) / H
    fr_z = _zscore_cs(fr)

    rho = float(np.clip(ic_target, -0.99, 0.99))
    c = float(factor_correlation)
    sqrt_c = float(np.sqrt(c))
    sqrt_1_c = float(np.sqrt(1.0 - c))
    sqrt_1_rho2 = float(np.sqrt(1.0 - rho * rho))

    common = rng.standard_normal((n_dates, n_assets))
    common_z = _zscore_cs(common)

    width = max(4, len(str(n_factors - 1)))
    factor_columns: dict[str, np.ndarray] = {}
    for k in range(n_factors):
        idio = rng.standard_normal((n_dates, n_assets))
        eta = sqrt_c * common_z + sqrt_1_c * _zscore_cs(idio)
        f_k = eta.copy()
        f_k[:last_valid] = rho * fr_z + sqrt_1_rho2 * _zscore_cs(eta[:last_valid])
        factor_columns[f"factor_{k:0{width}d}"] = f_k

    dates = _daily_date_index(start_date, n_dates)
    assets = _asset_ids(n_assets)
    date_col = np.repeat(dates.to_numpy(), n_assets)
    asset_col = np.tile(np.asarray(assets, dtype=object), n_dates)

    base = pl.DataFrame(
        {
            "date": date_col,
            "asset_id": asset_col,
            "price": prices.flatten(),
        },
        schema={
            "date": pl.Datetime("ms"),
            "asset_id": pl.String,
            "price": pl.Float64,
        },
    )
    factor_series = [
        pl.Series(name, arr.flatten(), dtype=pl.Float64)
        for name, arr in factor_columns.items()
    ]
    return base.with_columns(factor_series)


def _to_long_panel(
    *,
    start_date: str,
    n_dates: int,
    n_assets: int,
    prices: np.ndarray,
    factor: np.ndarray,
) -> pl.DataFrame:
    dates = _daily_date_index(start_date, n_dates)
    assets = _asset_ids(n_assets)

    date_col = np.repeat(dates.to_numpy(), n_assets)
    asset_col = np.tile(np.asarray(assets, dtype=object), n_dates)

    return pl.DataFrame(
        {
            "date": date_col,
            "asset_id": asset_col,
            "price": prices.flatten(),
            "factor": factor.flatten(),
        },
        schema={
            "date": pl.Datetime("ms"),
            "asset_id": pl.String,
            "price": pl.Float64,
            "factor": pl.Float64,
        },
    )
