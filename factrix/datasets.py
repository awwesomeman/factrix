"""Synthetic panels for examples, tests, and documentation.

Both generators emit **raw canonical-column panels** (``date, asset_id,
price, factor``). Callers run ``fl.preprocess`` before ``fl.evaluate`` —
this matches the rest of factrix where preprocess and evaluate are
deliberately two visible steps so config binding can't silently drift.

Usage:

    import factrix as fl

    raw = fl.datasets.make_cs_panel(n_assets=100, n_dates=500)
    cfg = fl.CrossSectionalConfig(forward_periods=5)
    prepared = fl.preprocess(raw, config=cfg)
    profile  = fl.evaluate(prepared, "synthetic", config=cfg)

The dataset's ``signal_horizon`` (default 5) is a property of the
synthetic signal, not a pipeline parameter. When
``cfg.forward_periods == signal_horizon`` the pipeline realizes the
nominal IC; other horizons realize a decayed signal. No disk I/O, no
external data — everything is seeded RNG.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl


_DEFAULT_START = "2020-01-02"


def _daily_date_index(start_date: str, n_dates: int) -> pl.Series:
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(days=n_dates - 1)
    return pl.datetime_range(
        start=start, end=end, interval="1d", eager=True,
    ).cast(pl.Datetime("ms"))


def _asset_ids(n_assets: int) -> list[str]:
    width = max(4, len(str(n_assets - 1)))
    return [f"A{i:0{width}d}" for i in range(n_assets)]


def _zscore_cs(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Plain Gaussian z along the last axis — NOT the MAD-z used by
    # factrix/preprocess/normalize.py. Needed here so the Pearson
    # correlation identity Corr(ρ·z(y) + √(1-ρ²)·z(η), y) = ρ holds
    # exactly at target; swapping in MAD-z breaks the calibration.
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, ddof=0, keepdims=True) + eps)


def make_cs_panel(
    *,
    n_assets: int = 50,
    n_dates: int = 252,
    ic_target: float = 0.04,
    signal_horizon: int = 5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pl.DataFrame:
    """Synthetic cross-sectional panel with a calibrated target IC.

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
           values there are pure noise and will be dropped by
           ``fl.preprocess`` along with the null forward returns.

    Args:
        n_assets: Cross-sectional width.
        n_dates: Number of calendar dates (daily index, includes
            weekends — factrix doesn't prescribe a calendar).
        ic_target: Target per-date Pearson CS correlation between
            factor and forward return at ``signal_horizon``. Realized
            ``ic_mean`` after ``fl.evaluate`` will fall near this within
            a couple of standard errors — overlapping forward returns
            reduce effective independent dates by ``signal_horizon`` so
            s.e. ≈ ``1 / √((n_dates / signal_horizon) · n_assets)``.
        signal_horizon: Horizon (in bars) at which the synthetic signal
            lives — a property of the generated data, not a pipeline
            parameter. Pipelines measuring at
            ``CrossSectionalConfig.forward_periods == signal_horizon``
            realize the nominal IC; different horizons realize a
            decayed IC (correct physics for a signal with a natural
            time-scale, not a bug).
        seed: RNG seed.
        start_date: ISO date for the first row.

    Returns:
        Long DataFrame with ``date, asset_id, price, factor`` and
        ``date`` dtype ``pl.Datetime("ms")`` so it drops directly into
        ``fl.validate_factor_data`` / ``fl.preprocess``.
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
    factor[:last_valid] = (
        rho * _zscore_cs(fr) + np.sqrt(1.0 - rho * rho) * _zscore_cs(noise[:last_valid])
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
    """Synthetic event-signal panel (factor ∈ ``{-1, 0, +1}``).

    Construction:
        1. Baseline returns as in ``make_cs_panel``.
        2. Independent ``Bernoulli(event_rate)`` per ``(t, i)``; sign
           ``±1`` with equal probability. Non-event cells get ``0``.
        3. Post-event drift: for each event with sign ``s``, add
           ``s · post_event_drift_bps / 1e4 / signal_horizon`` to the
           ``signal_horizon`` bars ``t+2 .. t+1+H`` of that asset — the
           exact window a pipeline measuring forward return at the same
           horizon will see. Drift magnitude is small (≈ bps-per-day)
           so the event signal is discoverable but not trivial.
        4. Prices are cumulated after drift injection.

    Suitable for ``EventConfig`` / ``factor_type="event_signal"``.

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
            ``EventConfig.forward_periods == signal_horizon`` realize
            the nominal drift; different horizons realize a weakened or
            diluted signal (correct physics for a signal with a natural
            time-scale, not a bug).
        seed: RNG seed.
        start_date: ISO date for the first row.

    Returns:
        Long DataFrame with ``date, asset_id, price, factor``. Factor
        is ``Float64`` with values in ``{-1.0, 0.0, +1.0}`` so the
        event_signal preprocess path accepts it verbatim.
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
