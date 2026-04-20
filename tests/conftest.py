"""Shared fixtures for factorlib unit tests."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.config import CrossSectionalConfig
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.evaluation.profiles import CrossSectionalProfile


@pytest.fixture
def tiny_panel() -> pl.DataFrame:
    """3 dates × 5 assets, perfect monotonic factor-return alignment.

    IC = 1.0 per date, monotonicity = 1.0, spread hand-computable.
    """
    dates = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(3)]
    assets = ["A", "B", "C", "D", "E"]
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]

    rows = []
    for d in dates:
        for a, f, r in zip(assets, factors, returns):
            rows.append({"date": d, "asset_id": a, "factor": f, "forward_return": r})

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def noisy_panel() -> pl.DataFrame:
    """40 dates × 30 assets, seeded random with positive IC.

    return = 0.3 * factor + 0.7 * noise → positive but noisy IC.

    40 dates keeps the non-overlapping sample (fp=5 → 8 periods) above
    MIN_PORTFOLIO_PERIODS=5 so spread/monotonicity tests actually exercise
    the compute path instead of short-circuiting to NaN.
    """
    rng = np.random.default_rng(42)
    n_dates, n_assets = 40, 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        r = 0.3 * f + 0.7 * noise
        for i, a in enumerate(assets):
            rows.append({
                "date": d,
                "asset_id": a,
                "factor": float(f[i]),
                "forward_return": float(r[i]),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def ic_series_positive() -> pl.DataFrame:
    """50-period positive IC series (date + value)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0.05, 0.02, 50)
    values = np.abs(values)  # ensure all positive
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    return pl.DataFrame({
        "date": dates,
        "value": values,
    }).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def ic_series_sign_flip() -> pl.DataFrame:
    """40-period IC: first 20 positive, last 20 negative."""
    rng = np.random.default_rng(42)
    pos = rng.normal(0.05, 0.01, 20)
    pos = np.abs(pos)
    neg = -np.abs(rng.normal(0.03, 0.01, 20))
    values = np.concatenate([pos, neg])
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    return pl.DataFrame({
        "date": dates,
        "value": values,
    }).with_columns(pl.col("date").cast(pl.Datetime("ms")))


# ---------------------------------------------------------------------------
# Profile-era fixtures (Phase A)
# ---------------------------------------------------------------------------

def _cs_panel(
    n_dates: int,
    n_assets: int,
    signal_coef: float,
    seed: int,
    *,
    include_price: bool = False,
) -> pl.DataFrame:
    """Build a cross-sectional panel with a tunable signal-to-noise mix.

    ``include_price=True`` adds a dummy ``price`` column — multi_horizon_ic
    (and any metric that recomputes forward returns from price) requires it.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        r = signal_coef * f + (1 - abs(signal_coef)) * noise
        for i in range(n_assets):
            row: dict[str, object] = {
                "date": d, "asset_id": f"a{i}",
                "factor": float(f[i]), "forward_return": float(r[i]),
            }
            if include_price:
                row["price"] = float(100 + rng.standard_normal())
            rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def make_macro_panel(
    n_dates: int, n_countries: int, signal: float, seed: int,
) -> pl.DataFrame:
    """Macro-panel factor panel (public — shared by profile/parity tests)."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        fvals = rng.standard_normal(n_countries)
        for i in range(n_countries):
            r = signal * fvals[i] + (1 - abs(signal)) * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"c{i}",
                "factor": float(fvals[i]), "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


@pytest.fixture
def cs_profile_strong() -> CrossSectionalProfile:
    """A cross-sectional profile with genuine positive IC signal."""
    df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.5, seed=101)
    art = build_artifacts(df, CrossSectionalConfig())
    art.factor_name = "cs_strong"
    profile, _ = CrossSectionalProfile.from_artifacts(art)
    return profile


@pytest.fixture
def cs_profile_weak() -> CrossSectionalProfile:
    """A cross-sectional profile with near-zero IC (should FAIL verdict)."""
    df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.02, seed=202)
    art = build_artifacts(df, CrossSectionalConfig())
    art.factor_name = "cs_weak"
    profile, _ = CrossSectionalProfile.from_artifacts(art)
    return profile


@pytest.fixture
def cs_profiles_and_artifacts() -> tuple[
    list[CrossSectionalProfile], dict[str, object]
]:
    """Four CS profiles ranging from strong to pure noise, plus their
    artifacts (keyed by factor_name) for redundancy_matrix tests."""
    specs = [
        ("strong", 0.5, 301),
        ("good", 0.3, 302),
        ("marginal", 0.1, 303),
        ("noise", 0.01, 304),
    ]
    profiles = []
    artifacts_map = {}
    for name, coef, seed in specs:
        df = _cs_panel(n_dates=60, n_assets=25, signal_coef=coef, seed=seed)
        art = build_artifacts(df, CrossSectionalConfig())
        art.factor_name = name
        profile, _ = CrossSectionalProfile.from_artifacts(art)
        profiles.append(profile)
        artifacts_map[name] = art
    return profiles, artifacts_map
