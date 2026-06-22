"""Shared synthetic raw-panel builders for slice-test verb tests.

The slice tests are data-first (raw panel + metric instance), so these
builders emit a long ``(date, asset_id, factor, forward_return,
<label>)`` panel. Each label is a disjoint asset universe sharing the
same dates — a cross-sectional partition — with a controllable
factor→return signal strength per label.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def build_labelled_raw_panel(
    *,
    n_dates: int,
    seed: int,
    signal: dict[str, float],
    label_col: str,
    n_assets: int = 40,
    noise: float = 0.3,
) -> pl.DataFrame:
    """Raw CS panel with one disjoint asset block per label.

    ``forward_return = signal[label] * factor + noise`` per (date, asset),
    so each label's per-date information coefficient (IC) has a
    controllable sign / strength. All labels share the same dates
    (cross-sectional partition). ``n_assets`` is per label and stays
    above the IC cross-section floor.
    """
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    date_series = pl.Series("date", dates, dtype=pl.Date)
    idx = np.repeat(np.arange(n_dates), n_assets)
    frames = []
    for lbl, s in signal.items():
        factor = rng.normal(size=n_dates * n_assets)
        fwd = s * factor + rng.normal(scale=noise, size=n_dates * n_assets)
        frames.append(
            pl.DataFrame(
                {
                    "date": date_series.gather(idx),
                    "asset_id": np.tile(
                        [f"{lbl}_{a:03d}" for a in range(n_assets)], n_dates
                    ),
                    "factor": factor,
                    "forward_return": fwd,
                    label_col: [lbl] * (n_dates * n_assets),
                }
            )
        )
    return pl.concat(frames)


def build_autocorrelated_ic_panel(
    *,
    n_dates: int,
    seed: int,
    signal: dict[str, float],
    label_col: str,
    n_assets: int = 40,
    phi: float = 0.85,
    noise: float = 0.3,
) -> pl.DataFrame:
    """Raw CS panel whose per-date IC is serially autocorrelated.

    Each label's factor→return signal follows an AR(1) around its base
    level, so the per-date IC inherits positive autocorrelation — the
    regime under which the HAC bandwidth choice matters (too-short a
    bandwidth under-estimates the variance). Same schema as
    :func:`build_labelled_raw_panel`.
    """
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    date_series = pl.Series("date", dates, dtype=pl.Date)
    idx = np.repeat(np.arange(n_dates), n_assets)
    frames = []
    for lbl, base in signal.items():
        m = np.empty(n_dates)
        m[0] = base
        for t in range(1, n_dates):
            m[t] = base + phi * (m[t - 1] - base) + rng.normal(scale=0.15)
        factor = rng.normal(size=(n_dates, n_assets))
        fwd = m[:, None] * factor + rng.normal(scale=noise, size=(n_dates, n_assets))
        frames.append(
            pl.DataFrame(
                {
                    "date": date_series.gather(idx),
                    "asset_id": np.tile(
                        [f"{lbl}_{a:03d}" for a in range(n_assets)], n_dates
                    ),
                    "factor": factor.reshape(-1),
                    "forward_return": fwd.reshape(-1),
                    label_col: [lbl] * (n_dates * n_assets),
                }
            )
        )
    return pl.concat(frames)
