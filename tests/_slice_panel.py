"""Shared synthetic-panel builder for slice-test verb tests."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def build_labelled_ic_panel(
    *,
    n_dates: int,
    seed: int,
    means: dict[str, float],
    label_col: str,
) -> pl.DataFrame:
    """Long-form ``(date, ic, <label>)`` panel with one block per label.

    Controlled per-label mean IC, fixed RNG seed. Used by the slice-test
    verb tests (pairwise / joint).
    """
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    frames = [
        pl.DataFrame(
            {
                "date": dates,
                "ic": rng.normal(mu, 0.05, n_dates),
                label_col: [lbl] * n_dates,
            }
        )
        for lbl, mu in means.items()
    ]
    return pl.concat(frames)
