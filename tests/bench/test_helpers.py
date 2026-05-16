"""Preset table invariants — dimensions and laptop / cloud split."""

from __future__ import annotations

from bench.scenarios._helpers import PRESETS, SPARSE_PRESETS


def test_xlarge_preset_dimensions():
    p = PRESETS["xlarge"]
    assert (p.n_factors, p.n_assets, p.n_dates) == (1000, 2000, 2000)


def test_user_realistic_high_preset_dimensions():
    p = PRESETS["user-realistic-high"]
    assert (p.n_factors, p.n_assets, p.n_dates) == (500, 3000, 2500)


def test_sparse_presets_track_continuous_presets():
    # Cloud-only presets must exist on both tables so a UX run with
    # --target xlarge / user-realistic-high resolves cleanly on the
    # sparse cell too.
    for name in ("xlarge", "user-realistic-high"):
        assert name in SPARSE_PRESETS, name


def test_laptop_presets_unchanged():
    # `small` and `large` size the laptop reference baseline; their
    # dimensions are pinned and must not drift when cloud presets get
    # added.
    assert PRESETS["small"].n_factors == 100
    assert PRESETS["large"].n_factors == 500
