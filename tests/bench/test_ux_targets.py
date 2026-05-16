"""UX target table — structure, version, stretch demotion."""

from __future__ import annotations

from bench.ux_targets import (
    STRETCH_N_FACTORS,
    UX_TARGETS,
    UX_TARGETS_VERSION,
    UxTarget,
    lookup_target,
)


def test_version_is_string():
    assert isinstance(UX_TARGETS_VERSION, str) and UX_TARGETS_VERSION


def test_core_scenarios_have_targets():
    for sid in ("S1", "S2", "S3", "P1"):
        target = UX_TARGETS[sid]
        assert isinstance(target, UxTarget)
        assert target.wall_s_max is not None
        assert target.wall_s_max > 0


def test_lookup_unknown_returns_none():
    assert lookup_target("nonexistent-scenario", n_factors=10) is None


def test_lookup_demotes_to_stretch_at_threshold():
    asserted = lookup_target("S2", n_factors=50)
    stretch = lookup_target("S2", n_factors=STRETCH_N_FACTORS)
    assert asserted is not None and asserted.wall_s_max is not None
    assert stretch is not None and stretch.wall_s_max is None


def test_lookup_no_demotion_below_threshold():
    target = lookup_target("S3", n_factors=STRETCH_N_FACTORS - 1)
    assert target is not None and target.wall_s_max is not None
