"""UX latency targets — absolute ``wall_s`` ceilings the harness asserts
records against in the UX validation lane.

These targets express *user-visible* wait time on multi-thread BLAS +
warm cache + realistic scale. They are deliberately distinct from the
single-thread / cold-cache ``compute_s`` ratios used by the daily
before/after lane and the release-time reference baseline — see
``bench/README.md`` for the lane split.

``UX_TARGETS_VERSION`` is bumped whenever the table is edited so an old
report can be re-interpreted against the version it was produced under.
The validator stamps the version it used into its markdown output.
"""

from __future__ import annotations

from dataclasses import dataclass

UX_TARGETS_VERSION = "1"

# Scale dimension above which a record is treated as "stretch" — its
# wall_s is recorded in the report but not asserted, regardless of the
# scenario's wall_s_max. Mirrors the "1000-factor stretch row" from the
# UX latency target table.
STRETCH_N_FACTORS = 1000


@dataclass(frozen=True)
class UxTarget:
    """One row of the UX latency target table.

    ``wall_s_max`` is the absolute upper bound in seconds. ``None``
    marks a stretch row — the validator records the measurement but
    does not raise a red flag if it overshoots.
    """

    wall_s_max: float | None
    description: str


# Keyed by ``scenario_id``. Scenarios not in this dict are reported as
# ``skip`` (no UX expectation registered yet). Adding a new scenario to
# the harness does not silently demote its measurement — the validator
# surfaces unknown ``scenario_id``s in its report.
UX_TARGETS: dict[str, UxTarget] = {
    "S1": UxTarget(
        wall_s_max=5.0,
        description="Single factor evaluate + run_metrics — interactive notebook",
    ),
    "S2": UxTarget(
        wall_s_max=30.0,
        description="50-factor screen — user willing to wait",
    ),
    "S3": UxTarget(
        wall_s_max=120.0,
        description="200-factor screen — user switches to other tasks",
    ),
    "P1": UxTarget(
        wall_s_max=600.0,
        description="Scaling probe (up to 500-factor) — not perceived as crashed",
    ),
}


def lookup_target(scenario_id: str, *, n_factors: int | None) -> UxTarget | None:
    """Return the target for ``scenario_id`` or ``None`` if unregistered.

    Stretch demotion (``n_factors >= STRETCH_N_FACTORS``) returns a
    target with ``wall_s_max=None`` so callers can render the row
    without asserting against it.
    """
    base = UX_TARGETS.get(scenario_id)
    if base is None:
        return None
    if n_factors is not None and n_factors >= STRETCH_N_FACTORS:
        return UxTarget(
            wall_s_max=None,
            description=f"{base.description} (stretch: n_factors≥{STRETCH_N_FACTORS})",
        )
    return base


__all__ = [
    "STRETCH_N_FACTORS",
    "UX_TARGETS",
    "UX_TARGETS_VERSION",
    "UxTarget",
    "lookup_target",
]
