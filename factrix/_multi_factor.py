"""v0.5 ``multi_factor`` namespace — collection-level FDR control (§7.4).

Currently exposes ``bhy`` only. ``redundancy_matrix`` /
``spanning_test`` / ``orthogonalize`` are listed in plan §7.4 and will
land alongside the v0.4 deletion sweep that retires the existing
v0.4 ``redundancy_matrix`` / ``spanning`` modules.

Family key for BHY = registry dispatch key (plan §5.6). Mode A and
Mode B never share a family — different null distributions and
effective sample sizes. Mode B sparse collapses scope into the
``_SCOPE_COLLAPSED`` sentinel so ``individual_sparse`` and
``common_sparse`` profiles at N=1 sit in the same family.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from factrix._codes import StatCode
from factrix._registry import _DispatchKey, _route_scope
from factrix.stats.multiple_testing import bhy_adjust

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


def _family_key(profile: "FactorProfile") -> _DispatchKey:
    """Derive the BHY family key from a ``FactorProfile``.

    Reuses ``_route_scope`` so the sparse-N=1 collapse mirrors
    ``_evaluate`` exactly — ``individual_sparse`` and ``common_sparse``
    profiles at N=1 sit in the same family (§5.4.1 / §5.6).
    """
    scope = _route_scope(
        profile.config.scope, profile.config.signal, profile.mode,
    )
    return _DispatchKey(
        scope=scope,
        signal=profile.config.signal,
        metric=profile.config.metric,
        mode=profile.mode,
    )


def _gate_p_value(
    profile: "FactorProfile", gate: StatCode | None,
) -> float:
    """Return ``primary_p`` (default) or ``stats[gate]`` for the BHY input."""
    if gate is None:
        return profile.primary_p
    return profile.stats[gate]


def bhy(
    profiles: Iterable["FactorProfile"],
    *,
    threshold: float = 0.05,
    gate: StatCode | None = None,
) -> list["FactorProfile"]:
    """BHY step-up FDR within each family; return the surviving subset.

    ``threshold`` is the FDR level (plan §7.5 invariant — never
    ``alpha``). ``gate`` chooses the p-value the test runs on:
    ``None`` = procedure-canonical ``primary_p``; otherwise the named
    ``StatCode`` is read from every profile's ``stats`` mapping
    (``KeyError`` if a family member does not populate it).

    Profiles are grouped by family key (= registry ``_DispatchKey``);
    each family runs an independent BHY step-up on its p-values.
    Cross-family aggregation is the user's responsibility and is
    deliberately not done here (§5.6).
    """
    families: dict[_DispatchKey, list["FactorProfile"]] = defaultdict(list)
    for profile in profiles:
        families[_family_key(profile)].append(profile)

    # UX-2 from review: BHY on a size-1 family is identical to a raw
    # threshold check — useful diagnostic for the (common bug) of
    # passing one profile per cell from a sweep and assuming FDR is
    # being controlled. Warn but do not raise; user may have only one
    # candidate in a family legitimately.
    singleton_families = sum(1 for fps in families.values() if len(fps) == 1)
    if singleton_families and len(families) > 1:
        warnings.warn(
            f"bhy: {singleton_families} of {len(families)} families "
            "contain a single profile — BHY on n=1 is identical to a "
            "raw threshold and provides no FDR correction. Group ≥2 "
            "same-family profiles per call for meaningful control.",
            RuntimeWarning,
            stacklevel=2,
        )

    survivors: list["FactorProfile"] = []
    for family_profiles in families.values():
        if not family_profiles:
            continue
        p_array = np.array(
            [_gate_p_value(p, gate) for p in family_profiles],
            dtype=np.float64,
        )
        mask = bhy_adjust(p_array, fdr=threshold)
        survivors.extend(
            fp for fp, accept in zip(family_profiles, mask) if accept
        )
    return survivors
