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

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from factrix._axis import FactorScope, Mode, Signal
from factrix._codes import StatCode
from factrix._registry import (
    _SCOPE_COLLAPSED,
    _DispatchKey,
    _ScopeCollapsedSentinel,
)
from factrix.stats.multiple_testing import bhy_adjust

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


def _family_key(profile: "FactorProfile") -> _DispatchKey:
    """Derive the BHY family key from a ``FactorProfile``.

    Mirrors ``_evaluate``'s sparse-N=1 collapse: when ``signal=SPARSE``
    and ``mode=TIMESERIES`` the user-facing scope is rewritten to the
    ``_SCOPE_COLLAPSED`` sentinel so ``individual_sparse`` and
    ``common_sparse`` profiles share one family (§5.4.1 / §5.6).
    """
    scope: FactorScope | _ScopeCollapsedSentinel = profile.config.scope
    if profile.config.signal is Signal.SPARSE and profile.mode is Mode.TIMESERIES:
        scope = _SCOPE_COLLAPSED
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
