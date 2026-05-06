"""v0.5 ``multi_factor`` namespace — collection-level FDR control (§7.4).

Currently exposes ``bhy`` only. ``redundancy_matrix`` /
``spanning_test`` / ``orthogonalize`` are listed in plan §7.4 and will
land alongside the v0.4 deletion sweep that retires the existing
v0.4 ``redundancy_matrix`` / ``spanning`` modules.

BHY family key = ``(_DispatchKey, forward_periods)``. The registry
dispatch key alone is *not* sufficient: pooling profiles across
horizons inflates family size and dilutes the step-up threshold,
silently destroying FDR control. PANEL and TIMESERIES never share a
family — different null distributions and effective sample sizes.
Sparse TIMESERIES collapses scope into the ``_SCOPE_COLLAPSED``
sentinel so ``individual_sparse`` and ``common_sparse`` profiles at
N=1 sit in the same family at matching horizon.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from factrix._codes import StatCode
from factrix._registry import _dispatch_key_for, _DispatchKey
from factrix.stats.multiple_testing import bhy_adjust

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


@dataclass(frozen=True, slots=True)
class _FamilyKey:
    """BHY family coordinate = procedure cell × forward-return horizon.

    Kept distinct from ``_DispatchKey`` (which the registry uses to
    route cells to procedures, horizon-agnostic) because BHY family
    membership *must* split on ``forward_periods``. Each horizon
    carries its own null distribution and effective sample size; mixing
    them in one step-up batch dilutes the per-rank threshold
    (``q × k / N``) and silently inflates FDR above the nominal level.
    """

    dispatch: _DispatchKey
    forward_periods: int


def _family_key(profile: FactorProfile) -> _FamilyKey:
    """Derive the BHY family key from a ``FactorProfile``.

    Routes through ``_dispatch_key_for`` so the sparse-N=1 collapse and
    key construction stay byte-identical to ``_evaluate`` — sharing the
    helper, not a copy of its body, is what makes
    ``individual_sparse`` / ``common_sparse`` profiles at N=1 land in the
    same family at matching ``forward_periods`` (§5.4.1 / §5.6).
    """
    dispatch = _dispatch_key_for(
        profile.config.scope,
        profile.config.signal,
        profile.config.metric,
        profile.mode,
    )
    return _FamilyKey(
        dispatch=dispatch,
        forward_periods=profile.config.forward_periods,
    )


def _gate_p_value(
    profile: FactorProfile,
    gate: StatCode | None,
) -> float:
    """Return ``primary_p`` (default) or ``stats[gate]`` for the BHY input.

    ``gate`` must be a p-value ``StatCode`` (``is_p_value`` is True).
    BHY step-up math requires probabilities; feeding a t-stat or HHI
    would silently produce incoherent FDR control. Validation lives at
    the caller (``bhy``) so the error fires once per call, not once per
    profile.
    """
    if gate is None:
        return profile.primary_p
    return profile.stats[gate]


def bhy(
    profiles: Iterable[FactorProfile],
    *,
    threshold: float = 0.05,
    gate: StatCode | None = None,
) -> list[FactorProfile]:
    """BHY step-up FDR within each family; return the surviving subset.

    Profiles are grouped by family key (= dispatch cell × forward
    horizon); each family runs an independent BHY step-up on its
    p-values. Cross-family aggregation is the user's responsibility
    and is deliberately not done here. A warning fires when most
    families are size-1 (BHY on a singleton is identical to a raw
    threshold and provides no FDR correction).

    Args:
        profiles: Iterable of ``FactorProfile`` to screen. Profiles
            from different cells / horizons partition into separate
            families automatically.
        threshold: FDR level (not ``alpha``). Default ``0.05``.
        gate: ``StatCode`` whose ``is_p_value`` is ``True`` selects
            an alternate p-value from each profile's ``stats``;
            ``None`` uses the procedure-canonical ``primary_p``.

    Returns:
        The subset of ``profiles`` that survive the BHY step-up
        within their respective families, in input order across
        families.

    Raises:
        ValueError: If ``gate`` is a ``StatCode`` whose
            ``is_p_value`` is ``False`` (BHY step-up requires
            probabilities).
        KeyError: If ``gate`` is set and any profile in a family
            does not populate that key in ``stats``.

    Warns:
        RuntimeWarning: If most families contain a single profile —
            BHY on n=1 provides no correction beyond a raw cutoff.
    """
    if gate is not None and not gate.is_p_value:
        raise ValueError(
            f"bhy(gate={gate.name}): BHY step-up requires p-value input, "
            f"but {gate.name} is not a probability. Pass a *_P StatCode "
            "(e.g. StatCode.IC_P, StatCode.FM_LAMBDA_P) or omit `gate=` "
            "to use the procedure-canonical primary_p.",
        )
    families: dict[_FamilyKey, list[FactorProfile]] = defaultdict(list)
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

    survivors: list[FactorProfile] = []
    for family_profiles in families.values():
        if not family_profiles:
            continue
        p_array = np.array(
            [_gate_p_value(p, gate) for p in family_profiles],
            dtype=np.float64,
        )
        mask = bhy_adjust(p_array, fdr=threshold)
        survivors.extend(
            fp for fp, accept in zip(family_profiles, mask, strict=False) if accept
        )
    return survivors
