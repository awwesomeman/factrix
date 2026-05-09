"""v0.5 ``multi_factor`` namespace — collection-level FDR control (§7.4).

Currently exposes ``bhy`` only. ``redundancy_matrix`` /
``spanning_test`` / ``orthogonalize`` are listed in plan §7.4 and will
land alongside the v0.4 deletion sweep that retires the existing
v0.4 ``redundancy_matrix`` / ``spanning`` modules.

Family declaration is now explicit: the input list ``profiles`` *is*
the family, optionally split per-bucket via ``expand_over`` (Benjamini
& Bogomolov 2014 selective-inference framework). The v0.4-era
auto-partition by dispatch cell × forward horizon was retired in #161 —
caller responsibility now, both because the implicit policy was opaque
and because v0.5 ``identity`` already encodes ``forward_periods`` (and
would silently flag mixed-cell inputs as duplicate identities).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from factrix._codes import StatCode
from factrix._family import _resolve_family
from factrix.stats.multiple_testing import bhy_adjust

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


_DEPRECATED_KWARGS = {
    "threshold": "q",
    "gate": "p_stat",
}
_DEFAULT_Q = 0.05


def bhy(
    profiles: Iterable[FactorProfile],
    *,
    expand_over: Sequence[str] | None = None,
    p_stat: StatCode | None = None,
    q: float | None = None,
    **deprecated: Any,
) -> list[FactorProfile]:
    """BHY step-up FDR within one declared family; return the survivors.

    The input list is treated as a single family. When ``expand_over``
    is supplied, one independent step-up runs per unique tuple of
    ``profile.context[k] for k in expand_over`` (Benjamini & Bogomolov
    2014 selective inference). Cell / horizon partitioning is the
    caller's responsibility — v0.5 retired the implicit auto-split.

    Args:
        profiles: Iterable of ``FactorProfile``. The full input is one
            family unless ``expand_over`` further partitions it.
        expand_over: Optional context keys whose distinct value tuples
            split the input into independent BHY step-up batches.
            ``None`` runs a single step-up over all profiles.
        p_stat: Alternate p-value :class:`StatCode` (must satisfy
            ``is_p_value``). ``None`` uses each profile's procedure-
            canonical ``primary_p`` (e.g. ``IC_P`` for IC,
            ``FM_LAMBDA_P`` for Fama–MacBeth, ``CAAR_P`` for event
            studies).
        q: Nominal false discovery rate target. The BHY step-up
            controls FDR ≤ q under positive-regression-dependence
            (PRDS); under arbitrary dependence the effective level is
            ``q / sum(1/k for k in 1..n)``. Default ``0.05``.

    Returns:
        Survivors in input order.

    Raises:
        UserInputError: On any family-resolution invariant failure
            (unknown / identity-shadowing ``expand_over`` name; missing
            or non-probability ``p_stat``; duplicate partition key —
            typically fixed by setting unique ``factor_id`` per profile
            or splitting via ``expand_over``).

    Warns:
        DeprecationWarning: When the v0.4 kwargs ``threshold=`` /
            ``gate=`` are used.
        RuntimeWarning: When the input mixes ``forward_periods`` while
            ``expand_over`` is ``None`` — pooling horizons in one
            step-up dilutes the per-rank threshold and silently
            inflates FDR. Or when most ``expand_over`` buckets contain
            a single profile (BHY on n=1 is a raw cutoff and provides
            no FDR correction).
    """
    expand_over, p_stat, q = _apply_deprecated_kwargs(
        expand_over=expand_over, p_stat=p_stat, q=q, deprecated=deprecated
    )

    profile_list = list(profiles)
    if not profile_list:
        return []

    _warn_on_mixed_horizons(profile_list, expand_over=expand_over)

    entries = _resolve_family(
        profile_list, verb="bhy", expand_over=expand_over, p_stat=p_stat
    )

    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        buckets[entry.expand_over_values].append(idx)

    singleton = sum(1 for ix in buckets.values() if len(ix) == 1)
    if singleton and len(buckets) > 1:
        warnings.warn(
            f"bhy: {singleton} of {len(buckets)} expand_over buckets "
            "contain a single profile — BHY on n=1 is identical to a "
            "raw threshold and provides no FDR correction.",
            RuntimeWarning,
            stacklevel=2,
        )

    survivor_idxs: list[int] = []
    for ix in buckets.values():
        p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
        mask = bhy_adjust(p_array, fdr=q)
        survivor_idxs.extend(i for i, accept in zip(ix, mask, strict=True) if accept)

    survivor_idxs.sort()
    return [entries[i].profile for i in survivor_idxs]


def _warn_on_mixed_horizons(
    profiles: list[FactorProfile],
    *,
    expand_over: Sequence[str] | None,
) -> None:
    if expand_over:
        return
    horizons = {p.config.forward_periods for p in profiles}
    if len(horizons) > 1:
        warnings.warn(
            f"bhy: input mixes forward_periods={sorted(horizons)} but "
            "expand_over is None — different horizons have different "
            "null distributions; pooling them in one step-up dilutes "
            "the per-rank threshold and silently inflates FDR. Either "
            "split the call per horizon, or set expand_over=[<context "
            "key>] to declare per-bucket families.",
            RuntimeWarning,
            stacklevel=3,
        )


def _apply_deprecated_kwargs(
    *,
    expand_over: Sequence[str] | None,
    p_stat: StatCode | None,
    q: float | None,
    deprecated: dict[str, Any],
) -> tuple[Sequence[str] | None, StatCode | None, float]:
    unknown = set(deprecated) - _DEPRECATED_KWARGS.keys()
    if unknown:
        raise TypeError(
            f"bhy() got unexpected keyword argument(s): {sorted(unknown)!r}"
        )

    if "threshold" in deprecated:
        if q is not None:
            raise TypeError(
                "bhy(): pass either `q=` or the deprecated `threshold=`, not both."
            )
        q = deprecated["threshold"]
    if "gate" in deprecated:
        if p_stat is not None:
            raise TypeError(
                "bhy(): pass either `p_stat=` or the deprecated `gate=`, not both."
            )
        p_stat = deprecated["gate"]

    if deprecated:
        renamed = ", ".join(
            f"{old}= → {new}="
            for old, new in _DEPRECATED_KWARGS.items()
            if old in deprecated
        )
        warnings.warn(
            f"bhy(): {renamed} (deprecated, removed next release).",
            DeprecationWarning,
            stacklevel=4,
        )

    return expand_over, p_stat, q if q is not None else _DEFAULT_Q
