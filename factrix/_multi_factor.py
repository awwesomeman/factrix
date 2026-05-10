"""v0.5 ``multi_factor`` namespace — collection-level FDR control (§7.4).

Currently exposes ``bhy`` only. ``redundancy_matrix`` /
``spanning_test`` / ``orthogonalize`` are listed in plan §7.4 and will
land alongside the v0.4 deletion sweep that retires the existing
v0.4 ``redundancy_matrix`` / ``spanning`` modules.

Family declaration is now explicit: the input list ``profiles`` *is*
the family, optionally split per-bucket via ``expand_over`` (Benjamini
& Bogomolov 2014 selective-inference framework). The previous
auto-partition by dispatch cell × forward horizon was retired in #161 —
caller responsibility now, both because the implicit policy was opaque
and because ``identity`` already encodes ``forward_periods`` (and would
silently flag mixed-cell inputs as duplicate identities).
"""

from __future__ import annotations

import html
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from factrix._codes import StatCode
from factrix._family import _resolve_family
from factrix.stats.multiple_testing import bhy_adjusted_p

if TYPE_CHECKING:
    from factrix._profile import FactorProfile


_DEPRECATED_KWARGS = {
    "threshold": "q",
    "gate": "p_stat",
}
_DEFAULT_Q = 0.05


@dataclass(frozen=True, slots=True, repr=False)
class Survivors:
    """Family-verb survivor container with rich Jupyter rendering.

    Procedure-agnostic: ``adj_q`` carries the verb's procedure-canonical
    adjusted p-value (BHY ``bhy_adjusted_p``, Holm step-down, Bonferroni
    ``min(p*m, 1)``, Romano-Wolf resampling, ...). The contract is
    ``survivor[i] iff adj_q[i] <= q`` — a duality every step-up /
    step-down family procedure satisfies.

    Invariants:
        ``len(profiles) == len(adj_q)`` and entries align in input
        order. Per-bucket independent step-up uses bucket-local ``n``
        and ``p_array``; ``adj_q[i]`` reflects ``profiles[i]``'s own
        bucket only (Benjamini & Bogomolov 2014 selective inference),
        not a global cross-bucket adjustment.

    Attributes:
        profiles: Survivors in input order.
        adj_q: Bucket-local adjusted p-values aligned with ``profiles``.
        q: Nominal FDR (or family-wise) target shared across all
            buckets.
        expand_over: Context keys used to partition the input into
            independent step-up buckets. Empty tuple when the full
            input is one family.
        n_total: Per-bucket family size fed into the step-up math,
            keyed by the bucket's ``expand_over_values`` tuple
            (``()`` for the single-bucket case). Records the ``m``
            two-stage screening uses.
    """

    profiles: list[FactorProfile]
    adj_q: np.ndarray
    q: float
    expand_over: tuple[str, ...]
    n_total: Mapping[tuple[Any, ...], int]

    def __len__(self) -> int:
        return len(self.profiles)

    def _columns(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """Return (headers, rows) as already-formatted strings.

        Single source of truth for both ``__repr__`` and ``_repr_html_``
        — shape drift between text and HTML branches caused real bugs
        in earlier iterations.
        """
        headers: tuple[str, ...]
        if self.expand_over:
            headers = ("expand_over_values", "identity", "primary_p", "adj_q")
        else:
            headers = ("identity", "primary_p", "adj_q")

        rows: list[tuple[str, ...]] = []
        for profile, adj in zip(self.profiles, self.adj_q, strict=True):
            cells = (
                repr(profile.identity),
                f"{profile.primary_p:.4g}",
                f"{float(adj):.4g}",
            )
            if self.expand_over:
                bucket_repr = repr(tuple(profile.context[k] for k in self.expand_over))
                rows.append((bucket_repr, *cells))
            else:
                rows.append(cells)
        return headers, rows

    def _header_summary(self) -> str:
        parts = [f"n={len(self.profiles)}", f"q={self.q:g}"]
        if self.expand_over:
            parts.append(f"expand_over={list(self.expand_over)!r}")
            n_total_repr = ", ".join(
                f"{k!r}: {v}" for k, v in sorted(self.n_total.items())
            )
            parts.append(f"n_total={{{n_total_repr}}}")
        else:
            parts.append(f"n_total={self.n_total.get((), len(self.profiles))}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        head = f"Survivors({self._header_summary()})"
        if not self.profiles:
            return head
        headers, rows = self._columns()
        widths = [
            max(len(col), *(len(row[i]) for row in rows))
            for i, col in enumerate(headers)
        ]
        header_line = "  ".join(
            c.ljust(w) for c, w in zip(headers, widths, strict=True)
        )
        body = "\n".join(
            "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True))
            for row in rows
        )
        return f"{head}\n{header_line}\n{body}"

    def _repr_html_(self) -> str:
        headers, rows = self._columns()
        thead = "".join(
            f"<th style='text-align:left'>{html.escape(h)}</th>" for h in headers
        )
        tbody = "".join(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        caption = html.escape(f"Survivors ({self._header_summary()})")
        return (
            "<table class='factrix-survivors'>"
            f"<caption>{caption}</caption>"
            f"<thead><tr>{thead}</tr></thead>"
            f"<tbody>{tbody}</tbody>"
            "</table>"
        )


def bhy(
    profiles: Iterable[FactorProfile],
    *,
    expand_over: Sequence[str] | None = None,
    p_stat: StatCode | None = None,
    q: float | None = None,
    **deprecated: Any,
) -> Survivors:
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
        ``Survivors`` container in input order; ``adj_q`` carries the
        bucket-local BHY-adjusted p-value and the survivor set is
        defined as ``adj_q <= q`` (single source of truth — no separate
        rejection mask path).

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
    expand_over_tuple: tuple[str, ...] = tuple(expand_over) if expand_over else ()

    profile_list = list(profiles)
    if not profile_list:
        return Survivors(
            profiles=[],
            adj_q=np.zeros(0, dtype=np.float64),
            q=q,
            expand_over=expand_over_tuple,
            n_total={},
        )

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

    adj_q_all = np.full(len(entries), np.nan, dtype=np.float64)
    n_total: dict[tuple[Any, ...], int] = {}
    for bucket_key, ix in buckets.items():
        p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
        adj_q_all[ix] = bhy_adjusted_p(p_array)
        n_total[bucket_key] = len(ix)

    survivor_idxs = np.flatnonzero(adj_q_all <= q)
    return Survivors(
        profiles=[entries[i].profile for i in survivor_idxs],
        adj_q=adj_q_all[survivor_idxs],
        q=q,
        expand_over=expand_over_tuple,
        n_total=n_total,
    )


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
        # BUMP-TIME: pin the actual SemVer cutoff in this string before
        # the next bump (e.g. "removed in v0.12.0"). #161 ships under the
        # release-train; the train's cz bump on main is where the cutoff
        # version becomes concrete. Mirror the same version in CHANGELOG
        # `### Deprecated` so user-facing message and changelog agree.
        warnings.warn(
            f"bhy(): {renamed} (deprecated, removed in a future release).",
            DeprecationWarning,
            stacklevel=4,
        )

    return expand_over, p_stat, q if q is not None else _DEFAULT_Q
