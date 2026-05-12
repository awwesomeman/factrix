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

from factrix._errors import UserInputError
from factrix._family import _resolve_family
from factrix.stats.multiple_testing import (
    bhy_adjusted_p,
    partial_conjunction_p,
    simes_p,
)

if TYPE_CHECKING:
    from factrix._profile import FactorProfile
    from factrix.stats import Estimator


@dataclass(frozen=True, slots=True, repr=False)
class Survivors:
    """Family-verb survivor container with rich Jupyter rendering.

    Procedure-agnostic: ``adj_p`` carries the verb's procedure-canonical
    adjusted p-value (BHY ``bhy_adjusted_p``, Holm step-down, Bonferroni
    ``min(p*m, 1)``, Romano-Wolf resampling, ...). The contract is
    ``survivor[i] iff adj_p[i] <= q`` — a duality every step-up /
    step-down family procedure satisfies.

    Invariants:
        ``len(profiles) == len(adj_p)`` and entries align in input
        order. Per-bucket independent step-up uses bucket-local ``n``
        and ``p_array``; ``adj_p[i]`` reflects ``profiles[i]``'s own
        bucket only (Benjamini & Bogomolov 2014 selective inference),
        not a global cross-bucket adjustment.

    Attributes:
        profiles: Survivors in input order.
        adj_p: Bucket-local adjusted p-values aligned with ``profiles``.
        q: Nominal FDR (or family-wise) target shared across all
            buckets.
        expand_over: Context keys used to partition the input into
            independent step-up buckets (``bhy``) or to aggregate
            conditions per identity (``partial_conjunction``). Empty
            tuple when the full input is one family.
        n_tests: Family size per bucket fed into the step-up math.
            Keying depends on the verb: ``bhy`` keys by
            ``expand_over_values`` tuple (``()`` for the single-bucket
            case); ``partial_conjunction`` keys by ``identity`` tuple
            (``(factor_id, forward_periods)``) and records the ``m``
            condition count per identity.
        pc_p: Raw partial-conjunction p-value per survivor (Benjamini
            & Heller 2008 Bonferroni-style: ``(m - min_pass + 1) *
            p_((min_pass))``, capped at 1). ``None`` when the verb is
            not ``partial_conjunction``.
        min_pass: ``k`` in the ``k`` of ``m`` partial conjunction test.
            ``None`` when the verb is not ``partial_conjunction``.
        n_passed_uncorr: Per-identity count of raw p-values strictly
            below ``q`` (descriptive — **not** used in inference; flags
            borderline cases and data gaps at a glance). The cutoff is
            the caller's ``q`` (same value driving the BHY step-up), so
            this count moves with ``q``; using it to override
            ``adj_p`` survivor selection is the anti-shopping failure
            mode this verb exists to prevent. ``None`` when the verb is
            not ``partial_conjunction``.
    """

    profiles: list[FactorProfile]
    adj_p: np.ndarray
    q: float
    expand_over: tuple[str, ...]
    n_tests: Mapping[tuple[Any, ...], int]
    pc_p: np.ndarray | None = None
    min_pass: int | None = None
    n_passed_uncorr: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.profiles)

    def _columns(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """Return (headers, rows) as already-formatted strings.

        Single source of truth for both ``__repr__`` and ``_repr_html_``
        — shape drift between text and HTML branches caused real bugs
        in earlier iterations.
        """
        headers: tuple[str, ...]
        pc_mode = self.pc_p is not None
        if pc_mode:
            headers = (
                "identity",
                "pc_p",
                "adj_p",
                "n_tests",
                "n_passed_uncorr",
            )
        elif self.expand_over:
            headers = ("expand_over_values", "identity", "primary_p", "adj_p")
        else:
            headers = ("identity", "primary_p", "adj_p")

        rows: list[tuple[str, ...]] = []
        if pc_mode:
            assert self.pc_p is not None
            assert self.n_passed_uncorr is not None
            for profile, pc, adj, n_passed in zip(
                self.profiles,
                self.pc_p,
                self.adj_p,
                self.n_passed_uncorr,
                strict=True,
            ):
                m = self.n_tests.get(profile.identity, 0)
                rows.append(
                    (
                        repr(profile.identity),
                        f"{float(pc):.4g}",
                        f"{float(adj):.4g}",
                        str(m),
                        str(int(n_passed)),
                    )
                )
            return headers, rows

        for profile, adj in zip(self.profiles, self.adj_p, strict=True):
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
        if self.min_pass is not None:
            parts.append(f"min_pass={self.min_pass}")
        if self.expand_over:
            parts.append(f"expand_over={list(self.expand_over)!r}")
            n_tests_repr = ", ".join(
                f"{k!r}: {v}" for k, v in sorted(self.n_tests.items())
            )
            parts.append(f"n_tests={{{n_tests_repr}}}")
        else:
            parts.append(f"n_tests={self.n_tests.get((), len(self.profiles))}")
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
    estimator: Estimator | None = None,
    q: float = 0.05,
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
        estimator: Optional inference-method override (#170). An
            :class:`~factrix.stats.Estimator` instance (e.g.
            ``factrix.stats.NeweyWest()``) names which p-value to feed
            the step-up; ``None`` uses each profile's ``primary_p``.
            The instance reports its applicability per cell and
            dispatches to a ``StatCode`` key in ``profile.stats``.
        q: Nominal false discovery rate target. The BHY step-up
            controls FDR ≤ q under positive-regression-dependence
            (PRDS); under arbitrary dependence the effective level is
            ``q / sum(1/k for k in 1..n)``. Default ``0.05``.

    Returns:
        ``Survivors`` container in input order; ``adj_p`` carries the
        bucket-local BHY-adjusted p-value and the survivor set is
        defined as ``adj_p <= q`` (single source of truth — no separate
        rejection mask path).

    Raises:
        UserInputError: On any family-resolution invariant failure
            (unknown / identity-shadowing ``expand_over`` name; an
            ``estimator`` not applicable to a profile's cell or whose
            dispatched ``StatCode`` is unpopulated; duplicate partition
            key — typically fixed by setting unique ``factor_id`` per
            profile or splitting via ``expand_over``).

    Warns:
        RuntimeWarning: When the input mixes ``forward_periods`` while
            ``expand_over`` is ``None`` — pooling horizons in one
            step-up dilutes the per-rank threshold and silently
            inflates FDR. Or when most ``expand_over`` buckets contain
            a single profile (BHY on n=1 is a raw cutoff and provides
            no FDR correction).
    """
    expand_over_tuple: tuple[str, ...] = tuple(expand_over) if expand_over else ()

    profile_list = list(profiles)
    if not profile_list:
        return Survivors(
            profiles=[],
            adj_p=np.zeros(0, dtype=np.float64),
            q=q,
            expand_over=expand_over_tuple,
            n_tests={},
        )

    _warn_on_mixed_horizons(profile_list, expand_over=expand_over)

    entries = _resolve_family(
        profile_list, verb="bhy", expand_over=expand_over, estimator=estimator
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

    adj_p_all = np.full(len(entries), np.nan, dtype=np.float64)
    n_tests: dict[tuple[Any, ...], int] = {}
    for bucket_key, ix in buckets.items():
        p_array = np.array([entries[i].p_value for i in ix], dtype=np.float64)
        adj_p_all[ix] = bhy_adjusted_p(p_array)
        n_tests[bucket_key] = len(ix)

    survivor_idxs = np.flatnonzero(adj_p_all <= q)
    return Survivors(
        profiles=[entries[i].profile for i in survivor_idxs],
        adj_p=adj_p_all[survivor_idxs],
        q=q,
        expand_over=expand_over_tuple,
        n_tests=n_tests,
    )


def partial_conjunction(
    profiles: Iterable[FactorProfile],
    *,
    min_pass: int,
    expand_over: Sequence[str],
    n_conditions: int | None = None,
    estimator: Estimator | None = None,
    q: float = 0.05,
) -> Survivors:
    """Partial conjunction screening: filter identities significant in
    at least ``min_pass`` of ``m`` expanded conditions, FDR-controlled.

    For "factor X is significant in universes A and B" style claims,
    naive ``set(survivors_A) & set(survivors_B)`` does not preserve FDR
    [Benjamini & Bogomolov 2014]. The partial conjunction test
    [Benjamini & Heller 2008] provides a contract-bearing path: per
    identity, combine the ``m`` per-condition p-values into a single
    PC p-value, then run BHY across identities.

    The PC p-value formula (Bonferroni-style, BH2008): for ``k`` =
    ``min_pass``, ``p_PC = (m - k + 1) * p_((k))`` capped at 1, where
    ``p_((k))`` is the ``k``-th smallest p-value across the identity's
    ``m`` conditions. ``k = m`` reduces to ``max(p)`` (full conjunction);
    ``k = 1`` reduces to ``m * min(p)`` (Bonferroni-union, forbidden
    here — see below).

    Args:
        profiles: Iterable of :class:`FactorProfile`. Conditions per
            identity come from ``expand_over``; multiple profiles
            sharing an identity must differ on at least one
            ``expand_over`` key (the standard ``_resolve_family``
            uniqueness check).
        min_pass: ``k`` in "k of m" — minimum number of conditions
            required to be significant. Must be ``>= 2``; ``min_pass=1``
            is union semantics (FDR ≈ ``2q`` under independence) and
            raises with a pointer to ``bhy(expand_over=...)``.
        expand_over: Required, non-empty. Context keys defining the
            condition axis. Identity dimensions (``factor_id`` /
            ``forward_periods``) are rejected by the family-resolution
            layer (#160 anti-shopping defense).
        n_conditions: Strict-mode declaration. ``None`` (lenient) lets
            ``m`` be inferred per identity from the data; an ``int``
            requires every identity to have exactly that many
            conditions and raises on mismatch (paper-grade — surfaces
            data gaps fail-loud).
        estimator: Optional inference-method override (#170). ``None``
            uses each profile's ``primary_p``.
        q: Nominal FDR target for the BHY step-up over PC p-values.
            Default ``0.05``.

    Returns:
        :class:`Survivors` in input order (deduplicated to one row per
        surviving identity, using the first profile of that identity as
        representative). ``adj_p`` is the BHY-adjusted PC p-value;
        ``pc_p`` is the raw PC p-value; ``n_tests[identity]`` is the
        condition count ``m``; ``n_passed_uncorr[i]`` is the count of
        raw p-values strictly below ``q`` for survivor ``i``.

    Raises:
        UserInputError: ``min_pass < 2`` (with ``min_pass=1`` flagged
            as union semantics); ``expand_over`` empty or ``None``;
            ``n_conditions < min_pass``; strict-mode ``n_conditions``
            mismatch with actual condition count; identity with fewer
            than ``min_pass`` conditions; family-resolution invariants
            (unknown ``expand_over`` key, identity-shadowing,
            duplicate partition key).
    """
    if min_pass < 2:
        if min_pass == 1:
            raise UserInputError(
                verb="partial_conjunction",
                field="min_pass",
                value=min_pass,
                expected=(
                    "min_pass >= 2. min_pass=1 reduces to union semantics "
                    "(rejects when any single condition is significant) "
                    "and inflates FDR to ~2q under independence; the "
                    "partial conjunction surface is contract-bearing only "
                    "for k >= 2. There is no drop-in 'any-significant' "
                    "verb — the closest available is "
                    "bhy(profiles, expand_over=[...]), which expands the "
                    "family across conditions instead of aggregating "
                    "(survivor unit becomes (factor, condition) pair, "
                    "not factor identity; see docs)"
                ),
                docs_path="api/partial-conjunction#min-pass-must-be-at-least-2",
            )
        raise UserInputError(
            verb="partial_conjunction",
            field="min_pass",
            value=min_pass,
            expected="positive integer >= 2",
            docs_path="api/partial-conjunction#min-pass-must-be-at-least-2",
        )

    if not expand_over:
        raise UserInputError(
            verb="partial_conjunction",
            field="expand_over",
            value=expand_over,
            expected=(
                "non-empty list of context keys naming the condition "
                "axis (e.g. ['universe_id'] for cross-universe "
                "replication, ['fwd_period'] for cross-horizon "
                "replication). partial_conjunction is undefined without "
                "a condition axis"
            ),
            docs_path="api/partial-conjunction#expand-over-required",
        )

    if n_conditions is not None and n_conditions < min_pass:
        raise UserInputError(
            verb="partial_conjunction",
            field="n_conditions",
            value=n_conditions,
            expected=(
                f"n_conditions >= min_pass ({min_pass}); requiring more "
                "passes than total conditions is unsatisfiable"
            ),
            docs_path="api/partial-conjunction#n-conditions",
        )

    expand_over_tuple: tuple[str, ...] = tuple(expand_over)
    profile_list = list(profiles)

    if not profile_list:
        return Survivors(
            profiles=[],
            adj_p=np.zeros(0, dtype=np.float64),
            q=q,
            expand_over=expand_over_tuple,
            n_tests={},
            pc_p=np.zeros(0, dtype=np.float64),
            min_pass=min_pass,
            n_passed_uncorr=np.zeros(0, dtype=np.int64),
        )

    entries = _resolve_family(
        profile_list,
        verb="partial_conjunction",
        expand_over=expand_over,
        estimator=estimator,
    )

    groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for entry in entries:
        groups[entry.identity].append(entry)

    identities_ordered: list[tuple[str, int]] = []
    seen_identities: set[tuple[str, int]] = set()
    for entry in entries:
        if entry.identity not in seen_identities:
            seen_identities.add(entry.identity)
            identities_ordered.append(entry.identity)

    pc_p_arr = np.empty(len(identities_ordered), dtype=np.float64)
    n_passed_arr = np.empty(len(identities_ordered), dtype=np.int64)
    n_tests_per_identity: dict[tuple[Any, ...], int] = {}
    rep_profiles: list[FactorProfile] = []

    for i, identity in enumerate(identities_ordered):
        group = groups[identity]
        m = len(group)

        if n_conditions is not None and m != n_conditions:
            raise UserInputError(
                verb="partial_conjunction",
                field="n_conditions",
                value=n_conditions,
                expected=(
                    f"identity {identity} has {m} condition(s) in data "
                    f"but n_conditions={n_conditions} declared (strict "
                    "mode). Pass n_conditions=None for lenient mode, or "
                    "fix the input so every identity has exactly "
                    f"{n_conditions} conditions"
                ),
                docs_path="api/partial-conjunction#strict-vs-lenient",
            )

        if m < min_pass:
            raise UserInputError(
                verb="partial_conjunction",
                field="profiles",
                value=identity,
                expected=(
                    f"identity {identity} has only {m} condition(s) but "
                    f"min_pass={min_pass} requires at least that many. "
                    "Either drop this identity from the input or lower "
                    "min_pass"
                ),
                docs_path="api/partial-conjunction#insufficient-conditions",
            )

        ps = np.array([e.p_value for e in group], dtype=np.float64)
        pc_p_arr[i] = partial_conjunction_p(ps, min_pass=min_pass)
        n_passed_arr[i] = int(np.sum(ps < q))
        n_tests_per_identity[identity] = m
        rep_profiles.append(group[0].profile)

    if n_conditions is None:
        m_values = set(n_tests_per_identity.values())
        if len(m_values) > 1:
            warnings.warn(
                "partial_conjunction: lenient mode (n_conditions=None) is "
                f"running with heterogeneous condition counts m={sorted(m_values)} "
                "across identities. PC p-values are valid marginally but the "
                "k/m bar differs per identity (an identity with m=8 faces a "
                "much stricter Bonferroni multiplier at k=2 than one with "
                "m=2). For cross-identity comparability pass "
                "n_conditions=<int> (strict mode) or split the call.",
                RuntimeWarning,
                stacklevel=2,
            )

    adj_p_all = bhy_adjusted_p(pc_p_arr)
    survivor_idx = np.flatnonzero(adj_p_all <= q)

    surviving_identities = [identities_ordered[i] for i in survivor_idx]
    return Survivors(
        profiles=[rep_profiles[i] for i in survivor_idx],
        adj_p=adj_p_all[survivor_idx],
        q=q,
        expand_over=expand_over_tuple,
        n_tests={ident: n_tests_per_identity[ident] for ident in surviving_identities},
        pc_p=pc_p_arr[survivor_idx],
        min_pass=min_pass,
        n_passed_uncorr=n_passed_arr[survivor_idx],
    )


def bhy_hierarchical(
    profiles: Iterable[FactorProfile],
    *,
    group: str,
    estimator: Estimator | None = None,
    q: float = 0.05,
) -> Survivors:
    """Hierarchical BHY: control FDR across groups then within groups.

    For factor sets with a natural group structure (momentum / value /
    quality families; cross-region universes), the Yekutieli 2008
    two-stage procedure controls *group-level* FDR ≤ ``q`` on the outer
    layer (Simes group representative + BHY) and *within-group* FDR
    ≤ ``q`` on the inner layer (BHY restricted to passing groups). Flat
    BHY across the whole input loses group-level interpretability and
    pays full m-correction even when most groups are dead; this verb
    keeps the group answer first-class.

    Args:
        profiles: Iterable of :class:`FactorProfile`. Each profile is
            assigned to one group via ``profile.context[group]``;
            within a group, ``(factor_id, forward_periods)`` must be
            unique (the standard ``_resolve_family`` partition-key
            check).
        group: Single context key naming the group axis (e.g.
            ``"family"`` for momentum / value / quality, ``"region"``
            for cross-region replication). Identity dimensions
            (``factor_id`` / ``forward_periods``) are rejected — using
            them as group keys collapses each cell to its own group
            and trivializes the procedure.
        estimator: Optional inference-method override. ``None`` uses
            each profile's ``primary_p``.
        q: Nominal FDR target shared by both layers. Default ``0.05``.

    Returns:
        :class:`Survivors` in input order. ``adj_p`` is the
        max-of-layers fold ``max(outer_adj_p[g], inner_adj_p[i])`` so
        the universal duality ``survivor[i] iff adj_p[i] <= q`` holds.
        ``n_tests`` maps group key to its inner family size;
        ``expand_over`` is ``(group,)`` and per-survivor group labels
        are recoverable via ``profile.context[group]``.

    Raises:
        UserInputError: ``group`` shadows an identity dimension
            (``factor_id`` / ``forward_periods``); ``group`` missing
            from a profile's ``context``; duplicate ``(identity,
            group_value)`` partition key; estimator applicability
            failures (per ``_resolve_family``).

    References:
        Yekutieli, D. (2008). "Hierarchical false discovery rate-
        controlling methodology." JASA 103(481), 309-316.
    """
    profile_list = list(profiles)
    expand_over_tuple: tuple[str, ...] = (group,)

    if not profile_list:
        return Survivors(
            profiles=[],
            adj_p=np.zeros(0, dtype=np.float64),
            q=q,
            expand_over=expand_over_tuple,
            n_tests={},
        )

    entries = _resolve_family(
        profile_list,
        verb="bhy_hierarchical",
        expand_over=[group],
        estimator=estimator,
    )

    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        groups[entry.expand_over_values[0]].append(idx)
    group_keys_ordered = list(
        dict.fromkeys(entry.expand_over_values[0] for entry in entries)
    )

    n_groups = len(group_keys_ordered)
    group_simes = np.empty(n_groups, dtype=np.float64)
    inner_adjs: list[np.ndarray] = []
    n_tests: dict[tuple[Any, ...], int] = {}
    for g_idx, gkey in enumerate(group_keys_ordered):
        member_idxs = groups[gkey]
        member_p = np.array([entries[i].p_value for i in member_idxs], dtype=np.float64)
        group_simes[g_idx] = simes_p(member_p)
        inner_adjs.append(bhy_adjusted_p(member_p))
        n_tests[(gkey,)] = len(member_idxs)

    outer_adj = bhy_adjusted_p(group_simes)

    adj_p_all = np.empty(len(entries), dtype=np.float64)
    for g_idx, gkey in enumerate(group_keys_ordered):
        for j, idx in enumerate(groups[gkey]):
            adj_p_all[idx] = max(outer_adj[g_idx], inner_adjs[g_idx][j])

    survivor_idxs = np.flatnonzero(adj_p_all <= q)
    return Survivors(
        profiles=[entries[i].profile for i in survivor_idxs],
        adj_p=adj_p_all[survivor_idxs],
        q=q,
        expand_over=expand_over_tuple,
        n_tests=n_tests,
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
