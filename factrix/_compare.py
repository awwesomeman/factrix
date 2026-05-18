"""``compare`` — leaderboard renderer for factrix artifacts (#177).

Pure projection: stacks ``FactorProfile`` / ``MetricsBundle`` /
``Survivors`` artifacts into a wide ``pl.DataFrame`` for sorting and
visual diff. No metric is recomputed; ``Survivors.adj_p`` is read
through, so Benjamini-Hochberg-Yekutieli (BHY) survivor leaderboards keep their adjusted p-values
without manual re-attach.

Heterogeneous context keys follow ``pl.concat(how="diagonal")`` —
union + null-fill — so an entry missing ``regime_id`` surfaces as a
``null`` cell rather than a silent drop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import polars as pl

from factrix._errors import UserInputError
from factrix._multi_factor import Survivors
from factrix._profile import FactorProfile
from factrix._run_metrics import MetricsBundle

CompareInput = list[FactorProfile] | list[MetricsBundle] | Survivors


def compare(
    artifacts: CompareInput,
    *,
    sort_by: str | None = None,
) -> pl.DataFrame:
    """Render a leaderboard ``pl.DataFrame`` for a list of artifacts.

    Args:
        artifacts: One of three input shapes (input-type dispatch — no
            ``compare_profiles`` / ``compare_bundles`` split):

            - ``list[FactorProfile]`` → identity + context +
              ``primary_stat`` / ``primary_stat_name`` / ``primary_p``
            - ``list[MetricsBundle]`` → identity + context + one column
              per standalone metric (``MetricOutput.value``)
            - :class:`~factrix.multi_factor.Survivors` → as the
              profile branch plus a final ``adj_p`` column (read from
              ``Survivors.adj_p``). ``expand_over`` dimensions surface
              as ordinary context columns (via ``profile.context[k]``).

            Mixed-type lists raise.
        sort_by: Column name to sort by. ``None`` keeps input order.
            ``nulls_last=True`` matches polars default so heterogeneous
            context / metric coverage stays robust.

    Returns:
        ``pl.DataFrame`` with columns laid out as ``factor_id``,
        ``forward_periods``, then context keys (union across entries,
        first-seen order), then branch-specific columns.

    Raises:
        UserInputError: Empty input; mixed artifact types; ``sort_by``
            not present in the output schema (with fuzzy suggestion).

    Examples:
        Leaderboard from a list of :class:`FactorProfile`:

        >>> import dataclasses
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> profiles = [
        ...     dataclasses.replace(
        ...         fx.evaluate(
        ...             compute_forward_return(
        ...                 fx.datasets.make_cs_panel(
        ...                     n_assets=100, n_dates=250, seed=i,
        ...                 ),
        ...                 forward_periods=5,
        ...             ),
        ...             cfg,
        ...         )["factor"],
        ...         factor_id=f"alpha_{i}",
        ...     )
        ...     for i in range(3)
        ... ]
        >>> leaderboard = fx.compare(profiles, sort_by="primary_p")

        Leaderboard from a :class:`~factrix.multi_factor.Survivors`
        (adds an ``adj_p`` column):

        >>> survivors = fx.multi_factor.bhy(profiles, q=0.5)
        >>> board = fx.compare(survivors, sort_by="adj_p")
    """
    if isinstance(artifacts, Survivors):
        if len(artifacts) == 0:
            raise UserInputError(
                func_name="compare",
                field="artifacts",
                value=artifacts,
                expected="non-empty Survivors",
                docs_path="api/compare/",
            )
        df = _profile_frame(artifacts.profiles).with_columns(
            pl.Series("adj_p", artifacts.adj_p)
        )
    else:
        entries = list(artifacts)
        if not entries:
            raise UserInputError(
                func_name="compare",
                field="artifacts",
                value=entries,
                expected="non-empty list of FactorProfile or MetricsBundle",
                docs_path="api/compare/",
            )
        kind = _classify(entries)
        if kind is FactorProfile:
            df = _profile_frame(cast("list[FactorProfile]", entries))
        else:
            df = _bundle_frame(cast("list[MetricsBundle]", entries))

    if sort_by is not None:
        if sort_by not in df.columns:
            raise UserInputError(
                func_name="compare",
                field="sort_by",
                value=sort_by,
                candidates=df.columns,
                docs_path="api/compare/",
            )
        df = df.sort(sort_by, nulls_last=True)

    return df


def _classify(entries: list[Any]) -> type:
    head = entries[0]
    if isinstance(head, FactorProfile):
        kind: type = FactorProfile
    elif isinstance(head, MetricsBundle):
        kind = MetricsBundle
    else:
        raise UserInputError(
            func_name="compare",
            field="artifacts[0]",
            value=head,
            expected="FactorProfile or MetricsBundle",
            docs_path="api/compare/",
        )
    mismatches = [
        (i, type(e).__name__)
        for i, e in enumerate(entries[1:], start=1)
        if not isinstance(e, kind)
    ]
    if mismatches:
        rendered = ", ".join(f"[{i}]={name}" for i, name in mismatches)
        raise UserInputError(
            func_name="compare",
            field="artifacts",
            value=f"mixed types ({kind.__name__} + {rendered})",
            expected=f"all entries to be {kind.__name__}",
            docs_path="api/compare/",
        )
    return kind


def _profile_frame(profiles: Sequence[FactorProfile]) -> pl.DataFrame:
    context_keys = _ordered_keys(p.context for p in profiles)
    rows: list[dict[str, Any]] = []
    for p in profiles:
        row: dict[str, Any] = {
            "factor_id": p.factor_id,
            "forward_periods": p.forward_periods,
        }
        for k in context_keys:
            row[k] = p.context.get(k)
        row["primary_stat"] = p.primary_stat
        row["primary_stat_name"] = p.primary_stat_name.value
        row["primary_p"] = p.primary_p
        rows.append(row)
    return pl.DataFrame(rows)


def _bundle_frame(bundles: Sequence[MetricsBundle]) -> pl.DataFrame:
    context_keys = _ordered_keys(b.context for b in bundles)
    metric_keys = _ordered_keys(b.metrics for b in bundles)
    rows: list[dict[str, Any]] = []
    for b in bundles:
        row: dict[str, Any] = {
            "factor_id": b.factor_id,
            "forward_periods": b.forward_periods,
        }
        for k in context_keys:
            row[k] = b.context.get(k)
        for m in metric_keys:
            output = b.metrics.get(m)
            row[m] = output.value if output is not None else None
        rows.append(row)
    return pl.DataFrame(rows)


def _ordered_keys(maps: Iterable[Mapping[str, Any]]) -> list[str]:
    """Union of mapping keys, ordered by first appearance.

    Matches ``pl.concat(how="diagonal")`` semantics without imposing
    alphabetic re-shuffling.
    """
    seen: dict[str, None] = {}
    for m in maps:
        for k in m:
            seen.setdefault(k, None)
    return list(seen)
