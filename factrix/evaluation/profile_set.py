"""ProfileSet — homogeneous collection of FactorProfile instances.

Polars-native: the df representation is kept alongside the profile
tuple and both are updated in lockstep. Users choose whichever view
fits their operation:

- ``filter(pl.Expr)`` — predicate on columns; bulk, fast, composable
- ``filter(Callable)`` — escape hatch for complex Python conditions
- ``iter_profiles()`` — typed dataclass access with IDE completion
- ``to_polars()`` — DataFrame view for export / joins / polars chains

The Expr path validates that the predicate is a pure row-wise Boolean
expression; aggregations or column mutations (which would break the
row-count invariant) are rejected with a targeted error.
"""

from __future__ import annotations

import dataclasses
import types
import typing
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Iterable, Iterator, Literal, TypeVar

import numpy as np
import polars as pl

from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p

if TYPE_CHECKING:
    from factrix.evaluation.profiles._base import FactorProfile

P = TypeVar("P")

# Column names multiple_testing_correct writes onto the DataFrame view.
# `_MT_METHOD_COL` doubles as the "already corrected" marker — presence
# of this column is what the double-apply guard keys on.
_MT_METHOD_COL = "mt_method"
_MT_P_SOURCE_COL = "mt_p_source"
_MT_FDR_COL = "mt_fdr"
_MT_N_TOTAL_COL = "mt_n_total"


class ProfileSet(Generic[P]):
    """Type-homogeneous collection of factor profiles.

    Invariants:
        - All elements are instances of the same concrete Profile class.
        - ``self._profiles`` and ``self._df`` are kept in lockstep: same
          row order, same length. Internal operations that change one
          must change the other via ``_derive``.
        - The df may carry extra columns injected by
          ``multiple_testing_correct`` (``p_adjusted``, ``bhy_significant``,
          ``canonical_p``); these are not on the dataclass schema.

    The set may be empty; an empty set still knows its profile class
    so downstream code can call ``to_polars()`` / ``iter_profiles()``
    without special-casing.
    """

    __slots__ = (
        "_profiles", "_df", "_profile_cls", "_canonical_override",
        "_warned_uncorrected",
    )

    _profiles: tuple[P, ...]
    _df: pl.DataFrame
    _profile_cls: type[P]
    _canonical_override: str | None
    # Sticky one-shot flag so a chain like `ps.filter(...).rank_by(...)`
    # warns once, not on every decision step.
    _warned_uncorrected: bool

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        profiles: Iterable[P],
        *,
        profile_cls: type[P] | None = None,
    ) -> None:
        plist = tuple(profiles)

        if plist:
            inferred_cls = type(plist[0])
            bad = {type(p).__name__ for p in plist if type(p) is not inferred_cls}
            if bad:
                observed = {inferred_cls.__name__, *bad}
                raise TypeError(
                    f"ProfileSet is single-type; got mix of {observed}. "
                    f"Run one ProfileSet per factor type (BHY requires a "
                    f"same-test-family batch)."
                )
            if profile_cls is not None and profile_cls is not inferred_cls:
                raise TypeError(
                    f"ProfileSet profile_cls={profile_cls.__name__} conflicts "
                    f"with inferred {inferred_cls.__name__} from profiles."
                )
            cls = inferred_cls
        else:
            if profile_cls is None:
                raise ValueError(
                    "Empty ProfileSet requires profile_cls= to preserve "
                    "the type identity. Pass the Profile class explicitly."
                )
            cls = profile_cls

        self._profiles = plist
        self._profile_cls = cls
        self._df = _profiles_to_df(plist, cls)
        self._canonical_override = None
        self._warned_uncorrected = False

    @classmethod
    def _with_df(
        cls,
        profiles: tuple[P, ...],
        df: pl.DataFrame,
        profile_cls: type[P],
        canonical_override: str | None = None,
        warned_uncorrected: bool = False,
    ) -> "ProfileSet[P]":
        """Internal constructor that preserves an existing df.

        Avoids regenerating the df (which would drop any extra columns
        added by multiple_testing_correct).
        """
        obj = cls.__new__(cls)
        obj._profiles = profiles
        obj._df = df
        obj._profile_cls = profile_cls
        obj._canonical_override = canonical_override
        obj._warned_uncorrected = warned_uncorrected
        return obj

    # Sentinel for ``_derive`` — ``None`` is a legitimate override value
    # meaning "clear any existing rebinding", so it can't double as the
    # "carry over from self" signal. Class-typed for mypy friendliness.
    class _Unchanged:
        __slots__ = ()

    _UNCHANGED: typing.ClassVar["ProfileSet._Unchanged"] = _Unchanged()

    def _derive(
        self,
        *,
        profiles: "tuple[P, ...] | None" = None,
        df: "pl.DataFrame | None" = None,
        canonical_override: "str | None | ProfileSet._Unchanged" = _UNCHANGED,
    ) -> "ProfileSet[P]":
        """Clone with select overrides; inherit the rest from ``self``.

        Single choke point so adding a new slot (future feature state)
        only updates this method, not every ``filter`` / ``rank_by`` /
        ``top`` / ``with_*`` / ``multiple_testing_correct`` return site.
        """
        resolved_override = (
            self._canonical_override
            if isinstance(canonical_override, ProfileSet._Unchanged)
            else canonical_override
        )
        return ProfileSet._with_df(
            self._profiles if profiles is None else profiles,
            self._df if df is None else df,
            self._profile_cls,
            resolved_override,
            warned_uncorrected=self._warned_uncorrected,
        )

    # ------------------------------------------------------------------
    # Dunder / basic access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[P]:
        return iter(self._profiles)

    def __repr__(self) -> str:
        return (
            f"ProfileSet[{self._profile_cls.__name__}]"
            f"(n={len(self._profiles)})"
        )

    @property
    def profile_cls(self) -> type[P]:
        return self._profile_cls

    def iter_profiles(self) -> Iterator[P]:
        return iter(self._profiles)

    def to_polars(self) -> pl.DataFrame:
        """Return the DataFrame view.

        Shape matches the dataclass schema plus any columns added by
        ``multiple_testing_correct``. Always includes a ``canonical_p``
        column (alias of ``CANONICAL_P_FIELD``) for convenience.
        """
        return self._df

    def to_list(self) -> list[P]:
        return list(self._profiles)

    # ------------------------------------------------------------------
    # Filter / rank / top
    # ------------------------------------------------------------------

    def filter(
        self,
        predicate: "pl.Expr | Callable[[P], bool]",
    ) -> "ProfileSet[P]":
        """Subset by a row-wise Boolean condition.

        Accepts either a polars ``pl.Expr`` (preferred; vectorized) or a
        Python callable (escape hatch for logic polars cannot express).

        When the Expr references a p-value column on a batch that has
        not been through ``multiple_testing_correct()``, emits a one-shot
        ``UserWarning`` — the Callable path does no such check (escape
        hatch implies advanced caller).
        """
        if isinstance(predicate, pl.Expr):
            self._check_uncorrected_p_decision(
                predicate.meta.root_names(), op="filter",
            )
            mask = self._mask_from_expr(predicate)
        elif callable(predicate):
            mask = [bool(predicate(p)) for p in self._profiles]
        else:
            raise TypeError(
                f"ProfileSet.filter expects pl.Expr or Callable; "
                f"got {type(predicate).__name__}."
            )

        new_profiles = tuple(
            p for p, keep in zip(self._profiles, mask) if keep
        )
        new_df = self._df.filter(pl.Series(mask))
        return self._derive(profiles=new_profiles, df=new_df)

    def _mask_from_expr(self, predicate: pl.Expr) -> list[bool]:
        """Evaluate a polars Expr and return a row-wise Boolean mask.

        Rejects expressions that change the row count (aggregations) or
        produce non-Boolean output.
        """
        result = self._df.select(predicate.alias("_mask"))
        if result.height != self._df.height:
            raise RuntimeError(
                f"ProfileSet.filter(pl.Expr) changed row count "
                f"({self._df.height} -> {result.height}). Use a pure "
                f"row-wise boolean expression (comparisons, .is_not_null, "
                f"& | ~); aggregations and with_columns mutations are "
                f"not allowed."
            )
        mask_col = result.get_column("_mask")
        if mask_col.dtype != pl.Boolean:
            raise TypeError(
                f"ProfileSet.filter(pl.Expr) produced dtype={mask_col.dtype}; "
                f"expected Boolean. Use comparisons (>=, <, .is_not_null, "
                f"& | ~) to build a Boolean expression."
            )
        return mask_col.to_list()

    def _check_uncorrected_p_decision(
        self,
        referenced: Iterable[str],
        *,
        op: str,
    ) -> None:
        """One-shot ``UserWarning`` on batch decisions over raw p-values.

        Fires when ``filter`` / ``rank_by`` reference a column in
        ``P_VALUE_FIELDS`` (or ``canonical_p``) on a set with
        ``len >= 2`` that has not been through
        ``multiple_testing_correct()``. Suppressed after one fire via
        ``_warned_uncorrected`` (propagated through ``_derive``), so
        ``ps.filter(p < 0.05).rank_by(p).top(n)`` warns once, not thrice.

        Rationale: with K ≥ 2 same-family hypotheses the family-wise
        error rate is already inflated (Harvey-Liu-Zhu 2016). Using a
        raw p-threshold to decide membership / ordering of the batch
        manufactures false discoveries in direct proportion to K.
        """
        if self._warned_uncorrected:
            return
        if len(self) < 2:
            return
        if _MT_METHOD_COL in self._df.columns:
            return
        p_cols = set(self._profile_cls.P_VALUE_FIELDS) | {"canonical_p"}
        hit = sorted(set(referenced) & p_cols)
        if not hit:
            return
        self._warned_uncorrected = True
        warnings.warn(
            f"ProfileSet.{op} is deciding on raw p-value column(s) "
            f"{hit} over {len(self)} factors without prior "
            f"multiple_testing_correct(). With K={len(self)} same-family "
            f"hypotheses the family-wise error rate inflates (Harvey-Liu-Zhu "
            f"2016 argue the published finance literature is dominated by "
            f"this). Apply BHY first:\n"
            f"    ps.multiple_testing_correct().filter(pl.col('bhy_significant'))\n"
            f"or sort on p_adjusted. If the decision is intentionally "
            f"uncorrected (single pre-registered hypothesis, exploratory "
            f"triage), suppress with warnings.filterwarnings.",
            UserWarning,
            stacklevel=3,
        )

    def rank_by(self, field: str, descending: bool = True) -> "ProfileSet[P]":
        """Reorder by a numeric field. Nulls go to the end."""
        if field not in self._df.columns:
            raise KeyError(
                f"ProfileSet.rank_by: {field!r} not in "
                f"{self._profile_cls.__name__}. "
                f"Available fields: {sorted(self._df.columns)}."
            )
        self._check_uncorrected_p_decision([field], op="rank_by")
        idx_df = (
            self._df.with_row_index("_rankby_idx")
            .sort(field, descending=descending, nulls_last=True)
        )
        new_order = idx_df.get_column("_rankby_idx").to_list()
        new_profiles = tuple(self._profiles[i] for i in new_order)
        new_df = idx_df.drop("_rankby_idx")
        return self._derive(profiles=new_profiles, df=new_df)

    def top(self, n: int) -> "ProfileSet[P]":
        if n < 0:
            raise ValueError(f"top(n): n must be non-negative; got {n}.")
        new_profiles = self._profiles[:n]
        new_df = self._df.head(n)
        return self._derive(profiles=new_profiles, df=new_df)

    # ------------------------------------------------------------------
    # Extensibility
    # ------------------------------------------------------------------

    def with_extra_columns(
        self,
        columns: "dict[str, Iterable] | pl.DataFrame",
    ) -> "ProfileSet[P]":
        """Append user-computed columns to the polars view.

        The dataclass tuples are not modified — extras live only on the
        DataFrame, which is how ``filter(pl.Expr)``, ``rank_by``,
        ``top``, ``multiple_testing_correct``, and ``to_polars`` all
        see them.

        Alignment is strictly **positional**: the i-th row of the input
        attaches to ``self._profiles[i]``. If your data is keyed by
        factor name, sort or join to ``self.to_polars()['factor_name']``
        order before passing it here — the ProfileSet deliberately does
        no name-based reindexing so it can never silently drop or
        reorder rows.

        Args:
            columns: Either a ``{col_name: values}`` dict (values must
                be an iterable of length ``len(self)``) or a polars
                ``DataFrame`` with the same number of rows as the
                ProfileSet.

        Returns:
            A new ``ProfileSet`` whose DataFrame view gains the
            requested columns.

        Raises:
            ValueError: row-count mismatch, or any column name already
                present in the internal DataFrame (dataclass fields,
                ``canonical_p``, MT output columns). Drop them from
                the input first if you intend to replace.
        """
        if isinstance(columns, pl.DataFrame):
            extra = columns
        else:
            extra = pl.DataFrame({k: list(v) for k, v in columns.items()})

        if extra.height != len(self):
            raise ValueError(
                f"with_extra_columns: row-count mismatch. "
                f"ProfileSet has {len(self)} rows; extras have {extra.height}. "
                f"Align to self.to_polars()['factor_name'] order before "
                f"passing — no name-based reindexing is performed."
            )

        overlap = set(extra.columns) & set(self._df.columns)
        if overlap:
            raise ValueError(
                f"with_extra_columns: column names already exist in "
                f"ProfileSet: {sorted(overlap)}. Drop them from the input "
                f"first if you want to replace (ProfileSet refuses silent "
                f"overwrites of schema-derived columns)."
            )

        new_df = self._df.hstack(extra)
        return self._derive(df=new_df)

    # ------------------------------------------------------------------
    # Diagnose / canonical rebinding
    # ------------------------------------------------------------------

    def diagnose_all(self) -> pl.DataFrame:
        """Flatten ``diagnose()`` output across the set.

        Returns one row per (profile, diagnostic). Profiles with no
        diagnostics contribute no rows — compare ``len(set)`` against
        ``diagnose_all()['factor_name'].n_unique()`` to see how many are
        clean. Columns:

            factor_name, severity, code, message, recommended_p_source

        Designed for zoo-scale triage: ``group_by('code').len()`` for a
        histogram of failing rules; ``filter(pl.col('severity')=='warn')``
        to isolate recommendations to act on before BHY.
        """
        factor_names: list[str] = []
        severities: list[str] = []
        codes: list[str | None] = []
        messages: list[str] = []
        recommended: list[str | None] = []
        for p in self._profiles:
            for d in p.diagnose():
                factor_names.append(p.factor_name)
                severities.append(d.severity)
                codes.append(d.code)
                messages.append(d.message)
                recommended.append(d.recommended_p_source)
        return pl.DataFrame(
            {
                "factor_name": factor_names,
                "severity": severities,
                "code": codes,
                "message": messages,
                "recommended_p_source": recommended,
            },
            schema={
                "factor_name": pl.Utf8,
                "severity": pl.Utf8,
                "code": pl.Utf8,
                "message": pl.Utf8,
                "recommended_p_source": pl.Utf8,
            },
        )

    def with_canonical(self, field: str) -> "ProfileSet[P]":
        """Rebind the canonical p-source for downstream operations.

        Affects:
        - ``multiple_testing_correct()`` default ``p_source`` resolves
          to ``field`` instead of the class-level ``CANONICAL_P_FIELD``.
        - The ``canonical_p`` alias column in ``to_polars()`` points at
          ``field``.

        The underlying profile dataclasses are untouched; individual
        ``profile.verdict()`` still uses the class-level canonical. For
        an alternative single-profile verdict, read ``getattr(p, field)``
        directly.

        ``field`` must be in ``P_VALUE_FIELDS``.
        """
        whitelist = self._profile_cls.P_VALUE_FIELDS
        if field not in whitelist:
            raise ValueError(
                f"with_canonical: {field!r} is not in "
                f"{self._profile_cls.__name__}.P_VALUE_FIELDS="
                f"{sorted(whitelist)}."
            )
        new_df = _attach_canonical_alias(self._df, field)
        return self._derive(df=new_df, canonical_override=field)

    # ------------------------------------------------------------------
    # Multiple testing
    # ------------------------------------------------------------------

    def multiple_testing_correct(
        self,
        p_source: str = "canonical_p",
        method: Literal["bhy"] = "bhy",
        fdr: float = 0.05,
        *,
        n_total: int | None = None,
    ) -> "ProfileSet[P]":
        """Apply a multiple-testing correction across the set.

        Validates ``p_source`` against the Profile class's
        ``P_VALUE_FIELDS`` whitelist — composed-p fields (e.g.
        ``min(ic_p, spread_p)``) are rejected because feeding them to
        BHY violates the same-test-family assumption.

        Args:
            p_source: Which p-value column to correct. Defaults to the
                profile's canonical p. Must be in ``P_VALUE_FIELDS``.
            method: Correction method. Only ``"bhy"`` is currently
                supported.
            fdr: Target false discovery rate (default 0.05).
            n_total: Full candidate family size for two-stage screening.
                Use this when the ProfileSet contains only survivors of
                a pre-screen over a larger candidate pool (e.g.
                ``len(self) = 50``, but you originally screened 1000):
                pass ``n_total=1000`` so BHY uses the correct denominator.
                Must be ``>= len(self)``. ``None`` (default) uses
                ``len(self)``, i.e. single-stage BHY.

        Returns a new ProfileSet whose DataFrame view gains:
            - ``p_adjusted`` (float) — BHY adjusted per-factor p-value
            - ``bhy_significant`` (bool) — rejection mask at the given fdr
            - ``mt_p_source`` (str) — which source was used
            - ``mt_method`` / ``mt_fdr`` — run metadata
            - ``mt_n_total`` (int) — BHY denominator m used
        The underlying profile dataclasses are unchanged.

        Note on two-stage screening:
            When ``n_total > len(self)``, BHY with ``m = n_total``
            controls the *marginal* FDR over the full candidate family
            on the assumption that the screening criterion is
            **independent** of the p-value being corrected. If the
            screen uses the same statistic family (e.g. screen on
            ``ic_p``, correct on ``ic_p``), you are conditioning on
            significance — the reported FDR is a lower bound on the
            true conditional FDR. Prefer orthogonal pre-filters (IC
            magnitude, turnover, coverage) to keep the marginal
            interpretation honest.

        Re-applying correction to an already-adjusted ProfileSet raises
        ``RuntimeError`` — chaining BHY on adjusted p-values is
        statistically meaningless.
        """
        if _MT_METHOD_COL in self._df.columns:
            # mt_method and mt_n_total are always written together, so
            # their presence is equivalent once len(self) > 0.
            if len(self):
                prev_method = self._df[_MT_METHOD_COL][0]
                prev_n_total = self._df[_MT_N_TOTAL_COL][0]
            else:
                prev_method = "?"
                prev_n_total = "?"
            raise RuntimeError(
                f"ProfileSet already has multiple-testing correction "
                f"applied (mt_method={prev_method!r}, "
                f"mt_n_total={prev_n_total}). Re-applying BHY to "
                f"already-adjusted p-values is statistically meaningless "
                f"(the first correction is the authoritative one). Build "
                f"a fresh ProfileSet from the raw profiles if you need "
                f"to redo the correction with different parameters."
            )

        whitelist = self._profile_cls.P_VALUE_FIELDS
        canonical_field = self._profile_cls.CANONICAL_P_FIELD
        # Re-check invariants at runtime: @register_profile validates these
        # at decoration time, but setattr on the class after registration
        # (e.g. an ill-advised monkey-patch in a notebook) could silently
        # put the profile class back into an invalid shape. A frozenset
        # membership test is essentially free.
        if canonical_field not in whitelist:
            raise RuntimeError(
                f"{self._profile_cls.__name__}.CANONICAL_P_FIELD was set "
                f"to {canonical_field!r}, which is not in P_VALUE_FIELDS="
                f"{sorted(whitelist)}. The @register_profile decorator "
                f"locked a valid value at class-definition time; this "
                f"mismatch means the ClassVar was overwritten later "
                f"(search your session for 'CANONICAL_P_FIELD =' / "
                f"setattr on {self._profile_cls.__name__}). Restore the "
                f"original value before calling multiple_testing_correct."
            )
        if p_source != "canonical_p" and p_source not in whitelist:
            raise ValueError(
                f"p_source={p_source!r} is not a valid p-value source for "
                f"{self._profile_cls.__name__}. "
                f"Valid: {{'canonical_p'}} union {sorted(whitelist)}. "
                f"This guard prevents composed-p (e.g. min(ic_p, spread_p)) "
                f"from being passed to BHY — doing so would violate the "
                f"same-test-family assumption and underestimate FDR."
            )

        if method != "bhy":
            raise ValueError(
                f"Unknown multiple-testing method {method!r}. Supported: 'bhy'."
            )

        effective_n_total = int(n_total) if n_total is not None else len(self)
        # Resolve "canonical_p" through any with_canonical() rebinding so
        # the recorded mt_p_source names the actual field pulled, not the
        # alias. Explicit non-canonical p_source (already validated above)
        # is passed through untouched.
        resolved_p_source = p_source
        if p_source == "canonical_p" and self._canonical_override is not None:
            resolved_p_source = self._canonical_override
        meta_cols = [
            pl.lit(resolved_p_source).alias(_MT_P_SOURCE_COL),
            pl.lit(method).alias(_MT_METHOD_COL),
            pl.lit(float(fdr)).alias(_MT_FDR_COL),
            pl.lit(effective_n_total).alias(_MT_N_TOTAL_COL),
        ]

        if not self._profiles:
            # Empty set: no adjustment to apply; still add the columns so
            # downstream filtering code does not need to special-case.
            new_df = self._df.with_columns([
                pl.Series("p_adjusted", [], dtype=pl.Float64),
                pl.Series("bhy_significant", [], dtype=pl.Boolean),
                *meta_cols,
            ])
            return self._derive(df=new_df)

        if resolved_p_source == "canonical_p":
            p_values = np.array(
                [float(p.canonical_p) for p in self._profiles], dtype=float,
            )
        else:
            p_values = np.array(
                [float(getattr(p, resolved_p_source)) for p in self._profiles],
                dtype=float,
            )

        significant = bhy_adjust(p_values, fdr=fdr, n_total=n_total)
        adjusted = bhy_adjusted_p(p_values, n_total=n_total)

        new_df = self._df.with_columns([
            pl.Series("p_adjusted", adjusted),
            pl.Series("bhy_significant", significant),
            *meta_cols,
        ])
        from factrix._logging import get_evaluation_logger
        get_evaluation_logger().info(
            "multiple_testing_correct: method=%s p_source=%s fdr=%.4f "
            "n_hypotheses=%d n_total=%d n_rejected=%d",
            method, resolved_p_source, fdr, len(self),
            effective_n_total, int(significant.sum()),
        )
        return self._derive(df=new_df)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _profiles_to_df(
    profiles: tuple[P, ...],
    profile_cls: type[P],
) -> pl.DataFrame:
    """Build the polars view from a tuple of profile dataclasses.

    Column-wise construction: we pull each field across the whole
    tuple at once. This avoids the per-row dict allocations and
    dataclasses.asdict's recursive deep-copy that would otherwise
    scale poorly to hundreds of profiles.
    """
    field_list = dataclasses.fields(profile_cls)

    if not profiles:
        # Empty: build schema-only frame so downstream filter / rank_by
        # keep working without special-casing for len-0 sets.
        hints = typing.get_type_hints(profile_cls, include_extras=False)
        return pl.DataFrame(
            {f.name: pl.Series(f.name, [], dtype=_polars_dtype_for(hints[f.name]))
             for f in field_list}
        )

    columns = {
        f.name: [getattr(p, f.name) for p in profiles]
        for f in field_list
    }
    df = pl.DataFrame(columns)

    return _attach_canonical_alias(df, profile_cls.CANONICAL_P_FIELD)


def _attach_canonical_alias(df: pl.DataFrame, field: str) -> pl.DataFrame:
    """Alias ``field`` as ``canonical_p`` on ``df``; no-op if absent."""
    if field in df.columns:
        return df.with_columns(pl.col(field).alias("canonical_p"))
    return df


_SCALAR_DTYPES: dict[type, pl.DataType] = {
    bool: pl.Boolean,  # bool is a subclass of int — must come first
    int: pl.Int64,
    float: pl.Float64,
    str: pl.Utf8,
}


def _polars_dtype_for(annotation: object) -> pl.DataType:
    """Resolve a Profile dataclass annotation to a polars dtype.

    Only consulted for empty ProfileSets (non-empty sets let polars
    infer from data). Handles:
      - scalars (int / float / bool / str);
      - ``PValue`` (NewType over float);
      - ``T | None`` unions by stripping ``None``;
      - ``tuple[T, ...]`` / ``list[T]`` → ``pl.List(scalar dtype of T)``;
      - ``Literal[...]`` / unknown → ``pl.Object``.
    """
    # Strip Optional / T | None down to the non-None arg.
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _polars_dtype_for(non_none[0])
        return pl.Object

    if origin in (tuple, list):
        if args:
            # tuple[T, ...] → args == (T, Ellipsis); list[T] → args == (T,)
            elem = args[0]
            inner = _polars_dtype_for(elem)
            # Polars nested lists need a concrete inner dtype; fall back
            # to Utf8 for Object to keep empty-set roundtrips valid.
            return pl.List(inner if inner != pl.Object else pl.Utf8)
        return pl.List(pl.Utf8)

    if isinstance(annotation, type):
        if annotation in _SCALAR_DTYPES:
            return _SCALAR_DTYPES[annotation]
        # PValue is a NewType; typing.get_type_hints resolves NewType to
        # its supertype on modern Python, but guard for the raw callable
        # form just in case.
        if issubclass(annotation, float):
            return pl.Float64

    # NewType wrappers expose __supertype__.
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return _polars_dtype_for(supertype)

    return pl.Object
