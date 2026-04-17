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
from typing import TYPE_CHECKING, Callable, Generic, Iterable, Iterator, Literal, TypeVar

import numpy as np
import polars as pl

from factorlib.stats.multiple_testing import bhy_adjust, bhy_adjusted_p

if TYPE_CHECKING:
    from factorlib.evaluation.profiles._base import FactorProfile

P = TypeVar("P")


class ProfileSet(Generic[P]):
    """Type-homogeneous collection of factor profiles.

    Invariants:
        - All elements are instances of the same concrete Profile class.
        - ``self._profiles`` and ``self._df`` are kept in lockstep: same
          row order, same length. Internal operations that change one
          must change the other via ``_with_df``.
        - The df may carry extra columns injected by
          ``multiple_testing_correct`` (``p_adjusted``, ``bhy_significant``,
          ``canonical_p``); these are not on the dataclass schema.

    The set may be empty; an empty set still knows its profile class
    so downstream code can call ``to_polars()`` / ``iter_profiles()``
    without special-casing.
    """

    __slots__ = ("_profiles", "_df", "_profile_cls")

    _profiles: tuple[P, ...]
    _df: pl.DataFrame
    _profile_cls: type[P]

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

    @classmethod
    def _with_df(
        cls,
        profiles: tuple[P, ...],
        df: pl.DataFrame,
        profile_cls: type[P],
    ) -> "ProfileSet[P]":
        """Internal constructor that preserves an existing df.

        Avoids regenerating the df (which would drop any extra columns
        added by multiple_testing_correct).
        """
        obj = cls.__new__(cls)
        obj._profiles = profiles
        obj._df = df
        obj._profile_cls = profile_cls
        return obj

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
        """
        if isinstance(predicate, pl.Expr):
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
        return ProfileSet._with_df(new_profiles, new_df, self._profile_cls)

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

    def rank_by(self, field: str, descending: bool = True) -> "ProfileSet[P]":
        """Reorder by a numeric field. Nulls go to the end."""
        if field not in self._df.columns:
            raise KeyError(
                f"ProfileSet.rank_by: {field!r} not in "
                f"{self._profile_cls.__name__}. "
                f"Available fields: {sorted(self._df.columns)}."
            )
        idx_df = (
            self._df.with_row_index("_rankby_idx")
            .sort(field, descending=descending, nulls_last=True)
        )
        new_order = idx_df.get_column("_rankby_idx").to_list()
        new_profiles = tuple(self._profiles[i] for i in new_order)
        new_df = idx_df.drop("_rankby_idx")
        return ProfileSet._with_df(new_profiles, new_df, self._profile_cls)

    def top(self, n: int) -> "ProfileSet[P]":
        if n < 0:
            raise ValueError(f"top(n): n must be non-negative; got {n}.")
        new_profiles = self._profiles[:n]
        new_df = self._df.head(n)
        return ProfileSet._with_df(new_profiles, new_df, self._profile_cls)

    # ------------------------------------------------------------------
    # Multiple testing
    # ------------------------------------------------------------------

    def multiple_testing_correct(
        self,
        p_source: str = "canonical_p",
        method: Literal["bhy"] = "bhy",
        fdr: float = 0.05,
    ) -> "ProfileSet[P]":
        """Apply a multiple-testing correction across the set.

        Validates ``p_source`` against the Profile class's
        ``P_VALUE_FIELDS`` whitelist — composed-p fields (e.g.
        ``min(ic_p, spread_p)``) are rejected because feeding them to
        BHY violates the same-test-family assumption.

        Returns a new ProfileSet whose DataFrame view gains:
            - ``p_adjusted`` (float) — BHY adjusted per-factor p-value
            - ``bhy_significant`` (bool) — rejection mask at the given fdr
            - ``mt_p_source`` (str) — which source was used
            - ``mt_method`` / ``mt_fdr`` — run metadata
        The underlying profile dataclasses are unchanged.
        """
        whitelist = self._profile_cls.P_VALUE_FIELDS
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

        if not self._profiles:
            # Empty set: no adjustment to apply; still add the columns so
            # downstream filtering code does not need to special-case.
            new_df = self._df.with_columns([
                pl.Series("p_adjusted", [], dtype=pl.Float64),
                pl.Series("bhy_significant", [], dtype=pl.Boolean),
                pl.lit(p_source).alias("mt_p_source"),
                pl.lit(method).alias("mt_method"),
                pl.lit(float(fdr)).alias("mt_fdr"),
            ])
            return ProfileSet._with_df(self._profiles, new_df, self._profile_cls)

        if p_source == "canonical_p":
            p_values = np.array(
                [float(p.canonical_p) for p in self._profiles], dtype=float,
            )
        else:
            p_values = np.array(
                [float(getattr(p, p_source)) for p in self._profiles],
                dtype=float,
            )

        significant = bhy_adjust(p_values, fdr=fdr)
        adjusted = bhy_adjusted_p(p_values)

        new_df = self._df.with_columns([
            pl.Series("p_adjusted", adjusted),
            pl.Series("bhy_significant", significant),
            pl.lit(p_source).alias("mt_p_source"),
            pl.lit(method).alias("mt_method"),
            pl.lit(float(fdr)).alias("mt_fdr"),
        ])
        return ProfileSet._with_df(self._profiles, new_df, self._profile_cls)


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
        return pl.DataFrame(
            {f.name: pl.Series(f.name, [], dtype=_polars_dtype_for(f.type))
             for f in field_list}
        )

    columns = {
        f.name: [getattr(p, f.name) for p in profiles]
        for f in field_list
    }
    df = pl.DataFrame(columns)

    # Expose canonical_p as a convenience column (not on the dataclass).
    canonical_field = profile_cls.CANONICAL_P_FIELD
    if canonical_field in df.columns:
        df = df.with_columns(pl.col(canonical_field).alias("canonical_p"))

    return df


def _polars_dtype_for(annotation: object) -> pl.DataType:
    """Best-effort dtype mapping for empty-ProfileSet schema building.

    For rare edge cases (empty set's DataFrame); non-empty sets infer
    from data. Conservative default: Object.
    """
    s = str(annotation)
    if "int" in s.lower():
        return pl.Int64
    if "float" in s.lower() or "PValue" in s:
        return pl.Float64
    if "bool" in s.lower():
        return pl.Boolean
    if "str" in s.lower():
        return pl.Utf8
    return pl.Object
