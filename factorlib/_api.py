"""High-level convenience API (profile-era).

    evaluate         — single factor → FactorProfile
    evaluate_batch   — multiple factors → ProfileSet
    list_factor_types — programmatic enumeration
    describe_profile — reflect per-type profile schema
    describe_factor_types — print supported factor types
    split_by_group   — sub-universe splitting with N-aware config
    redundancy_matrix — pairwise |ρ| across a ProfileSet
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

from factorlib._types import FactorType

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts
from factorlib.config import (
    BaseConfig,
    CrossSectionalConfig,
    EventConfig,
    MacroCommonConfig,
    MacroPanelConfig,
)
from factorlib.metrics._helpers import _median_universe_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor type registry
# ---------------------------------------------------------------------------

FACTOR_TYPES: dict[FactorType, type[BaseConfig]] = {
    FactorType.CROSS_SECTIONAL: CrossSectionalConfig,
    FactorType.EVENT_SIGNAL: EventConfig,
    FactorType.MACRO_PANEL: MacroPanelConfig,
    FactorType.MACRO_COMMON: MacroCommonConfig,
}

_DESCRIPTIONS: dict[FactorType, str] = {
    FactorType.CROSS_SECTIONAL: "截面因子（每期每資產有 signal，N ≥ 30）→ 選股",
    FactorType.EVENT_SIGNAL: "事件訊號（離散觸發）→ 事件交易",
    FactorType.MACRO_PANEL: "宏觀 panel（小截面 N < 30）→ 跨國配置",
    FactorType.MACRO_COMMON: "宏觀共用（單一時序）→ 風險歸因",
}


def describe_factor_types() -> None:
    """Print supported factor types with descriptions."""
    for ft, desc in _DESCRIPTIONS.items():
        print(f"  {ft.value:<20s}: {desc}")


def list_factor_types() -> list[str]:
    """Enumerate supported factor type string identifiers."""
    return [ft.value for ft in FACTOR_TYPES]


def describe_profile(
    factor_type: str | FactorType = "cross_sectional",
) -> None:
    """Print the profile schema for a factor type.

    Reflects the registered Profile dataclass's field annotations and
    ClassVar metadata, plus a short help text pointing at key methods.
    The dataclass is the single source of truth.
    """
    # Lazy imports: profiles package triggers all 4 @register_profile
    # decorators on first access.
    import dataclasses as _dc
    from factorlib.evaluation.profiles import _PROFILE_REGISTRY

    if isinstance(factor_type, str):
        try:
            factor_type = FactorType(factor_type)
        except ValueError:
            valid = ", ".join(list_factor_types())
            raise ValueError(
                f"Unknown factor_type {factor_type!r}. Valid: {valid}."
            ) from None

    cls = _PROFILE_REGISTRY.get(factor_type)
    if cls is None:
        raise KeyError(
            f"No profile registered for {factor_type.value!r}. "
            f"Registered: {sorted(ft.value for ft in _PROFILE_REGISTRY)}."
        )

    print(f"\n  {factor_type.value} — {cls.__name__}")
    print(f"  {'─' * 50}")
    if cls.__doc__:
        first_line = (cls.__doc__.strip().splitlines() or [""])[0]
        print(f"  {first_line}")

    fields = _dc.fields(cls)
    name_w = max(len(f.name) for f in fields) + 2
    print(f"\n  Fields:")
    for f in fields:
        typ = getattr(f.type, "__name__", str(f.type))
        print(f"    {f.name:<{name_w}} {typ}")

    print(f"\n  Canonical p-value (for BHY):")
    print(f"    CANONICAL_P_FIELD = {cls.CANONICAL_P_FIELD!r}")
    print(f"    P_VALUE_FIELDS    = {sorted(cls.P_VALUE_FIELDS)}")

    print(f"\n  Methods:")
    print(f"    .canonical_p     → PValue (property)")
    print(f"    .verdict(t=2.0)  → 'PASS' | 'FAILED'")
    print(f"    .diagnose()      → list[Diagnostic]")
    print()


def _config_for_type(
    factor_type: str | FactorType,
    **overrides: Any,
) -> BaseConfig:
    if isinstance(factor_type, str):
        try:
            factor_type = FactorType(factor_type)
        except ValueError:
            raise ValueError(
                f"Unknown factor_type '{factor_type}'. "
                f"Supported: {', '.join(ft.value for ft in FactorType)}. "
                f"Use fl.describe_factor_types() for details."
            ) from None

    config_cls = FACTOR_TYPES[factor_type]
    return config_cls(**overrides)


# ---------------------------------------------------------------------------
# split_by_group
# ---------------------------------------------------------------------------

_N_GROUP_TIERS = [
    (1000, 10),
    (200, 5),
    (0, 3),
]


def _recommend_n_groups(median_n: int) -> int:
    for min_n, groups in _N_GROUP_TIERS:
        if median_n >= min_n:
            return groups
    return 3


def split_by_group(
    df: pl.DataFrame,
    definitions: dict[str, pl.Expr],
    base_config: CrossSectionalConfig,
    auto_n_groups: bool = True,
    unify_n_groups: bool = True,
) -> dict[str, tuple[pl.DataFrame, CrossSectionalConfig]]:
    """Split DataFrame by filter conditions with N-aware config adjustment.

    Args:
        df: Preprocessed panel with ``date, asset_id, ...``.
        definitions: Mapping of group name → Polars filter expression.
        base_config: Base pipeline configuration.
        auto_n_groups: Adjust n_groups per group based on median N.
        unify_n_groups: All groups use same n_groups (minimum).

    Returns:
        Mapping of group name → (filtered DataFrame, adjusted config).
    """
    per_group: dict[str, tuple[pl.DataFrame, int, int]] = {}

    for name, expr in definitions.items():
        filtered = df.filter(expr)

        if filtered.is_empty():
            logger.warning("split_by_group: group '%s' is empty after filtering", name)
            continue

        median_n = _median_universe_size(filtered)
        recommended = _recommend_n_groups(median_n)
        per_group[name] = (filtered, median_n, recommended)

    if not per_group:
        return {}

    if auto_n_groups:
        if unify_n_groups:
            per_group_rec = {n: rec for n, (_, _, rec) in per_group.items()}
            unified = min(per_group_rec.values())
            if unified != base_config.n_groups:
                bottleneck = [n for n, r in per_group_rec.items() if r == unified]
                logger.warning(
                    "split_by_group: n_groups reduced %d → %d to unify "
                    "across groups (bottleneck: %s, per-group recommended: %s). "
                    "Pass unify_n_groups=False to keep independent n_groups.",
                    base_config.n_groups, unified,
                    bottleneck, per_group_rec,
                )
        else:
            unified = None
    else:
        unified = None

    for name, (_, median_n, _) in per_group.items():
        if median_n < 50:
            logger.warning(
                "split_by_group: group '%s' has median N=%d (< 50) "
                "— quantile results may be unstable",
                name, median_n,
            )

    result: dict[str, tuple[pl.DataFrame, CrossSectionalConfig]] = {}
    for name, (filtered, median_n, recommended) in per_group.items():
        if auto_n_groups:
            target = unified if unified is not None else recommended
            if target != base_config.n_groups:
                cfg = replace(base_config, n_groups=target)
            else:
                cfg = base_config
        else:
            cfg = base_config
            if base_config.n_groups > recommended:
                logger.warning(
                    "split_by_group: group '%s' median N=%d, "
                    "n_groups=%d may be too high (recommended: %d)",
                    name, median_n, base_config.n_groups, recommended,
                )

        result[name] = (filtered, cfg)

    return result


# ---------------------------------------------------------------------------
# Profile-era API
# ---------------------------------------------------------------------------

@overload
def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    preprocess: bool = ...,
    return_artifacts: Literal[False] = ...,
    **config_overrides: Any,
) -> Any: ...


@overload
def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    preprocess: bool = ...,
    return_artifacts: Literal[True],
    **config_overrides: Any,
) -> tuple[Any, "Artifacts"]: ...


def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = "cross_sectional",
    config: BaseConfig | None = None,
    preprocess: bool = True,
    return_artifacts: bool = False,
    **config_overrides: Any,
):
    """Evaluate a single factor and return a typed ``FactorProfile``.

    Dispatches to the per-type Profile class via ``_PROFILE_REGISTRY``.

    Args:
        df: Raw panel. If ``preprocess=True`` (default), must contain
            ``price``; pre-processing computes ``forward_return`` and
            applies the per-type cleaning steps. If ``preprocess=False``,
            must already contain ``forward_return``.
        factor_name: Identifier for the factor being evaluated; written
            onto the returned profile.
        factor_type: Factor type string or FactorType enum. Ignored if
            ``config`` is supplied.
        config: Explicit config instance; overrides ``factor_type``.
        preprocess: Run preprocessing before evaluation (default True).
        return_artifacts: If True, return ``(profile, artifacts)`` where
            ``artifacts`` exposes ``prepared`` + ``intermediates`` for
            user-defined metrics (MI, dCor, regime splits, etc.) without
            re-running ``build_artifacts``.
        **config_overrides: Forwarded to the config constructor when
            ``config`` is None.

    Returns:
        By default a per-type FactorProfile. When ``return_artifacts``
        is True, returns ``(profile, artifacts)``.
    """
    from factorlib.evaluation.pipeline import build_artifacts
    from factorlib.evaluation.profiles import _PROFILE_REGISTRY

    if config is None:
        config = _config_for_type(factor_type, **config_overrides)
    elif config_overrides:
        raise TypeError(
            "evaluate: cannot pass both config= and config overrides "
            f"({list(config_overrides)}). Pick one."
        )

    if preprocess:
        from factorlib.preprocess.pipeline import preprocess as _prep
        df = _prep(df, config=config)

    artifacts = build_artifacts(df, config)
    artifacts.factor_name = factor_name

    cls = _PROFILE_REGISTRY.get(type(config).factor_type)
    if cls is None:
        raise KeyError(
            f"No profile registered for factor_type "
            f"{type(config).factor_type!r}."
        )
    profile = cls.from_artifacts(artifacts)
    if return_artifacts:
        return profile, artifacts
    return profile


@overload
def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    preprocess: bool = ...,
    stop_on_error: bool = ...,
    on_result: Callable[[str, object], object] | None = ...,
    on_error: Callable[[str, BaseException], None] | None = ...,
    keep_artifacts: Literal[False] = ...,
    compact: bool = ...,
    **config_overrides: Any,
) -> Any: ...


@overload
def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    preprocess: bool = ...,
    stop_on_error: bool = ...,
    on_result: Callable[[str, object], object] | None = ...,
    on_error: Callable[[str, BaseException], None] | None = ...,
    keep_artifacts: Literal[True],
    compact: bool = ...,
    **config_overrides: Any,
) -> tuple[Any, dict[str, "Artifacts"]]: ...


def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = "cross_sectional",
    config: BaseConfig | None = None,
    preprocess: bool = True,
    stop_on_error: bool = False,
    on_result: Callable[[str, object], object] | None = None,
    on_error: Callable[[str, BaseException], None] | None = None,
    keep_artifacts: bool = False,
    compact: bool = False,
    **config_overrides: Any,
):
    """Evaluate many factors and return a ``ProfileSet``.

    Args:
        factors: ``{name: DataFrame}`` or ``[(name, DataFrame), ...]``.
        factor_type: Default factor type for all factors (ignored if
            ``config`` supplied).
        config: Shared config (overrides ``factor_type``).
        preprocess: Whether to preprocess each factor.
        stop_on_error: Raise on first failure (True) or log+skip (False).
        on_result: Optional callback ``(name, profile)`` after each ok.
            May return ``bool | None``; returning ``False`` stops the
            batch loop (logged at INFO level). ``None`` and ``True``
            both continue, so a plain side-effect lambda (e.g.
            ``lambda n, _: print(n)``) keeps working unchanged.
        on_error: Optional callback ``(name, exception)`` on each failure;
            only consulted when ``stop_on_error=False``.
        keep_artifacts: If True, return
            ``(ProfileSet, dict[name -> Artifacts])`` so callers can run
            ``redundancy_matrix(method="factor_rank")`` or user-defined
            metrics without a manual for-loop.
        compact: When ``keep_artifacts=True``, drop each retained
            Artifacts' ``prepared`` panel (replaced with a sentinel) to
            save memory on large-batch runs. ``compact=True`` without
            ``keep_artifacts=True`` raises ``ValueError`` — compact on a
            discarded artifact is a no-op indicating caller confusion.

    Returns:
        By default a ``ProfileSet`` homogeneous in the profile class
        matching ``factor_type``. When ``keep_artifacts=True``, returns
        ``(ProfileSet, dict[str, Artifacts])``. Factors that raised are
        absent from both when ``stop_on_error=False``.
    """
    from factorlib.evaluation._protocol import _COMPACTED_PREPARED
    from factorlib.evaluation.pipeline import build_artifacts
    from factorlib.evaluation.profile_set import ProfileSet
    from factorlib.evaluation.profiles import _PROFILE_REGISTRY

    if compact and not keep_artifacts:
        raise ValueError(
            "evaluate_batch: compact=True requires keep_artifacts=True "
            "(compact is a no-op when artifacts are discarded)."
        )

    if isinstance(factors, dict):
        factors = list(factors.items())

    if config is None:
        config = _config_for_type(factor_type, **config_overrides)
    elif config_overrides:
        raise TypeError(
            "evaluate_batch: cannot pass both config= and config overrides "
            f"({list(config_overrides)}). Pick one."
        )

    profile_cls = _PROFILE_REGISTRY.get(type(config).factor_type)
    if profile_cls is None:
        raise KeyError(
            f"No profile registered for factor_type "
            f"{type(config).factor_type!r}."
        )

    if preprocess:
        from factorlib.preprocess.pipeline import preprocess as _prep
    else:
        _prep = None

    profiles = []
    artifacts_map: dict[str, Any] = {}
    for name, factor_df in factors:
        try:
            prepared = _prep(factor_df, config=config) if _prep else factor_df
            artifacts = build_artifacts(prepared, config)
            artifacts.factor_name = name
            p = profile_cls.from_artifacts(artifacts)
        except Exception as exc:
            if stop_on_error:
                raise
            logger.exception("evaluate_batch: failed on '%s'", name)
            if on_error is not None:
                on_error(name, exc)
            continue

        if compact:
            # WHY: from_artifacts already read prepared; drop it now
            # to free memory before the next iteration.
            artifacts.prepared = _COMPACTED_PREPARED
            artifacts.compact = True
        if keep_artifacts:
            artifacts_map[name] = artifacts

        profiles.append(p)
        if on_result is not None:
            signal = on_result(name, p)
            if signal is False:
                logger.info(
                    "evaluate_batch: on_result returned False after %r, "
                    "stopping early",
                    name,
                )
                break

    profile_set = ProfileSet(profiles, profile_cls=profile_cls)
    if keep_artifacts:
        return profile_set, artifacts_map
    return profile_set


def redundancy_matrix(
    profiles,
    method: str = "factor_rank",
    *,
    artifacts: dict[str, object] | None = None,
):
    """See ``factorlib.metrics.redundancy.redundancy_matrix``."""
    from factorlib.metrics.redundancy import redundancy_matrix as _rm
    return _rm(profiles, method=method, artifacts=artifacts)
