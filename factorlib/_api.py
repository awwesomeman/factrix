"""High-level convenience API (profile-era).

    evaluate         — single factor → FactorProfile
    evaluate_batch   — multiple factors → ProfileSet
    factor           — single factor → Factor (research session with cache)
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

from factorlib._types import UNSET, FactorType, coerce_factor_type

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts
    from factorlib.evaluation.profile_set import ProfileSet
    from factorlib.evaluation.profiles._base import FactorProfile
    from factorlib.factor import Factor
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

# WHY: `forward_return` is produced by every preprocess_* variant; its
# absence is the strict gate that makes evaluate raise instead of
# silently auto-preprocessing (which would enable silent config mismatch).
_PREPROCESSED_MARKER = "forward_return"

_FP_MARKER = "_fl_forward_periods"

_PREPROCESSED_GATE_MSG = (
    "fl.{caller} expects a preprocessed panel — "
    f"'{_PREPROCESSED_MARKER}' column not found. Call\n"
    "    prepared = fl.preprocess(df, config=cfg)\n"
    "then pass 'prepared' to fl.{caller} with the SAME cfg "
    "instance. Running preprocess and {caller} with different "
    "configs (e.g. mismatched forward_periods) would silently "
    "poison downstream metrics."
)


def _build_artifacts_strict(
    df: pl.DataFrame,
    factor_name: str,
    config: BaseConfig,
    *,
    caller: str,
) -> "Artifacts":
    """Strict-gated ``build_artifacts`` shared by ``evaluate`` / ``factor``.

    Raises ``ValueError`` if ``df`` is not preprocessed (``forward_return``
    missing) or if the preprocessed ``forward_periods`` disagrees with
    ``config.forward_periods``. Caller identifier is embedded so users
    see ``fl.evaluate`` vs ``fl.factor`` in the error.
    """
    from factorlib.evaluation.pipeline import build_artifacts

    if _PREPROCESSED_MARKER not in df.columns:
        raise ValueError(_PREPROCESSED_GATE_MSG.format(caller=caller))
    if _FP_MARKER in df.columns:
        embedded_fp = int(df[_FP_MARKER][0])
        if embedded_fp != config.forward_periods:
            raise ValueError(
                f"fl.{caller}: forward_periods mismatch — df was "
                f"preprocessed with forward_periods={embedded_fp}, but "
                f"config.forward_periods={config.forward_periods}. Re-run "
                f"    prepared = fl.preprocess(df, config=cfg)\n"
                f"with the cfg that has forward_periods="
                f"{config.forward_periods}, or update the cfg to match "
                f"the existing panel."
            )
        # Drop after gate — the marker served its purpose; retaining it on
        # artifacts.prepared wastes ~4 bytes × rows × N_factors in batch runs.
        df = df.drop(_FP_MARKER)
    artifacts = build_artifacts(df, config)
    artifacts.factor_name = factor_name
    return artifacts

def _resolve_config(
    factor_type: "FactorType | str",
    config: "BaseConfig | None",
    config_overrides: dict[str, Any],
    *,
    caller: str,
) -> "BaseConfig":
    """Shared config resolution with mismatch / double-pass detection.

    Raises ``TypeError`` when the caller supplies contradictory inputs
    (``factor_type`` vs ``config.factor_type``, or ``config`` + ad-hoc
    overrides). Otherwise returns the resolved config — constructing
    from ``factor_type`` when ``config`` is None, or returning ``config``
    unchanged.

    ``factor_type is UNSET`` means the caller did not explicitly supply
    one; fall through to either ``config``'s factor_type (when config
    is provided) or the library default ``cross_sectional``.
    """
    if config is not None:
        if config_overrides:
            raise TypeError(
                f"{caller}: cannot pass both config= and config overrides "
                f"({list(config_overrides)}). Pick one."
            )
        if factor_type is not UNSET:
            requested = coerce_factor_type(factor_type)
            actual = type(config).factor_type
            if requested != actual:
                raise TypeError(
                    f"{caller}: factor_type={requested.value!r} and "
                    f"config.factor_type={actual.value!r} disagree. Drop "
                    f"one — they must refer to the same factor type."
                )
        return config

    ft = factor_type if factor_type is not UNSET else "cross_sectional"
    return _config_for_type(ft, **config_overrides)


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

    factor_type = coerce_factor_type(factor_type)
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
    config_cls = FACTOR_TYPES[coerce_factor_type(factor_type)]
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

def _evaluate_one(
    df: pl.DataFrame,
    factor_name: str,
    config: BaseConfig,
    *,
    profile_cls: type | None = None,
) -> tuple[Any, Any]:
    """Single-factor pipeline shared by evaluate() and evaluate_batch().

    Strict precondition: ``df`` must already be preprocessed (have a
    ``forward_return`` column). Raises ``ValueError`` otherwise. Users
    call ``fl.preprocess`` explicitly so the preprocess step is visible
    in their code (audit trail) and so the same config instance is
    physically bound to both steps (avoids silent config mismatch).

    Args:
        df: Preprocessed factor panel (has ``forward_return``).
        factor_name: Written onto the returned ``Artifacts`` and profile.
        config: Pipeline config; its ``factor_type`` picks the Profile
            class if ``profile_cls`` is not supplied.
        profile_cls: Optional pre-resolved Profile class. Passing it
            avoids a registry lookup per call when the caller already
            resolved it (e.g. evaluate_batch hoists it out of the loop).

    Returns:
        ``(profile, artifacts)`` — callers that don't need artifacts
        should discard the second element.
    """
    from factorlib.evaluation.profiles import _PROFILE_REGISTRY

    artifacts = _build_artifacts_strict(df, factor_name, config, caller="evaluate")

    if profile_cls is None:
        profile_cls = _PROFILE_REGISTRY.get(type(config).factor_type)
        if profile_cls is None:
            raise KeyError(
                f"No profile registered for factor_type "
                f"{type(config).factor_type!r}."
            )

    from factorlib.evaluation.profiles._base import _run_profile_and_attach
    profile = _run_profile_and_attach(profile_cls, artifacts)
    return profile, artifacts

@overload
def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    return_artifacts: Literal[False] = ...,
    **config_overrides: Any,
) -> "FactorProfile": ...


@overload
def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    return_artifacts: Literal[True],
    **config_overrides: Any,
) -> tuple["FactorProfile", "Artifacts"]: ...


def evaluate(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = UNSET,  # type: ignore[assignment]
    config: BaseConfig | None = None,
    return_artifacts: bool = False,
    **config_overrides: Any,
):
    """Evaluate a single factor and return a typed ``FactorProfile``.

    Dispatches to the per-type Profile class via ``_PROFILE_REGISTRY``.

    Expects a preprocessed panel (``forward_return`` column present).
    Call ``fl.preprocess(df, config=cfg)`` first with the same ``cfg``
    you'll pass here — factorlib intentionally keeps preprocess and
    evaluate as separate steps so the preprocess call is visible in
    your code and the physical binding of one ``cfg`` across both
    steps prevents silent config mismatch (e.g. mismatched
    ``forward_periods`` would silently poison every downstream metric).

    Args:
        df: Preprocessed factor panel (has ``forward_return``). Call
            ``fl.preprocess(raw_df, config=cfg)`` to produce one.
        factor_name: Identifier for the factor being evaluated; written
            onto the returned profile.
        factor_type: Factor type string or FactorType enum. Ignored if
            ``config`` is supplied.
        config: Explicit config instance; overrides ``factor_type``.
        return_artifacts: If True, return ``(profile, artifacts)`` where
            ``artifacts`` exposes ``prepared`` + ``intermediates`` for
            user-defined metrics (MI, dCor, regime splits, etc.) without
            re-running ``build_artifacts``.
        **config_overrides: Forwarded to the config constructor when
            ``config`` is None.

    Returns:
        By default a per-type FactorProfile. When ``return_artifacts``
        is True, returns ``(profile, artifacts)``.

    Note:
        Need per-metric drill-down (per-regime / per-horizon /
        spanning betas)? Pass ``return_artifacts=True`` and read
        ``artifacts.metric_outputs[key].metadata`` — the renderer
        ``fl.describe_profile_values(profile)`` intentionally stays
        scalar-only and needs no artifacts argument.

        Short-circuited metrics (insufficient data, missing input) return
        ``value=NaN`` — rendered as ``—`` by describe_profile_values and
        NaN-propagating through ``.sum()`` / ``.mean()``. Distinguishes
        "couldn't compute" from a legitimate zero (IC or β exactly 0).

        Low-cardinality factors (binary, bucketed, categorical) exercise
        ``CrossSectionalConfig(tie_policy='average')`` — keeps tied assets
        in the same bucket instead of the default ``"ordinal"`` which
        breaks ties by row order. A ``UserWarning`` fires on the first
        evaluate when the median tie_ratio exceeds 0.3 under ordinal.
    """
    config = _resolve_config(
        factor_type, config, config_overrides, caller="evaluate",
    )

    profile, artifacts = _evaluate_one(df, factor_name, config)
    if return_artifacts:
        return profile, artifacts
    return profile


def factor(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = UNSET,  # type: ignore[assignment]
    config: BaseConfig | None = None,
    **config_overrides: Any,
) -> "Factor":
    """Build a research-session ``Factor`` for interactive metric exploration.

    Returns a Factor bound to ``df``, ``factor_name``, and ``config`` with a
    pre-built ``Artifacts`` cache. Call metrics as methods (``f.ic()``,
    ``f.quantile_spread()``, ...); repeated calls reuse the cache.
    ``f.evaluate()`` collapses into a ``FactorProfile`` and shares the
    same cache. Use ``f.artifacts`` as an escape hatch to downstream
    tools that take ``Artifacts`` directly.

    Expects a preprocessed panel (``forward_return`` column present) —
    same strict gate as ``fl.evaluate``. Call ``fl.preprocess(df, config=cfg)``
    first with the same ``cfg`` to avoid silent config mismatch.

    Args:
        df: Preprocessed factor panel (has ``forward_return``).
        factor_name: Identifier written onto the returned Factor / profile.
        factor_type: Factor type string or enum. Ignored if ``config`` is
            supplied.
        config: Explicit config instance; overrides ``factor_type``.
        **config_overrides: Forwarded to the config constructor when
            ``config`` is None.

    Returns:
        A per-type ``Factor`` subclass (e.g. ``CrossSectionalFactor``)
        dispatched via ``_FACTOR_REGISTRY`` from the config's factor_type.

    Example:
        >>> f = fl.factor(prepared, "Mom_20D", factor_type="cross_sectional")
        >>> f.ic()                          # MetricOutput
        >>> f.quantile_spread(n_groups=10)  # per-call override
        >>> profile = f.evaluate()          # reuses the cache
    """
    from factorlib.factor import _FACTOR_REGISTRY

    config = _resolve_config(
        factor_type, config, config_overrides, caller="factor",
    )

    artifacts = _build_artifacts_strict(df, factor_name, config, caller="factor")

    ft = type(config).factor_type
    factor_cls = _FACTOR_REGISTRY.get(ft)
    if factor_cls is None:
        raise KeyError(
            f"No Factor subclass registered for factor_type {ft.value!r}. "
            f"Registered: {sorted(f.value for f in _FACTOR_REGISTRY)}."
        )
    return factor_cls(artifacts=artifacts)


@overload
def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    stop_on_error: bool = ...,
    on_result: Callable[[str, "FactorProfile"], bool | None] | None = ...,
    on_error: Callable[[str, BaseException], None] | None = ...,
    keep_artifacts: Literal[False] = ...,
    compact: bool = ...,
    **config_overrides: Any,
) -> "ProfileSet": ...


@overload
def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = ...,
    config: BaseConfig | None = ...,
    stop_on_error: bool = ...,
    on_result: Callable[[str, "FactorProfile"], bool | None] | None = ...,
    on_error: Callable[[str, BaseException], None] | None = ...,
    keep_artifacts: Literal[True],
    compact: bool = ...,
    **config_overrides: Any,
) -> tuple["ProfileSet", dict[str, "Artifacts"]]: ...


def evaluate_batch(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str | FactorType = UNSET,  # type: ignore[assignment]
    config: BaseConfig | None = None,
    stop_on_error: bool = False,
    on_result: Callable[[str, "FactorProfile"], bool | None] | None = None,
    on_error: Callable[[str, BaseException], None] | None = None,
    keep_artifacts: bool = False,
    compact: bool = False,
    **config_overrides: Any,
):
    """Evaluate many factors and return a ``ProfileSet``.

    Each input DataFrame must be preprocessed (``forward_return``
    present); strict gate matches ``evaluate``. Typical usage:

    .. code-block:: python

        cfg = fl.CrossSectionalConfig(forward_periods=5)
        prepared = {name: fl.preprocess(df, config=cfg)
                    for name, df in raw_factors.items()}
        ps = fl.evaluate_batch(prepared, config=cfg)

    Args:
        factors: ``{name: DataFrame}`` or ``[(name, DataFrame), ...]``.
        factor_type: Default factor type for all factors (ignored if
            ``config`` supplied).
        config: Shared config (overrides ``factor_type``).
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
    from factorlib.evaluation.profile_set import ProfileSet
    from factorlib.evaluation.profiles import _PROFILE_REGISTRY

    if compact and not keep_artifacts:
        raise ValueError(
            "evaluate_batch: compact=True requires keep_artifacts=True "
            "(compact is a no-op when artifacts are discarded)."
        )

    if isinstance(factors, dict):
        factors = list(factors.items())

    config = _resolve_config(
        factor_type, config, config_overrides, caller="evaluate_batch",
    )

    profile_cls = _PROFILE_REGISTRY.get(type(config).factor_type)
    if profile_cls is None:
        raise KeyError(
            f"No profile registered for factor_type "
            f"{type(config).factor_type!r}."
        )

    profiles = []
    artifacts_map: dict[str, Any] = {}
    for name, factor_df in factors:
        try:
            p, artifacts = _evaluate_one(
                factor_df, name, config,
                profile_cls=profile_cls,
            )
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
