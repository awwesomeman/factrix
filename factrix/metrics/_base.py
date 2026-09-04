from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from factrix._axis import (
    Aggregation,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._data_input import (
    _read_overlap_periods_stamp,
    _validate_named_columns,
    _validate_overlap_periods,
    _validate_panel_key_columns,
)
from factrix._metric_index import Cell, MetricSpec, SampleThreshold

# Parameters that ``evaluate`` injects at dispatch rather than the user
# configuring per metric: ``overlap_periods`` comes from the data's overlap
# stamp, ``expected_warnings`` from the caller's study-level declaration on
# ``evaluate``. On metrics whose body declares them they remain dataclass
# fields, kept out of the user-facing ``_param_names`` so neither shows up as
# a configured knob on ``spec()`` / ``_params()``.
_INJECTED_PARAMS: frozenset[str] = frozenset({"overlap_periods", "expected_warnings"})

# The injected param the public constructor refuses. ``overlap_periods`` is a
# property of the *data* — the panel's own overlap horizon — so a per-metric
# value could only disagree with the panel it runs on, and there is nothing a
# caller could correctly pass. ``expected_warnings`` is not in this set: it is
# a declaration *about* the run, every metric carries it keyword-only (see
# :ref:`the contract <expected-warnings-contract>` in the module docstring of
# ``factrix.metrics``), and both the constructor and the direct-call form take
# it, with ``evaluate`` overriding it at dispatch with the study-level value.
_REJECTED_PARAMS: frozenset[str] = frozenset({"overlap_periods"})


def _log_exception_once(
    logger: logging.Logger, msg: str, *args: Any, exc: BaseException
) -> None:
    """Log *exc* at INFO level the first time it surfaces; no-op if already logged."""
    if not getattr(exc, "_logged", False):
        logger.info(msg, *args, exc_info=True)
        with contextlib.suppress(AttributeError):
            exc._logged = True  # type: ignore[attr-defined]


class MetricMeta(type):
    """Metaclass that intercepts calling the Metric class.

    If the call provides the first parameter (input DataFrame/series) in args
    or kwargs, it instantiates a temporary instance with the remaining kwargs
    and calls it immediately. Otherwise, it performs standard dataclass instantiation.
    """

    def __call__(cls, *args, **kwargs):
        # Retrieve the pre-cached first parameter name
        first_param_name = getattr(cls, "_first_param_name", None)
        if not first_param_name:
            return super().__call__(*args, **kwargs)

        # Determine if the first parameter is present in the call
        has_first_param = False
        first_arg = None
        if len(args) > 0:
            has_first_param = True
            first_arg = args[0]
            remaining_args = args[1:]
        elif first_param_name in kwargs:
            has_first_param = True
            first_arg = kwargs[first_param_name]
            kwargs = kwargs.copy()
            kwargs.pop(first_param_name)
            remaining_args = args
        else:
            remaining_args = args

        if has_first_param:
            # Direct-call form takes the data positionally and every knob by
            # keyword. Positional knobs are rejected rather than mapped: the
            # old mapping followed ``_param_names`` order, which is NOT the
            # body's signature order — dataclass field rules re-sort
            # non-default fields first and ``overlap_periods`` is removed as
            # an injected param — so ``quantile_spread(df, 3)`` silently set
            # ``n_groups=3`` where the signature promises
            # ``overlap_periods=3``. No repo-internal caller, doc, or test
            # ever passed a second positional; failing loud costs nothing
            # and ends the misalignment class outright.
            if remaining_args:
                raise TypeError(
                    f"{cls.__name__}: pass metric parameters by keyword — "
                    f"got {len(remaining_args)} extra positional "
                    f"argument(s) after the data. Positional parameters "
                    f"were previously matched against an internal field "
                    f"order that differs from the signature, silently "
                    f"binding values to the wrong parameter."
                )
            resolved_kwargs = kwargs.copy()
            # A direct call may carry the injected horizon (``overlap_periods``) —
            # it is not a constructor field, so route it to the call as the
            # per-invocation horizon rather than into ``cls(**...)``.
            injected = getattr(cls, "_injected_param_names", ())
            call_kwargs = {
                k: resolved_kwargs.pop(k)
                for k in list(resolved_kwargs)
                if k in injected
            }
            instance = cls(**resolved_kwargs)
            return instance(first_arg, **call_kwargs)
        else:
            # Standard instantiation
            cls._reject_injected_params(kwargs)
            return super().__call__(*args, **kwargs)

    def _reject_injected_params(cls, supplied: dict[str, Any]) -> None:
        """Reject a user-supplied ``overlap_periods`` on the constructor.

        The panel's overlap horizon is dispatch-injected, not a per-metric
        knob: ``evaluate`` reads the data's stamp and hands it to every metric
        in the call. A metric never carries its own copy, so there is no
        per-metric value left to diverge from the panel — the guarantee is
        structural, enforced here at the constructor boundary.

        ``expected_warnings`` is deliberately *not* rejected. It is a
        declaration about the run rather than a property of the data, every
        metric accepts it keyword-only, and a direct call is a legitimate
        place to make it; ``evaluate`` still overrides it at dispatch with the
        study-level declaration so a whole study speaks with one voice.
        """
        offending = [name for name in supplied if name in _REJECTED_PARAMS]
        if offending:
            from factrix._errors import UserInputError

            name = offending[0]
            raise UserInputError(
                func_name=cls.__name__,
                field=name,
                value=supplied[name],
                expected=(
                    "'overlap_periods' is not a metric parameter — it is the "
                    "overlap of adjacent observations on the panel's "
                    "evaluation grid, read from the data. "
                    "factrix.preprocess.compute_forward_return stamps it "
                    "(equal to forward_periods on the full grid, derived "
                    "from dates= on a coarser one); evaluate reads it from "
                    "there, or takes overlap_periods= for an unstamped panel."
                ),
                docs_path="api/evaluate#forward_periods-and-overlap_periods",
            )


class MetricBase(metaclass=MetricMeta):
    """Abstract Base Class for all metrics.

    Provides ClassVar attributes for metadata and builds the MetricSpec dynamically.
    Calling an instance evaluates the underlying metric implementation.
    """

    cell: ClassVar[Cell]
    aggregation: ClassVar[Aggregation]
    input_shape: ClassVar[InputShape]
    output_shape: ClassVar[OutputShape]
    role: ClassVar[SpecRole]
    requires: ClassVar[dict[str, Any]]
    batchable: ClassVar[bool]
    # Per-metric sample floor, resolved against a metric instance. The decorator
    # normalizes both declaration forms — a static :class:`SampleThreshold`
    # constant and a dynamic ``Callable[[MetricBase], SampleThreshold]`` (a floor
    # that scales with run-time params such as ``overlap_periods``) — into this
    # single resolver, so no consumer ever sees the
    # ``SampleThreshold | Callable`` union.
    _resolve_sample_threshold: ClassVar[Callable[[MetricBase], SampleThreshold]]
    # Floor at the metric's default configuration: the resolver applied to a
    # default-built instance, baked once at class creation. ``spec()`` /
    # ``inspect_data`` pre-flight and the static-floor run-time gate
    # (:func:`_enforce_min_floor`) all read this one value. A metric whose floor
    # depends on run-time params re-derives it in-body from the same source the
    # resolver uses (e.g. ``_scaled_min_periods``), so the pre-flight floor and
    # the run-time floor stay numerically identical.
    sample_threshold: ClassVar[SampleThreshold]
    # Declares that the metric needs a continuous-magnitude factor (``|factor|``
    # must vary across events). A discrete ±k indicator makes it undefined; the
    # metric short-circuits ``not_applicable_discrete_signal`` at run time and
    # ``inspect_data`` blocks it pre-flight. Default ``False`` — most metrics
    # accept any cardinality.
    requires_continuous_magnitude: ClassVar[bool] = False
    # Declares that independent date-axis slices alter ordered-history,
    # sampling-phase, or time-series-model semantics. ``by_slice`` reads this
    # capability directly instead of guessing from the aggregation category.
    slice_boundary_sensitive: ClassVar[bool] = False

    # Canonical injected horizon. Declared here (not a real attribute on the
    # base) so a floor resolver typed ``Callable[[MetricBase], SampleThreshold]``
    # can read ``self.overlap_periods`` — every metric that sub-samples carries it
    # as a dataclass field; metrics without it never resolve a stride-scaled floor.
    overlap_periods: int
    # Rebalance stride the turnover metrics pair at, in evaluation-grid
    # observations. Declared here for the same reason as ``overlap_periods``:
    # a floor resolver typed ``Callable[[MetricBase], SampleThreshold]`` reads
    # it off the instance. Unlike the horizon it is a user knob — ``None``
    # means "fall back to the injected overlap" — and only the metrics that
    # pair consecutive rebalances carry it as a field.
    rebalance_lag: int | None
    _impl: ClassVar[Callable]
    _first_param_name: ClassVar[str | None]
    _param_names: ClassVar[tuple[str, ...]]
    # Params dispatch injects rather than the user configuring per metric:
    # ``overlap_periods`` (read from the data, rejected at the constructor) and
    # ``expected_warnings`` (the study-level declaration; the constructor and
    # the direct call both take it, ``evaluate`` overrides it). Both are kept
    # out of ``_param_names`` and injected at dispatch into the metrics whose
    # ``_impl`` declares them. Empty for a metric that takes neither.
    _injected_param_names: ClassVar[tuple[str, ...]] = ()
    # Closed-set knobs: ``(field, Literal alias)`` for every user-configurable
    # field annotated with a ``Literal``, resolved once by the decorator.
    _literal_fields: ClassVar[tuple[tuple[str, Any], ...]] = ()
    # The metric's own knob validator, declared as ``@metric(validate=...)``.
    _knob_validator: ClassVar[Callable[[MetricBase], None] | None] = None
    _logger: ClassVar[logging.Logger]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._logger = logging.getLogger(f"factrix.metric.{cls.__name__}")

    def __post_init__(self) -> None:
        """Validate every configured knob, once, at construction.

        The single site every metric's knob contract is enforced at. Both call
        forms reach it: ``metric(n_groups=1)`` is plain dataclass
        instantiation, and ``metric(data, n_groups=1)`` builds the same
        instance first (:meth:`MetricMeta.__call__`) and only then runs the
        body. So a bad knob raises where the caller wrote it — a config object
        cannot be built now and fail three metrics into an ``evaluate`` — and
        raises identically whichever form was used.

        Three rules, in the order a caller meets them:

        1. a field annotated with a ``Literal`` alias is checked against that
           alias, so the annotation is the runtime contract;
        2. an ``inference=`` field is checked against the module's
           ``applicable_inference`` allowlist;
        3. the metric's own ``@metric(validate=...)`` hook runs, for the
           numeric bounds the first two cannot express.

        ``overlap_periods`` is deliberately absent: it is injected from the
        panel at dispatch rather than configured, so it is validated where it
        enters, in :meth:`__call__`.
        """
        from factrix.metrics._helpers import (
            _check_applicable_inference,
            _validate_choice,
        )
        from factrix.metrics._metric_capabilities import resolve_applicable_inference

        func_name = type(self).__name__
        for field, alias in self._literal_fields:
            _validate_choice(
                getattr(self, field),
                alias,
                func_name=func_name,
                field=field,
                docs_path=self._docs_path(),
            )
        applicable = resolve_applicable_inference(self)
        if applicable is not None:
            _check_applicable_inference(
                self.inference,  # type: ignore[attr-defined]
                applicable,
                func_name=func_name,
            )
        validator = type(self)._knob_validator
        if validator is not None:
            validator(self)

    @classmethod
    def spec(cls) -> MetricSpec:
        """Dynamically build and return the MetricSpec for this metric.

        The floor carried into the spec is ``cls.sample_threshold`` — the single
        resolver applied to a default-built instance, baked at class creation
        (see :attr:`sample_threshold`). There is no static-vs-dynamic branch: a
        dynamic floor was already resolved at default config, and ``inspect_data``
        pre-flights that value. A metric whose floor depends on run-time params
        re-derives it in-body from the same source the resolver uses, so the
        pre-flight and run-time floors stay numerically identical at a given
        configuration.
        """
        return MetricSpec(
            name=cls.__name__,
            cell=cls.cell,
            aggregation=cls.aggregation,
            input_shape=cls.input_shape,
            output_shape=cls.output_shape,
            role=cls.role,
            requires=cls.requires,
            batchable=cls.batchable,
            sample_threshold=cls.sample_threshold,
            requires_continuous_magnitude=cls.requires_continuous_magnitude,
            slice_boundary_sensitive=cls.slice_boundary_sensitive,
        )

    def _params(self) -> dict[str, Any]:
        """Configured parameter values, pulled from the instance's slots."""
        return {name: getattr(self, name) for name in self._param_names}

    def _resolved_sample_threshold(
        self, overlap_periods: int | None = None
    ) -> SampleThreshold:
        """Resolve this instance's floor at the panel's overlap horizon.

        ``overlap_periods`` is injected from the data at dispatch, never
        configured, so the instance always carries the body's signature
        default; a stride-scaled floor resolved against the bare instance is
        the default-horizon floor, not the one the in-body gate applies at
        run time. Pass the panel's horizon (its stamp, or the caller's
        explicit declaration) to resolve the floor the run will actually
        gate on; ``None`` (or a metric that takes no injected horizon)
        resolves the instance as configured.
        """
        inst: MetricBase = self
        if overlap_periods is not None and "overlap_periods" in (
            self._injected_param_names
        ):
            inst = copy.copy(self)
            object.__setattr__(inst, "overlap_periods", overlap_periods)
        return type(self)._resolve_sample_threshold(inst)

    def _stamped_overlap_periods(self, data: Any) -> int | None:
        """Read the panel's overlap horizon from the reserved stamp column.

        ``compute_forward_return`` stamps the horizon it built ``forward_return``
        with, and ``evaluate`` injects that value at dispatch. A standalone call
        bypasses ``evaluate``, so without this the metric would silently fall
        back to its signature default — a different non-overlapping sample, and
        a different p-value, from the same panel. Returns ``None`` for an
        unstamped panel (the caller's explicit argument, then the signature
        default, still applies) and for a non-frame input (series consumers).
        """
        return _read_overlap_periods_stamp(data, func_name=self.__class__.__name__)

    def _inject(
        self,
        overlap_periods: int | None,
        expected_warnings: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Dispatch-time injected kwargs for ``_impl``.

        Each injected param is forwarded only when the body declares it. A
        dispatch-supplied value wins; ``None`` (a standalone call) falls back
        to the instance's own field, so a direct-call declaration such as
        ``ic(expected_warnings=("high_tie_ratio",))(panel)`` reaches the body
        instead of being silently dropped. For ``overlap_periods`` the field is
        always the signature default — the constructor rejects a user value —
        so the fallback is exactly the old "leave it at the default" behaviour.
        """
        supplied = {
            "overlap_periods": overlap_periods,
            "expected_warnings": expected_warnings,
        }
        return {
            name: (
                value if (value := supplied[name]) is not None else getattr(self, name)
            )
            for name in self._injected_param_names
        }

    def _named_column_params(self) -> dict[str, Any]:
        """Column names this call *chose* — the ones the caller can mis-type.

        Every metric already follows one naming convention: a parameter ending
        in ``_col`` carries one column name, one ending in ``_cols`` a sequence
        of them. Only values that differ from the signature default are
        returned. A param left at its default names a column the metric
        documents (``factor``, ``forward_return``, ``price``, ``market_cap``),
        and its absence is a fact about the data — a short-circuit verdict the
        body already reports — not a typo the caller can fix by renaming.
        """
        named: dict[str, Any] = {}
        for name in self._param_names:
            if not name.endswith(("_col", "_cols")):
                continue
            value = getattr(self, name)
            field = self.__dataclass_fields__[name]  # type: ignore[attr-defined]
            if value != field.default:
                named[name] = value
        return named

    def _docs_path(self) -> str:
        """Docs page for this metric — one page per public metrics module.

        A private (underscore-prefixed) module holds a producer the DAG
        configures rather than a documented metric, so it has no page of its
        own; those point at the metrics index.
        """
        module = self.__class__.__module__.rsplit(".", 1)[-1]
        if module.startswith("_"):
            return "api/metrics"
        return f"api/metrics/{module}"

    def __call__(
        self,
        *args: Any,
        overlap_periods: int | None = None,
        expected_warnings: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Evaluate the metric on a single input (one factor's view / upstream)."""
        if overlap_periods is not None:
            # A direct call declares the horizon here rather than through
            # ``evaluate``, which type/range-checks it at the boundary. Run the
            # same validator so the same mistake raises the same
            # ``UserInputError`` either way, instead of surfacing as an
            # ``OverflowError`` or a polars error from inside a stride
            # computation. ``evaluate`` injects an already-validated value, so
            # re-checking it is a no-op.
            _validate_overlap_periods(
                overlap_periods, func_name=self.__class__.__name__
            )
        if args and self.input_shape is InputShape.PANEL and not self.requires:
            # Standalone call on the raw panel: ``evaluate`` validated the
            # schema once and projected a view; a direct call skips that gate,
            # so the panel's key columns are checked here, before any polars
            # expression can turn a mis-named ``asset_id`` into a
            # ColumnNotFoundError from deep inside a quantile join. A
            # ``requires`` consumer takes a producer's derived frame instead —
            # its schema is the producer's contract, not the panel's.
            _validate_panel_key_columns(args[0], func_name=self.__class__.__name__)
            # Then the columns this call named itself: a mis-typed
            # ``factor_col`` / ``return_col`` / ``weight_col`` is the same
            # class of mistake as a mis-typed ``asset_id`` and fails the same
            # way, instead of reaching a polars expression or being answered
            # with a NaN "insufficient data" envelope.
            _validate_named_columns(
                args[0],
                self._named_column_params(),
                func_name=self.__class__.__name__,
                docs_path=self._docs_path(),
            )
        if overlap_periods is None and args:
            # Standalone call: the horizon is a property of the data, so read
            # the stamp rather than letting the signature default diverge from
            # what ``evaluate`` would use on the same panel.
            overlap_periods = self._stamped_overlap_periods(args[0])
        try:
            # Accessed via __class__ to avoid binding ``_impl`` as a method.
            return self.__class__._impl(
                *args,
                **{
                    **self._params(),
                    **self._inject(overlap_periods, expected_warnings),
                    **kwargs,
                },
            )
        except Exception as e:
            _log_exception_once(
                self._logger,
                "Metric %s failed with exception: %s",
                self.__class__.__name__,
                str(e),
                exc=e,
            )
            raise

    def __call_batch__(
        self,
        panel: Any,
        factor_cols: Sequence[str],
        *,
        project: Callable[[str], Any],
        upstream: dict[str, dict[str, Any]],
        overlap_periods: int | None = None,
        expected_warnings: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Run this metric across a factor batch; return ``{factor: output}``.

        The single dispatch entry the DAG executor calls for every metric.
        ``project(col)`` returns the thin per-factor view (executor-memoised,
        auto-projected); ``upstream[requires_key][factor]`` is an upstream
        producer's per-factor output. The three historical call shapes —
        ``batchable`` (whole panel), ``requires`` (consume upstream), plain
        (thin view) — are unified in :func:`_dispatch_batch`.
        """
        inj = self._inject(overlap_periods, expected_warnings)
        if self.requires:

            def run_batch() -> dict[str, Any]:
                return self.__class__._impl(**{**self._params(), **inj, **upstream})
        else:

            def run_batch() -> dict[str, Any]:
                return self.__class__._impl(
                    panel,
                    **{**self._params(), **inj, "factor_cols": list(factor_cols)},
                )

        return _dispatch_batch(
            name=self.__class__.__name__,
            call_one=self,
            run_batch=run_batch,
            batchable=self.batchable,
            requires=tuple(self.requires),
            input_shape=self.input_shape,
            factor_cols=factor_cols,
            project=project,
            upstream=upstream,
            inject=inj,
        )


def _dispatch_batch(
    *,
    name: str | None = None,
    call_one: Callable[..., Any],
    run_batch: Callable[[], dict[str, Any]],
    batchable: bool,
    requires: tuple[str, ...],
    input_shape: InputShape,
    factor_cols: Sequence[str],
    project: Callable[[str], Any],
    upstream: dict[str, dict[str, Any]],
    inject: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single source of truth for metric batch dispatch.

    Shared by :meth:`MetricBase.__call_batch__` and the DAG executor's
    bare-callable (``fn_resolver``) path. ``inject`` is the dispatch-time
    injected kwargs (the data's overlap horizon), already resolved by the
    caller against the callable's signature.
    """
    logger = logging.getLogger(f"factrix.metric.{name}") if name else None
    inj = inject or {}

    if batchable:
        try:
            res = run_batch()
            if isinstance(res, dict):
                return res
            return {c: res for c in factor_cols}
        except Exception as e:
            if logger:
                _log_exception_once(
                    logger, "Metric %s failed with exception: %s", name, str(e), exc=e
                )
            raise

    out: dict[str, Any] = {}
    for c in factor_cols:
        try:
            if requires:
                # Metric consumes upstream data via kwargs, replacing the raw panel
                c_kwargs = {k: upstream[k][c] for k in requires}
                out[c] = call_one(**c_kwargs, **inj)
            else:
                # Metric consumes the raw thin view
                out[c] = call_one(project(c), **inj)
        except Exception as e:
            if logger:
                _log_exception_once(
                    logger,
                    "Metric %s failed for factor %s with exception: %s",
                    name,
                    c,
                    str(e),
                    exc=e,
                )
            raise
    return out
