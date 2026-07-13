from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from factrix._axis import (
    Aggregation,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import Cell, MetricSpec, SampleThreshold


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
            # Instantiate with the remaining fields, then run on the first argument
            param_names = getattr(cls, "_param_names", ())
            resolved_kwargs = kwargs.copy()
            for name, val in zip(param_names, remaining_args, strict=False):
                resolved_kwargs[name] = val
            # A direct call may carry the injected horizon (``forward_periods``) —
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
        """Reject user-supplied injected params (e.g. ``forward_periods``).

        These are data-derived: ``evaluate`` injects the panel's stamped overlap
        horizon at dispatch. A metric never carries its own ``forward_periods``,
        so there is no per-metric knob left to diverge from the data — the
        guarantee is structural, enforced here at the constructor boundary.
        """
        injected = getattr(cls, "_injected_param_names", ())
        offending = [name for name in supplied if name in injected]
        if offending:
            from factrix._errors import UserInputError

            name = offending[0]
            raise UserInputError(
                func_name=cls.__name__,
                field=name,
                value=supplied[name],
                expected=(
                    f"{name!r} is no longer a metric parameter — it is the "
                    f"panel's overlap horizon, read from the data. Set it once "
                    f"via factrix.preprocess.compute_forward_return(data, "
                    f"forward_periods=<forward_periods>); evaluate reads it from there."
                ),
                docs_path="api/evaluate#forward_periods",
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
    # that scales with run-time params such as ``forward_periods``) — into this
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
    # can read ``self.forward_periods`` — every metric that sub-samples carries it
    # as a dataclass field; metrics without it never resolve a stride-scaled floor.
    forward_periods: int
    _impl: ClassVar[Callable]
    _first_param_name: ClassVar[str | None]
    _param_names: ClassVar[tuple[str, ...]]
    # Params injected from the data rather than configured per metric
    # (``forward_periods``): kept out of ``_param_names``, rejected at the
    # constructor, and injected at dispatch into the metrics whose ``_impl``
    # declares them. Empty for a metric that takes no injected param.
    _injected_param_names: ClassVar[tuple[str, ...]] = ()
    _logger: ClassVar[logging.Logger]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._logger = logging.getLogger(f"factrix.metric.{cls.__name__}")

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

    def _inject(self, forward_periods: int | None) -> dict[str, Any]:
        """Dispatch-time injected kwargs for ``_impl`` (the data's horizon).

        Only the horizon, and only when the body declares it; a standalone call
        (``forward_periods=None``) leaves the metric at its signature default.
        """
        if (
            forward_periods is not None
            and "forward_periods" in self._injected_param_names
        ):
            return {"forward_periods": forward_periods}
        return {}

    def __call__(
        self, *args: Any, forward_periods: int | None = None, **kwargs: Any
    ) -> Any:
        """Evaluate the metric on a single input (one factor's view / upstream)."""
        try:
            # Accessed via __class__ to avoid binding ``_impl`` as a method.
            return self.__class__._impl(
                *args, **{**self._params(), **self._inject(forward_periods), **kwargs}
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
        forward_periods: int | None = None,
    ) -> dict[str, Any]:
        """Run this metric across a factor batch; return ``{factor: output}``.

        The single dispatch entry the DAG executor calls for every metric.
        ``project(col)`` returns the thin per-factor view (executor-memoised,
        auto-projected); ``upstream[requires_key][factor]`` is an upstream
        producer's per-factor output. The three historical call shapes —
        ``batchable`` (whole panel), ``requires`` (consume upstream), plain
        (thin view) — are unified in :func:`_dispatch_batch`.
        """
        inj = self._inject(forward_periods)
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
