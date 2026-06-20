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
            instance = cls(**resolved_kwargs)
            return instance(first_arg)
        else:
            # Standard instantiation
            return super().__call__(*args, **kwargs)


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
    sample_threshold: ClassVar[SampleThreshold]
    # Optional dynamic-threshold hook. When a metric's floor is a function of
    # its own parameters (e.g. ``ic``'s floor scales with ``forward_periods``),
    # the static ``sample_threshold`` cannot express it. Such a metric supplies
    # ``sample_threshold_for`` instead — it reads the configured params off the
    # instance and returns a concrete :class:`SampleThreshold`. ``spec()``
    # resolves it against a default-constructed instance so ``inspect_data``
    # sees a real floor; ``None`` means the static ``sample_threshold`` holds.
    sample_threshold_for: ClassVar[Callable[[MetricBase], SampleThreshold] | None] = (
        None
    )
    # Declares that the metric needs a continuous-magnitude factor (``|factor|``
    # must vary across events). A discrete ±k indicator makes it undefined; the
    # metric short-circuits ``not_applicable_discrete_signal`` at run time and
    # ``inspect_data`` blocks it pre-flight. Default ``False`` — most metrics
    # accept any cardinality.
    requires_continuous_magnitude: ClassVar[bool] = False

    _impl: ClassVar[Callable]
    _first_param_name: ClassVar[str | None]
    _param_names: ClassVar[tuple[str, ...]]
    _logger: ClassVar[logging.Logger]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._logger = logging.getLogger(f"factrix.metric.{cls.__name__}")

    @classmethod
    def spec(cls) -> MetricSpec:
        """Dynamically build and return the MetricSpec for this metric.

        When the metric declares a dynamic ``sample_threshold_for`` hook, the
        floor is resolved here against a default-constructed instance (its
        params take their declared defaults) so the spec carries a concrete
        :class:`SampleThreshold` for ``inspect_data`` rather than the empty
        static placeholder. Static metrics keep ``cls.sample_threshold``.

        A metric declaring the hook must therefore be default-constructible —
        every param other than the input frame needs a default. The resolved
        floor reflects those defaults; ``inspect_data`` pre-flights the
        default-config floor, while run-time enforcement uses the body's actual
        params (the two share one computation, e.g. ``min_input_periods``).
        """
        threshold = cls.sample_threshold
        hook = cls.sample_threshold_for
        if hook is not None:
            # Call the unbound hook with a default-built instance as ``self``;
            # its configured param fields take their declared defaults.
            threshold = hook(cls())
        return MetricSpec(
            name=cls.__name__,
            cell=cls.cell,
            aggregation=cls.aggregation,
            input_shape=cls.input_shape,
            output_shape=cls.output_shape,
            role=cls.role,
            requires=cls.requires,
            batchable=cls.batchable,
            sample_threshold=threshold,
            requires_continuous_magnitude=cls.requires_continuous_magnitude,
        )

    def _params(self) -> dict[str, Any]:
        """Configured parameter values, pulled from the instance's slots."""
        return {name: getattr(self, name) for name in self._param_names}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the metric on a single input (one factor's view / upstream)."""
        try:
            # Accessed via __class__ to avoid binding ``_impl`` as a method.
            return self.__class__._impl(*args, **{**self._params(), **kwargs})
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
    ) -> dict[str, Any]:
        """Run this metric across a factor batch; return ``{factor: output}``.

        The single dispatch entry the DAG executor calls for every metric.
        ``project(col)`` returns the thin per-factor view (executor-memoised,
        auto-projected); ``upstream[requires_key][factor]`` is an upstream
        producer's per-factor output. The three historical call shapes —
        ``batchable`` (whole panel), ``requires`` (consume upstream), plain
        (thin view) — are unified in :func:`_dispatch_batch`.
        """
        if self.requires:

            def run_batch() -> dict[str, Any]:
                return self.__class__._impl(**{**self._params(), **upstream})
        else:

            def run_batch() -> dict[str, Any]:
                return self.__class__._impl(
                    panel,
                    **{**self._params(), "factor_cols": list(factor_cols)},
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
) -> dict[str, Any]:
    """Single source of truth for metric batch dispatch.

    Shared by :meth:`MetricBase.__call_batch__` and the DAG executor's
    bare-callable (``fn_resolver``) path.
    """
    logger = logging.getLogger(f"factrix.metric.{name}") if name else None

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
                out[c] = call_one(**c_kwargs)
            else:
                # Metric consumes the raw thin view
                out[c] = call_one(project(c))
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
