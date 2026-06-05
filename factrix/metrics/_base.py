from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from factrix._axis import (
    Aggregation,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._metric_index import Cell, MetricSpec, SampleThreshold


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
    test_method: ClassVar[TestMethod]
    se_method: ClassVar[SEMethod]
    input_shape: ClassVar[InputShape]
    output_shape: ClassVar[OutputShape]
    role: ClassVar[SpecRole]
    requires: ClassVar[dict[str, Any]]
    batchable: ClassVar[bool]
    sample_threshold: ClassVar[SampleThreshold]

    _impl: ClassVar[Callable]
    _first_param_name: ClassVar[str | None]
    _param_names: ClassVar[tuple[str, ...]]

    @classmethod
    def spec(cls) -> MetricSpec:
        """Dynamically build and return the MetricSpec for this metric."""
        return MetricSpec(
            name=cls.__name__,
            cell=cls.cell,
            aggregation=cls.aggregation,
            test_method=cls.test_method,
            se_method=cls.se_method,
            input_shape=cls.input_shape,
            output_shape=cls.output_shape,
            role=cls.role,
            requires=cls.requires,
            batchable=cls.batchable,
            sample_threshold=cls.sample_threshold,
        )

    def _params(self) -> dict[str, Any]:
        """Configured parameter values, pulled from the instance's slots."""
        return {name: getattr(self, name) for name in self._param_names}

    def __call__(self, df: Any) -> Any:
        """Evaluate the metric on a single input (one factor's view / upstream)."""
        # Accessed via __class__ to avoid binding ``_impl`` as a method.
        return self.__class__._impl(df, **self._params())

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
        return _dispatch_batch(
            call_one=self,
            run_batch=lambda: self.__class__._impl(
                panel,
                **{**self._params(), "factor_cols": list(factor_cols), **upstream},
            ),
            batchable=self.batchable,
            requires=tuple(self.requires),
            factor_cols=factor_cols,
            project=project,
            upstream=upstream,
        )


def _dispatch_batch(
    *,
    call_one: Callable[[Any], Any],
    run_batch: Callable[[], dict[str, Any]],
    batchable: bool,
    requires: tuple[str, ...],
    factor_cols: Sequence[str],
    project: Callable[[str], Any],
    upstream: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Single source of truth for the three metric dispatch shapes.

    Shared by :meth:`MetricBase.__call_batch__` and the DAG executor's
    bare-callable (``fn_resolver``) path so the batchable / requires / thin-view
    distinction lives in exactly one place.

    - ``batchable`` → one whole-panel call returning ``{factor: output}``.
    - ``requires``  → per factor, feed the upstream producer's output.
    - otherwise     → per factor, feed the thin per-factor view.
    """
    if batchable:
        return run_batch()
    out: dict[str, Any] = {}
    for c in factor_cols:
        if requires:
            # Every current consumer declares exactly one upstream; the impl's
            # first parameter is that requires key (validated at load time).
            out[c] = call_one(upstream[requires[0]][c])
        else:
            out[c] = call_one(project(c))
    return out
