from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from factrix._axis import (
    Aggregation,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import Cell, SampleThreshold
from factrix.metrics._base import MetricBase
from factrix.metrics._registry import register

_F = TypeVar("_F", bound=Callable[..., Any])

# Parameters that ``evaluate`` injects from the data rather than the user
# configuring per metric. They remain dataclass fields (so threshold hooks can
# read them and standalone calls keep the signature default) but are kept out of
# the user-facing ``_param_names`` — the public constructor rejects them.
_INJECTED_PARAMS: frozenset[str] = frozenset({"forward_periods"})


def _normalize_threshold(
    declared: SampleThreshold | Callable[[MetricBase], SampleThreshold] | None,
) -> tuple[Callable[[MetricBase], SampleThreshold], SampleThreshold | None]:
    """Collapse a floor declaration into ``(resolver, const_or_None)``.

    A :class:`SampleThreshold` (or ``None`` → empty floor) yields a resolver
    returning it verbatim, plus the constant itself so the caller can bake the
    default-config floor without building an instance (a constant floor may sit
    on a metric with required params, which is not default-constructible). A
    callable is returned unchanged with ``None`` — its default-config floor is
    resolved against a default-built instance.
    """
    if declared is None or isinstance(declared, SampleThreshold):
        const = declared or SampleThreshold()

        def _const_resolver(
            _self: MetricBase, _t: SampleThreshold = const
        ) -> SampleThreshold:
            return _t

        return _const_resolver, const
    return declared, None


def metric(
    cell: Cell,
    aggregation: Aggregation,
    *,
    input_shape: InputShape = InputShape.PANEL,
    output_shape: OutputShape = OutputShape.SCALAR,
    role: SpecRole = SpecRole.METRIC,
    requires: dict[str, Any] | None = None,
    batchable: bool = False,
    sample_threshold: SampleThreshold
    | Callable[[MetricBase], SampleThreshold]
    | None = None,
    requires_continuous_magnitude: bool = False,
) -> Callable[[_F], _F]:
    """Decorator to define a Metric class from a function definition.

    Constructs a frozen dataclass inheriting from MetricBase and registers it.

    The returned object is a ``MetricBase`` subclass whose metaclass makes
    calling it run the underlying implementation, so to callers it behaves
    like the original function. It is typed as the wrapped callable (``_F``)
    so direct calls and ``requires=`` references type-check against the real
    signature; the class identity is an internal registry concern.
    """

    def decorator(fn: _F) -> _F:
        # 1. Inspect the function signature to determine fields (skipping the first argument)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        first_param_name = params[0].name if params else None

        # Sort fields to put non-default arguments before default arguments
        non_default_fields = []
        default_fields = []

        for param in params[1:]:
            # Ignore *args and **kwargs in signature (if any)
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else Any
            )
            if param.default is not inspect.Parameter.empty:
                default_fields.append((param.name, annotation, param.default))
            else:
                non_default_fields.append((param.name, annotation))

        fields = non_default_fields + default_fields

        # ``forward_periods`` is the panel's overlap horizon — a property of the
        # data, not a per-metric knob. It stays a dataclass field (so threshold
        # hooks can read ``self.forward_periods`` and the body keeps its
        # signature default for standalone calls), but it is removed from the
        # user-configurable ``_param_names``: ``evaluate`` injects the data's
        # stamped horizon at dispatch time and the public constructor rejects it.
        injected_param_names = tuple(
            name for name, *_ in fields if name in _INJECTED_PARAMS
        )
        user_param_names = tuple(
            name for name, *_ in fields if name not in _INJECTED_PARAMS
        )

        # 2. Normalize the floor declaration into the single resolver type.
        # A ``SampleThreshold`` constant (or ``None``) becomes a resolver that
        # returns it verbatim; a callable is taken as-is. The ``SampleThreshold |
        # Callable`` union lives only here at the decorator boundary and never
        # reaches a consumer — every reader sees ``_resolve_sample_threshold``
        # (a resolver) or ``sample_threshold`` (a resolved constant).
        resolver, const_threshold = _normalize_threshold(sample_threshold)

        # 3. Build the class namespace with metadata ClassVars
        cls_attrs = {
            "cell": cell,
            "aggregation": aggregation,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "role": role,
            "requires": requires or {},
            "batchable": batchable,
            "_resolve_sample_threshold": staticmethod(resolver),
            "requires_continuous_magnitude": requires_continuous_magnitude,
            "_impl": fn,
            "_first_param_name": first_param_name,
            "_param_names": user_param_names,
            "_injected_param_names": injected_param_names,
            "__module__": fn.__module__,
            "__doc__": fn.__doc__,
        }

        # 4. Create the frozen dataclass dynamically
        cls = dataclasses.make_dataclass(
            cls_name=fn.__name__,
            fields=fields,
            bases=(MetricBase,),
            namespace=cls_attrs,
            frozen=True,
            slots=True,
        )
        cls.__module__ = fn.__module__

        # 5. Bake the default-config floor: the constant verbatim, or the
        # resolver applied to a default-built instance. Constructing a default
        # instance is only required for a dynamic floor — those metrics are
        # default-constructible by contract — so a constant floor on a metric
        # with required params (no default instance) is never constructed here.
        cls.sample_threshold = (  # type: ignore[attr-defined]
            const_threshold if const_threshold is not None else resolver(cls())
        )

        # 6. Register the class
        register(cls)

        # Runtime object is the MetricBase subclass; typed as the wrapped
        # callable so callers see the real signature (see docstring).
        return cls  # type: ignore[return-value]

    return decorator
