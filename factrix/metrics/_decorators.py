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


def metric(
    cell: Cell,
    aggregation: Aggregation,
    *,
    input_shape: InputShape = InputShape.PANEL,
    output_shape: OutputShape = OutputShape.SCALAR,
    role: SpecRole = SpecRole.METRIC,
    requires: dict[str, Any] | None = None,
    batchable: bool = False,
    sample_threshold: SampleThreshold | None = None,
    sample_threshold_for: Callable[[Any], SampleThreshold] | None = None,
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

        # 2. Build the class namespace with metadata ClassVars
        cls_attrs = {
            "cell": cell,
            "aggregation": aggregation,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "role": role,
            "requires": requires or {},
            "batchable": batchable,
            "sample_threshold": sample_threshold or SampleThreshold(),
            # Plain function (not staticmethod) so it binds ``self`` — the hook
            # reads the instance's configured param fields. ``None`` falls back
            # to the inherited static ``sample_threshold``.
            "sample_threshold_for": sample_threshold_for,
            "requires_continuous_magnitude": requires_continuous_magnitude,
            "_impl": fn,
            "_first_param_name": first_param_name,
            "_param_names": user_param_names,
            "_injected_param_names": injected_param_names,
            "__module__": fn.__module__,
            "__doc__": fn.__doc__,
        }

        # 3. Create the frozen dataclass dynamically
        cls = dataclasses.make_dataclass(
            cls_name=fn.__name__,
            fields=fields,
            bases=(MetricBase,),
            namespace=cls_attrs,
            frozen=True,
            slots=True,
        )
        cls.__module__ = fn.__module__

        # 4. Register the class
        register(cls)

        # Runtime object is the MetricBase subclass; typed as the wrapped
        # callable so callers see the real signature (see docstring).
        return cls  # type: ignore[return-value]

    return decorator
