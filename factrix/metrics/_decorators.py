from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from factrix._axis import (
    Aggregation,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import Cell, SampleThreshold
from factrix.metrics._base import _INJECTED_PARAMS, MetricBase
from factrix.metrics._registry import register

_F = TypeVar("_F", bound=Callable[..., Any])


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


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolved annotations for ``fn``, or ``{}`` when they cannot be resolved.

    Metric modules use ``from __future__ import annotations``, so a signature
    carries annotation *strings*; the closed-set rule below needs the real
    ``Literal`` alias behind the name. A third-party metric may annotate with a
    name that only exists under ``TYPE_CHECKING``, which no first-party metric
    does — resolution failing there degrades to the raw annotation object
    rather than breaking the decorator.
    """
    try:
        return typing.get_type_hints(fn)
    except Exception:
        return {}


def _literal_alias(annotation: Any) -> Any | None:
    """The ``Literal`` alias behind ``annotation``, or ``None`` if there is none.

    Unwraps a union (``Literal[...] | None``, ``Optional[Literal[...]]``) so an
    optional closed-set knob is still recognised as one; the ``None`` member is
    not a legal token, so an optional knob declares it in its own ``Literal``
    if it wants one.
    """
    if typing.get_origin(annotation) is typing.Literal:
        return annotation
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if typing.get_origin(arg) is typing.Literal:
                return arg
    return None


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
    slice_boundary_sensitive: bool = False,
    validate: Callable[[MetricBase], None] | None = None,
) -> Callable[[_F], _F]:
    """Decorator to define a Metric class from a function definition.

    Constructs a frozen dataclass inheriting from MetricBase and registers it.

    The returned object is a ``MetricBase`` subclass whose metaclass makes
    calling it run the underlying implementation, so to callers it behaves
    like the original function. It is typed as the wrapped callable (``_F``)
    so direct calls and ``requires=`` references type-check against the real
    signature; the class identity is an internal registry concern.

    ``validate`` is the metric's own knob validator: a callable taking the
    constructed instance and raising :class:`~factrix._errors.UserInputError`
    for a knob outside its bounds. It runs from
    :meth:`MetricBase.__post_init__`, alongside the two rules the decorator
    derives on its own (``Literal``-annotated fields against their alias, and
    ``inference=`` against the module's ``applicable_inference`` allowlist), so
    every knob mistake fails at construction on both the config and the
    direct-call path. Bounds a ``Literal`` cannot express are the only thing it
    is for — a closed set belongs in the annotation.
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

        # ``overlap_periods`` is the panel's overlap horizon — a property of the
        # data, not a per-metric knob. It stays a dataclass field (so threshold
        # hooks can read ``self.overlap_periods`` and the body keeps its
        # signature default for standalone calls), but it is removed from the
        # user-configurable ``_param_names``: ``evaluate`` injects the data's
        # stamped horizon at dispatch time and the public constructor rejects it.
        injected_param_names = tuple(
            name for name, *_ in fields if name in _INJECTED_PARAMS
        )
        user_param_names = tuple(
            name for name, *_ in fields if name not in _INJECTED_PARAMS
        )

        # Closed-set knobs, resolved once at class creation: every field
        # annotated with a ``Literal`` alias, paired with that alias. The
        # constructor validates against it, so the annotation *is* the runtime
        # contract and the two cannot drift.
        hints = _resolve_hints(fn)
        literal_fields = tuple(
            (name, alias)
            for name, annotation, *_ in fields
            if name in user_param_names
            and (alias := _literal_alias(hints.get(name, annotation))) is not None
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
            "slice_boundary_sensitive": slice_boundary_sensitive,
            "_impl": fn,
            "_first_param_name": first_param_name,
            "_param_names": user_param_names,
            "_injected_param_names": injected_param_names,
            "_literal_fields": literal_fields,
            "_knob_validator": staticmethod(validate) if validate else None,
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

        # Expose the implementation's real signature. ``MetricMeta.__call__``
        # otherwise shadows it with ``(*args, **kwargs)``, leaving editors,
        # ``help()`` and ``inspect.signature`` with no parameter list. The class
        # forwards calls to ``fn`` (config or direct-run form), so ``fn``'s
        # signature is the truthful call surface; underscore-prefixed params are
        # dispatch internals (e.g. ``_precomputed_series``) and are hidden.
        cls.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=[p for p in params if not p.name.startswith("_")]
        )

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
