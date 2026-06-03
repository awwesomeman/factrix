from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from factrix.metrics._base import MetricBase

# Central registry mapping metric class name to its subclass of MetricBase
REGISTRY: Dict[str, Type[MetricBase]] = {}

def register(cls: Type[MetricBase]) -> None:
    """Register a Metric class.

    Adds it to the central registry, exposes it in the factrix.metrics namespace,
    and clears caches in the discovery module.
    """
    from factrix.metrics._base import MetricBase

    if not isinstance(cls, type) or not issubclass(cls, MetricBase):
        raise TypeError(f"register(): expected a subclass of MetricBase, got {cls}")

    name = cls.__name__
    if name in REGISTRY:
        if REGISTRY[name] is cls:
            return  # Idempotent registration on repeated imports
        raise ValueError(f"register(): metric {name!r} is already registered.")

    REGISTRY[name] = cls

    # Expose in factrix.metrics namespace
    import factrix.metrics as _metrics_pkg
    if not hasattr(_metrics_pkg, name):
        setattr(_metrics_pkg, name, cls)

    # Proactively clear caches in discovery index and DAG modules
    try:
        import factrix._metric_index as _index
        if hasattr(_index._all_specs, "cache_clear"):
            _index._all_specs.cache_clear()
        if hasattr(_index.public_specs, "cache_clear"):
            _index.public_specs.cache_clear()
        if hasattr(_index._first_party_spec_by_name, "cache_clear"):
            _index._first_party_spec_by_name.cache_clear()
    except (ImportError, AttributeError):
        pass

    try:
        import factrix._dag as _dag
        if hasattr(_dag._registry_callable_table, "cache_clear"):
            _dag._registry_callable_table.cache_clear()
    except (ImportError, AttributeError):
        pass
