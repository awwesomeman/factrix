from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factrix.metrics._base import MetricBase

# Central registry mapping metric class name to its subclass of MetricBase
REGISTRY: dict[str, type[MetricBase]] = {}


def register(cls: type[MetricBase]) -> None:
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
    import factrix._metric_index as _index

    _index._all_specs.cache_clear()
    _index.public_specs.cache_clear()
    _index._first_party_spec_by_name.cache_clear()

    # WHY the guard here but not above: ``factrix._dag`` imports
    # ``factrix.metrics._helpers`` near the top of its module body, which pulls
    # in every metric module, whose ``@metric`` decorators call this function —
    # all while ``_dag`` is still executing and ``_registry_callable_table``
    # (defined much further down) does not exist yet. AttributeError on that
    # partially-initialised module is expected and safe to swallow: a module
    # that has not finished importing has no populated cache to clear.
    # ``factrix._metric_index`` has no such cycle, so it is called directly.
    try:
        import factrix._dag as _dag

        _dag._registry_callable_table.cache_clear()
    except (ImportError, AttributeError):
        pass
