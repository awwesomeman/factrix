"""``import factrix`` must not offer stdlib names alongside the real API."""

from __future__ import annotations

import factrix

_STDLIB_NAMES = {
    "Any",
    "NoReturn",
    "TYPE_CHECKING",
    "MappingProxyType",
    "dataclasses",
    "math",
    "pl",
}


def _public_extras() -> set[str]:
    return {
        name
        for name in dir(factrix)
        if not name.startswith("_") and name not in factrix.__all__
    }


def test_no_stdlib_names_leak_into_the_namespace() -> None:
    """``dir(factrix)`` used to offer Any, math, pl, dataclasses, ..."""
    assert _public_extras().isdisjoint(_STDLIB_NAMES)


def test_remaining_public_names_are_factrix_submodules_or_api() -> None:
    """What is left is deliberate: subpackages plus API not in ``__all__``.

    Membership is checked by provenance rather than by an exact name set:
    importing ``factrix.stats`` (or any other subpackage) anywhere in the
    session binds it as an attribute of the package, so the exact set depends
    on which tests ran first.
    """
    import types

    for name in _public_extras():
        obj = getattr(factrix, name)
        if isinstance(obj, types.ModuleType):
            assert obj.__name__.startswith("factrix"), name
        else:
            assert getattr(obj, "__module__", "").startswith("factrix"), name


def test_star_import_is_confined_to_all() -> None:
    namespace: dict[str, object] = {}
    exec("from factrix import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(factrix.__all__)
