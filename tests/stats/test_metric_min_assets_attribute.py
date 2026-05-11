"""Contract: each Layer-B-eligible metric module declares ``min_assets_per_group``.

Drift guard for #153 §5 — if a metric module is touched and the
attribute is dropped (or its type changes from ``int | None``), the
slice-test verbs in #176 silently treat the metric as non-bucketing
and skip the downscale step. This test fails loudly instead.
"""

from __future__ import annotations

import importlib

import pytest

# (module path, expected attribute value)
_EXPECTED: list[tuple[str, int | None]] = [
    ("factrix.metrics.ic", None),
    ("factrix.metrics.fama_macbeth", None),
    ("factrix.metrics.hit_rate", None),
    ("factrix.metrics.caar", None),
    ("factrix.metrics.monotonicity", 50),
]


@pytest.mark.parametrize(("module_path", "expected"), _EXPECTED)
def test_metric_declares_min_assets_per_group(
    module_path: str, expected: int | None
) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "min_assets_per_group"), (
        f"{module_path}.min_assets_per_group is missing — slice-test "
        f"verbs (#176) cannot resolve the downscale floor."
    )
    actual = mod.min_assets_per_group
    assert actual == expected, (
        f"{module_path}.min_assets_per_group changed from {expected} to "
        f"{actual}; update the contract here AND the rationale comment "
        f"on the metric module + cite the literature anchoring the new value."
    )
    if actual is not None:
        assert isinstance(actual, int) and actual >= 1, (
            f"{module_path}.min_assets_per_group must be int >= 1 or None; "
            f"got {actual!r}."
        )
