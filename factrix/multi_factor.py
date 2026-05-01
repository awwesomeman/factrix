"""Public ``multi_factor`` namespace (plan §7.4).

Currently exposes ``bhy`` only. ``redundancy_matrix`` /
``spanning_test`` / ``orthogonalize`` from §7.4 land alongside the
v0.4 deletion sweep that retires the existing v0.4 implementations
of those primitives.
"""

from factrix._multi_factor import bhy

__all__ = ["bhy"]
