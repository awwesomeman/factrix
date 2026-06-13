"""Estimator protocol — inference-method identity for slice-test instances.

``Estimator`` (base): identity protocol naming an inference method.
Carries no compute logic; the numerics live in the slice-test path
(``factrix.slicing.inference``) and the ``factrix._stats`` kernels.

Retained for slice-test instances (``WaldNWCluster`` /
``WaldTwoWayCluster`` / ``DriscollKraay`` / ``BlockBootstrap``) whose
compute path is multivariate and lives outside the family-function
axis; ``factrix.slicing.inference`` consumes them as
``estimator: Estimator | None`` and narrows via ``isinstance`` on the
concrete classes. Rewriting these onto the ``factrix.inference``
``Inference`` Protocol (member-dataclass self-compute) is a follow-up;
until then this base identity protocol stays.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ``InferenceResult`` now lives in ``factrix.inference._base``; re-exported
# here so existing ``factrix.stats.InferenceResult`` importers keep resolving.
from factrix.inference._base import InferenceResult

__all__ = ["Estimator", "InferenceResult"]


@runtime_checkable
class Estimator(Protocol):
    """Inference-method identity for slice-test instances.

    Implementations supply identity (``name``) and a human-readable
    summary (``description``). The protocol is deliberately silent on
    how the value is computed — that lives in the slice-test path that
    consumes the instance.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in error messages and diagnostics."""
        ...

    @property
    def description(self) -> str:
        """One-line summary of the inference method (cell-agnostic)."""
        ...
