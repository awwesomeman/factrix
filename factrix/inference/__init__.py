"""``factrix.inference`` — curated statistical inference methods.

A small, curated set of named inference methods, each a frozen dataclass
that carries its own ``compute`` plus identity labels. Discovery is by
the ``fx.inference.*`` namespace (autocomplete + docs); there is no
flat global ``list()`` — the contextual question "which methods does this
metric accept?" is answered per-metric (e.g. ``ic`` accepts
``NON_OVERLAPPING`` / ``NEWEY_WEST``).

This release scopes the namespace to the **series-mean** family
(``compute(data, *, value_col, forward_periods)``). Slice / panel methods
keep their multivariate compute in ``factrix.slicing`` until they move
onto the same ``metric(inference=...)`` path; they are deliberately not
listed here to avoid re-creating a heterogeneous discovery surface.

Which metrics expose ``inference=``
-----------------------------------
A selectable ``inference`` is offered **only** to the metrics whose
headline test is "average an overlapping per-date series and test
``mean != 0``" — currently ``ic``, ``quantile_spread`` and ``k_spread``.
There the choice between non-overlap sub-sampling and a HAC SE genuinely
changes the standard error, so it is the caller's to make.

Every other metric carries a **fixed** estimator dictated by its own
statistical shape, and so takes no ``inference`` knob — the absence is by
design, not an omission:

- **Event-study** (``caar``, ``corrado_rank``, ``event_quality.*``,
  ``mfe_mae``, ``clustering_hhi``) — Brown-Warner / standardized-AR on the
  event axis; a different inference family from the series-mean one.
- **Cross-sectional regression aggregates** (``fm_beta`` family) —
  Fama-MacBeth / Driscoll-Kraay SE built into the estimator.
- **Per-asset time-series** (``common_beta``) — its own SE.
- **Fixed-distribution tests** (``directional_hit_rate`` is
  Pesaran-Timmermann, ``positive_rate`` is a binomial) — no SE to choose.
- **Descriptive diagnostics** (``oos_decay``, ``concentration``,
  ``trend`` = Theil-Sen, ``common_asymmetry``) — no single headline H0, or an
  estimator-specific CI.

Closed-union policy
-------------------
The ``inference=`` parameter is typed as a **closed union** of named
members (e.g. ``NonOverlapping | NeweyWest``), never an open ``Inference``
``Protocol`` the caller can implement. This is deliberate for a statistics
library: an unvetted user-supplied SE estimator (wrong-axis HAC,
mis-calibrated bandwidth) would silently emit invalid p-values. Each
curated member instead ships a calibrated ``min_input_periods`` and a
vetted ``compute``. The ``Inference`` ``Protocol`` (``_base.py``) exists
to constrain *member identity*, not to invite external implementations.
The union grows only when a new member is validated **for that metric
family** — extension is gated, not open.

Per-metric allowlist enforcement
--------------------------------
The closed union is a type annotation, not a runtime gate, so every
``inference=``-bearing metric additionally declares a module-level
``applicable_inference`` frozenset and validates against it on entry (via
``_check_applicable_inference``). A method outside the set raises
:class:`~factrix.IncompatibleInferenceError` listing the allowed members,
rather than running an unintended test or silently falling back to the
default. ``quantile_spread`` / ``k_spread`` allow
``{NON_OVERLAPPING, NEWEY_WEST}``; ``ic`` additionally allows
``STATIONARY_BOOTSTRAP`` (see below); ``resolve_applicable_inference``
reads the set back for discovery.

``HANSEN_HODRICK`` / ``STATIONARY_BOOTSTRAP`` vs the metric allowlists
-----------------------------------------------------------------------
``HansenHodrick`` and ``StationaryBootstrap`` are complete series-mean
members (same ``compute`` contract as the others), but neither is in
every metric's ``applicable_inference``, for reasons that differ per
dispatch style:

- ``ic`` dispatches **polymorphically** (``inference.compute(...)`` /
  ``inference.min_input_periods(...)``), so it could in principle run
  ``HANSEN_HODRICK`` — its allowlist is narrower than its capability. The
  vetted HAC pair is kept: ``NeweyWest`` (Bartlett kernel, PSD-guaranteed) is
  the recommended HAC, while ``HansenHodrick``'s rectangular kernel has no
  PSD guarantee (it can clamp a negative variance — see
  ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``). ``StationaryBootstrap``
  runs through the same polymorphic path and *is* admitted — it makes no
  asymptotic-variance assumption at all, so it is the recommended fallback
  when the IC series is too short or too heavy-tailed for either HAC
  member to be trusted.
- ``quantile_spread`` / ``k_spread`` dispatch through
  ``_spread_significance_with_inference``, which hard-branches on
  ``isinstance(inference, NeweyWest)`` for the HAC path. The allowlist is
  **load-bearing** there: it admits exactly the two members that dispatch
  handles, so widening it to ``HANSEN_HODRICK`` or ``STATIONARY_BOOTSTRAP``
  requires making that dispatch polymorphic first — not done here.
"""

from __future__ import annotations

from factrix.inference._base import Inference, InferenceResult
from factrix.inference.series_mean import (
    HANSEN_HODRICK,
    NEWEY_WEST,
    NON_OVERLAPPING,
    STATIONARY_BOOTSTRAP,
    HansenHodrick,
    NeweyWest,
    NonOverlapping,
    StationaryBootstrap,
)

__all__ = [
    "HANSEN_HODRICK",
    "NEWEY_WEST",
    "NON_OVERLAPPING",
    "STATIONARY_BOOTSTRAP",
    "HansenHodrick",
    "Inference",
    "InferenceResult",
    "NeweyWest",
    "NonOverlapping",
    "StationaryBootstrap",
]
