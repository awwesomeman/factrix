"""``factrix.inference`` — curated statistical inference methods.

A small, curated set of named inference methods, each a frozen dataclass
that carries its own ``compute`` plus identity labels. Discovery is by
the ``fx.inference.*`` namespace (autocomplete + docs); there is no
flat global ``list()`` — the contextual question "which methods does this
metric accept?" is answered per-metric (e.g. ``ic`` accepts
``NON_OVERLAPPING`` / ``NEWEY_WEST``).

The namespace is scoped to the **series-mean** family
(``compute(data, *, value_col, overlap_periods)``). Slice / panel methods
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
``_check_applicable_inference``). Membership is by the member's exact
type, not its value, so a configured ``StationaryBootstrap(n_resamples=,
seed=)`` is admitted wherever the method is. A method outside the set
raises :class:`~factrix.IncompatibleInferenceError` listing the allowed
members, rather than running an unintended test or silently falling back
to the default. ``ic`` / ``quantile_spread`` / ``quantile_spread_vw`` /
``k_spread`` all allow ``{NON_OVERLAPPING, NEWEY_WEST,
STATIONARY_BOOTSTRAP}``; ``resolve_applicable_inference`` reads the set
back for discovery.

The allowlist is a **vetting record, not a dispatch table**. Every
``inference=``-bearing metric dispatches polymorphically — it calls
``inference.compute(...)`` and reads the ``InferenceResult`` back — so it
could in principle run any series-mean member. What the allowlist says is
which members have been *measured on that metric's series*: the IC series
and the long-short spread series have different distributions, so each
carries its own size table (see ``reference/statistical-methods``) and
each admits a member only on the strength of it.

``HANSEN_HODRICK`` vs the metric allowlists
-------------------------------------------
``HansenHodrick`` is a complete series-mean member (same ``compute``
contract as the others) and is in no metric's ``applicable_inference``.
The reason is statistical, not structural: its rectangular kernel has no
PSD guarantee and can clamp a negative variance (see
``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``), so the vetted HAC is
``NeweyWest``'s PSD-guaranteed Bartlett kernel. ``StationaryBootstrap``
*is* admitted everywhere the family is offered — it makes no
asymptotic-variance assumption at all, so it is the recommended read when
a series is too short or too heavy-tailed for a HAC member to be trusted.

Who builds which series
-----------------------
Members differ in the series they consume, and each one declares that
itself through the ``consumes_full_series`` ``ClassVar``:
``NonOverlapping`` sub-samples, so a metric that already strided its own
panel hands it the strided series; ``NeweyWest`` / ``HansenHodrick`` /
``StationaryBootstrap`` correct or resample the dependence and need every
period, so the metric builds the full overlapping series for them. The
flag is series-mean-specific — a slice- or panel-shaped member has no
"full series" to speak of — so it stays on the family's dataclasses and
out of the base ``Inference`` Protocol, which constrains only the
dimension-agnostic identity fields.
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
