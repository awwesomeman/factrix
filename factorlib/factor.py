"""Research-facade Factor sessions.

Factor bundles a preprocessed panel, a factor name, a config, and a
pre-built ``Artifacts`` cache into one object exposing every standalone
metric as a uniform method. Designed for research workflows where the
same factor is interrogated through multiple metrics and then collapsed
into a ``FactorProfile``.

Usage (preferred, via factory)::

    f = fl.factor(df, "Mom_20D", factor_type="cross_sectional", config=cfg)
    f.ic()                                  # MetricOutput
    f.quantile_spread(n_groups=10)          # per-call n_groups override
    f.monotonicity()
    profile = f.evaluate()                  # reuses same cache; no rebuild

Cache semantics:
    - Default metric calls read / write ``artifacts.metric_outputs``. Second
      call to the same metric returns the cached ``MetricOutput`` without
      recomputing.
    - ``f.evaluate()`` reuses the same cache via ``from_artifacts`` — any
      metric called standalone before ``evaluate()`` is not recomputed.
    - Per-call kwarg overrides (e.g. ``n_groups=10``) bypass the cache
      (because the result no longer reflects the bound config) and do NOT
      write back. The next default call (``f.quantile_spread()``) returns
      the config-bound value.

Escape hatches:
    - ``f.artifacts`` — the underlying ``Artifacts`` bundle, for tools that
      operate at that level (``fl.describe_profile_values``,
      ``fl.redundancy_matrix``, user-defined custom metrics).
    - ``factorlib.metrics.*`` — low-level primitive functions that take
      prepared panel / processed intermediates directly (for library
      authors, unit tests, deeply custom research).

Thread safety: ``Factor`` is not thread-safe. Metric methods write to
``artifacts.metric_outputs``; concurrent calls from multiple threads can
race. Use one ``Factor`` instance per worker, or call ``fl.evaluate_batch``
for parallel batch evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from collections.abc import Callable
from typing import Any

from factorlib._types import FactorType, MetricOutput
from factorlib.evaluation._protocol import Artifacts
from factorlib.evaluation.profiles._base import _memoized
from factorlib.metrics._helpers import _short_circuit_output

if TYPE_CHECKING:
    from factorlib.config import BaseConfig
    from factorlib.evaluation.profiles._base import FactorProfile


@dataclass
class Factor:
    """Base class for per-factor-type research sessions.

    Not intended for direct instantiation. Use ``fl.factor(df, name, ...)``
    which picks the right subclass via ``_FACTOR_REGISTRY``. Direct
    construction (``CrossSectionalFactor(artifacts=...)``) is an advanced
    path — ``__post_init__`` enforces that the artifacts' config's
    factor_type matches the subclass's ``EXPECTED_FACTOR_TYPE`` and that
    ``artifacts.factor_name`` is set.
    """

    artifacts: Artifacts

    # Subclasses MUST override with their concrete factor type.
    EXPECTED_FACTOR_TYPE: ClassVar[FactorType]

    def __post_init__(self) -> None:
        actual_ft = type(self.artifacts.config).factor_type
        expected = getattr(type(self), "EXPECTED_FACTOR_TYPE", None)
        if expected is None:
            raise TypeError(
                f"{type(self).__name__} does not declare EXPECTED_FACTOR_TYPE. "
                f"Factor must be subclassed for each factor type."
            )
        if actual_ft != expected:
            raise TypeError(
                f"{type(self).__name__} expects factor_type="
                f"{expected.value!r}; got {actual_ft.value!r}. "
                f"Use fl.factor(df, name, factor_type=...) to pick the "
                f"correct Factor subclass automatically."
            )
        if not self.artifacts.factor_name:
            raise ValueError(
                "Factor.artifacts.factor_name is empty. "
                "Use fl.factor(df, name=...) instead of direct instantiation."
            )

    @property
    def config(self) -> "BaseConfig":
        return self.artifacts.config

    @property
    def factor_name(self) -> str:
        return self.artifacts.factor_name

    def evaluate(self) -> "FactorProfile":
        """Compute the full typed ``FactorProfile`` for this factor.

        Reuses cached metric computations: any metric previously invoked
        standalone (``f.ic()`` etc.) is read from ``artifacts.metric_outputs``
        rather than recomputed. Conversely, metrics computed here populate
        the cache, so subsequent standalone calls also hit the cache.
        """
        from factorlib.evaluation.profiles import _PROFILE_REGISTRY

        ft = type(self.config).factor_type
        profile_cls = _PROFILE_REGISTRY.get(ft)
        if profile_cls is None:
            raise KeyError(
                f"No profile registered for factor_type {ft.value!r}."
            )
        profile, metric_outputs = profile_cls.from_artifacts(self.artifacts)
        self.artifacts.metric_outputs = metric_outputs
        return profile

    # ----- subclass helpers -----

    def _cached_or_compute(
        self,
        name: str,
        fn: Callable[..., MetricOutput],
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> MetricOutput:
        """Read ``metric_outputs[name]`` or compute via ``fn(*args, **kwargs)``.

        When ``override=True``, bypasses the cache entirely (computes fresh
        and does NOT write back) — per-call overrides must not pollute the
        config-bound cache that ``evaluate()`` reads.
        """
        if override:
            return fn(*args, **kwargs)
        return _memoized(
            self.artifacts.metric_outputs, name, fn, *args, **kwargs,
        )

    def _l2_short_circuit(self, name: str, field: str) -> MetricOutput:
        """Return cached L2 MetricOutput or a cached short-circuit.

        Level-2 opt-in metrics (regime_ic / multi_horizon_ic / spanning_alpha)
        are pre-populated at ``build_artifacts`` time when the corresponding
        config field is set. If the field is absent, stash a short-circuit
        ``MetricOutput`` on first access so repeated calls hit cache too.
        """
        cached = self.artifacts.metric_outputs.get(name)
        if cached is not None:
            return cached
        return _memoized(
            self.artifacts.metric_outputs, name,
            _short_circuit_output, name, f"no_{field}_configured",
        )


@dataclass
class CrossSectionalFactor(Factor):
    """Research session for a cross-sectional factor panel.

    Level-2 opt-in metrics (``regime_ic`` / ``multi_horizon_ic`` /
    ``spanning_alpha``) are pre-populated at ``build_artifacts`` time when
    their corresponding config fields are set; otherwise the method
    returns a cached short-circuit ``MetricOutput`` with
    ``metadata["reason"]="no_<field>_configured"``.
    """

    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.CROSS_SECTIONAL

    # ----- Level 1: core metrics -----

    def ic(self, forward_periods: int | None = None) -> MetricOutput:
        """Spearman rank IC mean significance (non-overlapping t-test).

        ``forward_periods`` override bypasses the cache and does not write
        back — the config-bound cache read by ``evaluate()`` stays pure.
        """
        from factorlib.metrics.ic import ic as _ic
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        return self._cached_or_compute(
            "ic", _ic, self.artifacts.get("ic_series"),
            forward_periods=fp, override=forward_periods is not None,
        )

    def ic_ir(self) -> MetricOutput:
        """IC_IR = mean(IC) / std(IC). Descriptive (no hypothesis test)."""
        from factorlib.metrics.ic import ic_ir as _ic_ir
        return self._cached_or_compute(
            "ic_ir", _ic_ir, self.artifacts.get("ic_series"),
        )

    def hit_rate(self, forward_periods: int | None = None) -> MetricOutput:
        """Binomial sign-hit rate on IC series."""
        from factorlib.metrics.hit_rate import hit_rate as _hr
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        return self._cached_or_compute(
            "hit_rate", _hr, self.artifacts.get("ic_values"),
            forward_periods=fp, override=forward_periods is not None,
        )

    def ic_trend(self) -> MetricOutput:
        """OLS trend slope test on IC series (structural drift check)."""
        from factorlib.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "ic_trend", _tr, self.artifacts.get("ic_values"),
        )

    def monotonicity(
        self,
        forward_periods: int | None = None,
        n_groups: int | None = None,
    ) -> MetricOutput:
        """Monotonic return ordering across factor quintiles."""
        from factorlib.metrics.monotonicity import monotonicity as _mono
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        ng = n_groups if n_groups is not None else self.config.n_groups
        override = forward_periods is not None or n_groups is not None
        return self._cached_or_compute(
            "monotonicity", _mono, self.artifacts.prepared,
            forward_periods=fp, n_groups=ng, override=override,
        )

    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput:
        """Q1-Q5 spread per-period mean, t-test against zero.

        ``n_groups`` override bypasses the cache AND the cached
        ``spread_series`` (which was built with the bound ``n_groups``) —
        the primitive recomputes spread_series internally.
        """
        from factorlib.metrics.quantile import quantile_spread as _qs
        if n_groups is not None:
            return _qs(
                self.artifacts.prepared,
                forward_periods=self.config.forward_periods,
                n_groups=n_groups,
            )
        return self._cached_or_compute(
            "q1_q5_spread", _qs, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
            _precomputed_series=self.artifacts.intermediates.get("spread_series"),
        )

    def q1_concentration(self, q_top: float | None = None) -> MetricOutput:
        """Q1 bucket weight concentration (effective-N / total-N ratio)."""
        from factorlib.metrics.concentration import q1_concentration as _qc
        qt = q_top if q_top is not None else 1.0 / self.config.n_groups
        return self._cached_or_compute(
            "q1_concentration", _qc, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            q_top=qt, override=q_top is not None,
        )

    def turnover(self) -> MetricOutput:
        """Period-over-period weight turnover fraction."""
        from factorlib.metrics.tradability import turnover as _tn
        return self._cached_or_compute(
            "turnover", _tn, self.artifacts.prepared,
        )

    def breakeven_cost(self) -> MetricOutput:
        """Per-side cost (bps) that fully erodes the Q1-Q5 spread."""
        from factorlib.metrics.tradability import breakeven_cost as _be
        return self._cached_or_compute(
            "breakeven_cost", _be,
            self.quantile_spread().value, self.turnover().value,
        )

    def net_spread(self) -> MetricOutput:
        """Spread minus ``estimated_cost_bps`` × turnover (signed)."""
        from factorlib.metrics.tradability import net_spread as _ns
        return self._cached_or_compute(
            "net_spread", _ns,
            self.quantile_spread().value, self.turnover().value,
            self.config.estimated_cost_bps,
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival ratio with sign-flip detection."""
        from factorlib.metrics.oos import multi_split_oos_decay as _oos
        return self._cached_or_compute(
            "oos_decay", _oos, self.artifacts.get("ic_values"),
        )

    # ----- Level 2: opt-in metrics (pre-populated by build_artifacts) -----

    def regime_ic(self) -> MetricOutput:
        """Per-regime IC (requires ``config.regime_labels``)."""
        return self._l2_short_circuit("regime_ic", "regime_labels")

    def multi_horizon_ic(self) -> MetricOutput:
        """IC retention across horizons (requires ``config.multi_horizon_periods``)."""
        return self._l2_short_circuit("multi_horizon_ic", "multi_horizon_periods")

    def spanning_alpha(self) -> MetricOutput:
        """Spanning regression alpha (requires ``config.spanning_base_spreads``)."""
        return self._l2_short_circuit("spanning_alpha", "spanning_base_spreads")


# ---------------------------------------------------------------------------
# Factor registry — factor_type → Factor subclass
# ---------------------------------------------------------------------------

_FACTOR_REGISTRY: dict[FactorType, type[Factor]] = {
    FactorType.CROSS_SECTIONAL: CrossSectionalFactor,
}
