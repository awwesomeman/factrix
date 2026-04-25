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
    - Per-call kwarg overrides (e.g. ``n_groups=10``) are **sweep overrides**
      for sensitivity analysis; they bypass the cache (the result no longer
      reflects the bound config) and do NOT write back. The next default
      call returns the config-bound value.

Override vs rebind (common gotcha)::

    # ✅ sweep: compare deciles to terciles, cfg untouched
    [f.quantile_spread(n_groups=k) for k in (3, 5, 10)]

    # ❌ override does NOT persist — evaluate() stays config-bound
    f.quantile_spread(n_groups=10)
    profile = f.evaluate()  # still n_groups=5!

    # ✅ persist by rebuilding with a new config
    from dataclasses import replace
    f2 = fl.factor(df, name, config=replace(cfg, n_groups=10))

First override per (Factor, key) emits a one-shot ``UserWarning``.

Escape hatches:
    - ``f.artifacts`` — the underlying ``Artifacts`` bundle, for tools that
      operate at that level (``fl.redundancy_matrix``, user-defined
      custom metrics, direct ``metric_outputs`` drill-down).
    - ``factrix.metrics.*`` — low-level primitive functions that take
      prepared panel / processed intermediates directly (for library
      authors, unit tests, deeply custom research).

Thread safety: ``Factor`` is not thread-safe. Metric methods write to
``artifacts.metric_outputs``; concurrent calls from multiple threads can
race. Use one ``Factor`` instance per worker, or call ``fl.evaluate_batch``
for parallel batch evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from collections.abc import Callable
from typing import Any

from factrix._types import FactorType, MetricOutput
from factrix.evaluation._protocol import Artifacts
from factrix.evaluation.profiles._base import _memoized
from factrix.metrics._helpers import _short_circuit_output

if TYPE_CHECKING:
    from factrix.config import BaseConfig
    from factrix.evaluation.profiles._base import FactorProfile


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

    # Dedup set for the override advisory — one warning per (Factor, key)
    # keeps sweep loops quiet. Method names are a fixed ~15-element set, so
    # unbounded growth isn't a concern.
    _override_log_seen: set[str] = field(default_factory=set, repr=False)

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
        from factrix.evaluation.profiles import _PROFILE_REGISTRY
        from factrix.evaluation.profiles._base import _run_profile_and_attach

        ft = type(self.config).factor_type
        profile_cls = _PROFILE_REGISTRY.get(ft)
        if profile_cls is None:
            raise KeyError(
                f"No profile registered for factor_type {ft.value!r}."
            )
        return _run_profile_and_attach(profile_cls, self.artifacts)

    # ----- subclass helpers -----

    def _cached_or_compute(
        self,
        key: str,
        fn: Callable[..., MetricOutput],
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> MetricOutput:
        """Read ``metric_outputs[key]`` or compute via ``fn(*args, **kwargs)``.

        When ``override=True``, bypasses the cache entirely (computes fresh
        and does NOT write back) — per-call overrides must not pollute the
        config-bound cache that ``evaluate()`` reads.

        Enforces the ``cache-key == primitive.MetricOutput.name`` contract:
        if the computed output's ``.name`` differs from ``key``, ``_stash``
        would key the entry under the primitive's name and leave the Factor
        side of the cache empty — a silent double-entry / cache-miss bug.
        ``_memoized`` + ``_stash`` already do the store; we assert here on
        the override (uncached) path because ``_memoized`` returns the
        stored view and can't double-check after stashing.

        Uses ``key`` — not ``name`` — because primitives may take their own
        ``name=`` kwarg (e.g. ``ic_trend(series, name="caar_trend")``) that
        would collide with a positional alias.
        """
        if override:
            self._log_override_once(key)
            result = fn(*args, **kwargs)
            assert result.name == key, (
                f"Factor cache-key contract violated: method requested "
                f"cache key {key!r} but primitive returned "
                f"MetricOutput.name={result.name!r}"
            )
            return result
        return _memoized(
            self.artifacts.metric_outputs, key, fn, *args, **kwargs,
        )

    def _log_override_once(self, key: str) -> None:
        """Emit a one-shot ``UserWarning`` when an override bypasses the cache.

        Uses ``warnings.warn`` rather than ``logger.info`` because the
        default Jupyter / REPL root logger is WARNING — an INFO advisory
        would be silently dropped in the very environment researchers
        live in. ``UserWarning`` surfaces in notebooks and CI without
        requiring explicit handler setup.

        Deduped per (Factor instance, cache key) via ``_override_log_seen``
        so a sweep loop ``[f.method(kw=k) for k in …]`` warns once, not N
        times. Message points at the rebuild recipe — ``dataclasses.replace(
        cfg, …)`` into a fresh ``fl.factor`` — which is the persistent path.
        """
        if key in self._override_log_seen:
            return
        self._override_log_seen.add(key)
        warnings.warn(
            f"Factor.{key}: per-call override detected — result is NOT "
            f"cached and will not appear in f.evaluate(). To persist, "
            f"rebuild with fl.factor(df, name, "
            f"config=dataclasses.replace(cfg, ...)).",
            UserWarning,
            stacklevel=3,
        )

    def _short_circuit_if(
        self,
        cache_key: str,
        condition: bool,
        reason: str,
    ) -> MetricOutput | None:
        """Return a cached short-circuit ``MetricOutput`` when ``condition``.

        Centralizes "if prepared lacks column X / factor is discrete /
        N==1, return a short-circuit MetricOutput with
        ``metadata["reason"]``" used by ``EventFactor`` sites. Returns
        ``None`` when the condition is False — callers fall through to
        ``_cached_or_compute``. ``_memoized`` handles the cache read /
        stash atomically, so repeat calls are O(1) without an explicit
        peek here.
        """
        if not condition:
            return None
        return _memoized(
            self.artifacts.metric_outputs, cache_key,
            _short_circuit_output, cache_key, reason,
        )

    # Shared tradability metrics. CS and MP both expose a portfolio
    # (quantile long-short) with the same cost model; Event / MC don't
    # call these — per-class method resolution keeps their surface clean.

    def turnover(self) -> MetricOutput:
        """Rank-autocorrelation turnover at the session's forward horizon.

        Sampled at stride ``config.forward_periods`` (non-overlapping) so
        the stability window matches the horizon used for forward returns
        — a factor reshuffled within the holding window counts as churn,
        but within-window noise does not.

        Diagnostic only — for the bps cost arithmetic in
        ``breakeven_cost`` / ``net_spread`` use ``turnover_jaccard`` (it
        measures notional Q1/Q_n churn, the units the formulas assume).
        """
        from factrix.metrics.tradability import turnover as _tn
        return self._cached_or_compute(
            "turnover", _tn, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
        )

    def turnover_jaccard(self, n_groups: int | None = None) -> MetricOutput:
        """Notional turnover (Novy-Marx & Velikov τ) at session horizon.

        Top/bottom quantile membership churn per rebalance — the
        position-replacement fraction that ``breakeven_cost`` /
        ``net_spread`` consume. ``n_groups`` override mirrors
        ``quantile_spread`` so a sensitivity sweep stays internally
        consistent (spread bucketing == turnover bucketing).
        """
        from factrix.metrics.tradability import turnover_jaccard as _tj
        if n_groups is not None:
            self._log_override_once("turnover_jaccard")
            return _tj(
                self.artifacts.prepared,
                forward_periods=self.config.forward_periods,
                n_groups=n_groups,
            )
        return self._cached_or_compute(
            "turnover_jaccard", _tj, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
        )

    def breakeven_cost(self, n_groups: int | None = None) -> MetricOutput:
        """Per-side cost (bps) that fully erodes the long-short spread.

        ``n_groups`` override threads through ``quantile_spread`` *and*
        ``turnover_jaccard`` so a sensitivity sweep like
        ``[f.breakeven_cost(n_groups=k) for k in (5, 10, 20)]`` reflects
        the spread *and* breakeven on the same bucketing. Default path
        reads cached quantile_spread / turnover_jaccard (no recompute).
        """
        from factrix.metrics.tradability import breakeven_cost as _be
        spread_val = self.quantile_spread(n_groups=n_groups).value  # type: ignore[attr-defined]
        turnover_val = self.turnover_jaccard(n_groups=n_groups).value
        return self._cached_or_compute(
            "breakeven_cost", _be, spread_val, turnover_val,
            forward_periods=self.config.forward_periods,
            override=n_groups is not None,
        )

    def net_spread(
        self,
        n_groups: int | None = None,
        estimated_cost_bps: float | None = None,
    ) -> MetricOutput:
        """Spread minus ``estimated_cost_bps`` × turnover_jaccard (signed).

        Both ``n_groups`` (bucketing) and ``estimated_cost_bps`` (market
        assumption) are overrideable so research can sweep either
        dimension without rebuilding the Factor session. Override path
        bypasses cache on the standard contract.
        """
        from factrix.metrics.tradability import net_spread as _ns
        spread_val = self.quantile_spread(n_groups=n_groups).value  # type: ignore[attr-defined]
        turnover_val = self.turnover_jaccard(n_groups=n_groups).value
        cost = (
            estimated_cost_bps if estimated_cost_bps is not None
            else self.config.estimated_cost_bps
        )
        override = n_groups is not None or estimated_cost_bps is not None
        return self._cached_or_compute(
            "net_spread", _ns, spread_val, turnover_val, cost,
            forward_periods=self.config.forward_periods,
            override=override,
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

    def ic(self) -> MetricOutput:
        """Spearman rank IC mean significance (non-overlapping t-test).

        Uses ``config.forward_periods`` — to test a different horizon,
        rebuild the session with a new config (the IC series itself is
        horizon-bound; a per-call override would only shift dof while the
        values stay computed at the original horizon, which is misleading).
        """
        from factrix.metrics.ic import ic as _ic
        return self._cached_or_compute(
            "ic", _ic, self.artifacts.get("ic_series"),
            forward_periods=self.config.forward_periods,
        )

    def ic_ir(self) -> MetricOutput:
        """IC_IR = mean(IC) / std(IC). Descriptive (no hypothesis test)."""
        from factrix.metrics.ic import ic_ir as _ic_ir
        return self._cached_or_compute(
            "ic_ir", _ic_ir, self.artifacts.get("ic_series"),
        )

    def hit_rate(self) -> MetricOutput:
        """Binomial sign-hit rate on IC series."""
        from factrix.metrics.hit_rate import hit_rate as _hr
        return self._cached_or_compute(
            "hit_rate", _hr, self.artifacts.get("ic_values"),
            forward_periods=self.config.forward_periods,
        )

    def ic_trend(self) -> MetricOutput:
        """OLS trend slope test on IC series (structural drift check)."""
        from factrix.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "ic_trend", _tr, self.artifacts.get("ic_values"),
        )

    def monotonicity(self, n_groups: int | None = None) -> MetricOutput:
        """Monotonic return ordering across factor quintiles.

        ``n_groups`` override is a legitimate per-call knob (re-groups
        the prepared panel); ``forward_periods`` is bound to the session
        and cannot be overridden — the prepared panel's ``forward_return``
        is baked at ``config.forward_periods``.
        """
        from factrix.metrics.monotonicity import monotonicity as _mono
        ng = n_groups if n_groups is not None else self.config.n_groups
        return self._cached_or_compute(
            "monotonicity", _mono, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=ng, tie_policy=self.config.tie_policy,
            override=n_groups is not None,
        )

    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput:
        """Top-bottom quantile spread per-period mean, t-test against zero.

        ``n_groups`` override bypasses the cache AND the cached
        ``spread_series`` (which was built with the bound ``n_groups``) —
        the primitive recomputes spread_series internally.
        """
        from factrix.metrics.quantile import quantile_spread as _qs
        if n_groups is not None:
            self._log_override_once("quantile_spread")
            return _qs(
                self.artifacts.prepared,
                forward_periods=self.config.forward_periods,
                n_groups=n_groups,
                tie_policy=self.config.tie_policy,
            )
        return self._cached_or_compute(
            "quantile_spread", _qs, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
            tie_policy=self.config.tie_policy,
            _precomputed_series=self.artifacts.intermediates.get("spread_series"),
        )

    def top_concentration(self, q_top: float | None = None) -> MetricOutput:
        """top bucket weight concentration (effective-N / total-N ratio)."""
        from factrix.metrics.concentration import top_concentration as _qc
        qt = q_top if q_top is not None else 1.0 / self.config.n_groups
        return self._cached_or_compute(
            "top_concentration", _qc, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            q_top=qt,
            override=q_top is not None,
        )

    # ``turnover`` / ``breakeven_cost`` / ``net_spread`` are inherited from
    # the ``Factor`` base — shared with MacroPanelFactor because the cost
    # model is identical for any portfolio-based Factor type.

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival ratio with sign-flip detection."""
        from factrix.metrics.oos import multi_split_oos_decay as _oos
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


@dataclass
class EventFactor(Factor):
    """Research session for an event-signal factor panel.

    Method set mirrors ``EventProfile.from_artifacts`` so that any metric
    surfaced on the Profile is reachable as a standalone method with the
    same cache key. Methods that need ``price`` (``mfe_mae_summary`` /
    ``event_around_return`` / ``multi_horizon_hit_rate``) return a short-
    circuit ``MetricOutput`` if the prepared panel lacks the column,
    matching the L2 opt-in contract on ``CrossSectionalFactor``.
    """

    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.EVENT_SIGNAL

    def _return_col(self) -> str:
        # Shared helper: matches EventProfile.from_artifacts + pipeline.
        from factrix.metrics._helpers import _pick_event_return_col
        return _pick_event_return_col(self.artifacts.prepared)

    # ----- Canonical + core -----

    def caar(self) -> MetricOutput:
        """CAAR non-overlapping t-test (canonical test).

        Uses ``config.forward_periods`` — the caar_series was sampled at
        that horizon; overriding fp per-call would only re-sample a series
        whose values are horizon-bound, producing misleading dof.
        """
        from factrix.metrics.caar import caar as _caar
        return self._cached_or_compute(
            "caar", _caar, self.artifacts.get("caar_series"),
            forward_periods=self.config.forward_periods,
        )

    def bmp_test(self) -> MetricOutput:
        """BMP standardized-AR z-test (variance-robust confirmation).

        Uses ``config.forward_periods`` — ``vol_scale = 1/sqrt(fp)`` must
        match the horizon used to build ``abnormal_return``.
        """
        from factrix.metrics.caar import bmp_test as _bmp
        return self._cached_or_compute(
            "bmp_test", _bmp, self.artifacts.prepared,
            return_col=self._return_col(),
            forward_periods=self.config.forward_periods,
        )

    def event_hit_rate(self) -> MetricOutput:
        """Fraction of events with signed_car > 0."""
        from factrix.metrics.event_quality import event_hit_rate as _hr
        return self._cached_or_compute(
            "event_hit_rate", _hr, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def profit_factor(self) -> MetricOutput:
        """sum(positive signed_car) / sum(negative signed_car)."""
        from factrix.metrics.event_quality import profit_factor as _pf
        return self._cached_or_compute(
            "profit_factor", _pf, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def event_skewness(self) -> MetricOutput:
        """D'Agostino skewness test on signed_car distribution."""
        from factrix.metrics.event_quality import event_skewness as _sk
        return self._cached_or_compute(
            "event_skewness", _sk, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def event_ic(self) -> MetricOutput:
        """Spearman corr(|factor|, signed_car). Short-circuits for discrete {±1}."""
        import polars as pl
        from factrix.metrics.event_quality import event_ic as _eic

        events = self.artifacts.prepared.filter(pl.col("factor") != 0)
        discrete = events["factor"].abs().n_unique() <= 1
        sc = self._short_circuit_if("event_ic", discrete, "no_magnitude_variance")
        if sc is not None:
            return sc
        return self._cached_or_compute(
            "event_ic", _eic, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def signal_density(self) -> MetricOutput:
        """Average bars per event per asset."""
        from factrix.metrics.event_quality import signal_density as _sd
        return self._cached_or_compute(
            "signal_density", _sd, self.artifacts.prepared,
        )

    def clustering_hhi(self, cluster_window: int | None = None) -> MetricOutput:
        """Event-date Herfindahl concentration. Short-circuits for N=1."""
        from factrix.metrics.clustering import clustering_diagnostic as _cl

        # Override path recomputes fresh regardless of the single-asset
        # branch; the cache peek below only helps the default path.
        if cluster_window is None:
            single_asset = self.artifacts.prepared["asset_id"].n_unique() <= 1
            sc = self._short_circuit_if("clustering_hhi", single_asset, "single_asset")
            if sc is not None:
                return sc
        cw = cluster_window if cluster_window is not None else self.config.cluster_window
        return self._cached_or_compute(
            "clustering_hhi", _cl, self.artifacts.prepared,
            cluster_window=cw, override=cluster_window is not None,
        )

    def corrado_rank_test(self) -> MetricOutput:
        """Nonparametric rank test for event abnormal returns."""
        from factrix.metrics.corrado import corrado_rank_test as _cr
        return self._cached_or_compute(
            "corrado_rank", _cr, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    # ----- Path-based (need price column) -----

    def mfe_mae_summary(self, window: int | None = None) -> MetricOutput:
        """MFE / MAE excursion ratio. Requires price column."""
        from factrix.metrics.mfe_mae import (
            compute_mfe_mae as _cm, mfe_mae_summary as _ms,
        )

        no_price = "price" not in self.artifacts.prepared.columns
        sc = self._short_circuit_if("mfe_mae_summary", no_price, "no_price_column")
        if sc is not None:
            return sc
        w = window if window is not None else self.config.event_window_post
        return self._cached_or_compute(
            "mfe_mae_summary", lambda df: _ms(_cm(df, window=w)),
            self.artifacts.prepared,
            override=window is not None,
        )

    def event_around_return(self) -> MetricOutput:
        """Per-offset return profile T-6..T+24 around events. Requires price."""
        from factrix.metrics.event_horizon import event_around_return as _ear

        no_price = "price" not in self.artifacts.prepared.columns
        sc = self._short_circuit_if("event_around_return", no_price, "no_price_column")
        if sc is not None:
            return sc
        return self._cached_or_compute(
            "event_around_return", _ear, self.artifacts.prepared,
        )

    def multi_horizon_hit_rate(self) -> MetricOutput:
        """Win rate at horizons [1, 6, 12, 24]. Requires price column."""
        from factrix.metrics.event_horizon import multi_horizon_hit_rate as _mh

        no_price = "price" not in self.artifacts.prepared.columns
        sc = self._short_circuit_if("multi_horizon_hit_rate", no_price, "no_price_column")
        if sc is not None:
            return sc
        return self._cached_or_compute(
            "multi_horizon_hit_rate", _mh, self.artifacts.prepared,
        )

    # ----- Stability -----

    def caar_trend(self) -> MetricOutput:
        """Theil-Sen trend on the CAAR value series."""
        from factrix.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "caar_trend", _tr, self.artifacts.get("caar_values"),
            name="caar_trend",
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on CAAR values."""
        from factrix.metrics.oos import multi_split_oos_decay as _oos
        return self._cached_or_compute(
            "oos_decay", _oos, self.artifacts.get("caar_values"),
        )


@dataclass
class MacroPanelFactor(Factor):
    """Research session for a macro-panel (cross-country / small-N) factor.

    Method set mirrors ``MacroPanelProfile.from_artifacts``.
    """

    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.MACRO_PANEL

    # ----- Canonical + core -----

    def fm_beta(self) -> MetricOutput:
        """Fama-MacBeth λ Newey-West t-test (canonical test).

        Threads ``config.forward_periods`` through so the NW bandwidth is
        floored at ``forward_periods - 1`` — consistent under the
        MA(h-1) overlap of h-period forward returns.
        """
        from factrix.metrics.fama_macbeth import fama_macbeth as _fm
        return self._cached_or_compute(
            "fm_beta", _fm, self.artifacts.get("beta_series"),
            forward_periods=self.config.forward_periods,
        )

    def pooled_beta(self) -> MetricOutput:
        """Pooled OLS with date-clustered SE (sign-consistency cross-check)."""
        from factrix.metrics.fama_macbeth import pooled_ols as _po
        return self._cached_or_compute(
            "pooled_beta", _po, self.artifacts.prepared,
        )

    def beta_sign_consistency(self) -> MetricOutput:
        """Fraction of periods with β in the dominant direction."""
        from factrix.metrics.fama_macbeth import beta_sign_consistency as _bsc
        return self._cached_or_compute(
            "beta_sign_consistency", _bsc, self.artifacts.get("beta_series"),
        )

    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput:
        """Top-bottom long-short spread t-test."""
        from factrix.metrics.quantile import quantile_spread as _qs
        if n_groups is not None:
            self._log_override_once("quantile_spread")
            return _qs(
                self.artifacts.prepared,
                forward_periods=self.config.forward_periods,
                n_groups=n_groups,
                tie_policy=self.config.tie_policy,
            )
        return self._cached_or_compute(
            "quantile_spread", _qs, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
            tie_policy=self.config.tie_policy,
            _precomputed_series=self.artifacts.intermediates.get("spread_series"),
        )

    # ``turnover`` / ``breakeven_cost`` / ``net_spread`` inherited from base.

    # ----- Stability -----

    def beta_trend(self) -> MetricOutput:
        """Theil-Sen trend on the FM β series."""
        from factrix.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "beta_trend", _tr, self.artifacts.get("beta_values"),
            name="beta_trend",
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on FM β values."""
        from factrix.metrics.oos import multi_split_oos_decay as _oos
        return self._cached_or_compute(
            "oos_decay", _oos, self.artifacts.get("beta_values"),
        )


@dataclass
class MacroCommonFactor(Factor):
    """Research session for a macro-common (single-time-series) factor.

    Method set mirrors ``MacroCommonProfile.from_artifacts``. The N=1
    degenerate case (single asset) is handled by the underlying primitive
    just as it is in the Profile path.
    """

    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.MACRO_COMMON

    # ----- Canonical + core -----

    def ts_beta(self) -> MetricOutput:
        """Cross-sectional t-test on per-asset TS β (canonical test).

        With N=1 the cross-sectional t-test is undefined; dispatch to
        ``ts_beta_single_asset_fallback`` which reports the single-asset
        regression's own t-stat with ``p_value=1.0`` (suppressed from
        BHY). Shared with ``MacroCommonProfile.from_artifacts`` so
        Profile-path and Factor-path produce bit-identical values.
        """
        from factrix.metrics.ts_beta import (
            ts_beta as _tsb,
            ts_beta_single_asset_fallback as _n1,
        )
        ts_betas = self.artifacts.get("beta_series")
        if len(ts_betas) <= 1:
            return self._cached_or_compute("ts_beta", _n1, ts_betas)
        return self._cached_or_compute("ts_beta", _tsb, ts_betas)

    def mean_r_squared(self) -> MetricOutput:
        """Mean per-asset regression R²."""
        from factrix.metrics.ts_beta import mean_r_squared as _mr2
        return self._cached_or_compute(
            "mean_r_squared", _mr2, self.artifacts.get("beta_series"),
        )

    def ts_beta_sign_consistency(self) -> MetricOutput:
        """Fraction of assets with β in the dominant direction."""
        from factrix.metrics.ts_beta import ts_beta_sign_consistency as _sc
        return self._cached_or_compute(
            "ts_beta_sign_consistency", _sc, self.artifacts.get("beta_series"),
        )

    # ----- Stability -----

    def beta_trend(self) -> MetricOutput:
        """Theil-Sen trend on the rolling mean β series."""
        from factrix.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "beta_trend", _tr, self.artifacts.get("beta_values"),
            name="beta_trend",
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on rolling mean β values."""
        from factrix.metrics.oos import multi_split_oos_decay as _oos
        return self._cached_or_compute(
            "oos_decay", _oos, self.artifacts.get("beta_values"),
        )


# ---------------------------------------------------------------------------
# Factor registry — factor_type → Factor subclass
# ---------------------------------------------------------------------------

_FACTOR_REGISTRY: dict[FactorType, type[Factor]] = {
    FactorType.CROSS_SECTIONAL: CrossSectionalFactor,
    FactorType.EVENT_SIGNAL: EventFactor,
    FactorType.MACRO_PANEL: MacroPanelFactor,
    FactorType.MACRO_COMMON: MacroCommonFactor,
}
