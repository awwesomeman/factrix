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
        from factorlib.evaluation.profiles._base import _run_profile_and_attach

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

        Enforces the ``cache-key == primitive.MetricOutput.name`` contract:
        if the computed output's ``.name`` differs from ``name``, ``_stash``
        would key the entry under the primitive's name and leave the Factor
        side of the cache empty — a silent double-entry / cache-miss bug.
        ``_memoized`` + ``_stash`` already do the store; we assert here on
        the override (uncached) path because ``_memoized`` returns the
        stored view and can't double-check after stashing.
        """
        if override:
            result = fn(*args, **kwargs)
            assert result.name == name, (
                f"Factor cache-key contract violated: method requested "
                f"cache key {name!r} but primitive returned "
                f"MetricOutput.name={result.name!r}"
            )
            return result
        return _memoized(
            self.artifacts.metric_outputs, name, fn, *args, **kwargs,
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
        """Period-over-period weight turnover fraction."""
        from factorlib.metrics.tradability import turnover as _tn
        return self._cached_or_compute(
            "turnover", _tn, self.artifacts.prepared,
        )

    def breakeven_cost(self, n_groups: int | None = None) -> MetricOutput:
        """Per-side cost (bps) that fully erodes the long-short spread.

        ``n_groups`` override threads through ``quantile_spread`` so a
        sensitivity sweep like
        ``[f.breakeven_cost(n_groups=k) for k in (5, 10, 20)]`` reflects
        the spread *and* breakeven on the same bucketing. Default path
        reads cached quantile_spread / turnover (no recompute).
        """
        from factorlib.metrics.tradability import breakeven_cost as _be
        spread_val = self.quantile_spread(n_groups=n_groups).value  # type: ignore[attr-defined]
        turnover_val = self.turnover().value
        return self._cached_or_compute(
            "breakeven_cost", _be, spread_val, turnover_val,
            override=n_groups is not None,
        )

    def net_spread(
        self,
        n_groups: int | None = None,
        estimated_cost_bps: float | None = None,
    ) -> MetricOutput:
        """Spread minus ``estimated_cost_bps`` × turnover (signed).

        Both ``n_groups`` (bucketing) and ``estimated_cost_bps`` (market
        assumption) are overrideable so research can sweep either
        dimension without rebuilding the Factor session. Override path
        bypasses cache on the standard contract.
        """
        from factorlib.metrics.tradability import net_spread as _ns
        spread_val = self.quantile_spread(n_groups=n_groups).value  # type: ignore[attr-defined]
        turnover_val = self.turnover().value
        cost = (
            estimated_cost_bps if estimated_cost_bps is not None
            else self.config.estimated_cost_bps
        )
        override = n_groups is not None or estimated_cost_bps is not None
        return self._cached_or_compute(
            "net_spread", _ns, spread_val, turnover_val, cost,
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
        """long-short spread per-period mean, t-test against zero.

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
            "long_short_spread", _qs, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
            _precomputed_series=self.artifacts.intermediates.get("spread_series"),
        )

    def top_concentration(self, q_top: float | None = None) -> MetricOutput:
        """top bucket weight concentration (effective-N / total-N ratio)."""
        from factorlib.metrics.concentration import top_concentration as _qc
        qt = q_top if q_top is not None else 1.0 / self.config.n_groups
        return self._cached_or_compute(
            "top_concentration", _qc, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            q_top=qt, override=q_top is not None,
        )

    # ``turnover`` / ``breakeven_cost`` / ``net_spread`` are inherited from
    # the ``Factor`` base — shared with MacroPanelFactor because the cost
    # model is identical for any portfolio-based Factor type.

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
        from factorlib.metrics._helpers import _pick_event_return_col
        return _pick_event_return_col(self.artifacts.prepared)

    # ----- Canonical + core -----

    def caar(self, forward_periods: int | None = None) -> MetricOutput:
        """CAAR non-overlapping t-test (canonical test)."""
        from factorlib.metrics.caar import caar as _caar
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        return self._cached_or_compute(
            "caar", _caar, self.artifacts.get("caar_series"),
            forward_periods=fp, override=forward_periods is not None,
        )

    def bmp_test(self, forward_periods: int | None = None) -> MetricOutput:
        """BMP standardized-AR z-test (variance-robust confirmation)."""
        from factorlib.metrics.caar import bmp_test as _bmp
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        return self._cached_or_compute(
            "bmp_sar", _bmp, self.artifacts.prepared,
            return_col=self._return_col(), forward_periods=fp,
            override=forward_periods is not None,
        )

    def event_hit_rate(self) -> MetricOutput:
        """Fraction of events with signed_car > 0."""
        from factorlib.metrics.event_quality import event_hit_rate as _hr
        return self._cached_or_compute(
            "event_hit_rate", _hr, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def profit_factor(self) -> MetricOutput:
        """sum(positive signed_car) / sum(negative signed_car)."""
        from factorlib.metrics.event_quality import profit_factor as _pf
        return self._cached_or_compute(
            "profit_factor", _pf, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def event_skewness(self) -> MetricOutput:
        """D'Agostino skewness test on signed_car distribution."""
        from factorlib.metrics.event_quality import event_skewness as _sk
        return self._cached_or_compute(
            "event_skewness", _sk, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    def event_ic(self) -> MetricOutput:
        """Spearman corr(|factor|, signed_car). Short-circuits for discrete {±1}."""
        import polars as pl
        from factorlib.metrics.event_quality import event_ic as _eic

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
        from factorlib.metrics.event_quality import signal_density as _sd
        return self._cached_or_compute(
            "signal_density", _sd, self.artifacts.prepared,
        )

    def clustering_hhi(self, cluster_window: int | None = None) -> MetricOutput:
        """Event-date Herfindahl concentration. Short-circuits for N=1."""
        from factorlib.metrics.clustering import clustering_diagnostic as _cl

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
        from factorlib.metrics.corrado import corrado_rank_test as _cr
        return self._cached_or_compute(
            "corrado_rank", _cr, self.artifacts.prepared,
            return_col=self._return_col(),
        )

    # ----- Path-based (need price column) -----

    def mfe_mae_summary(self, window: int | None = None) -> MetricOutput:
        """MFE / MAE excursion ratio. Requires price column.

        Cache key is ``"mfe_mae"`` (the primitive's ``MetricOutput.name``),
        not ``"mfe_mae_summary"`` — ``_stash`` keys by ``.name`` so the
        method/key drift is intentional to avoid double-cache entries.
        """
        from factorlib.metrics.mfe_mae import (
            compute_mfe_mae as _cm, mfe_mae_summary as _ms,
        )

        no_price = "price" not in self.artifacts.prepared.columns
        sc = self._short_circuit_if("mfe_mae", no_price, "no_price_column")
        if sc is not None:
            return sc
        w = window if window is not None else self.config.event_window_post
        return self._cached_or_compute(
            "mfe_mae", lambda df: _ms(_cm(df, window=w)),
            self.artifacts.prepared,
            override=window is not None,
        )

    def event_around_return(self) -> MetricOutput:
        """Per-offset return profile T-6..T+24 around events. Requires price."""
        from factorlib.metrics.event_horizon import event_around_return as _ear

        no_price = "price" not in self.artifacts.prepared.columns
        sc = self._short_circuit_if("event_around_return", no_price, "no_price_column")
        if sc is not None:
            return sc
        return self._cached_or_compute(
            "event_around_return", _ear, self.artifacts.prepared,
        )

    def multi_horizon_hit_rate(self) -> MetricOutput:
        """Win rate at horizons [1, 6, 12, 24]. Requires price column."""
        from factorlib.metrics.event_horizon import multi_horizon_hit_rate as _mh

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
        # WHY: EventProfile stashes under "ic_trend" key (shared trend helper).
        from factorlib.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "ic_trend", _tr, self.artifacts.get("caar_values"),
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on CAAR values."""
        from factorlib.metrics.oos import multi_split_oos_decay as _oos
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
        """Fama-MacBeth λ Newey-West t-test (canonical test)."""
        from factorlib.metrics.fama_macbeth import fama_macbeth as _fm
        return self._cached_or_compute(
            "fm_beta", _fm, self.artifacts.get("beta_series"),
        )

    def pooled_beta(self) -> MetricOutput:
        """Pooled OLS with date-clustered SE (sign-consistency cross-check)."""
        from factorlib.metrics.fama_macbeth import pooled_ols as _po
        return self._cached_or_compute(
            "pooled_beta", _po, self.artifacts.prepared,
        )

    def beta_sign_consistency(self) -> MetricOutput:
        """Fraction of periods with β in the dominant direction."""
        from factorlib.metrics.fama_macbeth import beta_sign_consistency as _bsc
        return self._cached_or_compute(
            "beta_sign_consistency", _bsc, self.artifacts.get("beta_series"),
        )

    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput:
        """Top-bottom long-short spread t-test."""
        from factorlib.metrics.quantile import quantile_spread as _qs
        if n_groups is not None:
            return _qs(
                self.artifacts.prepared,
                forward_periods=self.config.forward_periods,
                n_groups=n_groups,
            )
        return self._cached_or_compute(
            "long_short_spread", _qs, self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=self.config.n_groups,
            _precomputed_series=self.artifacts.intermediates.get("spread_series"),
        )

    # ``turnover`` / ``breakeven_cost`` / ``net_spread`` inherited from base.

    # ----- Stability -----

    def beta_trend(self) -> MetricOutput:
        """Theil-Sen trend on the FM β series.

        Cache key is ``"ic_trend"`` because the primitive's
        ``MetricOutput.name`` is ``ic_trend`` (shared ``ic_trend`` /
        ``trend.py`` helper applied to the β series). Same reuse pattern
        as ``EventFactor.caar_trend`` and ``MacroCommonFactor.beta_trend``.
        """
        from factorlib.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "ic_trend", _tr, self.artifacts.get("beta_values"),
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on FM β values."""
        from factorlib.metrics.oos import multi_split_oos_decay as _oos
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
        from factorlib.metrics.ts_beta import (
            ts_beta as _tsb,
            ts_beta_single_asset_fallback as _n1,
        )
        ts_betas = self.artifacts.get("beta_series")
        if len(ts_betas) <= 1:
            return self._cached_or_compute("ts_beta", _n1, ts_betas)
        return self._cached_or_compute("ts_beta", _tsb, ts_betas)

    def mean_r_squared(self) -> MetricOutput:
        """Mean per-asset regression R²."""
        from factorlib.metrics.ts_beta import mean_r_squared as _mr2
        return self._cached_or_compute(
            "mean_r_squared", _mr2, self.artifacts.get("beta_series"),
        )

    def ts_beta_sign_consistency(self) -> MetricOutput:
        """Fraction of assets with β in the dominant direction."""
        from factorlib.metrics.ts_beta import ts_beta_sign_consistency as _sc
        return self._cached_or_compute(
            "ts_beta_sign_consistency", _sc, self.artifacts.get("beta_series"),
        )

    # ----- Stability -----

    def beta_trend(self) -> MetricOutput:
        """Theil-Sen trend on the rolling mean β series."""
        from factorlib.metrics.trend import ic_trend as _tr
        return self._cached_or_compute(
            "ic_trend", _tr, self.artifacts.get("beta_values"),
        )

    def oos_decay(self) -> MetricOutput:
        """Multi-split OOS survival on rolling mean β values."""
        from factorlib.metrics.oos import multi_split_oos_decay as _oos
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
