"""v0.5 ``FactorProfile`` — single unified result type (§4.4.2 B3).

Replaces the per-cell ``CrossSectionalProfile`` / ``EventProfile`` /
``MacroPanelProfile`` / ``MacroCommonProfile`` proliferation: every cell
produces an instance of this dataclass, with cell-specific scalars
keyed in the ``stats`` mapping. Adding a new metric does not grow the
schema (§7.5).
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from factrix._axis import Mode
from factrix._codes import InfoCode, StatCode, WarningCode

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


@dataclass(frozen=True, slots=True, repr=False)
class FactorProfile:
    """Procedure-canonical analysis result for one factor.

    Reading the four sample axes (``n_obs`` / ``n_pairs`` /
    ``n_periods`` / ``n_assets``) side by side disambiguates whether a
    small ``n_obs`` came from a short series, a thin cross-section, or
    a sparse panel. Each axis answers one question and never overlaps
    with another.

    The four sample axes are paired with ``primary_*``, not with
    secondary diagnostic entries in ``stats`` — e.g. an ADF run on the
    factor reports its own n inside its ``stats`` / ``metadata`` entry,
    not via ``n_obs``.

    Hashing is disabled (``__hash__ = None``) because ``context``
    defaults to ``dict`` (unhashable). Equality is field-by-field via
    the auto-generated ``__eq__``; bhy family partitioning uses
    ``identity`` directly without needing the profile to be hashable.

    Attributes:
        config: The ``AnalysisConfig`` that produced this profile.
        mode: Evaluation mode (``PANEL`` / ``TIMESERIES``).
        primary_p: Procedure-canonical p-value driving
            ``multi_factor.bhy``.
        primary_stat: Test statistic value paired with ``primary_p``
            (e.g. ``t_nw`` value for an NW HAC t-test). ``None`` when
            the primary procedure produces no test statistic (e.g.
            empirical-p block bootstrap). Invariant:
            ``stats[primary_stat_name] == primary_stat`` whenever
            ``primary_stat is not None``.
        primary_stat_name: ``stats``-key pointer for ``primary_stat``
            (e.g. ``StatCode.T_NW`` / ``StatCode.WALD_NWCL`` /
            ``StatCode.P_BOOT``). Always populated; for a no-test-stat
            primary (e.g. ``StatCode.P_BOOT``) it points at the
            p-value entry itself. Serialised to its ``.value`` slug by
            ``diagnose()``.
        n_obs: Cell-canonical final-stage test denominator — the
            sample size the primary estimator actually saw after
            procedure-internal trimming. Reflects the n that
            ``primary_p`` is computed against, not the raw panel
            envelope. Effective-DoF adjustments (NW HAC autocorrelation,
            overlapping windows) live inside ``stats`` / ``metadata``,
            not here. ``n_obs = 0`` is a legal degenerate value.
        n_pairs: Raw count of non-null (period, asset) pairs entering
            the cell's first stage. Sparsity numerator
            (``n_pairs / (n_periods * n_assets)``). Always
            ``>= n_obs`` (first-stage count vs. final-stage count).
        n_periods: Unique periods in the raw panel under the
            any-non-null union. Calendar time, not event time.
        n_assets: Unique assets in the raw panel under the
            any-non-null union. ``n_assets = 1`` is a legal signal
            (single-asset TIMESERIES).
        factor_id: User-supplied factor name; stamped by ``_evaluate``
            from ``factor_col``. Defaults to ``"factor"`` when a
            profile is constructed directly.
        context: Sample-restriction / conditioning dimensions
            (``universe_id``, ``regime_id``, future axes). Populated
            by higher-level verbs via ``dataclasses.replace``.
        warnings, info_notes, stats: per-procedure flags / scalars.
        metadata: Hyperparameter-selection records the procedure made
            internally, keyed by the ``StatCode`` they produced.
            Symmetric with ``stats`` — for any populated entry,
            ``stats[code]`` is the value and ``metadata[code]`` is the
            inner dict of hyperparameters that produced it (e.g.
            ``{"nw_lags": 5}`` for ``T_NW`` / ``P`` under an NW HAC
            test). Stats with no hyperparameter (e.g. plain ``MEAN``)
            are absent from the mapping rather than mapping to an
            empty dict. Tests that share a hyperparameter (NW
            populates both ``T_NW`` and ``P`` from one bandwidth
            choice) duplicate the inner dict under both keys to keep
            single-key lookup honest (#188).

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> profile = fx.evaluate(panel, cfg)
        >>> isinstance(profile, fx.FactorProfile)
        True
        >>> 0.0 <= profile.primary_p <= 1.0
        True
        >>> profile.n_assets == 20
        True
    """

    config: AnalysisConfig
    mode: Mode
    primary_p: float
    primary_stat: float | None
    primary_stat_name: StatCode
    n_obs: int
    n_pairs: int
    n_periods: int
    n_assets: int
    factor_id: str = "factor"
    context: Mapping[str, Any] = field(default_factory=dict)
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)
    metadata: Mapping[StatCode, Mapping[str, Any]] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    @property
    def forward_periods(self) -> int:
        return self.config.forward_periods

    @property
    def identity(self) -> tuple[str, int]:
        return (self.factor_id, self.forward_periods)

    def _summary_rows(self) -> list[tuple[str, Any]]:
        stat_repr = "None" if self.primary_stat is None else f"{self.primary_stat:.4g}"
        rows: list[tuple[str, Any]] = [
            ("factor_id", self.factor_id),
            ("forward_periods", self.forward_periods),
            ("mode", self.mode.value),
            ("n_obs", self.n_obs),
            ("n_pairs", self.n_pairs),
            ("n_periods", self.n_periods),
            ("n_assets", self.n_assets),
            (
                "primary_p",
                f"{self.primary_p:.4g} (stat={stat_repr}, name={self.primary_stat_name.value})",
            ),
        ]
        for k in sorted(self.context):
            rows.append((f"context.{k}", self.context[k]))
        if self.warnings:
            rows.append(("warnings", ", ".join(sorted(w.value for w in self.warnings))))
        return rows

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self._summary_rows()]
        return f"FactorProfile({', '.join(parts)})"

    def _repr_html_(self) -> str:
        body = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in self._summary_rows()
        )
        return (
            "<table class='factrix-factor-profile'>"
            "<caption>FactorProfile</caption>"
            f"{body}</table>"
        )

    def diagnose(self) -> dict[str, Any]:
        """JSON-shaped view for human / AI agent triage.

        Key order follows the reader-flow: identity → context →
        dispatch cell → sample axes → primary significance → flag
        sets → raw stats / metadata.

        Returns:
            A plain-Python dict with ``cell`` (scope / signal / metric
            / mode), the four sample axes, the ``primary_*`` family
            (``primary_p`` / ``primary_stat`` / ``primary_stat_name``),
            sorted warning / info code names, and the full ``stats``
            mapping with enum keys converted to their string values.

        Examples:
            >>> import factrix as fx
            >>> from factrix.preprocess import compute_forward_return
            >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
            >>> panel = compute_forward_return(raw, forward_periods=5)
            >>> profile = fx.evaluate(panel, fx.AnalysisConfig.individual_continuous())
            >>> d = profile.diagnose()
            >>> isinstance(d, dict)
            True
            >>> set(["identity", "cell", "primary_p", "stats"]).issubset(d)
            True
            >>> d["cell"]["scope"] == "individual"
            True
        """
        return {
            "identity": {
                "factor_id": self.factor_id,
                "forward_periods": self.forward_periods,
            },
            "context": dict(self.context),
            "cell": {
                "scope": self.config.scope.value,
                "signal": self.config.signal.value,
                "metric": (
                    self.config.metric.value if self.config.metric is not None else None
                ),
                "mode": self.mode.value,
            },
            "n_obs": self.n_obs,
            "n_pairs": self.n_pairs,
            "n_periods": self.n_periods,
            "n_assets": self.n_assets,
            "primary_p": self.primary_p,
            "primary_stat": self.primary_stat,
            "primary_stat_name": self.primary_stat_name.value,
            "warnings": sorted(w.value for w in self.warnings),
            "info_notes": sorted(i.value for i in self.info_notes),
            "stats": {k.value: v for k, v in self.stats.items()},
            "metadata": {k.value: dict(v) for k, v in self.metadata.items()},
        }
