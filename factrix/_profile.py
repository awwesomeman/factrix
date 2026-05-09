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
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


@dataclass(frozen=True, slots=True, repr=False)
class FactorProfile:
    """Procedure-canonical analysis result for one factor.

    Reading ``n_obs`` and ``n_assets`` side by side disambiguates
    whether a small ``n_obs`` came from a short series or a thin
    cross-section.

    Attributes:
        config: The ``AnalysisConfig`` that produced this profile.
        mode: Evaluation mode (``PANEL`` / ``TIMESERIES``).
        primary_p: Procedure-canonical p-value driving ``verdict()``
            and ``multi_factor.bhy``.
        n_obs: Cell-canonical effective sample size.
        n_assets: Cross-section width of the raw panel.
        factor_id: User-supplied factor name; stamped by ``_evaluate``
            from ``factor_col``. Defaults to ``"factor"`` when a
            profile is constructed directly.
        context: Sample-restriction / conditioning dimensions
            (``universe_id``, ``regime_id``, future axes). Populated
            by higher-level verbs via ``dataclasses.replace``.
        warnings, info_notes, stats: per-procedure flags / scalars.

    Hashing is disabled (``__hash__ = None``) because ``context``
    defaults to ``dict`` (unhashable). Equality is field-by-field via
    the auto-generated ``__eq__``; bhy family partitioning uses
    ``identity`` directly without needing the profile to be hashable.
    """

    config: AnalysisConfig
    mode: Mode
    primary_p: float
    n_obs: int
    n_assets: int
    factor_id: str = "factor"
    context: Mapping[str, Any] = field(default_factory=dict)
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    @property
    def forward_periods(self) -> int:
        return self.config.forward_periods

    @property
    def identity(self) -> tuple[str, int]:
        return (self.factor_id, self.forward_periods)

    def verdict(
        self,
        *,
        threshold: float = 0.05,
        gate: StatCode | None = None,
    ) -> Verdict:
        """Pass/fail at ``threshold`` against ``primary_p`` (or ``gate``).

        ``threshold`` is a generic cutoff — not tied to Type-I-error
        semantics, because ``gate`` may name a non-p stat (t-stat,
        HHI, etc.). The comparison ``value < threshold`` is
        interpreted by the caller.

        Args:
            threshold: Cutoff applied to the gated value. Default
                ``0.05``.
            gate: ``StatCode`` whose value is read from ``stats``;
                ``None`` uses the procedure-canonical ``primary_p``.

        Returns:
            ``Verdict.PASS`` if the gated value is below ``threshold``,
            otherwise ``Verdict.FAIL``.

        Raises:
            KeyError: If ``gate`` is not populated in ``stats``.
        """
        p = self.primary_p if gate is None else self.stats[gate]
        return Verdict.PASS if p < threshold else Verdict.FAIL

    def _summary_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("factor_id", self.factor_id),
            ("forward_periods", self.forward_periods),
            ("mode", self.mode.value),
            ("primary_p", f"{self.primary_p:.4g}"),
            ("n_obs", self.n_obs),
            ("n_assets", self.n_assets),
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
        """Secondary stats + flag sets for human / AI agent triage.

        Returns:
            A plain-Python dict with mode, sample sizes, primary p,
            warning / info code names sorted alphabetically, and the
            full ``stats`` mapping with enum keys converted to their
            string values.
        """
        return {
            "identity": {
                "factor_id": self.factor_id,
                "forward_periods": self.forward_periods,
            },
            "context": dict(self.context),
            "mode": self.mode.value,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "primary_p": self.primary_p,
            "warnings": sorted(w.value for w in self.warnings),
            "info_notes": sorted(i.value for i in self.info_notes),
            "stats": {k.value: v for k, v in self.stats.items()},
        }
