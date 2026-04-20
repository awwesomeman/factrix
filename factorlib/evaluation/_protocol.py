"""Shared evaluation primitives.

``Artifacts`` is a read-only container of pre-computed intermediates
shared by every per-type ``from_artifacts`` classmethod — the runtime
data bridge between ``build_artifacts`` (pipeline) and the Profile
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from factorlib._types import MetricOutput
from factorlib.config import BaseConfig


class _CompactedPrepared:
    """Placeholder substituted for ``Artifacts.prepared`` in compact mode.

    Any access -- attribute, indexing, truth-testing, iteration -- raises
    a targeted ``RuntimeError``. Slot-based dunders bypass ``__getattr__``
    so each must be listed explicitly; otherwise ``bool(prepared)`` would
    silently return True and ``if art.prepared:`` would give the wrong
    answer.
    """

    __slots__ = ()

    @staticmethod
    def _format_msg(target: str) -> str:
        """Build the compact-mode error message for a specific access site.

        Single source of truth so ``__getattr__``'s attribute-specific
        wording and the container-dunder generic wording stay in lock-step.
        """
        return (
            f"Cannot access {target}: Artifacts is in compact mode "
            f"(fl.evaluate_batch(..., compact=True) dropped the prepared "
            f"DataFrame to save memory). Either re-run evaluate_batch "
            f"without compact=True, or use metrics that read only "
            f"artifacts.intermediates / artifacts.metric_outputs (e.g. "
            f"redundancy_matrix(method='value_series'))."
        )

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(self._format_msg(f"'prepared.{name}'"))

    def __bool__(self) -> bool:
        raise RuntimeError(self._format_msg("'Artifacts.prepared'"))

    def __len__(self) -> int:
        raise RuntimeError(self._format_msg("'Artifacts.prepared'"))

    def __iter__(self):
        raise RuntimeError(self._format_msg("'Artifacts.prepared'"))

    def __getitem__(self, key):
        raise RuntimeError(self._format_msg("'Artifacts.prepared'"))

    def __contains__(self, item) -> bool:
        raise RuntimeError(self._format_msg("'Artifacts.prepared'"))

    def __repr__(self) -> str:
        return "<CompactedPrepared: prepared DataFrame dropped>"


_COMPACTED_PREPARED = _CompactedPrepared()


@dataclass
class Artifacts:
    """Pre-computed intermediate results consumed by Profile builders.

    Built once by ``build_artifacts`` and handed to
    ``Profile.from_artifacts``. ``intermediates`` holds type-specific
    DataFrames (e.g. ``ic_series``, ``spread_series`` for
    cross-sectional). Use ``.get(key)`` for access with a helpful
    KeyError on missing keys.

    ``metric_outputs`` is the parallel channel for raw ``MetricOutput``
    objects keyed by ``MetricOutput.name``; power users read per-metric
    ``.metadata`` dicts (per_regime / per_horizon / betas) directly for
    drill-down. Shares lifecycle with ``intermediates`` (both dropped
    by ``keep_artifacts=False``).

    ``factor_name`` identifies which factor this instance represents —
    consumed by per-type Profile ``from_artifacts`` classmethods and by
    downstream plotting / reporting that needs to label outputs.

    ``compact`` toggles memory-saving mode: when True, ``prepared`` is
    replaced with a sentinel that raises on any attribute access. Use
    for 1000-factor batches where the prepared panel (~MB per factor)
    would exhaust memory. ``intermediates`` and ``metric_outputs``
    (small DataFrames / MetricOutput objects) are kept because metrics
    and diagnose() need them.
    """

    prepared: pl.DataFrame
    config: BaseConfig
    intermediates: dict[str, pl.DataFrame] = field(default_factory=dict)
    metric_outputs: dict[str, MetricOutput] = field(default_factory=dict)
    factor_name: str = ""
    compact: bool = False

    def __post_init__(self) -> None:
        if self.compact:
            object.__setattr__(self, "prepared", _COMPACTED_PREPARED)

    def get(self, key: str) -> pl.DataFrame:
        if key not in self.intermediates:
            ft = type(self.config).factor_type
            raise KeyError(
                f"Artifacts has no '{key}'. "
                f"Available for {ft}: {list(self.intermediates.keys())}"
            )
        return self.intermediates[key]
