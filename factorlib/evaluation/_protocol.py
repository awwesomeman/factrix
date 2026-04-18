"""Shared evaluation primitives.

``Artifacts`` is a read-only container of pre-computed intermediates
shared by every per-type ``from_artifacts`` classmethod — the runtime
data bridge between ``build_artifacts`` (pipeline) and the Profile
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

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

    _MSG = (
        "Cannot use 'Artifacts.prepared': Artifacts is in compact mode "
        "(prepared DataFrame was dropped to save memory). Rebuild the "
        "Artifacts without compact=True if you need the prepared panel."
    )

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(
            f"Cannot access 'prepared.{name}': Artifacts is in compact mode "
            f"(prepared DataFrame was dropped to save memory). Rebuild the "
            f"Artifacts without compact=True if you need the prepared panel."
        )

    def __bool__(self) -> bool:
        raise RuntimeError(self._MSG)

    def __len__(self) -> int:
        raise RuntimeError(self._MSG)

    def __iter__(self):
        raise RuntimeError(self._MSG)

    def __getitem__(self, key):
        raise RuntimeError(self._MSG)

    def __contains__(self, item) -> bool:
        raise RuntimeError(self._MSG)

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

    ``factor_name`` identifies which factor this instance represents —
    consumed by per-type Profile ``from_artifacts`` classmethods and by
    downstream plotting / reporting that needs to label outputs.

    ``compact`` toggles memory-saving mode: when True, ``prepared`` is
    replaced with a sentinel that raises on any attribute access. Use
    for 1000-factor batches where the prepared panel (~MB per factor)
    would exhaust memory. Intermediates (small DataFrames) are kept
    because metrics and diagnose() need them.
    """

    prepared: pl.DataFrame
    config: BaseConfig
    intermediates: dict[str, pl.DataFrame] = field(default_factory=dict)
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
