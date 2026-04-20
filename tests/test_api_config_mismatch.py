"""Raise on silent / asymmetric config handling.

Covers four correctness fixes:

- ``fl.evaluate`` / ``fl.factor`` / ``fl.evaluate_batch`` — ``factor_type``
  vs ``config.factor_type`` mismatch must raise instead of silently
  letting the config win.
- ``preprocess_cs_factor`` — passing both ``config=`` and ad-hoc kwargs
  (``forward_periods`` / ``mad_n`` / ``return_clip_pct``) must raise,
  matching ``fl.evaluate``'s error surface.
- ``_CompactedPrepared`` — error message must name the root cause
  (``fl.evaluate_batch(..., compact=True)``) so the trace points at the
  fix, not at the sentinel internals.

These guards all exist because ``factorlib``'s north star prioritizes
statistical correctness (U4) over convenience: a silently-overridden
``forward_periods`` poisons every downstream metric, and a silent
``factor_type`` mismatch lets BHY batch two incompatible test families.
"""

from __future__ import annotations

import polars as pl
import pytest

import factorlib as fl
from factorlib.config import (
    CrossSectionalConfig,
    EventConfig,
    MacroPanelConfig,
    MacroCommonConfig,
)
from factorlib.preprocess.pipeline import preprocess_cs_factor


# ---------------------------------------------------------------------------
# factor_type ↔ config.factor_type mismatch
# ---------------------------------------------------------------------------

class TestFactorTypeMismatch:
    """Passing both ``factor_type=`` and an inconsistent ``config=`` raises."""

    def test_evaluate_mismatch_raises(self, noisy_panel):
        with pytest.raises(TypeError, match="disagree.*must refer to the same factor type"):
            fl.evaluate(
                noisy_panel, "f",
                factor_type="event_signal",
                config=CrossSectionalConfig(),
            )

    def test_factor_mismatch_raises(self, noisy_panel):
        with pytest.raises(TypeError, match="disagree.*must refer to the same factor type"):
            fl.factor(
                noisy_panel, "f",
                factor_type="macro_panel",
                config=CrossSectionalConfig(),
            )

    def test_evaluate_batch_mismatch_raises(self, noisy_panel):
        with pytest.raises(TypeError, match="disagree.*must refer to the same factor type"):
            fl.evaluate_batch(
                {"f": noisy_panel},
                factor_type="macro_common",
                config=CrossSectionalConfig(),
            )

    def test_matching_factor_type_still_works(self, noisy_panel):
        # Explicit redundant supply is allowed as long as it matches.
        p = fl.evaluate(
            noisy_panel, "f",
            factor_type="cross_sectional",
            config=CrossSectionalConfig(),
        )
        assert p.factor_name == "f"

    def test_default_factor_type_works_without_config(self, noisy_panel):
        # No explicit factor_type, no config: falls back to cross_sectional.
        p = fl.evaluate(noisy_panel, "f")
        assert type(p).__name__ == "CrossSectionalProfile"


# ---------------------------------------------------------------------------
# preprocess_cs_factor config + kwargs double-pass
# ---------------------------------------------------------------------------

class TestPreprocessDoublePass:
    """Mirror fl.evaluate's TypeError when both config and kwargs are passed."""

    def _raw(self) -> pl.DataFrame:
        from datetime import datetime, timedelta
        import numpy as np
        rng = np.random.default_rng(0)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        rows = []
        for d in dates:
            for a in range(10):
                rows.append({
                    "date": d, "asset_id": f"a{a}",
                    "price": 100.0 + rng.standard_normal(),
                    "factor": float(rng.standard_normal()),
                })
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_config_plus_forward_periods_raises(self):
        df = self._raw()
        cfg = CrossSectionalConfig(forward_periods=5)
        # Error must name the offending kwarg so the fix is obvious.
        with pytest.raises(TypeError, match=r"cannot pass both config=.*forward_periods"):
            preprocess_cs_factor(df, config=cfg, forward_periods=3)

    def test_config_plus_mad_n_raises(self):
        df = self._raw()
        cfg = CrossSectionalConfig()
        with pytest.raises(TypeError, match=r"cannot pass both config=.*mad_n"):
            preprocess_cs_factor(df, config=cfg, mad_n=2.0)

    def test_config_plus_return_clip_raises(self):
        df = self._raw()
        cfg = CrossSectionalConfig()
        with pytest.raises(TypeError, match=r"cannot pass both config=.*return_clip_pct"):
            preprocess_cs_factor(df, config=cfg, return_clip_pct=(0.05, 0.95))

    def test_config_only_works(self):
        df = self._raw()
        cfg = CrossSectionalConfig(forward_periods=3, mad_n=2.5)
        out = preprocess_cs_factor(df, config=cfg)
        assert "forward_return" in out.columns

    def test_kwargs_only_works(self):
        df = self._raw()
        out = preprocess_cs_factor(df, forward_periods=3, mad_n=2.5)
        assert "forward_return" in out.columns

    def test_no_args_uses_defaults(self):
        df = self._raw()
        out = preprocess_cs_factor(df)
        assert "forward_return" in out.columns


# ---------------------------------------------------------------------------
# _CompactedPrepared error message points at the root cause
# ---------------------------------------------------------------------------

class TestPublicExportSurface:
    """Q14: ``Factor`` base class is not in ``fl.__all__`` — users should
    reach for ``fl.factor()`` factory, not construct the base directly."""

    def test_factor_base_not_in_all(self):
        import factorlib as fl
        assert "Factor" not in fl.__all__, (
            "Factor base is intentionally excluded — use fl.factor() factory. "
            "Import from factorlib.factor directly if you need it for type hints."
        )

    def test_factor_factory_and_subclasses_are_in_all(self):
        import factorlib as fl
        for name in ("factor", "CrossSectionalFactor", "EventFactor",
                     "MacroPanelFactor", "MacroCommonFactor"):
            assert name in fl.__all__, f"{name} missing from fl.__all__"

    def test_factor_base_still_importable_for_type_hints(self):
        # Not in __all__ but still reachable from the submodule.
        from factorlib.factor import Factor
        assert Factor.__name__ == "Factor"


class TestCompactedErrorAttribution:
    def test_attr_error_names_evaluate_batch_compact(self):
        from factorlib.evaluation._protocol import _COMPACTED_PREPARED
        with pytest.raises(RuntimeError, match=r"evaluate_batch\(.*compact=True"):
            _COMPACTED_PREPARED.columns

    def test_bool_error_names_evaluate_batch_compact(self):
        from factorlib.evaluation._protocol import _COMPACTED_PREPARED
        with pytest.raises(RuntimeError, match=r"evaluate_batch\(.*compact=True"):
            bool(_COMPACTED_PREPARED)
