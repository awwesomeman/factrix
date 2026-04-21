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

from tests.conftest import _cs_panel


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


class TestPreprocessTimeFieldMismatch:
    """Full preprocess-time fingerprint gate — extends the fp-only check.

    Prepared panels embed every preprocess-time field in a
    ``_fl_preprocess_sig`` marker. ``fl.evaluate`` / ``fl.factor`` re-
    derive the sig from the supplied config and raise on ANY disagreement,
    not just ``forward_periods``. Evaluate-time fields (``n_groups``,
    ``tie_policy``, ``estimated_cost_bps``, ``ortho``, ``regime_labels``,
    ``multi_horizon_periods``, ``spanning_base_spreads``) are intentionally
    NOT part of the sig so sweep patterns remain cheap.
    """

    def test_mad_n_mismatch_raises(self):
        df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.3, seed=101, include_price=True)
        prepared = fl.preprocess(df, config=CrossSectionalConfig(mad_n=3.0))
        with pytest.raises(
            ValueError,
            match=r"(?s)preprocess-time fields mismatch.*mad_n",
        ):
            fl.evaluate(prepared, "x", config=CrossSectionalConfig(mad_n=2.5))

    def test_return_clip_pct_mismatch_raises(self):
        df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.3, seed=102, include_price=True)
        prepared = fl.preprocess(
            df, config=CrossSectionalConfig(return_clip_pct=(0.01, 0.99)),
        )
        with pytest.raises(
            ValueError,
            match=r"(?s)preprocess-time fields mismatch.*return_clip_pct",
        ):
            fl.evaluate(
                prepared, "x",
                config=CrossSectionalConfig(return_clip_pct=(0.05, 0.95)),
            )

    def test_evaluate_time_field_sweep_does_not_raise(self):
        """n_groups / tie_policy / estimated_cost_bps differing between
        preprocess-cfg and evaluate-cfg is the sweep pattern — must pass."""
        df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.3, seed=103, include_price=True)
        prepared = fl.preprocess(
            df, config=CrossSectionalConfig(forward_periods=5, n_groups=5),
        )
        profile = fl.evaluate(
            prepared, "x",
            config=CrossSectionalConfig(
                forward_periods=5,
                n_groups=3,
                tie_policy="average",
                estimated_cost_bps=50.0,
            ),
        )
        assert profile.factor_name == "x"

    def test_macro_panel_demean_mismatch_raises(self):
        df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.3, seed=104, include_price=True)
        prepared = fl.preprocess(
            df, config=MacroPanelConfig(demean_cross_section=False),
        )
        with pytest.raises(
            ValueError,
            match=r"(?s)preprocess-time fields mismatch.*demean_cross_section",
        ):
            fl.evaluate(
                prepared, "x",
                config=MacroPanelConfig(demean_cross_section=True),
            )

    def test_factor_type_mismatch_across_preprocess_and_evaluate_raises(self):
        """Prepared panel pinned to one factor_type cannot be re-evaluated
        under a different type's config — catches the class of silent
        wrongness where column presence passes but the canonical test is
        wrong (e.g. IC computed on event {-1, 0, +1} signals).
        """
        df = _cs_panel(n_dates=60, n_assets=30, signal_coef=0.3, seed=105, include_price=True)
        prepared = fl.preprocess(df, config=EventConfig(forward_periods=5))
        with pytest.raises(
            ValueError,
            match=r"(?s)preprocess-time fields mismatch.*factor_type",
        ):
            fl.evaluate(
                prepared, "x",
                config=CrossSectionalConfig(forward_periods=5),
            )

    def test_preprocess_without_config_raises(self):
        """Silent default CrossSectionalConfig() would CS-preprocess an
        event / macro panel with no downstream marker to catch the
        mismatch — require explicit config so user intent is visible.
        """
        df = _cs_panel(n_dates=30, n_assets=10, signal_coef=0.1, seed=106, include_price=True)
        with pytest.raises(
            TypeError,
            match=r"(?s)fl\.preprocess requires an explicit config=.*CrossSectionalConfig.*EventConfig",
        ):
            fl.preprocess(df)


class TestSingleAssetFactorTypeMismatch:
    """N=1 panel on CS / MP raises with actionable message instead of
    silently short-circuiting to FAILED verdict. Event / macro_common
    tolerate N=1 by design and must keep working.
    """

    def test_cross_sectional_n1_raises_at_preprocess(self):
        """Guard fires early at preprocess, not later at evaluate — user
        should not waste a preprocess pass before learning the factor_type
        is wrong for their data shape."""
        df = _cs_panel(n_dates=60, n_assets=1, signal_coef=0.0, seed=200, include_price=True)
        with pytest.raises(
            ValueError,
            match=r"(?s)cross_sectional expects a multi-asset panel.*MacroCommonConfig",
        ):
            fl.preprocess(df, config=CrossSectionalConfig(forward_periods=5))

    def test_cross_sectional_n1_still_raises_if_preprocess_skipped(self):
        """Backstop: callers that hand a raw-ish DataFrame to build_artifacts
        directly (bypassing preprocess) still get the guard at evaluate time.
        Exercised by supplying a prepared-shaped panel manually."""
        import numpy as np, polars as pl
        from datetime import datetime, timedelta
        rng = np.random.default_rng(280)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        # Hand-build a panel that passes _validate_columns but trips the
        # build_artifacts backstop (forward_return present, N=1).
        rows = [
            {"date": d, "asset_id": "A1",
             "factor": float(rng.standard_normal()),
             "forward_return": float(rng.standard_normal())}
            for d in dates
        ]
        prepared_like = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        from factorlib.evaluation.pipeline import build_artifacts
        with pytest.raises(
            ValueError,
            match=r"(?s)cross_sectional expects a multi-asset panel",
        ):
            build_artifacts(prepared_like, CrossSectionalConfig(forward_periods=5))

    def test_macro_panel_n1_raises_at_preprocess(self):
        df = _cs_panel(n_dates=60, n_assets=1, signal_coef=0.0, seed=201, include_price=True)
        with pytest.raises(
            ValueError,
            match=r"(?s)macro_panel expects a small cross-section.*MacroCommonConfig",
        ):
            fl.preprocess(df, config=MacroPanelConfig(forward_periods=5))

    def test_macro_panel_staggered_schedule_raises(self):
        """Staggered-schedule case the global-only check used to miss:
        N=3 overall but each date only has 1 asset (different asset each
        day). Per-date max n_unique < 3 → raise with staggered message."""
        import numpy as np, polars as pl
        from datetime import datetime, timedelta
        rng = np.random.default_rng(290)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        rows = []
        for d_i, d in enumerate(dates):
            asset = f"a{d_i % 3}"  # Rotate through 3 assets, one per date
            rows.append({
                "date": d, "asset_id": asset,
                "factor": float(rng.standard_normal()),
            })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        with pytest.raises(
            ValueError,
            match=r"(?s)macro_panel.*max per-date n_unique.*staggered",
        ):
            fl.preprocess(df, config=MacroPanelConfig(forward_periods=1))

    def test_event_signal_n1_does_not_raise(self):
        """Event signal explicitly supports single-asset event studies."""
        from datetime import datetime, timedelta
        import numpy as np
        rng = np.random.default_rng(202)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(120)]
        rows = []
        for d in dates:
            rows.append({
                "date": d, "asset_id": "A1",
                "price": 100.0 + float(rng.standard_normal()),
                "factor": float(rng.choice([-1.0, 0.0, 0.0, 0.0, 1.0])),
            })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        prepared = fl.preprocess(df, config=EventConfig(forward_periods=5))
        # Should not raise — single-asset event study is a legitimate use case
        profile = fl.evaluate(prepared, "f", config=EventConfig(forward_periods=5))
        assert profile.factor_name == "f"

    def test_macro_common_n1_does_not_raise(self):
        """macro_common has a dedicated N=1 fallback path."""
        df = _cs_panel(n_dates=60, n_assets=1, signal_coef=0.0, seed=203, include_price=True)
        prepared = fl.preprocess(df, config=MacroCommonConfig(forward_periods=5))
        # Should not raise — N=1 fallback produces a conservative verdict
        # (canonical p → 1.0) but still returns a Profile
        profile = fl.evaluate(prepared, "f", config=MacroCommonConfig(forward_periods=5))
        assert profile.factor_name == "f"
        assert profile.n_assets == 1


class TestCompactedErrorAttribution:
    def test_attr_error_names_evaluate_batch_compact(self):
        from factorlib.evaluation._protocol import _COMPACTED_PREPARED
        with pytest.raises(RuntimeError, match=r"evaluate_batch\(.*compact=True"):
            _COMPACTED_PREPARED.columns

    def test_bool_error_names_evaluate_batch_compact(self):
        from factorlib.evaluation._protocol import _COMPACTED_PREPARED
        with pytest.raises(RuntimeError, match=r"evaluate_batch\(.*compact=True"):
            bool(_COMPACTED_PREPARED)
