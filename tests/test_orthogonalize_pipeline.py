"""Integration tests for Step 6 orthogonalization wired into evaluate().

Covers the T3.S1 Spike surface:
  - config.ortho=None (default) produces the same Profile as before
  - config.ortho=base_df (DataFrame shortcut) residualizes the factor
    and populates orthogonalize_r2_mean / orthogonalize_n_base
  - config.ortho=OrthoConfig(...) form equivalent to the shortcut
  - coverage below the gate raises ValueError
  - the new diagnose rule fires when R² > 0.7
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib.evaluation.diagnostics import clear_custom_rules


@pytest.fixture(autouse=True)
def _isolate_custom_rules():
    clear_custom_rules()
    yield
    clear_custom_rules()


def _panel_with_base(
    n_dates: int = 80,
    n_assets: int = 40,
    basis_weight: float = 0.0,
    seed: int = 101,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build (factor_panel_with_price, basis_df).

    The factor is ``basis_weight * size + (1 - basis_weight) * noise``,
    so ``basis_weight=0`` gives an orthogonal factor (R² ≈ 0) and
    ``basis_weight=1`` gives one fully explained by size (R² ≈ 1).
    Forward return is lightly correlated with the noise component so
    the factor is not trivially dead after residualization at
    ``basis_weight=0``.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    prices = {f"a{i}": 100.0 for i in range(n_assets)}
    factor_rows: list[dict] = []
    basis_rows: list[dict] = []

    for d in dates:
        size = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = basis_weight * size + (1 - basis_weight) * noise
        # return driven by noise (the residual alpha) + size premium component.
        ret = 0.01 * noise + 0.005 * rng.standard_normal(n_assets)
        for i in range(n_assets):
            aid = f"a{i}"
            prices[aid] *= (1 + ret[i])
            factor_rows.append({
                "date": d, "asset_id": aid,
                "factor": float(factor[i]), "price": float(prices[aid]),
            })
            basis_rows.append({
                "date": d, "asset_id": aid, "size": float(size[i]),
            })

    factor_df = pl.DataFrame(factor_rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    basis_df = pl.DataFrame(basis_rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    # Preprocess the factor panel so evaluate's strict gate passes. All
    # callers here use default forward_periods; ortho runs inside
    # build_artifacts after preprocess, so this ordering matches the
    # production pipeline.
    factor_df = fl.preprocess(factor_df, config=fl.CrossSectionalConfig())
    return factor_df, basis_df


class TestDefaultOff:
    def test_no_ortho_info_populated(self):
        factor_df, _ = _panel_with_base(basis_weight=0.0, seed=42)
        p = fl.evaluate(factor_df, "x", factor_type="cross_sectional")
        assert p.orthogonalize_r2_mean is None
        assert p.orthogonalize_n_base == 0


class TestOrthoInputValidation:
    def test_non_dataframe_non_orthoconfig_raises(self):
        # `ortho` accepts OrthoConfig | pl.DataFrame | None; anything
        # else must raise at construction time, not deep in the pipeline.
        with pytest.raises(TypeError, match="expects OrthoConfig"):
            fl.CrossSectionalConfig(ortho="not_a_frame")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="expects OrthoConfig"):
            fl.CrossSectionalConfig(ortho={"base_factors": 1})  # type: ignore[arg-type]


class TestBasicIntegration:
    def test_residualization_shifts_ic(self):
        # Strong loading on size → IC without residualization reflects
        # both the noise component and the size-premium echo. After
        # residualizing against size, only the noise-driven IC remains.
        factor_df, basis_df = _panel_with_base(basis_weight=0.7, seed=7)

        raw = fl.evaluate(factor_df, "raw", factor_type="cross_sectional")
        # DataFrame shortcut — covered for the common case.
        cfg = fl.CrossSectionalConfig(ortho=basis_df)
        orth = fl.evaluate(factor_df, "orth", config=cfg)

        # r2 populated, n_base=1, mean is a real number
        assert orth.orthogonalize_n_base == 1
        assert orth.orthogonalize_r2_mean is not None
        assert 0.3 < orth.orthogonalize_r2_mean < 0.95
        # IC values differ: raw leans on the size contribution, orth
        # sees only the residual.
        assert raw.ic_mean != pytest.approx(orth.ic_mean, abs=1e-6)

    def test_zero_loading_leaves_ic_close(self):
        # basis_weight=0 → factor is pure noise wrt size.
        # Residualization subtracts ~nothing systematic; IC should be
        # in the same ballpark, and R² should be low.
        factor_df, basis_df = _panel_with_base(basis_weight=0.0, seed=123)

        raw = fl.evaluate(factor_df, "raw", factor_type="cross_sectional")
        cfg = fl.CrossSectionalConfig(ortho=basis_df)
        orth = fl.evaluate(factor_df, "orth", config=cfg)

        assert orth.orthogonalize_r2_mean < 0.2
        assert orth.ic_mean == pytest.approx(raw.ic_mean, abs=0.03)

    def test_shortcut_and_explicit_forms_equivalent(self):
        # CrossSectionalConfig(ortho=df) and
        # CrossSectionalConfig(ortho=OrthoConfig(base_factors=df)) must
        # produce identical Profiles — the shortcut is pure sugar. Use
        # asdict equality so any future Profile field is covered without
        # updating this test.
        import dataclasses

        factor_df, basis_df = _panel_with_base(basis_weight=0.5, seed=42)

        shortcut = fl.evaluate(
            factor_df, "x",
            config=fl.CrossSectionalConfig(ortho=basis_df),
        )
        explicit = fl.evaluate(
            factor_df, "x",
            config=fl.CrossSectionalConfig(
                ortho=fl.OrthoConfig(base_factors=basis_df),
            ),
        )
        assert dataclasses.asdict(shortcut) == dataclasses.asdict(explicit)


class TestCoverageGate:
    def test_partial_coverage_raises_below_threshold(self):
        factor_df, basis_df = _panel_with_base(basis_weight=0.5, seed=9)
        # Knock out 20% of basis rows — coverage drops below default 0.95
        small_basis = basis_df.head(int(basis_df.height * 0.80))

        cfg = fl.CrossSectionalConfig(ortho=small_basis)
        with pytest.raises(ValueError, match="coverage"):
            fl.evaluate(factor_df, "x", config=cfg)

    def test_explicit_low_threshold_accepts_partial(self):
        factor_df, basis_df = _panel_with_base(basis_weight=0.5, seed=9)
        small_basis = basis_df.head(int(basis_df.height * 0.80))

        cfg = fl.CrossSectionalConfig(
            ortho=fl.OrthoConfig(base_factors=small_basis, min_coverage=0.5),
        )
        p = fl.evaluate(factor_df, "x", config=cfg)
        assert p.orthogonalize_n_base == 1


class TestDiagnoseRule:
    def test_absorbed_most_fires_when_r2_high(self):
        # basis_weight=0.95 → R² very high
        factor_df, basis_df = _panel_with_base(basis_weight=0.95, seed=11)
        cfg = fl.CrossSectionalConfig(ortho=basis_df)
        p = fl.evaluate(factor_df, "x", config=cfg)

        codes = [d.code for d in p.diagnose()]
        assert "cs.orthogonalize_absorbed_most" in codes

    def test_absorbed_most_silent_when_not_applied(self):
        factor_df, _ = _panel_with_base(basis_weight=0.0, seed=13)
        p = fl.evaluate(factor_df, "x", factor_type="cross_sectional")
        codes = [d.code for d in p.diagnose()]
        assert "cs.orthogonalize_absorbed_most" not in codes
