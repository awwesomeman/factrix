"""Contract tests for ``Artifacts.metric_outputs``.

The parallel channel (alongside ``intermediates``) stashes raw
``MetricOutput`` objects keyed by ``MetricOutput.name``. Consumers —
``describe_profile_values``, user drill-down — assume:

- Canonical keys per factor_type are always present (tied to the metrics
  each Profile's ``from_artifacts`` calls)
- Opt-in keys appear only when the corresponding config field is set
- ``MetricOutput.metadata`` is wrapped read-only (``MappingProxyType``)
  — writes raise ``TypeError``

The tests here pin those invariants so a future refactor of
``from_artifacts`` (new metric / renamed key / dropped metric) cannot
silently break consumers.
"""

from __future__ import annotations

import polars as pl
import pytest

import factorlib as fl

from tests.conftest import _cs_panel


def _panel(n_dates: int = 80, n_assets: int = 30, seed: int = 7) -> pl.DataFrame:
    # Signal strong enough for opt-in metrics to fire; price included so
    # multi_horizon_ic (which recomputes forward returns) can run.
    return _cs_panel(
        n_dates=n_dates, n_assets=n_assets,
        signal_coef=0.3, seed=seed, include_price=True,
    )


# ---------------------------------------------------------------------------
# Canonical keys (CS) — always present
# ---------------------------------------------------------------------------

_CS_CANONICAL_KEYS = frozenset({
    "ic", "ic_ir", "hit_rate", "ic_trend", "monotonicity",
    "quantile_spread", "turnover", "breakeven_cost", "net_spread",
    "top_concentration",
})


class TestCanonicalKeysCS:
    def test_all_canonical_keys_present(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_canonical", return_artifacts=True,
        )
        missing = _CS_CANONICAL_KEYS - arts.metric_outputs.keys()
        assert not missing, (
            f"CS canonical metric keys missing: {sorted(missing)}. "
            f"from_artifacts changed without updating the contract."
        )

    def test_opt_in_keys_absent_by_default(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_default", return_artifacts=True,
        )
        opt_in = {"regime_ic", "multi_horizon_ic", "spanning_alpha"}
        present = opt_in & arts.metric_outputs.keys()
        assert not present, (
            f"opt-in metric keys appeared without config: {sorted(present)}"
        )


# ---------------------------------------------------------------------------
# Opt-in keys (CS) — appear only when config enables them
# ---------------------------------------------------------------------------

class TestOptInKeysCS:
    def test_regime_ic_appears_when_regime_labels_set(self):
        df = _panel(n_dates=80)
        # Build simple regime labels from date parity — enough to exercise
        # the opt-in branch.
        dates = df["date"].unique().sort()
        regime_df = pl.DataFrame({
            "date": dates,
            "regime": ["bull" if i % 2 == 0 else "bear" for i in range(len(dates))],
        })
        cfg = fl.CrossSectionalConfig(regime_labels=regime_df)
        _, arts = fl.evaluate(
            df, "cs_regime",
            config=cfg, return_artifacts=True,
        )
        assert "regime_ic" in arts.metric_outputs
        meta = arts.metric_outputs["regime_ic"].metadata
        assert "per_regime" in meta
        assert set(meta["per_regime"]) == {"bull", "bear"}

    def test_multi_horizon_ic_appears_when_periods_set(self):
        df = _panel(n_dates=120)
        cfg = fl.CrossSectionalConfig(multi_horizon_periods=[1, 5, 10])
        _, arts = fl.evaluate(
            df, "cs_mh",
            config=cfg, return_artifacts=True,
        )
        assert "multi_horizon_ic" in arts.metric_outputs
        assert "per_horizon" in arts.metric_outputs["multi_horizon_ic"].metadata


# ---------------------------------------------------------------------------
# Metadata read-only enforcement
# ---------------------------------------------------------------------------

class TestMetadataIsReadOnly:
    def test_assignment_raises_type_error(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_readonly", return_artifacts=True,
        )
        meta = arts.metric_outputs["ic"].metadata
        with pytest.raises(TypeError):
            meta["new_key"] = 1  # type: ignore[index]

    def test_deletion_raises_type_error(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_readonly_del", return_artifacts=True,
        )
        meta = arts.metric_outputs["ic"].metadata
        with pytest.raises(TypeError):
            del meta["p_value"]  # type: ignore[arg-type]

    def test_reads_still_work(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_readonly_read", return_artifacts=True,
        )
        # Reads must keep working — proxy only blocks writes.
        meta = arts.metric_outputs["ic"].metadata
        assert "p_value" in meta
        assert isinstance(meta["p_value"], float)
        assert list(meta)  # iteration works


# ---------------------------------------------------------------------------
# Same MetricOutput name identifier
# ---------------------------------------------------------------------------

class TestKeysMatchMetricOutputName:
    def test_keys_equal_name(self):
        df = _panel()
        _, arts = fl.evaluate(
            df, "cs_name_match", return_artifacts=True,
        )
        for key, m in arts.metric_outputs.items():
            assert key == m.name, (
                f"Artifacts.metric_outputs['{key}'].name == {m.name!r}; "
                f"keys must equal MetricOutput.name."
            )
