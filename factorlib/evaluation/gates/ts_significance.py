"""Gate: Time-series beta significance for macro_common."""

from __future__ import annotations

from factorlib.evaluation._protocol import Artifacts, GateResult
from factorlib.metrics.ts_beta import ts_beta


def ts_significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """Cross-sectional mean |β| t-stat >= threshold.

    Tests whether assets, on average, have significant exposure
    to the common factor.
    """
    ts_betas_df = artifacts.get("beta_series")
    result = ts_beta(ts_betas_df)
    t = result.stat or 0.0

    return GateResult(
        name="ts_significance",
        status="PASS" if abs(t) >= threshold else "FAILED",
        detail={
            "mean_beta": result.value,
            "tstat": t,
            "threshold": threshold,
            "n_assets": result.metadata.get("n_assets", 0),
        },
    )
