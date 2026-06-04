"""MFE/MAE — per-event price path excursion analysis.

Answers: "what does the price path look like after events?"

Requires bar-by-bar ``price`` data within the event window.
If ``price`` is not available, ``compute_mfe_mae`` returns an empty
DataFrame and ``mfe_mae_summary`` returns a short-circuit ``MetricResult``
(``value=NaN``, ``metadata["reason"]``) — never ``None``.

Metrics:
    mfe_mae_summary   — aggregate summary (p50, p75, ratio)

Notes:
    **Pipeline.** Per-event MFE / MAE excursion over a fixed window
    (per-event step), then cross-event quantile / ratio summary;
    descriptive (no formal H₀).
"""

from __future__ import annotations

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    SEMethod,
    TestMethod,
)
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._types import EPSILON, MIN_EVENTS_HARD
from factrix.metrics import metric
from factrix.metrics._helpers import _short_circuit_output
from factrix.metrics._primitives import compute_mfe_mae

__all__ = [
    "mfe_mae_summary",
]

_MFE_CELL = cell(None, FactorDensity.SPARSE, structure=DataStructure.PANEL)

DEFAULT_MIN_ESTIMATION_SAMPLES: int = 20


@metric(
    cell=_MFE_CELL,
    aggregation=Aggregation.EVENT_TIME,
    test_method=TestMethod.DESCRIPTIVE,
    se_method=SEMethod.NONE,
    requires={"mfe_mae_df": compute_mfe_mae},
)
def mfe_mae_summary(mfe_mae_df: pl.DataFrame) -> MetricResult:
    """Aggregate MFE/MAE statistics.

    Reports MFE/MAE ratio as the primary value — higher is better
    (favorable excursion exceeds adverse excursion).

    Args:
        mfe_mae_df: Output of ``compute_mfe_mae()``.

    Returns:
        MetricResult with value=MFE_p50/|MAE_p75| ratio. On insufficient
        data (empty input or fewer than ``MIN_EVENTS_HARD`` rows), returns a
        short-circuit MetricResult (``value=NaN``, ``metadata["reason"]``
        set) so all metrics share a single return contract.

    Notes:
        Headline ``ratio = quantile(mfe, 0.50) / |quantile(mae, 0.75)|``.
        Z-normalised siblings ``mfe_z_p50`` / ``mae_z_p75`` /
        ``mfe_mae_ratio_z`` are reported when ``mfe_z`` / ``mae_z`` are
        present and pass the same minimum-events threshold.

        factrix pairs the MFE median against the MAE 75th percentile
        (not the median) because the asymmetric quantile pair captures
        risk-adjusted favourability: a strategy with median favourable
        excursion that exceeds typical adverse excursion in the worst
        quartile is the practically useful regime.

    Examples:
        Chain from :func:`compute_mfe_mae` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae_summary
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> per_event = compute_mfe_mae(panel, window=20)
        >>> result = mfe_mae_summary(per_event)
        >>> result.name == ""
        True
    """
    if mfe_mae_df.is_empty():
        return _short_circuit_output(
            "mfe_mae_summary",
            "no_price_data",
            mfe_mae_ratio=float("nan"),
            n_events=0,
        )

    mfe = mfe_mae_df["mfe"].drop_nulls().drop_nans()
    mae = mfe_mae_df["mae"].drop_nulls().drop_nans()

    n_events = min(len(mfe), len(mae))
    if n_events < MIN_EVENTS_HARD:
        return _short_circuit_output(
            "mfe_mae_summary",
            "insufficient_events",
            mfe_mae_ratio=float("nan"),
            n_events=n_events,
            min_required=MIN_EVENTS_HARD,
        )

    mfe_p50 = float(mfe.quantile(0.50))  # type: ignore[arg-type]
    mae_p75 = float(mae.quantile(0.75))  # type: ignore[arg-type]

    ratio = mfe_p50 / abs(mae_p75) if abs(mae_p75) > EPSILON else 0.0

    metadata = {
        "mfe_p50": mfe_p50,
        "mae_p75": mae_p75,
        "mfe_mae_ratio": ratio,
        "n_events": n_events,
    }

    # Normalized quantiles (apples-to-apples across horizons / vol regimes).
    if "mfe_z" in mfe_mae_df.columns:
        mfe_z = mfe_mae_df["mfe_z"].drop_nulls().drop_nans()
        mae_z = mfe_mae_df["mae_z"].drop_nulls().drop_nans()
        if len(mfe_z) >= MIN_EVENTS_HARD and len(mae_z) >= MIN_EVENTS_HARD:
            mfe_z_p50 = float(mfe_z.quantile(0.50))  # type: ignore[arg-type]
            mae_z_p75 = float(mae_z.quantile(0.75))  # type: ignore[arg-type]
            metadata["mfe_z_p50"] = mfe_z_p50
            metadata["mae_z_p75"] = mae_z_p75
            metadata["mfe_mae_ratio_z"] = (
                mfe_z_p50 / abs(mae_z_p75) if abs(mae_z_p75) > EPSILON else 0.0
            )
            metadata["n_events_z"] = int(min(len(mfe_z), len(mae_z)))

    return MetricResult(
        p=metadata.get("p_value"),
        value=ratio,
        metadata=metadata,
    )
