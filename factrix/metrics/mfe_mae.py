"""MFE/MAE — per-event price path excursion analysis.

Answers: "what does the price path look like after events?"

Requires bar-by-bar ``price`` data within the event window.
If ``price`` is not available, ``compute_mfe_mae`` returns an empty
DataFrame and ``mfe_mae`` returns a short-circuit ``MetricResult``
(``value=NaN``, ``metadata["reason"]``) — never ``None``.

Metrics:
    mfe_mae           — aggregate summary (MFE p50, MAE p25, ratio)

Notes:
    **Pipeline.** Per-event MFE / MAE excursion over a fixed window
    (per-event step), then cross-event quantile / ratio summary;
    descriptive (no formal H₀).
"""

from __future__ import annotations

import polars as pl

from factrix._axis import (
    Aggregation,
    FactorDensity,
    InputShape,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import EPSILON, MIN_EVENTS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _enforce_min_floor, _short_circuit_output
from factrix.metrics._primitives import compute_mfe_mae

__all__ = [
    "mfe_mae",
]

# structure=None (event-axis): MFE/MAE excursions aggregate over events, so a
# single name with enough events is valid. Density stays SPARSE; the event floor
# guards thin samples.
_MFE_CELL = cell(None, FactorDensity.SPARSE, structure=None)


def _excursion_ratio(mfe: float, mae: float) -> tuple[float, str]:
    """MFE / |MAE| with the no-adverse-excursion case reported, not scored 0.

    A factor whose events never traded against entry is the **best** possible
    outcome, and not rare on short windows for gap-driven event factors. The
    ratio there is unbounded, and returning ``0.0`` — the worst possible score,
    indistinguishable from "MFE is zero, this factor is worthless" — inverted
    the ranking at the top of the distribution. ``profit_factor`` already
    faced the analogous no-losses case and reports ``inf`` with a status flag;
    this mirrors it exactly, including the both-zero case, which is undefined
    rather than either extreme.

    Returns ``(ratio, status)`` where status is ``"finite"``,
    ``"unbounded_no_adverse_excursion"`` or
    ``"undefined_no_excursion"``.
    """
    no_adverse = abs(mae) <= EPSILON
    no_favourable = abs(mfe) <= EPSILON
    if no_adverse and no_favourable:
        return float("nan"), "undefined_no_excursion"
    if no_adverse:
        return float("inf"), "unbounded_no_adverse_excursion"
    return mfe / abs(mae), "finite"


@metric(
    cell=_MFE_CELL,
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"mfe_mae_df": compute_mfe_mae},
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def mfe_mae(mfe_mae_df: pl.DataFrame) -> MetricResult:
    """Aggregate MFE/MAE statistics.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the summary on the per-event MFE/MAE count. Pre-flight reads the raw non-zero factor count as a loose upper bound.

    Reports MFE/MAE ratio as the primary value — higher is better
    (favorable excursion exceeds adverse excursion).

    Args:
        mfe_mae_df: Output of ``compute_mfe_mae()``.

    Returns:
        MetricResult with value=MFE_p50/|MAE_p25| ratio. On insufficient
        data (empty input or fewer than ``MIN_EVENTS_HARD`` rows), returns a
        short-circuit MetricResult (``value=NaN``, ``metadata["reason"]``
        set) so all metrics share a single return contract.

    Notes:
        Headline ``ratio = MFE_p50 / |MAE_p25|`` =
        ``quantile(mfe, 0.50) / |quantile(mae, 0.25)|``, reported with
        ``mfe_mae_ratio_status``: ``inf`` /
        ``"unbounded_no_adverse_excursion"`` when the events never traded
        against entry, ``NaN`` / ``"undefined_no_excursion"`` when neither
        excursion exists (see :func:`_excursion_ratio` — the best possible
        outcome must not share a score with the worst).
        Z-normalised siblings ``mfe_z_p50`` / ``mae_z_p25`` /
        ``mfe_mae_ratio_z`` are reported when ``mfe_z`` / ``mae_z`` are
        present and pass the same minimum-events threshold.

        factrix pairs the MFE median against the **worst adverse
        quartile** (not the median) because the asymmetric quantile pair
        captures risk-adjusted favourability: a strategy whose median
        favourable excursion exceeds the adverse excursion of its worst
        quartile of events is the practically useful regime.

        MAE is stored as a *signed non-positive* excursion, so the worst
        quartile is the **25th** percentile — the most negative tail — and
        ``|MAE_p25| >= |MAE_p50|``. Earlier versions read
        ``quantile(mae, 0.75)``, which on a non-positive series is the
        *mildest* adverse quartile (nearest zero); pairing it with the
        stated "worst quartile" intent both mislabelled the statistic and
        inflated ``ratio`` by dividing by the smallest adverse magnitude
        instead of the largest. The alternative convention — storing MAE
        as a positive magnitude, where p75 would be correct — was not
        adopted: the signed form keeps ``mfe``/``mae`` on the same signed
        return axis as every other event primitive and lets
        ``mfe + |mae|`` read as total excursion range.

        ``compute_mfe_mae`` floors the excursions at the entry price
        (``mfe >= 0``, ``mae <= 0``, the Sweeney/Tharp definition), so the
        sign of ``mae`` is guaranteed here rather than assumed. Entry is
        the event bar's own close — see ``compute_mfe_mae`` for why this
        primitive enters one bar earlier than the return-profile ones.

    Examples:
        Chain from :func:`compute_mfe_mae` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> per_event = compute_mfe_mae(panel, window=20)
        >>> result = mfe_mae(per_event)
        >>> result.name == ""
        True
    """
    if mfe_mae_df.is_empty():
        return _short_circuit_output(
            "mfe_mae",
            "no_price_data",
            mfe_mae_ratio=float("nan"),
            n_events=0,
        )

    mfe = mfe_mae_df["mfe"].drop_nulls().drop_nans()
    mae = mfe_mae_df["mae"].drop_nulls().drop_nans()

    n_events = min(len(mfe), len(mae))
    sc = _enforce_min_floor(
        mfe_mae,
        "mfe_mae",
        n_events,
        "insufficient_events",
        axis="events",
        mfe_mae_ratio=float("nan"),
        n_events=n_events,
    )
    if sc is not None:
        return sc

    mfe_p50 = float(mfe.quantile(0.50))  # type: ignore[arg-type]
    # MAE is a signed non-positive excursion, so the *worst* adverse quartile
    # is the 25th percentile (most negative), not the 75th.
    mae_p25 = float(mae.quantile(0.25))  # type: ignore[arg-type]

    ratio, status = _excursion_ratio(mfe_p50, mae_p25)

    metadata: dict[str, object] = {
        "mfe_p50": mfe_p50,
        "mae_p25": mae_p25,
        "mfe_mae_ratio": ratio,
        "mfe_mae_ratio_status": status,
        "n_events": n_events,
    }

    # Normalized quantiles (apples-to-apples across horizons / vol regimes).
    if "mfe_z" in mfe_mae_df.columns:
        mfe_z = mfe_mae_df["mfe_z"].drop_nulls().drop_nans()
        mae_z = mfe_mae_df["mae_z"].drop_nulls().drop_nans()
        if len(mfe_z) >= MIN_EVENTS_HARD and len(mae_z) >= MIN_EVENTS_HARD:
            mfe_z_p50 = float(mfe_z.quantile(0.50))  # type: ignore[arg-type]
            mae_z_p25 = float(mae_z.quantile(0.25))  # type: ignore[arg-type]
            metadata["mfe_z_p50"] = mfe_z_p50
            metadata["mae_z_p25"] = mae_z_p25
            ratio_z, status_z = _excursion_ratio(mfe_z_p50, mae_z_p25)
            metadata["mfe_mae_ratio_z"] = ratio_z
            metadata["mfe_mae_ratio_z_status"] = status_z
            metadata["n_events_z"] = int(min(len(mfe_z), len(mae_z)))

    return MetricResult(
        p_value=None,  # descriptive metric — no hypothesis test
        value=ratio,
        n_obs=n_events,
        n_obs_axis="events",
        metadata=metadata,
    )
