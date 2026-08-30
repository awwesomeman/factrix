"""Top-bucket concentration analysis for cross-sectional panels.

Measures whether top-bucket (long-leg) alpha is concentrated in a few
stocks or broadly distributed, using Herfindahl-Hirschman index (HHI)
inverse.

Notes:
    **Pipeline.** Per-period HHI inverse on top-bucket weights
    (cross-section step) → per-period ratio series, then non-overlapping
    sample. Descriptive: no p-value is reported (see the ``Notes`` on
    ``top_concentration`` for the measurement that withdrew it).

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode, _emit_warning
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._types import (
    DEFAULT_FORWARD_PERIODS,
    EPSILON,
    MIN_PORTFOLIO_PERIODS_HARD,
    MIN_PORTFOLIO_PERIODS_WARN,
    ConcentrationWeight,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _compute_tie_ratio,
    _enforce_scaled_floor,
    _finite_expr,
    _finite_values,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _validate_open_unit_interval,
    _warn_below_scaled_floor,
)

__all__ = [
    "top_concentration",
]


_DOCS_CONCENTRATION = "api/metrics/concentration"


def _validate_top_concentration(m: MetricBase) -> None:
    """``q_top`` is a fraction of the cross-section, strictly inside (0, 1).

    ``weight_by`` is a closed set the constructor reads off its ``Literal``
    annotation.
    """
    _validate_open_unit_interval(
        m.q_top,  # type: ignore[attr-defined]
        func_name="top_concentration",
        field="q_top",
        detail=(
            "0 leaves no top bucket and 1 selects the whole cross-section, "
            "so no top-bucket concentration is defined."
        ),
        docs_path=_DOCS_CONCENTRATION,
    )


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    # Periods floor scales with the non-overlap stride (see ``quantile``): the
    # per-period HHI series is sub-sampled at ``overlap_periods``, so the HARD and
    # WARN floors and their in-body gates share ``MIN_PORTFOLIO_PERIODS_*`` +
    # ``_scaled_min_periods``.
    sample_threshold=_scaled_periods_threshold(
        MIN_PORTFOLIO_PERIODS_HARD, warn=MIN_PORTFOLIO_PERIODS_WARN
    ),
    validate=_validate_top_concentration,
)
def top_concentration(
    data: pl.DataFrame,
    overlap_periods: int = DEFAULT_FORWARD_PERIODS,
    q_top: float = 0.2,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_by: ConcentrationWeight = "abs_factor",
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Top-bucket concentration via Herfindahl-Hirschman index (HHI) inverse.

    Per date, selects top ``q_top`` stocks by factor rank, computes
    HHI of their weights, and returns 1/HHI as the effective number of
    independent bets.

    Args:
        data: Panel with ``date, asset_id, factor`` (and ``forward_return``
            if ``weight_by="alpha_contribution"``).
        q_top: Fraction of top-ranked stocks to include (default 0.2 =
            top 20%). Must lie strictly inside ``(0, 1)``; 0 leaves no top
            bucket and 1 selects the whole cross-section, so neither
            defines a top-bucket concentration and both raise
            ``UserInputError``.
        weight_by: HHI weight convention. Either ``"abs_factor"`` or
            ``"alpha_contribution"``; anything else raises
            ``UserInputError``.
            - ``"abs_factor"`` (default): weight by ``|factor|``. Answers
              "how concentrated is the density itself in the top bucket".
              Conservative, density-level. **Assumes a zero-centred
              factor**: ``|f|`` reads as signal strength only when zero is
              the neutral point. The HHI of ``|f|`` is not
              location-invariant — the same top bucket reads eff_n 9.7
              z-scored, 7.8 shifted by −1, 10.0 shifted by −10 — so on a
              raw factor whose values never change sign the number is an
              artefact of the level. Such a factor raises
              ``WarningCode.ONE_SIGNED_FACTOR``; centre it first
              (cross-sectional z-score) or use ``alpha_contribution``.
            - ``"alpha_contribution"``: weight by the magnitude of each
              name's realised contribution ``|sign(factor) · forward_return|``.
              Captures **risk-concentration**: the top bucket's realised
              return is dominated by a few outliers. Note the absolute
              value — a single big *winner* and a single big *loser*
              both register as concentration, which is the right
              framing for risk but NOT for signed-alpha attribution.
              If you need the latter, apply HHI downstream on the
              signed ``sign(factor) · forward_return`` series yourself.

    Returns:
        MetricResult with value = mean(1/HHI) across dates.
        Higher = more diversified top bucket. **Descriptive**:
        ``p_value``, ``stat`` and ``alternative`` are ``None`` — see Notes.

    Notes:
        **The metric is descriptive; there is no p-value.** Earlier
        versions reported a one-sided ``t`` against
        $H_0: \mathbb{E}[r] \geq 0.5$. That test is withdrawn because the
        0.5 reference ratio sits far inside the null for any realistic
        panel, not near it: on a Gaussian factor with ``abs_factor``
        weights the null diversification ratio is ~0.91, roughly nine
        null standard deviations above the boundary. Measured on
        ``make_cs_panel(ic_target=0.0)``, 50 assets, ``q_top=0.2``,
        ``overlap_periods=5``, seed 20260830 + replication, it rejected
        **0 of 300** draws at a nominal 5% at 10 tested periods and 0 of
        300 at 48 tested periods. A p-value that cannot reject at any
        sample size carries no information, so it is no longer emitted.

        This is a *boundary* choice, not a degrees-of-freedom effect.
        Earlier releases attributed the conservatism to the ``df = 2``
        admitted by ``MIN_PORTFOLIO_PERIODS_HARD = 3``; the measurement
        above shows the rejection rate is 0 at every sample size, so
        short samples were never the cause. Read ``value`` (mean
        effective number of names) and ``ratio_eff_to_total`` against a
        reference *you* choose for your universe.

        Per non-overlap date $t$ with top-bucket members $Q^{\mathrm{top}}(t)$
        (size $n^{\mathrm{top}}$), define weights $w_i$ by ``weight_by``
        and form the Herfindahl
        $\mathrm{HHI}_t = \sum_i (w_i / \sum_j w_j)^2$. Effective
        independent bets $n^{\mathrm{eff}}_t = 1 / \mathrm{HHI}_t$. The
        per-period diversification ratio
        $r_t = n^{\mathrm{eff}}_t / n^{\mathrm{top}}$ is averaged into
        ``metadata["ratio_eff_to_total"]`` and reported as is.

        **Membership is a count, not a percent-rank threshold.** With
        $n_t$ *finite* factor values on date $t$, the bucket is the
        $k_t = \max(1, \lfloor n_t \cdot q_{\mathrm{top}} \rfloor)$
        highest by descending ordinal rank. The alternative — an
        inclusive percent-rank cutoff
        $\mathrm{rank}/n \geq 1 - q_{\mathrm{top}}$ — is off by one
        (n=10, q=0.2 selects 3 names; n=100 selects 21), because the
        boundary rank itself satisfies the inequality. factrix takes the
        strict count so the bucket size matches the requested fraction at
        every $n$. Null and NaN factor values are excluded from $n_t$ and
        can never be selected; counting them (``pl.len()``) would shrink
        every bucket on a partially missing date and empty it outright
        once more than $1 - q_{\mathrm{top}}$ of the names are missing.

        ``tie_policy`` from Config does not apply to that cutoff —
        ordinal ranking is used unconditionally, because HHI measures
        concentration *among* the selected stocks rather than their
        bucketing, and an average-rank cutoff would return a variable
        number of names. ``tie_ratio`` is still recorded in metadata as a
        data-quality diagnostic (high tie_ratio → unstable membership
        across re-rankings).

        **Non-finite weights.** Under ``weight_by="alpha_contribution"`` a
        selected name with no realised return has no weight. Such names
        are removed from the HHI *and* from ``n_top``, so
        $n^{\mathrm{eff}}_t / n^{\mathrm{top}}_t$ compares like with like;
        ``metadata["n_top_members_dropped"]`` records how many
        (date, asset) pairs went that way.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.concentration import top_concentration
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = top_concentration(panel, overlap_periods=5, q_top=0.2)
        >>> result.name == ""
        True
    """
    if weight_by == "alpha_contribution" and return_col not in data.columns:
        return _short_circuit_output(
            "top_concentration",
            "no_return_column",
            descriptive=True,
            missing_column=return_col,
            weight_by=weight_by,
        )

    filtered = _sample_non_overlapping(data, overlap_periods)
    tie_ratio = _compute_tie_ratio(filtered, factor_col)

    # Top-bucket membership: the strict count cutoff ``k = max(1, floor(n·q_top))``
    # on the *finite* per-period cross-section (``n`` counts neither nulls nor
    # NaNs — ``pl.len()`` would, shrinking or emptying the bucket on a partially
    # missing date), selected by descending ordinal rank so exactly ``k`` names
    # are taken. A percent-rank threshold (``rank/n >= 1 - q_top``) is off by one
    # in the inclusive direction: at n=10, q=0.2 it takes 3 names, at n=100 it
    # takes 21.
    finite_factor = _finite_expr(factor_col)
    q1 = (
        filtered.with_columns(
            pl.when(finite_factor).then(pl.col(factor_col)).alias("_f_valid")
        )
        .with_columns(
            # ``count`` (non-null) not ``len`` (all rows): the whole point of
            # ``_f_valid`` is that non-finite factors are nulled out.
            pl.col("_f_valid").count().over("date").alias("_n_valid"),
            pl.col("_f_valid")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("_top_rank"),
        )
        .with_columns(
            ((pl.col("_n_valid") * q_top).floor().cast(pl.Int64))
            .clip(lower_bound=1)
            .alias("_k_top")
        )
        .filter(
            pl.col("_top_rank").is_not_null()
            & (pl.col("_top_rank") <= pl.col("_k_top"))
        )
    )

    if weight_by == "alpha_contribution":
        weighted = q1.with_columns(
            (pl.col(factor_col).sign() * pl.col(return_col)).abs().alias("_raw_weight")
        )
    else:
        weighted = q1.with_columns(pl.col(factor_col).abs().alias("_raw_weight"))

    # A null / NaN weight (``alpha_contribution`` on a name with no realised
    # return) contributes nothing to the HHI numerator but would still be
    # counted by ``pl.len()`` in ``n_top`` — biasing the eff_n / n_top
    # diversification ratio downward. Drop such names from BOTH sides and
    # record how many went.
    n_top_selected = weighted.height
    weighted = weighted.filter(_finite_expr("_raw_weight"))
    n_top_dropped = n_top_selected - weighted.height

    hhi_per_date = (
        weighted.with_columns(
            (pl.col("_raw_weight") / pl.col("_raw_weight").sum().over("date")).alias(
                "_weight"
            )
        )
        .group_by("date")
        .agg(
            (pl.col("_weight") ** 2).sum().alias("hhi"),
            pl.len().alias("n_top"),
        )
        .filter(pl.col("hhi") > EPSILON)
        .with_columns((1.0 / pl.col("hhi")).alias("eff_n"))
        .sort("date")
    )

    # Raw (pre-sampling) date count: the axis the stride-scaled periods floors
    # are calibrated against.
    n_raw_periods = data["date"].n_unique()
    sc = _enforce_scaled_floor(
        "top_concentration",
        n_raw_periods,
        MIN_PORTFOLIO_PERIODS_HARD,
        overlap_periods,
        "insufficient_portfolio_periods",
        descriptive=True,
        tie_ratio=tie_ratio,
    )
    if sc is not None:
        return sc

    warning_codes: list[str] = []
    one_signed_meta: dict[str, int] = {}
    if weight_by == "abs_factor":
        finite_f = _finite_values(data[factor_col])
        n_pos = int((finite_f > 0).sum())
        n_neg = int((finite_f < 0).sum())
        if finite_f.len() and (n_pos == 0 or n_neg == 0):
            one_signed_meta = {
                "n_positive_factor_values": n_pos,
                "n_negative_factor_values": n_neg,
            }
            # Constant message, counts in metadata: Python de-duplicates
            # warnings per (text, category, module, lineno), so interpolating
            # the counts made every call a distinct warning and a by_slice
            # sweep or multi-factor evaluate emitted one per call instead of
            # one per session.
            _emit_warning(
                WarningCode.ONE_SIGNED_FACTOR,
                "weight_by='abs_factor' on a factor that "
                "never changes sign. |factor| is a density weight only when "
                "zero is the neutral point, and the HHI of |f| moves with an "
                "arbitrary level shift. Centre the factor (cross-sectional "
                "z-score) or use weight_by='alpha_contribution'. Counts are "
                "in metadata['n_positive_factor_values'] / "
                "['n_negative_factor_values'].",
                label="top_concentration",
                expected_warnings=expected_warnings,
                warning_codes=warning_codes,
                stacklevel=2,
            )
    warn_code = _warn_below_scaled_floor(
        n_raw_periods,
        MIN_PORTFOLIO_PERIODS_WARN,
        overlap_periods,
        f"{n_raw_periods} raw dates below "
        f"MIN_PORTFOLIO_PERIODS_WARN*overlap_periods="
        f"{MIN_PORTFOLIO_PERIODS_WARN * overlap_periods}; the mean effective "
        f"number of names is averaged over few periods and moves substantially "
        f"between draws. The metric is descriptive (no p-value at any sample "
        f"size), so read value and ratio_eff_to_total as a noisy estimate here "
        f"and lengthen the history before comparing panels.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS,
        label="top_concentration",
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    if hhi_per_date.is_empty():
        # Every date lost its top bucket (all-null factor, or all-null weights
        # under ``alpha_contribution``): there is no ratio series to average.
        return _short_circuit_output(
            "top_concentration",
            "insufficient_top_bucket_periods",
            descriptive=True,
            n_obs=0,
            n_obs_axis="periods",
            tie_ratio=tie_ratio,
            weight_by=weight_by,
            n_top_members_dropped=n_top_dropped,
        )

    eff_n_arr = hhi_per_date["eff_n"].to_numpy()
    n_top_arr = hhi_per_date["n_top"].to_numpy()
    mean_eff_n = float(np.mean(eff_n_arr))
    mean_n_top = float(np.mean(n_top_arr))
    ratio = mean_eff_n / max(mean_n_top, 1)
    n = len(eff_n_arr)

    # No hypothesis test. The one-sided t against ``H₀: ratio >= 0.5`` this
    # metric used to report never rejected — the null ratio is ~0.91 on a
    # Gaussian factor with ``abs_factor`` weights, so 0.5 is ~9 null SDs
    # inside H₀ and the measured rejection rate is 0/300 at both 10 and 48
    # tested periods (docstring Notes). A test that cannot reject at any
    # sample size is not inference, so ``value`` and ``ratio_eff_to_total``
    # are returned descriptively instead.
    metadata: dict = {
        "method": "descriptive: mean 1/HHI on the top bucket",
        "mean_n_top": mean_n_top,
        "ratio_eff_to_total": ratio,
        "tie_ratio": tie_ratio,
        "weight_by": weight_by,
        "q_top": q_top,
        # Top-bucket membership bookkeeping: how many (date, asset) pairs the
        # count cutoff selected, and how many of those were dropped for a
        # non-finite weight (both HHI and n_top exclude them).
        "n_top_members_selected": n_top_selected,
        "n_top_members_dropped": n_top_dropped,
        # Present only when ONE_SIGNED_FACTOR fired; the sign split behind it.
        **one_signed_meta,
    }
    return MetricResult(
        p_value=None,
        alternative=None,
        value=mean_eff_n,
        n_obs=n,
        n_obs_axis="periods",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
