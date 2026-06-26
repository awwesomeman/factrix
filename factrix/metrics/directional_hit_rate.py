"""Directional hit rate — small-N robust sibling of ``hit_rate``.

Notes:
    **Pipeline.** Pool sign agreement between predicted (``factor``) and
    realised (``forward_return``) direction over a non-overlapping
    subsample; Pesaran-Timmermann (1992) market-timing test against
    ``H₀: predicted and realised direction are independent``.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    The small-N robust counterpart of
    :func:`~factrix.metrics.hit_rate.hit_rate` (and a directional sibling
    of :func:`~factrix.metrics.event_quality.event_hit_rate`). ``hit_rate``
    runs a naive two-sided binomial against ``p = 0.5`` on a single
    pre-aggregated per-date series, implicitly assuming each call is an
    independent Bernoulli draw with a fixed 0.5 success rate. The
    Pesaran-Timmermann test instead conditions on the *marginal* up/down
    frequencies of both the prediction and the realisation, so a factor
    that is simply long a persistently-rising market is not credited with
    skill. This makes it the appropriate directional test for small,
    sign-imbalanced samples (the N < 30 allocation regime). The headline
    ``value`` is itself a hit rate — the fraction of correctly-signed
    calls — hence the shared ``hit_rate`` name.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    FactorDensity,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import MIN_DIRECTIONAL_PAIRS_HARD, MIN_DIRECTIONAL_PAIRS_WARN
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _enforce_min_floor,
    _estimate_within_date_icc,
    _kp_cluster_scale,
    _sample_non_overlapping,
    _short_circuit_output,
    _warn_below_floor,
)

__all__ = [
    "directional_hit_rate",
]


@metric(
    cell=cell(None, FactorDensity.DENSE, structure=None),
    aggregation=Aggregation.TS_ONLY,
    input_shape=InputShape.PANEL,
    sample_threshold=SampleThreshold(
        min_pairs=MIN_DIRECTIONAL_PAIRS_HARD,
        warn_pairs=MIN_DIRECTIONAL_PAIRS_WARN,
    ),
)
def directional_hit_rate(
    df: pl.DataFrame,
    forward_periods: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Directional hit rate via the Pesaran-Timmermann (1992) test.

    Proportion of observations whose predicted direction
    ``sign(factor)`` matches the realised direction
    ``sign(forward_return)``, tested for predictive ability against the
    PT independence null.

    Args:
        df: Panel with ``date, asset_id``, ``factor_col`` and
            ``return_col``. Pooled across all ``(date, asset)``
            observations on the non-overlapping subsample.
        forward_periods: Sampling stride for non-overlapping dates;
            match the forward-return horizon so overlapping windows do
            not inflate the test.
        factor_col: Prediction column (default ``"factor"``).
        return_col: Realisation column (default ``"forward_return"``).

    Returns:
        MetricResult with value = directional hit rate ``P̂`` (0.0-1.0),
        ``stat`` = the PT statistic ``S_n`` (deflated for within-date
        cross-sectional correlation on a panel — see Notes), one-sided
        p-value. ``metadata["kolari_pynnonen_applied"]`` records whether
        the deflation fired and ``stat_uncorrected`` carries the raw
        ``S_n`` when it did.

    Notes:
        On the non-overlapping subsample, drop observations where either
        sign is zero (direction undefined), leaving $n$ pooled pairs.
        With $\hat P_x = \Pr(\text{factor} > 0)$,
        $\hat P_y = \Pr(\text{return} > 0)$ and the realised hit rate
        $\hat P = \frac1n \sum_t \mathbb{1}[\operatorname{sign}(x_t) =
        \operatorname{sign}(y_t)]$, the rate expected under independence is

        $$P_* = \hat P_x \hat P_y + (1 - \hat P_x)(1 - \hat P_y).$$

        The Pesaran-Timmermann statistic

        $$S_n = \frac{\hat P - P_*}{\sqrt{\widehat{\operatorname{var}}(\hat P)
        - \widehat{\operatorname{var}}(P_*)}}$$

        is asymptotically $N(0, 1)$ under $H_0$, with
        $\widehat{\operatorname{var}}(\hat P) = P_*(1 - P_*)/n$ and
        $\widehat{\operatorname{var}}(P_*) = \frac1n (2\hat P_y - 1)^2
        \hat P_x (1 - \hat P_x) + \frac1n (2\hat P_x - 1)^2 \hat P_y
        (1 - \hat P_y) + \frac{4}{n^2} \hat P_x \hat P_y (1 - \hat P_x)
        (1 - \hat P_y)$.

        **Cross-sectional correlation.** Pooling every ``(date, asset)``
        trial as independent over-states the effective sample on a panel —
        same-date returns share shocks, so the trials are correlated and
        $\widehat{\operatorname{var}}(\hat P)$ understates the true
        variance. $S_n$ is therefore deflated by the Kolari-Pynnönen (2010)
        factor $\sqrt{(1 - \hat r)/(1 + (\bar m - 1)\hat r)}$, where
        $\hat r$ is the within-date intraclass correlation of the sign-hit
        indicator and $\bar m$ the mean assets-per-date — the same
        correction :func:`~factrix.metrics.caar.bmp_z` applies to clustered
        SARs. A single-asset series has one trial per date, so $\hat r$ is
        undefined and $S_n$ is left exact (the canonical PT setting).

        The test is **one-sided**: a large positive $S_n$ signals genuine
        directional skill, so $p = \Pr(Z > S_n)$. A factor expected to be
        *negatively* related to forward returns scores poorly here — flip
        its sign before testing. Degenerate samples (all predictions or
        all realisations one-signed, or a non-positive variance estimate)
        short-circuit: $P_*$ is then 1 and the statistic is undefined.

        The sample floor is on the **pairs** axis — the $n$ pooled
        ``(date, asset)`` directional trials, not the period count. Below
        ``MIN_DIRECTIONAL_PAIRS_HARD`` the metric short-circuits; between
        the HARD and ``MIN_DIRECTIONAL_PAIRS_WARN`` floors it returns the
        hit rate but flags ``WarningCode.FEW_DIRECTIONAL_PAIRS`` because the
        normal approximation to $S_n$ is power-thin on few pooled trials.

    References:
        Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric
        test of predictive performance. *Journal of Business & Economic
        Statistics*, 10(4), 461-465.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.directional_hit_rate import directional_hit_rate
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = directional_hit_rate(panel, forward_periods=5)
        >>> result.name == ""
        True
    """
    if return_col not in df.columns:
        return _short_circuit_output(
            "directional_hit_rate",
            "no_return_column",
            missing_column=return_col,
        )

    sampled = _sample_non_overlapping(df, forward_periods)
    paired = sampled.select(
        pl.col("date"),
        pl.col(factor_col).sign().alias("_x_sign"),
        pl.col(return_col).sign().alias("_y_sign"),
    ).filter(
        pl.col("_x_sign").is_not_null()
        & pl.col("_y_sign").is_not_null()
        & (pl.col("_x_sign") != 0)
        & (pl.col("_y_sign") != 0)
    )

    n = paired.height
    # Pairs floor is NOT stride-scaled (unlike the periods-axis samplers that
    # route through _enforce_scaled_floor): n here is the PT test's effective
    # sample size and feeds var_s directly, and sampled_pairs cannot be derived
    # from a param-only resolver — Σ assets-per-sampled-date drifts unboundedly
    # from n_pairs/forward_periods on unbalanced panels. The full-panel
    # pre-flight stays deliberately loose; this post-sampling count is the
    # authoritative gate.
    sc = _enforce_min_floor(
        directional_hit_rate,
        "directional_hit_rate",
        n,
        "insufficient_directional_samples",
        axis="pairs",
    )
    if sc is not None:
        return sc

    x_up = paired["_x_sign"].to_numpy() > 0
    y_up = paired["_y_sign"].to_numpy() > 0

    p_correct = float(np.mean(x_up == y_up))
    p_x = float(np.mean(x_up))
    p_y = float(np.mean(y_up))
    p_star = p_x * p_y + (1.0 - p_x) * (1.0 - p_y)

    var_p_hat = p_star * (1.0 - p_star) / n
    var_p_star = (
        (2.0 * p_y - 1.0) ** 2 * p_x * (1.0 - p_x) / n
        + (2.0 * p_x - 1.0) ** 2 * p_y * (1.0 - p_y) / n
        + 4.0 * p_x * p_y * (1.0 - p_x) * (1.0 - p_y) / n**2
    )
    var_s = var_p_hat - var_p_star

    # Degenerate: one-signed predictions or realisations collapse P* to 1
    # and drive var_s to (numerically) zero or below — S_n is undefined.
    if var_s <= 0.0:
        return _short_circuit_output(
            "directional_hit_rate",
            "degenerate_directional_variance",
            n_obs=n,
            n_obs_axis="pairs",
            p_correct=p_correct,
            p_up_pred=p_x,
            p_up_real=p_y,
        )

    s_n = (p_correct - p_star) / np.sqrt(var_s)

    # Cross-sectional correlation: pooling every (date, asset) trial as
    # independent over-states the effective sample because same-date returns
    # share shocks. Deflate S_n by the Kolari-Pynnönen (2010) factor built from
    # the within-date ICC of the sign-hit indicator — the same correction
    # bmp_z applies to clustered SARs. A single-asset series has no within-date
    # cross-section (one trial per date), so r̂ is undefined and S_n is left
    # exact (the canonical PT setting).
    hit_by_date = paired.select(
        pl.col("date"),
        (pl.col("_x_sign") == pl.col("_y_sign")).cast(pl.Float64).alias("_hit"),
    )
    r_hat, n_eff, _ = _estimate_within_date_icc(hit_by_date, "_hit")
    if r_hat is not None and n_eff > 1.0:
        s_stat = s_n * _kp_cluster_scale(r_hat, n_eff)
        kolari_pynnonen_applied = True
    else:
        s_stat = s_n
        kolari_pynnonen_applied = False
    p = float(sp_stats.norm.sf(s_stat))

    warning_codes: list[str] = []
    warn_code = _warn_below_floor(
        directional_hit_rate,
        n,
        f"directional_hit_rate: n_pairs={n} below "
        f"MIN_DIRECTIONAL_PAIRS_WARN={MIN_DIRECTIONAL_PAIRS_WARN}; the "
        f"Pesaran-Timmermann hit rate is returned but n counts pooled "
        f"non-overlapping (date, asset) directional trials, and the normal "
        f"approximation to S_n is power-thin below ~30 pooled pairs. Read "
        f"borderline p-values cautiously.",
        WarningCode.FEW_DIRECTIONAL_PAIRS,
        axis="pairs",
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    metadata: dict = {
        "stat_type": "z",
        "h0": "independent_direction",
        "method": "Pesaran-Timmermann (1992)",
        "p_correct": p_correct,
        "p_expected": p_star,
        "p_up_pred": p_x,
        "p_up_real": p_y,
        "kolari_pynnonen_r": r_hat,
        "kolari_pynnonen_n_eff": n_eff,
        "kolari_pynnonen_applied": kolari_pynnonen_applied,
    }
    if kolari_pynnonen_applied:
        metadata["stat_uncorrected"] = float(s_n)
        metadata["method"] = (
            "Pesaran-Timmermann (1992) + Kolari-Pynnönen (2010) "
            "cross-sectional-correlation adjustment"
        )

    return MetricResult(
        value=p_correct,
        p_value=p,
        n_obs=n,
        n_obs_axis="pairs",
        stat=float(s_stat),
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
