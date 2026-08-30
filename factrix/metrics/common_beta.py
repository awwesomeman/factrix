"""Time-series beta metrics for macro common factors.

macro_common factors (VIX, gold, USD index) are a single time series
shared across all assets. Per-asset time-series regression measures
each asset's sensitivity (β) to the common factor.

``compute_common_betas``: per-asset full-sample TS regression → ``{factor: per-asset DataFrame}``.
``common_beta``: cross-sectional test on the β distribution.
``common_beta_r_squared``: average explanatory power across assets.
``compute_rolling_common_beta``: rolling window mean β for stability analysis.

The common factor here is an exogenous, given series — not a
jointly-estimated Barra/APT factor return.

Notes:
    **Pipeline.** Per-asset full-sample ordinary least squares (OLS) β (time-series step), then
    cross-asset t on the β distribution; rolling-window variant slices
    the time axis before the per-asset step.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
)
from factrix._stats.constants import MIN_ASSETS_WARN
from factrix._types import DDOF, EPSILON
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_min_floor,
    _finite_expr,
    _short_circuit_output,
    _surface_drop_stats,
    _warn_below_floor,
)
from factrix.metrics._primitives import compute_common_betas

__all__ = [
    "common_beta",
    "common_beta_profile",
    "common_beta_r_squared",
    "common_beta_sign_consistency",
    "compute_rolling_common_beta",
]

_COMMON_BETA_CELL = cell(
    FactorScope.COMMON,
    FactorDensity.DENSE,
    structure=DataStructure.PANEL,
)


def _calendar_time_se(
    common_betas_df: pl.DataFrame,
    betas: np.ndarray,
    std_b: float,
    metadata: dict[str, object],
) -> tuple[float, int] | None:
    r"""Standard error and df of the mean beta from the calendar-time portfolio.

    Reads ``ew_portfolio_beta_var`` — the Newey-West($h-1$) variance of the
    equal-weight portfolio's slope on the factor, measured by
    :func:`~factrix.metrics._primitives._common_betas.compute_common_betas`
    where the panel exists — and adds the cross-asset dispersion of the *true*
    betas the random-effects null needs:

    $$\mathrm{SE}^2 = V_{\mathrm{EW}} + \hat\tau^2 / N, \qquad
    \hat\tau^2 = \max\bigl(0,\; s^2_\beta - \mathrm{mean}_i\,
    \widehat{\mathrm{Var}}(\hat\beta_i)\bigr)$$

    $V_{\mathrm{EW}}$ is the sampling variance of $\bar{\hat\beta}$ given
    the betas — it carries any cross-asset residual covariance and
    heteroskedasticity without an equicorrelation estimate; $\hat\tau^2$ is
    the excess of the observed cross-sectional variance over the estimation
    noise (per-asset OLS variance, recovered from ``t_stat``), the
    DerSimonian-Laird moment estimator. The reference distribution is $t$
    with the Welch-Satterthwaite df of the two components
    ($T - 2$ behind $V_{\mathrm{EW}}$, $N - 1$ behind $\hat\tau^2 / N$), so
    a dispersion-dominated SE is read against the cross-section and a
    noise-dominated one against the time series.

    Returns ``(se, dof)``, or ``None`` when the frame carries no calendar-time
    estimate (a hand-built beta table), in which case the caller falls back
    to the iid cross-asset $t$ and says so.
    """
    if "ew_portfolio_beta_var" not in common_betas_df.columns:
        metadata["calendar_time_se_applied"] = False
        metadata["calendar_time_se_source"] = "unavailable_hand_built_frame"
        return None
    var_ew = common_betas_df["ew_portfolio_beta_var"][0]
    n_periods = common_betas_df["ew_portfolio_periods"][0]
    if var_ew is None or not math.isfinite(var_ew) or n_periods is None:
        metadata["calendar_time_se_applied"] = False
        metadata["calendar_time_se_source"] = "too_few_shared_periods"
        return None
    n = len(betas)
    # Estimation-noise variance per asset from the OLS t; assets whose t is
    # undefined (exact fit) contribute nothing, and with none at all the whole
    # cross-sectional variance is read as dispersion — the conservative side.
    frame = common_betas_df.filter(
        pl.col("beta").is_not_null()
        & pl.col("beta").is_finite()
        & pl.col("t_stat").is_not_null()
        & pl.col("t_stat").is_finite()
        & (pl.col("t_stat").abs() > EPSILON)
    )
    est_var = (frame["beta"] / frame["t_stat"]) ** 2
    mean_est_var = float(est_var.mean()) if frame.height else 0.0  # type: ignore[arg-type]
    tau2 = max(0.0, std_b * std_b - mean_est_var)
    v_tau = tau2 / n
    var_total = float(var_ew) + v_tau
    metadata["calendar_time_se_applied"] = True
    metadata["ew_portfolio_beta"] = common_betas_df["ew_portfolio_beta"][0]
    metadata["ew_portfolio_beta_se"] = float(np.sqrt(var_ew))
    metadata["ew_portfolio_periods"] = int(n_periods)
    metadata["beta_dispersion_excess"] = tau2
    if var_total <= EPSILON * EPSILON:
        return 0.0, n - 1
    # Welch-Satterthwaite df for the sum of two variance components.
    df_ew = max(int(n_periods) - 2, 1)
    df_tau = max(n - 1, 1)
    denom = float(var_ew) ** 2 / df_ew + v_tau**2 / df_tau
    dof = int(max(1, math.floor(var_total**2 / denom))) if denom > 0 else df_ew
    return float(np.sqrt(var_total)), dof


@metric(
    cell=_COMMON_BETA_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"common_betas_df": compute_common_betas},
    sample_threshold=SampleThreshold(min_assets=3, warn_assets=MIN_ASSETS_WARN),
)
def common_beta(
    common_betas_df: pl.DataFrame,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Test $H_0: \mathrm{mean}(\beta) = 0$ across assets.

    ``value`` is the cross-sectional mean of the per-asset betas; the test
    reads its standard error off the calendar-time portfolio (see Notes).

    Notes:
        Stage 2 of the BJS-style aggregation order:
        $\overline{\beta} = \mathrm{mean}_i \hat\beta_i$ with
        $H_0: \mathbb{E}[\beta] = 0$ across assets. The textbook form is the
        iid cross-asset $t = \overline{\beta} / (\mathrm{std}(\beta) /
        \sqrt{N})$; it is kept as ``metadata["stat_uncorrected"]`` and is the
        statistic reported when the input is a hand-built beta table
        (``calendar_time_se_applied = False``).

        **What factrix tests with, and why.** $\mathrm{std}(\beta)/\sqrt{N}$
        is the SE of a mean over independent draws. Assets loading on a
        common component do not give independent betas, and the
        understatement is unbounded in the correlation: at $N = 8$ the iid
        test rejected 3.0% of true nulls at $\rho = 0$, 44.8% at 0.5 and
        79.2% at 0.9. With one regressor shared by every asset,
        $\overline{\hat\beta}$ is exactly the OLS slope of the equal-weight
        portfolio return on the factor (the calendar-time construction
        ``caar`` uses; [Fama (1998)][fama-1998]), and that regression's
        Newey-West($h - 1$) variance $V_{\mathrm{EW}}$ carries whatever
        cross-asset residual covariance, heteroskedasticity and overlap there
        is, with no equicorrelation estimate. The random-effects null adds
        the dispersion of the true betas, so
        $\mathrm{SE}^2 = V_{\mathrm{EW}} + \hat\tau^2 / N$ with
        $\hat\tau^2 = \max(0, s^2_\beta - \mathrm{mean}_i
        \widehat{\mathrm{Var}}(\hat\beta_i))$ (DerSimonian-Laird), and the
        $t$ is read against a Welch-Satterthwaite df across the two
        components (:func:`_calendar_time_se`).

        An earlier version deflated the iid $t$ by the full Kolari-Pynnönen
        factor $\sqrt{(1 - \bar r)/(1 + (N - 1)\bar r)}$ on the mean
        residual correlation $\bar r$. That factor is exact only when the
        betas are equal, the residuals homoskedastic and equicorrelated, and
        the cross-sectional spread is estimation noise; with true beta
        dispersion (sd 0.5 around a mean of 0.2, $\rho = 0.5$) it rejected
        **0.0%** against 41% uncorrected, and 0.7% on a heteroskedastic null.
        Measured with the calendar-time SE (300 draws, $T = 300$, nominal
        5%): null size 4.0 / 5.0 / 5.3% at $N = 20$ and $\rho = 0 / 0.5 /
        0.9$ (5.7 / 6.7% at $N = 5$, $\rho = 0.5 / 0.9$; 5.0 / 5.3% at
        $N = 100$), 5.3% on the heteroskedastic null, 7.0% with beta sd 1
        around a zero mean at $\rho = 0.5$; power at mean 0.2 / 0.4, sd 0.5,
        $\rho = 0.5$: 28.7% / 80.7% (0.0% under the old factor). Per-asset
        $\widehat{\mathrm{Var}}(\hat\beta_i)$
        is the homoskedastic OLS variance behind ``t_stat``, which at
        $h > 1$ understates the overlap-inflated noise, so $\hat\tau^2$ is
        over-stated and the SE errs conservative there. On an unbalanced
        panel the equal-weight slope (``ew_portfolio_beta``) can differ from
        the mean of the per-asset slopes; both are reported. On a
        COMMON-scope panel null the test measures 2.0-7.3% across
        $T \in \{60, 120, 240\}$ and $h \in \{1, 5\}$
        (``statistical-methods`` section 6).

    References:
        [Black-Jensen-Scholes 1972][black-jensen-scholes-1972]:
        beta-sorted-portfolio time-series CAPM tests. factrix's
        cross-asset t on mean β is a simplified analogue of the BJS
        aggregation order, not a replication of the grouped-portfolio
        intercept test BJS run on assets sorted into beta deciles.

    Examples:
        Chain from :func:`compute_common_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_beta import compute_common_betas, common_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> common_betas_df = compute_common_betas(panel)["factor"]
        >>> result = common_beta(common_betas_df)
        >>> result.name == ""
        True
    """
    # ``drop_nans`` as well as ``drop_nulls``: polars keeps float NaN, and a NaN
    # beta reaching ``_calc_t_stat`` yields a NaN t — which now withholds the
    # test as ``degenerate_variance``, mislabelling missing data as a
    # dispersion-free sample.
    betas = common_betas_df["beta"].drop_nulls().drop_nans().to_numpy()
    n = len(betas)

    sc = _enforce_min_floor(
        common_beta, "common_beta", n, "insufficient_assets", axis="assets"
    )
    if sc is not None:
        return sc

    mean_b = float(np.mean(betas))
    std_b = float(np.std(betas, ddof=DDOF))
    t_iid = _calc_t_stat(mean_b, std_b, n)

    metadata: dict[str, object] = {
        "stat_type": "t",
        "h0": "mean(β)=0",
        "method": "cross-sectional t-test on per-asset TS betas",
        "n_assets": n,
        "beta_std": std_b,
        "median_beta": float(np.median(betas)),
    }
    warning_codes: list[str] = []
    # std(beta)/sqrt(N) is the SE of a mean over INDEPENDENT draws, which
    # assets loading on a common component are not. The calendar-time
    # portfolio SE (see _calendar_time_se) is the test's SE whenever the
    # upstream primitive could measure it; a hand-built beta table falls back
    # to the iid t and says so.
    calendar = _calendar_time_se(common_betas_df, betas, std_b, metadata)
    if calendar is None:
        t = t_iid
        p = _p_value_from_t(t, n)
    else:
        se, dof = calendar
        metadata["stat_uncorrected"] = t_iid
        metadata["dof"] = dof
        metadata["method"] = (
            "cross-sectional t-test on per-asset TS betas with the "
            "calendar-time portfolio SE (Newey-West on the equal-weight "
            "portfolio slope plus beta dispersion / N)"
        )
        t = mean_b / se if se > EPSILON else float("nan")
        p = _p_value_from_t(t, n, dof=dof) if math.isfinite(t) else float("nan")
    # The headline is a cross-asset t-test on E[beta], so its critical value
    # inflates as the cross-section thins — the regime FEW_ASSETS exists for.
    # The estimator does not change; the code is the record that df = n - 1 was
    # small. Same floor and same helper ic / fm_beta use, so a thin panel reads
    # the same across the cross-asset family.
    warn_code = _warn_below_floor(
        common_beta,
        n,
        f"common_beta: n_assets={n} below MIN_ASSETS_WARN={MIN_ASSETS_WARN}; "
        f"the cross-asset t-test on the mean per-asset beta runs on df={n - 1}, "
        f"so its critical value is well above the asymptotic one. mean(beta) is "
        f"returned but read borderline p-values cautiously.",
        WarningCode.FEW_ASSETS,
        axis="assets",
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)
    _surface_drop_stats(
        common_betas_df,
        "common_beta",
        metadata,
        warning_codes,
        axis="assets",
        expected_warnings=expected_warnings,
    )
    # Every asset carrying an identical β leaves no cross-asset dispersion:
    # ``mean_b`` is still the profile, the t is not computable.
    stat, p_out, alternative = _degenerate_test_fields(
        t, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=mean_b,
        n_obs=n,
        n_obs_axis="assets",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Descriptive beta sign / dispersion profile
# ---------------------------------------------------------------------------


def _validate_common_beta_profile(m: MetricBase) -> None:
    """``neutral_epsilon`` is an absolute-beta width, so it cannot be negative."""
    from factrix._errors import UserInputError

    value = m.neutral_epsilon  # type: ignore[attr-defined]
    numeric = not isinstance(value, bool) and isinstance(value, int | float)
    if not numeric or float(value) < 0.0:
        raise UserInputError(
            func_name="common_beta_profile",
            field="neutral_epsilon",
            value=value,
            expected=(
                "a non-negative number. It is the half-width of the |beta| "
                "band counted as neutral, so a negative width has no meaning."
            ),
            docs_path="api/metrics/common_beta",
        )


@metric(
    cell=_COMMON_BETA_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"common_betas_df": compute_common_betas},
    sample_threshold=SampleThreshold(min_assets=1),
    validate=_validate_common_beta_profile,
)
def common_beta_profile(
    common_betas_df: pl.DataFrame,
    *,
    neutral_epsilon: float = EPSILON,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Descriptive sign and dispersion profile of per-asset common-factor betas.

    ``value`` is the positive-minus-negative beta mean spread when both sides
    exist; otherwise it is ``NaN`` and the side counts in metadata explain why.
    No hypothesis test is run.

    Args:
        common_betas_df: Per-asset beta table from
            :func:`compute_common_betas`.
        neutral_epsilon: Absolute-beta threshold treated as neutral. Betas with
            ``abs(beta) <= neutral_epsilon`` are counted as neutral rather than
            positive or negative.

    Returns:
        MetricResult with descriptive beta-profile metadata and
        ``p_value=None``.
    """
    if "beta" not in common_betas_df.columns:
        return _short_circuit_output(
            "common_beta_profile",
            "no_beta_column",
            missing_column="beta",
            descriptive=True,
        )

    betas = common_betas_df["beta"].drop_nulls().drop_nans().to_numpy()
    n = len(betas)
    sc = _enforce_min_floor(
        common_beta_profile,
        "common_beta_profile",
        n,
        "no_asset_beta_observations",
        axis="assets",
        descriptive=True,
    )
    if sc is not None:
        return sc

    positive = betas > neutral_epsilon
    negative = betas < -neutral_epsilon
    neutral = ~(positive | negative)

    pos_betas = betas[positive]
    neg_betas = betas[negative]
    positive_mean = float(np.mean(pos_betas)) if len(pos_betas) else float("nan")
    negative_mean = float(np.mean(neg_betas)) if len(neg_betas) else float("nan")
    spread = (
        positive_mean - negative_mean
        if len(pos_betas) and len(neg_betas)
        else float("nan")
    )
    beta_std = float(np.std(betas, ddof=DDOF)) if n >= 2 else float("nan")

    metadata: dict[str, object] = {
        "n_assets": n,
        "n_positive_beta": int(np.sum(positive)),
        "n_negative_beta": int(np.sum(negative)),
        "n_neutral_beta": int(np.sum(neutral)),
        "positive_beta_mean": positive_mean,
        "negative_beta_mean": negative_mean,
        "abs_beta_mean": float(np.mean(np.abs(betas))),
        "beta_std": beta_std,
        "positive_minus_negative_beta_spread": spread,
        "neutral_epsilon": neutral_epsilon,
        "method": "descriptive per-asset beta sign and dispersion profile",
    }
    if math.isnan(spread):
        metadata["spread_status"] = "requires_positive_and_negative_betas"

    warning_codes: list[str] = []
    _surface_drop_stats(
        common_betas_df,
        "common_beta_profile",
        metadata,
        warning_codes,
        axis="assets",
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=spread,
        p_value=None,
        n_obs=n,
        n_obs_axis="assets",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Mean R²
# ---------------------------------------------------------------------------


@metric(
    cell=_COMMON_BETA_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"common_betas_df": compute_common_betas},
    sample_threshold=SampleThreshold(min_assets=1),
)
def common_beta_r_squared(
    common_betas_df: pl.DataFrame,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Average $R^2$ across per-asset TS regressions — ``value`` $= \mathrm{mean}_i R^2_i$.

    $R^2_i$ comes from asset $i$'s regression
    $R_{i,t} = \alpha_i + \beta_i \cdot F_t + \varepsilon$ (computed
    upstream in ``compute_common_betas``). Metadata carries
    ``median_r_squared`` as well — useful when a few high-$R^2$ assets
    pull the mean. Low values ($< 0.05$) indicate the factor is too
    weak or noisy to drive individual-asset returns even when its
    cross-asset mean $\beta$ looks nonzero.

    Short-circuits to NaN when no assets have a non-null $R^2$.

    Notes:
        ``value`` $= \mathrm{mean}_i R^2_i$ and ``median_r_squared``
        $= \mathrm{median}_i R^2_i$ on the per-asset ordinary least squares (OLS) fits from
        ``compute_common_betas``. Pure descriptive statistic — no formal
        $H_0$.

        factrix reports both mean and median because a few high-$R^2$
        assets can dominate the mean; large mean-vs-median gaps density
        the factor explains a small subset of assets rather than the
        cross-section as a whole.

    Examples:
        Chain from :func:`compute_common_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_beta import compute_common_betas, common_beta_r_squared
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> common_betas_df = compute_common_betas(panel)["factor"]
        >>> result = common_beta_r_squared(common_betas_df)
        >>> result.name == ""
        True
    """
    r2_vals = common_betas_df["r_squared"].drop_nulls().drop_nans().to_numpy()
    n = len(r2_vals)

    sc = _enforce_min_floor(
        common_beta_r_squared,
        "common_beta_r_squared",
        n,
        "no_asset_r_squared_observations",
        axis="assets",
    )
    if sc is not None:
        return sc

    metadata: dict[str, object] = {
        "n_assets": n,
        "median_r_squared": float(np.median(r2_vals)),
        "min_r_squared": float(np.min(r2_vals)),
        "max_r_squared": float(np.max(r2_vals)),
    }
    warning_codes: list[str] = []
    _surface_drop_stats(
        common_betas_df,
        "common_beta_r_squared",
        metadata,
        warning_codes,
        axis="assets",
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=float(np.mean(r2_vals)),
        n_obs=n,
        n_obs_axis="assets",
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Rolling mean beta for stability / OOS analysis
# ---------------------------------------------------------------------------


@metric(
    cell=_COMMON_BETA_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    slice_boundary_sensitive=True,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
    # Pipeline producer: window-specific eligibility is enforced in-body; no
    # static panel-shape floor can pre-flight how many rolling dates survive.
    sample_threshold=SampleThreshold(),
)
def compute_rolling_common_beta(
    data: pl.DataFrame,
    *,
    window: int = 60,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Rolling-window mean β across assets — time-series input for out-of-sample (OOS) / trend.

    Formula (per period t ≥ ``window``):
        For each asset i, take the trailing window of dates
        ``[dates[i - window], dates[i - 1]]`` — the ``window`` dates
        **strictly before** t; date t itself is excluded.
        If ≥ 10 valid (factor, return) pairs, run ordinary least squares (OLS):
            R_{i,s} = α_i + β_i·F_s + ε   (s in window)
        β_t = mean_i β_i   (cross-asset mean of this window's βs)

    Dates with fewer than ``window`` trailing dates are skipped. Assets
    with < 10 valid obs in the window are dropped from that date's β
    calculation. If no asset qualifies at a given date, that date is
    absent from the output entirely.

    Returns:
        DataFrame with ``date, value`` where ``value`` is the rolling
        cross-asset mean β. Shape compatible with ``oos`` / ``ic_trend``.

    Notes:
        **Window convention — lag-1, out-of-sample.** The value stamped on
        output date ``t`` is estimated on the half-open history
        ``[t - window, t - 1]``: it uses only dates strictly *before* t and
        never t's own observation. That makes each ``value_t`` usable as a
        signal *at* t without look-ahead — the series can be joined directly
        onto date t and fed to ``oos`` / ``ic_trend`` (or traded) with no
        further shifting. The alternative and more common convention (pandas
        ``rolling(window)``, which closes the window *at* t and therefore
        includes t) would make ``value_t`` contemporaneous and quietly
        look-ahead-biased for exactly the OOS/trend use this producer exists
        to feed; factrix pays one date of history for that guarantee.

        **Zero-variance regressors are skipped, not solved.** An asset whose
        factor is constant over the window has no identifiable slope;
        ``lstsq`` does not raise there, it returns the minimum-norm solution
        — slope exactly 0.0 — which would enter the cross-asset mean as a
        real "no relationship" estimate and shrink ``value_t`` toward zero.
        Such assets are dropped from that date's mean exactly like the
        < 10-obs assets, so only identifiable slopes are averaged; a date on
        which every asset is degenerate is absent from the output.

        Per date ``t >= window``, run the per-asset TS OLS over the
        trailing ``window`` rows and compute ``value_t = mean_i beta_i``.
        Output schema matches the time-series tools (``oos`` /
        ``ic_trend``), so callers can pipe rolling betas into stability
        and trend diagnostics.

        factrix requires at least 10 valid rows per asset within each
        rolling window; below that, the asset is dropped from that
        date's mean rather than imputed — keeps each ``value_t`` an
        average over identifiable per-asset slopes.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_beta import compute_rolling_common_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> rolling = compute_rolling_common_beta(panel, window=60)
        >>> set(rolling.columns) >= {"date", "value"}
        True
    """
    dates = data["date"].unique().sort()
    if len(dates) < window:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )

    # Partition by asset once into date-sorted numpy arrays, dropping rows with a
    # non-finite factor or return up front: an incomplete pair is unobserved, and
    # leaving it in would feed a NaN into the per-asset OLS and poison that
    # asset's slope (and the cross-asset mean). ``is_not_null`` alone is not
    # enough — polars keeps float NaN. The trailing date window for each
    # ``t`` is the closed interval ``[dates[i-window], dates[i-1]]`` — every
    # asset row whose date lands in it, located by ``searchsorted`` on the
    # asset's sorted dates — which replaces the per-period ``is_in`` filter and the
    # per-asset ``asset_id ==`` filter the loop used to run.
    valid = data.filter(_finite_expr(factor_col) & _finite_expr(return_col))
    asset_arrays: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, a_data in (
        valid.sort("date")
        .partition_by("asset_id", as_dict=True, maintain_order=True)
        .items()
    ):
        asset_arrays[key[0]] = (
            a_data["date"].to_numpy(),
            a_data[factor_col].to_numpy().astype(np.float64),
            a_data[return_col].to_numpy().astype(np.float64),
        )

    date_vals = dates.to_numpy()
    rows: list[dict] = []
    for i in range(window, len(dates)):
        lo = date_vals[i - window]  # first date in the trailing window (inclusive)
        hi = date_vals[i - 1]  # last date in the trailing window (inclusive)

        betas_per_asset: list[float] = []
        for a_dates, x_all, y_all in asset_arrays.values():
            left = int(np.searchsorted(a_dates, lo, side="left"))
            right = int(np.searchsorted(a_dates, hi, side="right"))
            n = right - left
            if n < 10:
                continue
            x = x_all[left:right]
            y = y_all[left:right]
            # Skip zero-variance regressors: lstsq does not raise on a constant
            # column, it returns the minimum-norm solution (slope exactly 0.0),
            # which would enter the cross-asset mean as a genuine "no
            # relationship" estimate and bias value_t toward 0.
            if float(np.std(x)) < EPSILON:
                continue
            X = np.column_stack([np.ones(n), x])
            try:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                betas_per_asset.append(float(b[1]))
            except np.linalg.LinAlgError:
                continue

        if betas_per_asset:
            rows.append(
                {
                    "date": dates[i],
                    "value": float(np.mean(betas_per_asset)),
                }
            )

    if not rows:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# β sign consistency (per-asset version)
# ---------------------------------------------------------------------------


@metric(
    cell=_COMMON_BETA_CELL,
    aggregation=Aggregation.TS_THEN_CS,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"common_betas_df": compute_common_betas},
    sample_threshold=SampleThreshold(min_assets=2),
)
def common_beta_sign_consistency(
    common_betas_df: pl.DataFrame,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Symmetric sign-agreement across per-asset βs — `value = max(pos, 1−pos)` where `pos = mean_i 1{β_i > 0}`.

    Range [0.5, 1.0]: 0.5 = βs evenly split (no directional consensus);
    1.0 = all βs share one sign. Unlike
    ``fm_beta.fm_beta_sign_consistency`` this is **direction-agnostic**
    — it does not require a prior on the factor's expected sign.

    Requires ``n_assets >= 2``: a single β is trivially "100% consistent with
    itself" (the max collapses to 1.0 for any nonzero β), which would
    read as strong evidence on a dashboard but carries zero information.
    Short-circuits to NaN in that case so the degenerate value never
    leaks into downstream inference.

    Notes:
        ``pos = mean_i 1{beta_i > 0}``; ``value = max(pos, 1 - pos)``.
        Direction-agnostic: returns 1 when all assets have positive
        beta or all negative.

        factrix gates this metric at ``n_assets >= 2`` so a single-asset
        ``max(pos, 1-pos) = 1.0`` cannot leak into downstream
        inference as spurious "perfect agreement". Pair with
        ``fm_beta.fm_beta_sign_consistency`` when a directional prior
        is available.

    Examples:
        Chain from :func:`compute_common_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.common_beta import (
        ...     compute_common_betas,
        ...     common_beta_sign_consistency,
        ... )
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> common_betas_df = compute_common_betas(panel)["factor"]
        >>> result = common_beta_sign_consistency(common_betas_df)
        >>> result.name == ""
        True
    """
    betas = common_betas_df["beta"].drop_nulls().drop_nans().to_numpy()
    n = len(betas)
    sc = _enforce_min_floor(
        common_beta_sign_consistency,
        "common_beta_sign_consistency",
        n,
        "insufficient_assets_for_sign_consistency",
        axis="assets",
    )
    if sc is not None:
        return sc

    positive = float(np.mean(betas > 0))
    consistency = max(positive, 1.0 - positive)

    metadata: dict[str, object] = {
        "n_assets": n,
        "fraction_positive": positive,
    }
    warning_codes: list[str] = []
    _surface_drop_stats(
        common_betas_df,
        "common_beta_sign_consistency",
        metadata,
        warning_codes,
        axis="assets",
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=consistency,
        n_obs=n,
        n_obs_axis="assets",
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
