---
title: factrix.metrics.concentration
---

::: factrix.metrics.concentration
    options:
      show_root_members_full_path: true
      members:
        - top_concentration

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Signal concentration in the long leg__

    ---

    `weight_by="abs_factor"` (default) — Herfindahl-Hirschman index (HHI) on $|\text{factor}|$
    inside the top-$q$ bucket. Answers "how concentrated is the
    *signal* itself in the long leg". Conservative; depends only on
    factor values, not on realised returns.

-   __Realised risk concentration__

    ---

    `weight_by="alpha_contribution"` — HHI on
    $|\text{sign}(\text{factor}) \cdot \text{forward\_return}|$.
    Captures whether the long-leg's realised return is dominated by a
    few outliers. Absolute value: a single big winner and a single
    big loser both register as concentration (the right framing for
    *risk*, not for signed-alpha attribution).

-   __Diversification test, one-sided__

    ---

    `value = mean(1/HHI)` (effective number of independent bets) and
    `metadata["ratio_eff_to_total"]` is
    $r = n^{\text{eff}} / n^{\text{top}}$. **Descriptive**: `p_value`,
    `stat` and `alternative` are `None` — the former one-sided $t$
    against $H_0: \mathbb{E}[r] \geq 0.5$ is withdrawn because 0.5 sits
    far inside the null (measured null ratio ~0.91, 0 of 300 rejections
    at a nominal 5%). Compare $r$ against a reference you choose for
    your universe.

</div>

## Worked example — top-bucket HHI on a synthetic panel

!!! example "top_concentration with both weighting modes"

    ```python
    import factrix as fx
    from factrix.metrics.concentration import top_concentration
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=500, n_dates=500, ic_target=0.08, rng=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # FactorDensity-level concentration (no return dependence)
    sig = top_concentration(panel, overlap_periods=5, q_top=0.2,
                            weight_by="abs_factor")
    print(sig.value, sig.metadata["ratio_eff_to_total"])
    # 90.1  0.90   (|factor| weights on a Gaussian factor are near-uniform)

    # Realised-return concentration (risk framing)
    risk = top_concentration(panel, overlap_periods=5, q_top=0.2,
                             weight_by="alpha_contribution")
    print(risk.value, risk.metadata["ratio_eff_to_total"])
    # 58.4  0.58   (realised contributions are the outlier-driven axis)

    # Both are descriptive: there is no p-value to read.
    assert sig.p_value is None and sig.stat is None
    ```

## See also

<div class="grid cards" markdown>

-   __`quantile_spread` / `quantile_spread_vw`__

    ---

    The long-short spread on the same top / bottom buckets — pair the
    EW-vs-VW spread gap with concentration to disentangle small-cap
    vs few-name alpha.

    [api/metrics/quantile →](quantile.md)

-   __`notional_turnover` / `breakeven_cost`__

    ---

    Implementation feasibility on the same long-short construction.

    [api/metrics/tradability →](tradability.md)

-   __Statistical methods__

    ---

    One-sided $t$ on the per-date diversification ratio, DDOF
    convention, sample-size guards.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_PORTFOLIO_PERIODS_HARD` / `MIN_PORTFOLIO_PERIODS_WARN`).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
