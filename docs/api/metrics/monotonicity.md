---
title: factrix.metrics.monotonicity
---

::: factrix.metrics.monotonicity
    options:
      show_root_members_full_path: true
      members:
        - monotonicity

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Decile-curve monotonicity test__

    ---

    The Patton-Timmermann (2010) MR test on the quantile-bucket return
    curve: $J = \min_i \overline{\Delta}_i$ over the adjacent bucket
    differences, with $H_0$ that the relation is *not* monotone and a
    stationary-bootstrap p. A strictly increasing decile curve is a
    stronger requirement than a positive long-short spread, which only
    needs the two end buckets to separate.

-   __Magnitude vs direction, as descriptive shape__

    ---

    `metadata["mean_abs_spearman"]` (magnitude, $\geq 0$) and
    `metadata["mean_signed"]` (direction consistency) still read
    separately: high magnitude with a near-zero signed mean flags a
    factor that sorts returns but flips direction across dates. They are
    metadata, not the headline — $\mathbb{E}|\rho| > 0$ under $H_0$ by
    Jensen, so mean $|\rho|$ has an `n_groups`-dependent noise floor
    (0.67 / 0.42 / 0.27 at $K = 3 / 5 / 10$) that reads like evidence.

</div>

!!! info "Per-bucket cardinality floor"
    `min_assets_per_group = 50` (Patton-Timmermann 2010) — the slice-
    test function downscales `n_groups` automatically so each bucket meets
    the floor; below it, per-date bucket means are noise-dominated and
    the rank statistic is unreliable. Defaults: `n_groups=10` for
    universes around 2000 stocks; drop to 5 for $n_{assets} < 1000$
    and 3 for $n_{assets} < 200$.

## Worked example — Spearman direction test on quantile-bucket returns

!!! example "monotonicity on a synthetic cross-sectional panel"

    ```python
    import factrix as fx
    from factrix.metrics.monotonicity import monotonicity
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=2000, n_dates=500, ic_target=0.08, rng=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # monotonicity returns dict[str, MetricResult], keyed by factor column
    out = monotonicity(panel, overlap_periods=5, n_groups=10, rng=0)["factor"]
    print(out.value, out.p_value,
          out.metadata["mr_adjacent_diffs"],
          out.metadata["mean_abs_spearman"])
    # value = stat = J, the smallest average adjacent bucket-return
    # difference (return units); p_value is its bootstrap p under
    # H0 "not monotonically increasing"; mr_adjacent_diffs shows which
    # step binds; mean_abs_spearman is the descriptive magnitude.
    ```

!!! warning "Declare the direction, do not search it"
    `direction="increasing"` (default) or `"decreasing"` states what $H_1$
    asserts. Running both and reporting the smaller p is a two-sided search
    charged at a one-sided level. For a factor hypothesised to be negatively
    related to returns, pass `direction="decreasing"` (or flip the factor's
    sign upstream) rather than reading the increasing test and inverting it.

!!! info "Deviation from the paper"
    Patton-Timmermann bootstrap the raw (unstudentised) adjacent
    differences, which is what runs here; their studentised variant is not
    implemented. The bootstrap is the library's shared stationary
    (Politis-Romano 1994) resampler with the Politis-White (2004) automatic
    block length, applied to the whole $(T, K-1)$ difference matrix under
    one row-index draw so within-period cross-bucket dependence is
    preserved. Empirical p uses the Davison-Hinkley $+1$ smoothing shared
    with the rest of the library, so p is never exactly 0.

## See also

<div class="grid cards" markdown>

-   __`quantile_spread` / `compute_group_returns`__

    ---

    The decile-curve chart input and the long-short headline spread
    over the same buckets.

    [api/metrics/quantile →](quantile.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice monotonicity summaries.

    [api/by-slice →](../by-slice.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice inference (Wald $\chi^2$ + Holm / Romano-Wolf adjusted $p$).

    [api/slice-test →](../slice-test.md)

-   __Statistical methods__

    ---

    The MR bootstrap, and the DDOF convention behind the descriptive
    signed-Spearman $t$.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    Per-bucket sample-size floor (Patton-Timmermann 2010) and the
    `n_groups` downscaling contract.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
