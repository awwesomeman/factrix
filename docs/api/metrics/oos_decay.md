---
title: factrix.metrics.oos_decay
---

::: factrix.metrics.oos_decay
    options:
      show_root_members_full_path: true
      members:
        - oos_decay
        - oos_decay_splits

<hr>

!!! info "Descriptive only — no formal $H_0$"
    `oos_decay` emits a survival ratio + sign-flip detail;
    no `p_value` is attached and `stat` is `None`. A $t$-test at the
    `MIN_OOS_PERIODS_HARD` floor would have power $\approx 0$ and would
    invite mis-reading the diagnostic as a significance test. Callers
    routing this output into Benjamini-Hochberg-Yekutieli (BHY) / gate logic must read `status`
    (`"PASS"` / `"VETOED"`) and `sign_flipped`, not a probability.

## Use cases

<div class="grid cards" markdown>

-   __Persistence read on a factor-return series__

    ---

    `oos_decay` is a standalone series diagnostic — input is a 1-D
    `(date, value)` series, typically information coefficient (IC)
    from `compute_ic`, spread from `compute_spread_series`, or any
    other factor-mimicking-portfolio return series. Reports
    $|\mathrm{mean}_{\text{OOS}}| / |\mathrm{mean}_{\text{IS}}|$ on a
    single `is_ratio` split.

-   __One row per period, finite values only__

    ---

    The split is a count of periods on the series' own distinct-date
    grid, taken positionally after a sort by date. A duplicated date has
    no defined position in that sort, so it could put the same
    chronological period on both sides of the boundary; it raises
    `UserInputError` rather than being aggregated under a rule the
    caller never chose. Null / NaN / $\pm\infty$ observations are
    dropped first and recorded in `metadata`, so `n_obs` and the split
    index count the same periods.

-   __`survival_threshold` is a retention fraction__

    ---

    Its domain is the finite half-open interval $(0, 1]$, validated at
    construction. Outside it the knob stops gating and starts forcing:
    $\le 0$ passes every series a ratio exists for, `float("nan")`
    vetoes every one of them, and `True` reads as `1.0`. A threshold
    above 1 would demand out-of-sample *amplification* rather than
    survival — read `value` directly for that question.

-   __Sign-flip veto__

    ---

    Any split with opposite-signed IS and out-of-sample (OOS) means flips
    `sign_flipped = True` and forces `status = "VETOED"` — IC
    sign-flip OOS means the factor predicts the wrong direction, not
    just a weaker one. McLean & Pontiff (2016) report average OOS
    decay around 32 %; factrix's default `survival_threshold = 0.5`
    sits inside that window.

-   __Sweep the split fraction with `oos_decay_splits`__

    ---

    One `oos_decay` call is one `is_ratio`, and a regime change landing
    near that cut point can reverse the gate.
    `oos_decay_splits` runs the same primitive over a set of fractions
    the caller declares up front (default `(0.6, 0.7, 0.8)`), purges the
    overlapping in-sample tail, and reports one aggregate verdict plus
    every individual split.

</div>

## Robustness sweep — `oos_decay_splits`

!!! warning "Robustness validation, not another model search"
    The split set is declared **before** the sweep runs, every split is
    reported, the aggregate rule is fixed, and no $p$-value is emitted
    anywhere — so there is nothing to correct for and no best split to
    select. Re-running with different fraction sets until one passes turns
    it back into an uncorrected search. The pre-declaration is the only
    thing preventing that, and the library cannot enforce it for you.

**Aggregate rule.** `status = "PASS"` requires all three of:

1. every declared split assessable (a split that could not be computed
   withholds the aggregate: `value` is NaN, `reason = "unassessable_splits"`
   — "cannot assess" must not read as "passed");
2. **no** split sign-flipped;
3. the **median** survival ratio at or above `survival_threshold`.

Direction and magnitude are aggregated differently on purpose. A sign flip
says the factor predicts the wrong way over some contiguous tail, which the
single-split primitive already treats as a hard veto — a majority vote over
splits would launder it. A magnitude below the bar at one cut point is
exactly the arbitrariness the median is there to absorb (50 % breakdown
point, and order-invariant, so the declaration order cannot move the
verdict). The median is fixed rather than exposed as a knob: choosing an
aggregate after seeing the per-split ratios is the search this closes off.
When the gate vetoes on a sign flip, `value` still carries the median that
ran — the veto lives in `status` and `n_sign_flips`, not in a corrupted
number.

**Purge gap.** A value stamped at period $t$ built from a
`forward_periods`-period forward return is realised over
$(t,\, t + \texttt{forward\_periods}]$, so the last in-sample observations
are partly realised inside the out-of-sample window. `oos_decay_splits`
drops `forward_periods` periods off the end of each in-sample window —
counted on the panel's distinct-date grid, never calendar time — which is
the purge of Lopez de Prado (2018). The gap comes off the **in-sample** side
only: the out-of-sample window is the thing being validated and is never
shortened to protect the window it is validated against.

```python title="Illustrative"
from factrix.metrics import oos_decay_splits

out = oos_decay_splits(
    ic_df, value_col="ic", split_fractions=(0.6, 0.7, 0.8), forward_periods=5,
)
print(out.value, out.metadata["status"])           # median ratio, aggregate gate
for split in out.metadata["splits"]:               # provenance, ascending
    print(split["split_fraction"], split["n_is"], split["n_oos"],
          split["survival"], split["sign_flipped"], split["status"])
```

## Choosing a function

| Goal                                                                          | Function                |
|-------------------------------------------------------------------------------|-------------------------|
| Single-split OOS survival + sign-flip gate on a `(date, value)` series        | `oos_decay` |
| Purged survival across a pre-declared set of splits, with provenance          | `oos_decay_splits` |

## Worked example — IC series fed into oos_decay

!!! example "compute_ic → oos_decay"

    ```python
    import factrix as fx
    from factrix.metrics.ic import compute_ic
    from factrix.metrics.oos_decay import oos_decay
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=1000, ic_target=0.08, rng=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # The series diagnostic consumes (date, value); the value column on
    # the compute_ic output is named ``ic``.
    ic_df = compute_ic(panel)["factor"]
    out   = oos_decay(ic_df, value_col="ic")
    print(out.value, out.metadata["status"], out.metadata["sign_flipped"])
    # 0.93   PASS   False   (approximate)
    print(out.metadata["is_ratio"],
          out.metadata["mean_is"], out.metadata["mean_oos"])
    # 0.7   0.0771   0.0719   (approximate)

    # One call is one split. For a fraction-robust read, declare the split
    # set up front and let oos_decay_splits purge and aggregate it.
    from factrix.metrics import oos_decay_splits

    swept = oos_decay_splits(ic_df, value_col="ic", forward_periods=5)
    print(swept.value, swept.metadata["status"], swept.metadata["n_sign_flips"])
    ```

## See also

<div class="grid cards" markdown>

-   __`compute_ic` / `compute_spread_series`__

    ---

    Canonical producers of the `(date, value)` series this diagnostic
    consumes.

    [api/metrics/ic →](ic.md)

-   __`positive_rate` / `trend`__

    ---

    Sibling series diagnostics on the same input shape — sign
    significance and slope detection. Pair with `oos` when both
    in-sample magnitude and out-of-sample persistence matter.

    [api/metrics/positive_rate →](positive_rate.md)

-   __`by_slice`__

    ---

    Per-slice survival summaries (regime / universe / sector).

    [api/by-slice →](../by-slice.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_OOS_PERIODS_HARD * 2` floor; per-split `MIN_OOS_PERIODS_HARD` on each
    side).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Series diagnostics landing__

    ---

    Adjacent axis-agnostic series diagnostics.

    [api/metrics/series-tools →](series-tools.md)

</div>
