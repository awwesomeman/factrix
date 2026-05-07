# event_horizon

Multi-horizon event analysis — per-event return profile across `k`
offsets, plus binomial test on per-horizon hit rate. Detects pre-event
leakage and identifies the strongest horizon.

## Offset conventions

Defaults: `offsets = [-6, -3, -1, 1, 6, 12, 24]`. Offset `0` is the
event date itself and is excluded from the defaults; user-supplied
`offsets` lists are honoured verbatim.

| `k` | Anchor | Formula | Sign-adjusted |
|---|---|---|---|
| `k > 0` (post-event) | Cumulative from `t+1` entry | `price[t+1+k] / price[t+1] − 1` | Yes — multiplied by `sign(factor)`. The reading is signal *quality*. |
| `k < 0` (pre-event) | Single bar at offset | `price[t+k] / price[t+k−1] − 1` | **No** — the reading is *leakage*, where the bar's directional response is what matters independent of the eventual signal sign. |
| `k == 0` (corner) | Single bar at event | `price[t] / price[t−1] − 1` | No — falls into the pre-event branch. Pass with care; the event-day bar is usually contaminated by the announcement itself. |

The pre/post asymmetry is intentional. Mixing the two conventions on a
single chart (post-event cumulative + pre-event single-bar) is the
default factrix presentation; downstream consumers should not
re-cumulate the pre-event leg.

The binomial null at each offset assumes per-event independence at
that offset. **Adjacent post-event offsets are serially correlated
within the same event** — `k=6` and `k=12` share the `t+1` entry
price and overlap on bars `[t+2, t+7]`. The reported per-offset
*p*-values therefore have understated variance under the joint null
across offsets; treat the curve as descriptive and read the binomial
*p* one offset at a time. See also the
[confounded-event note](../../reference/metric-applicability.md#confounded-event-handling)
on within-asset event clustering, which compounds the same issue.

::: factrix.metrics.event_horizon

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual sparse landing page](individual-sparse.md) — adjacent event-study metrics.
