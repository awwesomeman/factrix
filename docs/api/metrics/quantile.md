# quantile

Quantile-bucket long-short spread on a panel: per-date
top-minus-bottom return → spread series, then non-overlapping *t* on
its mean. Equal-weight and value-weight variants.

::: factrix.metrics.quantile
    options:
      members:
        - quantile_spread
        - quantile_spread_vw
        - compute_spread_series
        - compute_group_returns
