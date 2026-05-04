# FactorProfile

Procedure-canonical analysis result for a single factor. Every
registered procedure produces an instance of this dataclass with
cell-specific scalars keyed in the `stats` mapping; adding a new
metric does not grow the schema.

See [Metric applicability](../metric-applicability.md) for `n_obs`
and `n_assets` thresholds per procedure.

::: factrix.FactorProfile
    options:
      show_root_heading: false
      members:
        - verdict
        - diagnose
