# AnalysisConfig

Three-axis spec for a single-factor analysis. Constructed via one of
the four factory methods on the class — direct construction works but
runs the same `__post_init__` validation, so the factories are the
documented user surface.

See [Methodology](../development/methodology.md) for the axis taxonomy
and [Statistical methods](../reference/statistical-methods.md) for procedure
selection rationale.

::: factrix.AnalysisConfig
    options:
      show_root_heading: false
      members:
        - individual_continuous
        - individual_sparse
        - common_continuous
        - common_sparse
        - to_dict
        - from_dict
