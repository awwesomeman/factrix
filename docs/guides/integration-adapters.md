---
title: Integration adapters
---

# Integration adapters

factrix returns ordinary Polars frames and frozen result records instead of
owning an experiment tracker or rendering backend. This page shows the adapter
boundary for MLflow. Plotly has one canonical source of truth in
[Plotting recipes](plotting-recipes.md).

Install the backend in the application environment that uses it; neither
package is a factrix dependency:

```console
python -m pip install mlflow
```

The Python fences are marked `Illustrative` because the factrix documentation
test environment intentionally does not install this optional backend. CI
syntax-compiles them, but only a caller-provisioned MLflow environment can
exercise tracking-server side effects.

## Log one `EvaluationResult` to MLflow

An auditable run needs more than the headline metric. The adapter below logs:

- the full `(scope, density, structure)` analysis cell in the experiment name
  and tags;
- factor, horizon, overlap, and every hypothesis `params` key;
- every finite metric value and p-value;
- every `n_obs` paired with its `n_obs_axis` tag;
- bundle `n_periods`, `n_pairs`, and `n_assets`;
- both the complete warning record and the unexpected-warning subset; and
- the complete nested `to_dict()` payload as the durable artifact.

```python title="Illustrative"
import json
import math

import mlflow


def _warning_record(warning):
    return {
        "code": warning.code.value,
        "source": warning.source,
        "message": warning.message,
        "expected": warning.expected,
    }


def log_evaluation(result):
    scope, density, structure = result.cell
    experiment = f"factrix.{scope.value}.{density.value}.{structure.value}"
    tag_namespace = "factrix"
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{result.factor}.h{result.forward_periods}"):
        mlflow.log_params(
            {
                "factor": result.factor,
                "forward_periods": result.forward_periods,
                "overlap_periods": result.overlap_periods,
                **{f"param.{key}": value for key, value in result.params.items()},
            }
        )

        values = {
            "sample.n_periods": float(result.n_periods),
            "sample.n_pairs": float(result.n_pairs),
            "sample.n_assets": float(result.n_assets),
        }
        axis_tags = {}
        for label, metric in result.metrics.items():
            for field in ("value", "p_value"):
                value = getattr(metric, field)
                if value is not None and math.isfinite(float(value)):
                    values[f"{label}.{field}"] = float(value)
            if metric.n_obs is not None:
                values[f"{label}.n_obs"] = float(metric.n_obs)
            axis_tags[f"{label}.n_obs_axis"] = metric.n_obs_axis or "none"
        mlflow.log_metrics(values)

        all_warnings = [_warning_record(w) for w in result.warnings]
        unexpected = [_warning_record(w) for w in result.unexpected_warnings]
        mlflow.set_tags(
            {
                f"{tag_namespace}.cell.scope": scope.value,
                f"{tag_namespace}.cell.density": density.value,
                f"{tag_namespace}.cell.structure": structure.value,
                f"{tag_namespace}.warning_codes": json.dumps(
                    [w["code"] for w in all_warnings]
                ),
                f"{tag_namespace}.unexpected_warning_codes": json.dumps(
                    [w["code"] for w in unexpected]
                ),
                **axis_tags,
            }
        )
        mlflow.log_dict(
            {"all": all_warnings, "unexpected": unexpected},
            "factrix/warnings.json",
        )
        mlflow.log_dict(result.to_dict(), "factrix/evaluation.json")
```

`expected=True` means a warning was declared as part of the study design; it
does not mean the warning disappeared. That is why the adapter retains
`warnings` and separately logs `unexpected_warnings` as the alert view.

The full cell triple belongs in the experiment name. Dropping
`DataStructure.PANEL` versus `TIMESERIES`, for example, would combine estimates
with different sample regimes. If an organization deliberately groups more
broadly, retain all three fields as tags and document the omitted axis in its
experiment convention.

## Scalar columns versus nested artifacts

Use `result.to_frame(metadata=(...))` when selected scalar estimator metadata
must travel beside each metric row:

```python title="Illustrative"
audit = result.to_frame(
    metadata=("n_groups", "rebalance_lag", "mean_tail_size")
)
```

Missing keys become null. Nested lists, dictionaries, event paths, complete
warnings, and the execution plan belong in `result.to_dict()` and therefore in
the JSON artifact above. Do not flatten them into lossy or ambiguous metric
names.

For end-to-end reproducibility, the caller must also stamp identifiers factrix
cannot infer, such as a dataset content hash, data vintage, application commit,
and research-plan ID, into `EvaluationResult.metadata` before logging. The
nested artifact preserves that mapping unchanged.

If a Plotly recipe produces `fig`, MLflow can store it without changing the
factrix adapter:

```python title="Illustrative"
mlflow.log_figure(fig, "factrix/charts/ic-path.html")
```

## Why integrations remain caller-owned

Bundling Plotly or MLflow would enlarge the core install, couple factrix
releases to fast-moving third-party dependency graphs, and still fail to match
each organization's tracking schema, artifact store, theme, and access policy.
Keeping a small adapter at the application boundary lets users replace MLflow
or Plotly without changing the quantitative result contract.

MLflow references: [tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)
and [Python API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html).
