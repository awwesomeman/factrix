# suggest_config

Heuristic introspection that inspects a raw panel and proposes an
[`AnalysisConfig`](analysis-config.md) plus the observations and
reasoning behind the suggestion. The proposal is never auto-applied —
the caller (or an AI agent) reads `reasoning` / `warnings` before
deciding to use, override, or reject it.

Canonical [`MissingConfigError`](errors.md) recovery:

```python
import factrix as fx

result  = fx.suggest_config(panel)
profile = fx.evaluate(panel, result.suggested)
```

## `SuggestConfigResult`

Frozen dataclass with four fields. The first three carry the
recommendation and its trace; the fourth is a list of pre-computed
risk codes the caller can act on before running `evaluate`.

| Field | Type | Read for |
|---|---|---|
| `suggested` | [`AnalysisConfig`](analysis-config.md) | Pass straight into `evaluate(panel, result.suggested)` |
| `detected` | `dict[str, Any]` | Branch on the observations that drove the suggestion |
| `reasoning` | `dict[str, str]` | Show the human-readable rationale per axis |
| `warnings` | `list[WarningCode]` | Pre-evaluate risk codes (small N, short series) |

### `detected` — structured observations

Always-present, type-stable keys — programmatic consumers can branch
without parsing strings. The key set is also exposed as
`factrix._describe.DETECTED_KEYS` for membership checks.

| Key | Type | Meaning |
|---|---|---|
| `scope` | `str` | `"individual"` / `"common"` |
| `signal` | `str` | `"continuous"` / `"sparse"` |
| `mode` | `str` | `"panel"` / `"timeseries"` |
| `n_assets` | `int` | Unique `asset_id` count |
| `n_periods` | `int` | Unique `date` count |
| `sparsity` | `float` | Zero-ratio in `factor` (NaN if panel is empty) |

```python
result.detected
# {
#     "scope":     "individual",
#     "signal":    "continuous",
#     "mode":      "panel",
#     "n_assets":  100,
#     "n_periods": 494,
#     "sparsity":  0.0,
# }
```

### `reasoning` — per-axis human-readable trace

Mirror of `detected`, one short sentence per axis. Keys are stable:
`"scope"`, `"signal"`, `"metric"`, `"mode"`.

```python
result.reasoning
# {
#     "scope":  "factor varies across assets at given date: YES → INDIVIDUAL",
#     "signal": "sparsity ratio = 0.00 (threshold 0.5): → CONTINUOUS",
#     "metric": "scope=INDIVIDUAL × signal=CONTINUOUS: defaulting metric=IC ...",
#     "mode":   "n_assets = 100 detected → PANEL",
# }
```

When the `(scope=COMMON, mode=PANEL)` cell's inference-stage
cross-section falls below `MIN_ASSETS_WARN` (minimum cross-section
size that warrants a warning), the `mode` line appends the matching
`WarningCode` name so the reader sees *why* a warning fired without
consulting the panel.

### `warnings` — pre-computed risk codes

`list[WarningCode]` enum values; see
[Warning / info / stat codes](../reference/warning-codes.md) for full
gate semantics.

| Code | Trigger |
|---|---|
| `UNRELIABLE_SE_SHORT_PERIODS` | `mode == TIMESERIES` and `MIN_PERIODS_HARD ≤ n_periods < MIN_PERIODS_WARN` (hard-error floor and soft-warning floor for the time axis) |
| `SMALL_CROSS_SECTION_N` / `BORDERLINE_CROSS_SECTION_N` | PANEL only. INDIVIDUAL cells threshold on raw `n_assets`; COMMON cells first apply the per-asset `MIN_TS_OBS` filter (minimum time-series observations per asset, mirroring `compute_ts_betas`) so the preview matches what `evaluate()` will emit |

## `result.diagnose()` — JSON-shape exit point

Python callers read `result.warnings` as a `list[WarningCode]`. For
cross-wire / log / AI-agent consumers, `result.diagnose()` returns a
plain-Python, JSON-serialisable dict. Calling `json.dumps(result)`
directly fails — `AnalysisConfig` and `WarningCode` are not JSON
primitives.

```python
import json

print(json.dumps(result.diagnose(), indent=2))
```

```json
{
  "suggested": {
    "scope": "individual",
    "signal": "continuous",
    "metric": "ic",
    "forward_periods": 5
  },
  "detected": {
    "scope": "individual",
    "signal": "continuous",
    "mode": "panel",
    "n_assets": 100,
    "n_periods": 494,
    "sparsity": 0.0
  },
  "reasoning": {
    "scope":  "factor varies across assets at given date: YES → INDIVIDUAL",
    "signal": "sparsity ratio = 0.00 (threshold 0.5): → CONTINUOUS",
    "metric": "scope=INDIVIDUAL × signal=CONTINUOUS: defaulting metric=IC (rank predictive ordering)",
    "mode":   "n_assets = 100 detected → PANEL"
  },
  "warnings": []
}
```

| Field on `SuggestConfigResult` | Field on `diagnose()` payload | Transform |
|---|---|---|
| `suggested: AnalysisConfig` | `"suggested": dict` | [`AnalysisConfig.to_dict()`](analysis-config.md) |
| `detected: dict` | `"detected": dict` | shallow copy |
| `reasoning: dict` | `"reasoning": dict` | shallow copy |
| `warnings: list[WarningCode]` | `"warnings": list[str]` | `sorted(w.value for w in warnings)` |

The `warnings` serialisation mirrors
[`FactorProfile.diagnose()`](factor-profile.md#example) — one parser
handles both payloads.

## Inference axes

Three axes are derived from the panel; the fourth is collapsed or
defaulted based on the other three. See
[Concepts](../getting-started/concepts.md) for what each axis means.

| Axis | Rule |
|---|---|
| `signal` | `sparsity = zero_ratio(factor)`; `≥ 0.5` → `SPARSE`, else `CONTINUOUS` |
| `scope` | `factor` constant across `asset_id` per `date` → `COMMON`, else `INDIVIDUAL`; collapses to `COMMON` when `n_assets ≤ 1` |
| `mode` | `n_assets ≤ 1` → `TIMESERIES`, else `PANEL` |
| `metric` | `INDIVIDUAL × CONTINUOUS` defaults to `Metric.IC` (Information Coefficient — rank predictive ordering); collapsed (`None`) on every other cell |

## Source

::: factrix.suggest_config
