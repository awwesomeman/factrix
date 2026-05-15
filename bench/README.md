# bench — benchmark harness

Internal dev tooling for the multi-factor scale-out benchmark.
**Not packaged into the factrix wheel** —
`tool.setuptools.packages.find` excludes `bench*`.

For the harness roadmap and open work, see issue #380 and its open
sub-issues. This README describes the current surface only.

## Modules

- `bench.schema` — pydantic v2 model for the JSONL record
  (`schema_version="1"`, open-schema `scale` keyed on `axis_cell`).
- `bench.preflight` — BLAS thread lock (`OMP_/OPENBLAS_/MKL_NUM_THREADS`),
  numpy seed, `gc.collect()`, env fingerprint (git SHA, factrix version,
  python/numpy/BLAS, CPU model, cores, RAM).
- `bench.wrapper` — `measure(setup, compute, …)` produces one
  `BenchRecord` per call with `setup_s` / `compute_s` / `wall_s` /
  `cpu_s` / `peak_rss_mb` / `peak_alloc_mb` / `status` /
  `started_at` / `is_warmup` / `cache_state`. `status="error"` rows
  are preserved (not raised).
- `bench.validator` — self-validates JSONL; fail-loud on
  `schema_version` mismatch, missing fields, enum violations, or
  scale ↔ axis_cell mismatch.
- `bench.metric_sets` — pinned `core` / `heavy` / `algo` / `event`
  bundles + `METRIC_SET_VERSION`, independent of
  `factrix.run_metrics` defaults so a default-set tweak in factrix
  cannot silently shift baselines.
- `bench.scenarios.*` — scenario implementations grouped by
  analysis-axis cell (see scenario reference below).

## Running

`bench/` is a top-level package next to `factrix/`; it is **not
installed**. Run from the repo root so `bench` resolves on
`sys.path`:

```bash
# run every scenario at the tiny preset (CI smoke)
python -m bench --target tiny --output out/

# baseline run (16 GB laptop) — cold-cache mode required for
# reference baselines so OS page cache + numpy import state resets
# between scenarios
python -m bench --target small --output out/baselines/v0.13.1/ --cold-cache

# sparse-cell only
python -m bench --target event --output out/

# re-run one scenario without spawning the full set
python -m bench --run-one S2 --preset small --output out/

# self-validate an existing JSONL
python -m bench.validate path/to/run.jsonl
```

If launching from elsewhere, set `PYTHONPATH=<repo-root>` explicitly.
CI smoke jobs should `cd` to the repo root or export `PYTHONPATH=.`
before invoking any `python -m bench.*` entry point.

### Targets

| `--target` | Preset | Scenarios |
|---|---|---|
| `tiny` | `tiny` | All scenarios listed below |
| `small` | `small` | All scenarios — 16 GB laptop baseline |
| `large` | `large` | All scenarios — 32 GB cloud, opt-in |
| `event` | `small` | Sparse × Individual cell only |

`--cold-cache` re-execs `python -m bench --run-one <id>` in a fresh
subprocess per scenario, resetting OS page cache / numpy import /
BLAS thread state. Reference baselines must run in cold-cache mode;
ad-hoc warm runs skip the subprocess overhead.

### Scenarios

| `scenario_id` | Cell | Compute |
|---|---|---|
| `S1` | Continuous × Individual | Single factor through `evaluate` + `run_metrics` + bootstrap CI on the IC series |
| `S2` | Continuous × Individual | 50-factor screen with the `core` metric set per factor |
| `S3` | Continuous × Individual | 200-factor screen with the `core` metric set per factor |
| `S4` | Continuous × Individual | Greedy forward selection over a candidate factor pool |
| `S5` | Sparse × Individual | Event-study bundle: corrado rank test + CAAR + MFE/MAE |
| `P1` | Continuous × Individual | Scaling probe — three sub-runs at 100 / 200 / 500 factors |
| `M-ic` | Continuous × Individual | Per-factor `ic` only — attributes cost to one metric |
| `M-ic-boot` | Continuous × Individual | Per-factor `compute_ic` + `bootstrap_mean_ci` |
| `M-quantile` | Continuous × Individual | Per-factor `quantile_spread` only |
| `M-mono` | Continuous × Individual | Per-factor `monotonicity` only |
| `M-corrado` | Sparse × Individual | `corrado_rank_test` only on the event panel |

## Scale presets

`bench.scenarios._helpers.PRESETS` / `SPARSE_PRESETS` pin the
dimensions used by every scenario:

| Preset | Cont: factors / assets / dates | Sparse: assets / dates / event_rate | Use |
|---|---|---|---|
| `tiny` | 8 / 20 / 60 | 20 / 60 / 0.05 | Tests + CI smoke; seconds-level |
| `small` | 100 / 1000 / 1250 | 200 / 1250 / 0.0001 | 16 GB laptop baseline |
| `large` | 500 / 1000 / 1250 | 500 / 1250 / 0.0002 | 32 GB cloud, opt-in |

Fixed-scale scenarios (`S2`=50 factors, `S3`=200, `M-*`=50 factors)
override the preset's `n_factors` so the workload stays fixed
regardless of preset choice. Sparse-cell `n_events` is reported back
from the realised event panel rather than configured directly — the
seeded `make_event_panel` produces Binomial events whose count
depends on `(n_dates × n_assets × event_rate)` and the seed.

## Schema / version invariants

`BenchRecord` carries three compatibility keys — comparison tools
must refuse to compare records that differ on any of them:

- `schema_version` (JSONL format itself) — pinned in `bench.schema.SCHEMA_VERSION`
- `metric_set_version` (metric set definitions) — pinned in `bench.metric_sets.METRIC_SET_VERSION`
- `env.dataset_spec_version` (synthetic generator recipe) — pinned in `factrix.datasets.DATASET_SPEC_VERSION`

The wrapper validates its own output via `pydantic.BaseModel.model_validate`
on construction; every scenario reads the written JSONL back through
`bench.validator.validate_file` before returning.
