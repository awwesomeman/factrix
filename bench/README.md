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
  `started_at` / `is_warmup` / `cache_state`. Regular exceptions
  become `status="error"` rows (preserved, not raised);
  `KeyboardInterrupt` and `SystemExit` propagate so an interrupted
  run does not poison the JSONL with partial state.
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
| `xlarge` | 1000 / 2000 / 2000 | 2000 / 2000 / 0.0002 | Cloud-only stress (UX validation) |
| `user-realistic-high` | 500 / 3000 / 2500 | 3000 / 2500 / 0.0002 | Cloud-only (factor researcher upper bound) |

`xlarge` and `user-realistic-high` will OOM a 32 GB laptop and are
excluded from `make bench-bump` — they belong to the UX validation
lane below, not to the reference baseline lane.

Fixed-scale scenarios (`S2`=50 factors, `S3`=200, `M-*`=50 factors)
override the preset's `n_factors` so the workload stays fixed
regardless of preset choice. Sparse-cell `n_events` is reported back
from the realised event panel rather than configured directly — the
seeded `make_event_panel` produces Binomial events whose count
depends on `(n_dates × n_assets × event_rate)` and the seed.

## UX validation

The reference baseline measures *algorithm cost lower bound* —
single-thread BLAS, cold cache, fixed seed — for cross-version
regression detection. It does not answer "would a real user hit a
perf wall?". UX validation is a separate lane on the same harness
that does.

| Axis | Reference baseline | UX validation |
|---|---|---|
| BLAS threads | locked to 1 | `--threads N` (typical 4 / 16) |
| Cache | cold | warm |
| Measurement target | `compute_s` ratio | `wall_s` absolute |
| Acceptance | ratio vs history | assert vs `bench.ux_targets.UX_TARGETS` |
| Frequency | per minor release | per release + ad-hoc |

Run UX validation by combining `--threads` with the cloud-only
presets, then pipe the output directory through `bench.ux_validate`:

```bash
python -m bench --target xlarge              --threads 16 --output /tmp/A1-xl
python -m bench --target user-realistic-high --threads 16 --output /tmp/A1-rh
python -m bench --target xlarge              --threads 4  --output /tmp/A2-xl
python -m bench --target user-realistic-high --threads 4  --output /tmp/A2-rh

for d in /tmp/A{1,2}-{xl,rh}; do python -m bench.ux_validate "$d"; done
```

`bench.ux_validate`:

- Reads every `*.jsonl` under the directory, picks measured rows
  (`is_warmup=false ∧ status="ok"`), and asserts each row's `wall_s`
  against `UX_TARGETS`.
- Prints a markdown table to stdout (scenario / `n_factors` /
  `wall_s` / target / verdict / `peak_rss_mb` / `n_threads`).
- Returns non-zero when any row fails its target (a *red flag*).
- Records OOM / error / unknown-scenario rows as non-blocking
  incidents — OOM on a 1000-factor screen is real data, not a CI
  failure.

Pair `--threads N` with `--cold-cache` whenever multi-thread effects
actually matter: BLAS picks its thread count at numpy import time, so
in single-process warm mode the in-process flag is partly cosmetic
(it stamps `env.omp_threads=N` into every record but cannot re-thread
an already-initialised BLAS). The cold-cache path forks a fresh
subprocess per scenario with the thread env vars set before import.

`bench.ux_targets.UX_TARGETS_VERSION` is bumped whenever the target
table is edited so a stored report can be re-interpreted against the
version it was produced under. The validator stamps the version it
used into its markdown header.

## Schema / version invariants

`BenchRecord` carries three compatibility keys — comparison tools
must refuse to compare records that differ on any of them:

- `schema_version` (JSONL format itself) — pinned in `bench.schema.SCHEMA_VERSION`
- `metric_set_version` (metric set definitions) — pinned in `bench.metric_sets.METRIC_SET_VERSION`
- `env.dataset_spec_version` (synthetic generator recipe) — pinned in `factrix.datasets.DATASET_SPEC_VERSION`

The wrapper validates its own output via `pydantic.BaseModel.model_validate`
on construction; every scenario reads the written JSONL back through
`bench.validator.validate_file` before returning.
