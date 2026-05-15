# bench — benchmark harness

Internal dev tooling for the multi-factor scale-out benchmark
(issue #380). **Not packaged into the factrix wheel** —
`tool.setuptools.packages.find` excludes `bench*`.

For the harness roadmap and open work, see issue #380 and its open
sub-issues. This README describes the current surface only.

## Modules

- `bench.schema` — pydantic v2 model for the JSONL record (`schema_version="1"`,
  open-schema `scale` keyed on `axis_cell`). Aligns with #380 §9.
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
  bundles + `METRIC_SET_VERSION` (independent of
  `factrix.run_metrics` defaults so a default-set tweak in factrix
  cannot silently shift baselines, #380 §3).
- `bench.scenarios.continuous` — mandatory Cont × Ind scenarios
  (#380 §4): S1 (single factor + `evaluate` + `run_metrics(heavy)`),
  S2 / S3 (50 / 200-factor screen, `core`), P1 (scaling probe), and
  per-metric micros M-ic / M-ic-boot / M-quantile / M-mono.
- `bench.scenarios.algo` — S4 greedy forward selection. Setup phase
  pre-computes per-factor spread series (rank → bucket → spread);
  compute phase runs the greedy + backward-elimination loop.
- `bench.scenarios.sparse` — Sparse × Individual scenarios (#380 §4):
  S5 (event-study bundle: `corrado_rank_test` + `compute_caar` +
  `compute_mfe_mae`) and the M-corrado micro.
- `bench.scenarios.dummy` — smoke scenario proving the wrapper →
  JSONL → validator loop.

## Running

`bench/` is a top-level package next to `factrix/`; it is **not
installed**. Run from the repo root so `bench` resolves on
`sys.path`:

```bash
# run every mandatory scenario at the tiny preset (CI smoke)
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
| `tiny` | `tiny` | All 11 mandatory scenarios (`#380` §4) |
| `small` | `small` | All 11 — 16 GB laptop baseline |
| `large` | `large` | All 11 — 32 GB cloud, opt-in |
| `event` | `small` | Sparse × Individual only (S5 + M-corrado) |

`--cold-cache` re-execs `python -m bench --run-one <id>` in a fresh
subprocess per scenario, resetting OS page cache / numpy import /
BLAS thread state. Reference baselines must run in cold-cache mode;
ad-hoc warm runs skip the subprocess overhead.

## Scale presets

`bench.scenarios._helpers.PRESETS` / `SPARSE_PRESETS` pin scales per
#380 §7:

| Preset | Cont: factors / assets / dates | Sparse: assets / dates / event_rate | Use |
|---|---|---|---|
| `tiny` | 8 / 20 / 60 | 20 / 60 / 0.05 | Tests + CI smoke; seconds-level |
| `small` | 100 / 1000 / 1250 | 200 / 1250 / 0.0001 | 16 GB laptop baseline |
| `large` | 500 / 1000 / 1250 | 500 / 1250 / 0.0002 | 32 GB cloud, opt-in |

Fixed-scale scenarios (S2 = 50 factors, S3 = 200, M-* = 50) override
the preset's `n_factors` so the workload stays fixed regardless of
preset choice. Sparse-cell `n_events` is reported back from the
realised event panel rather than configured directly — the seeded
`make_event_panel` produces Binomial events whose count depends on
`(n_dates × n_assets × event_rate)`.

## Schema / version invariants

`BenchRecord` carries three compatibility keys — comparison tools
must refuse to compare records that differ on any of them:

- `schema_version` (JSONL format itself) — pinned in `bench.schema.SCHEMA_VERSION`
- `metric_set_version` (metric set definitions) — pinned in `bench.metric_sets.METRIC_SET_VERSION`
- `env.dataset_spec_version` (synthetic generator recipe) — pinned in `factrix.datasets.DATASET_SPEC_VERSION`

The wrapper validates its own output via `pydantic.BaseModel.model_validate`
on construction; the dummy scenario additionally reads the written
JSONL back through `bench.validator.validate_file`.
