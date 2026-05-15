# bench — benchmark harness

Internal dev tooling for the multi-factor scale-out benchmark
(issue #380). **Not packaged into the factrix wheel** —
`tool.setuptools.packages.find` excludes `bench*`.

## What this foundation ships

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
- `bench.scenarios.dummy` — smoke scenario proving the wrapper →
  JSONL → validator loop.

## Running

`bench/` is a top-level package next to `factrix/`; it is **not
installed**. Run from the repo root so `bench` resolves on
`sys.path`:

```bash
# self-validate an existing JSONL
python -m bench.validate path/to/run.jsonl

# end-to-end smoke (dummy scenario)
python -m bench.scenarios.dummy --output out/dummy.jsonl
```

If launching from elsewhere, set `PYTHONPATH=<repo-root>` explicitly.
CI smoke jobs should `cd` to the repo root or export `PYTHONPATH=.`
before invoking any `python -m bench.*` entry point.

## Scale presets

`bench.scenarios._helpers.PRESETS` pins three scales per #380 §7:

| Preset | n_factors | n_assets | n_dates | Use |
|---|---|---|---|---|
| `tiny` | 8 | 20 | 60 | Tests + CI smoke; seconds-level |
| `small` | 100 | 1000 | 1250 | 16 GB laptop baseline (mandatory) |
| `large` | 500 | 1000 | 1250 | 32 GB cloud baseline (opt-in) |

Fixed-scale scenarios (S2 = 50 factors, S3 = 200) override the
preset's `n_factors` so the workload stays fixed regardless of
preset choice.

## What it does *not* do (yet)

- Sparse / event scenarios S5 + M-corrado — follow-up sub-PR of #382
- CLI dispatcher `python -m bench --target small|large|event|tiny` — follow-up sub-PR of #382
- Cold-cache subprocess re-exec — follow-up sub-PR of #382
- Reference baselines in `bench/baselines/` + release-flow rerun (#383)
- CI `bench-tiny` smoke (#383)
- Ratio / summary markdown post-processing

## Schema / version invariants

`BenchRecord` carries three compatibility keys — comparison tools
must refuse to compare records that differ on any of them:

- `schema_version` (JSONL format itself) — pinned in `bench.schema.SCHEMA_VERSION`
- `metric_set_version` (metric set definitions) — pinned in `bench.metric_sets.METRIC_SET_VERSION`
- `env.dataset_spec_version` (synthetic generator recipe) — pinned in `factrix.datasets.DATASET_SPEC_VERSION`

The wrapper validates its own output via `pydantic.BaseModel.model_validate`
on construction; the dummy scenario additionally reads the written
JSONL back through `bench.validator.validate_file`.
