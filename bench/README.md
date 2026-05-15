# bench — benchmark harness foundation

Internal dev tooling for the multi-factor scale-out benchmark
(issue #380, foundation sub-issue #381). **Not packaged into the
factrix wheel** — `tool.setuptools.packages.find` excludes `bench*`.

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
- `bench.scenarios.dummy` — smoke scenario proving the wrapper →
  JSONL → validator loop. Mandatory S/M scenarios from #380 §4
  follow in a separate sub-issue.

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

## What it does *not* do (yet)

Per #381 scope, this foundation does **not** ship:

- Mandatory peak/probe scenarios S1–S5 + P1
- Per-metric micro scenarios M-ic / M-ic-boot / M-quantile / M-mono / M-corrado
- Reference baselines in `bench/baselines/`
- Release-flow baseline rerun integration
- CI `bench-tiny` smoke
- Ratio / summary markdown post-processing

Those live in follow-up sub-issues of #380.

## Schema / version invariants

`BenchRecord` carries three compatibility keys — comparison tools
must refuse to compare records that differ on any of them:

- `schema_version` (JSONL format itself) — pinned in `bench.schema.SCHEMA_VERSION`
- `metric_set_version` (metric set definitions) — to be pinned in `bench/metric_sets.py` (follow-up sub-issue)
- `env.dataset_spec_version` (synthetic generator recipe) — pinned in `factrix.datasets.DATASET_SPEC_VERSION`

The wrapper validates its own output via `pydantic.BaseModel.model_validate`
on construction; the dummy scenario additionally reads the written
JSONL back through `bench.validator.validate_file`.
