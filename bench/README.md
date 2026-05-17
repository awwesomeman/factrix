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

`scenario_id` prefix encodes the role rather than running order:

- `S*` — baseline screen scenarios (full pipeline at one workload size).
- `P*` — scaling probe (one scenario emits multiple records across scale steps).
- `M-*` — per-metric micro attribution (isolates one metric so its cost is
  separable from the bundled screen scenarios).

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

## Ratio reading

JSONL rows store **raw observations only** — ratios are computed
post-hoc by the comparison tool, not written by the harness. The
rule of thumb when reading them:

- **Default to `compute_s` ratios**, not `wall_s`. `setup_s` covers
  synthetic-data generation + disk I/O, which varies with NVMe vs.
  network-mounted home dirs and CI cache state; folding it into the
  ratio dilutes the signal you care about (the metric computation).
- **`wall_s = setup_s + compute_s`** is a sanity check, not a
  primary metric. Use it to spot scenarios where setup dominates
  (e.g. CI runners with slow disk) before drawing conclusions.
- **`peak_rss_mb` vs. `peak_alloc_mb`**: RSS includes BLAS scratch
  and C-level numpy buffers; `tracemalloc`'s `peak_alloc_mb` only
  counts Python-side allocation. Expect `peak_rss_mb ≫ peak_alloc_mb`
  on numeric workloads; the two are **not** interchangeable and
  must not be subtracted.
- **Comparison is gated on three keys**: `schema_version`,
  `metric_set_version`, `env.dataset_spec_version`. Different on
  any of them → refuse to compare, do not silently coerce.
- **Cross-machine ratios** are valid; cross-machine absolute
  numbers are not. See "Cross-machine rebaseline" below.

### Ad-hoc ratio table

`scripts/bench_diff.py` renders a markdown ratio table from two output
dirs for direct paste into PR descriptions:

```
python scripts/bench_diff.py <before-dir> <after-dir>
```

Gates on `schema_version` / `metric_set_version` /
`env.dataset_spec_version` / `axis_cell` / `cache_state` — mismatched
runs exit non-zero rather than silently coerce. Multi-scale scenarios
(P1) align per `scale`. This is the PR-description helper for the
#378 optimisation loop, **not** a long-term CLI; promoting to a
proper `bench.compare` module is a separate decision.

## `cache_state` operating rules

Every record carries `cache_state ∈ {"cold", "warm", "unknown"}`.
The harness never infers it — the caller (CLI or Makefile)
declares it; analysis tools refuse to compare across cache states.

| Mode | When | How |
|---|---|---|
| `cold` | Reference baselines committed under `bench/baselines/` | `--cold-cache` flag (or `make bench-bump`). Re-execs one subprocess per scenario so OS page cache, numpy import, BLAS thread state reset between scenarios. |
| `warm` | Ad-hoc iteration during a `#378`-style optimisation PR | Default (no flag). All scenarios share one process; faster, but absolute numbers are not comparable to a cold-cache baseline. |
| `unknown` | Imported / replayed runs whose provenance is unclear | Only used when ingesting external JSONL with missing context. Avoid producing new `unknown` rows from the harness itself. |

Rule of thumb: if it lands in `bench/baselines/`, it must be
`cold`. If it lands in a PR description as an ad-hoc ratio, it is
usually `warm` — say so explicitly and compare warm-to-warm.

## Synthetic-data caveat

`factrix.datasets.make_cs_panel` / `make_event_panel` approximate
**computational load**, not **statistical reality**. The
generators are designed to exercise the same code paths as
production data at controlled scale, with a tunable IC strength
and cross-sectional correlation knob — but the IC, autocorrelation,
and event-clustering structure of real markets are not reproduced.

- Use bench numbers to compare **harness configurations** (cold vs.
  warm, before vs. after an optimisation, machine A vs. machine B).
- Do **not** use them to predict production runtime on real
  factors, or to validate that a metric's statistical behaviour is
  correct — that is what `tests/` and `factrix.run_metrics`
  invariants are for.
- `dataset_spec_version` is bumped any time the generator default
  parameters move (IC strength, correlation level, etc.); baselines
  on different `dataset_spec_version` are explicitly refused by the
  comparison gate above.

## Cross-machine rebaseline

The primary baseline machine will eventually retire. The procedure
that keeps history comparable:

1. **Anchor run** — on the **same `git_sha`**, run the `small`
   target cold-cache on both the outgoing machine and the incoming
   machine. Commit both under `bench/baselines/v<version>-<machine-id>/`.
2. **Compute the per-scenario ratio** between the two machines'
   `compute_s` (one row per scenario, ideally
   `is_warmup=false ∧ status="ok"`). Record those ratios in
   `bench/baselines/README.md` under a "Cross-machine anchors"
   subsection.
3. **From that point on**, the new machine is the reference. Old
   absolute numbers stay readable via the anchor ratios; new PRs
   compare against the new-machine baseline directly.
4. **Never re-run an old commit on a new machine and overwrite a
   prior baseline file** — keep both, dated by directory slug.

If the outgoing machine has already gone offline before the anchor
was captured, history before the swap becomes
absolute-incomparable. Document the gap in
`bench/baselines/README.md` rather than fabricating ratios.

## Release flow

Reference-baseline rerun is a **mandatory** step of every minor
release. The hooks:

- **`Makefile`** target `bench-bump` — runs the small target
  cold-cache, derives the machine-id slug from `platform` + `psutil`,
  writes to `bench/baselines/v<version>-<machine-id>/`. Invoke
  **after** `cz bump` (so `factrix.__version__` resolves to the new
  version) and **before** pushing the release tag.
- **Release PR template** — `.github/PULL_REQUEST_TEMPLATE/release.md`
  carries a "Reference baseline rerun" checkbox alongside the
  CHANGELOG-polish checklist; the release author must tick it
  before merge.
- **`bench/baselines/README.md`** — append one row per new baseline
  (version / commit SHA / machine slug / `dataset_spec_version` /
  `metric_set_version` / `cache_state`).

Operationally:

```bash
# on main, after CHANGELOG entries are polished and committed
cz bump --changelog
make bench-bump                          # writes bench/baselines/v<new>-<machine-id>/
# edit bench/baselines/README.md index row
git add bench/baselines CHANGELOG.md
git commit --amend --no-edit             # fold baseline + index row into the release commit
git tag -fa v<new> -m "v<new>"
git push origin main --follow-tags
```

If `bench-bump` regresses a scenario into `status="error"` /
`status="oom"`, **do not** tag — investigate first. The whole point
of the rerun is to catch harness or factrix rot before it ships.
