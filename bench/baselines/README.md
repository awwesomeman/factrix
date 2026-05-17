# bench/baselines — reference-baseline index

Each subdirectory is a frozen reference baseline: one JSONL per
scenario, produced by `make bench-bump` (cold-cache, small preset)
on a specific commit and machine.

Naming: `v<factrix-version>-<machine-id>/`, where `<machine-id>` is
`<os>-<arch>-<ram-rounded-down>g` from `make bench-machine-id`.

The harness writes only **raw observations**; ratios are computed
post-hoc. See `bench/README.md` for ratio-reading rules and the
cross-machine rebaseline procedure.

## Index

| Directory | factrix version | Commit SHA | Machine ID | CPU | RAM | `dataset_spec_version` | `metric_set_version` | `cache_state` | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `v0.14.0-linux-x86_64-125g/` | 0.13.0 (pre-release pin) | `1dce8fb` | `linux-x86_64-125g` | x86_64 (8 phys / 16 logical) | 125.8 GB | 1 | 1 | cold | Captured against `1dce8fb` (factrix 0.13.0) ahead of the v0.14.0 release; the slug names the target version this baseline serves. `make bench-bump` rerun at v0.14.0 release time overwrites these files with HEAD-versioned records — the captured `compute_s` shape reflects the commit it was pinned to (pre-batch-dispatch), not the current HEAD. |

> The **JSONL `env` block** is the source of truth for any given
> record's machine / version provenance. This index is a discovery
> aid — if the index and the JSONL disagree, the JSONL wins.

## Cross-machine anchors

When the primary baseline machine retires, capture per-scenario
`compute_s` ratios on **the same `git_sha`** between the outgoing
and incoming machine, and record them here. Until then, this
section stays empty.

| Anchor commit | From machine | To machine | Scenario | Ratio (`to / from`) |
|---|---|---|---|---|

## Compatibility gates

Comparison tools must refuse to compare records that differ on any
of:

- `schema_version` (`bench.schema.SCHEMA_VERSION`)
- `metric_set_version` (`bench.metric_sets.METRIC_SET_VERSION`)
- `env.dataset_spec_version` (`factrix.datasets.DATASET_SPEC_VERSION`)
- `cache_state` (cold ↔ warm comparisons are meaningless)
- `axis_cell` (a Continuous × Individual record is not comparable
  to a Sparse × Individual one)
