# Changelog

All notable changes to **factorlib** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR**: API-breaking changes (field rename, signature change, removed metric)
- **MINOR**: New metrics, new Profile fields, new optional parameters
- **PATCH**: Bug fixes, docstring/test fixes, internal refactors

While the version is below `1.0.0`, the public API should be considered
unstable — breaking changes may occur in **MINOR** bumps. Consumers are
expected to pin by commit SHA (e.g., via git submodule) rather than relying
on semver range constraints until `1.0.0` is cut.

---

## [0.1.0] - 2026-04-20

### Note
First standalone release. Extracted from the `awwesomeman/factor-analysis`
research workspace via `git filter-repo`; 149 commits of prior development
history are preserved in this repository's git log.

Earlier version tags (`3.x`) existed only in the originating workspace and
are not reproduced here — version numbering restarts from `0.1.0` to
honestly reflect that the API is still iterating. The pre-extraction
snapshot is anchored in the source workspace as the tag
`pre-extraction-backup`.

### State at extraction
- **555 tests passing** across four factor types (cross-sectional, event
  signal, macro panel, macro common)
- `fl.evaluate` / `fl.evaluate_batch` / `fl.factor()` public API
- Typed `Profile` dataclasses (`frozen=True, slots=True`) with binary
  `verdict()` (PASS / FAILED) and `diagnose()` diagnostic list
- `ProfileSet` container with BHY multiple-testing correction
  (`multiple_testing_correct`)
- Artifacts retention via `return_artifacts=True` and `evaluate_batch(keep_artifacts=, compact=)`
- External rule registration: `register_rule` / `clear_custom_rules`
- `on_result` callback supports `bool | None` early-stopping
- Level 2 helpers integrated into Profile pipeline: `regime_ic`,
  `multi_horizon_ic`, `spanning_alpha`, orthogonalization
- Shared tradability primitives (`turnover` / `breakeven_cost` /
  `net_spread`) hoisted to `Factor` base class with per-call override
  support

### Known caveats
- `examples/demo.ipynb` stored outputs may reflect an earlier naming
  (`q1_q5_spread` → `long_short_spread` rename landed 2026-04-20); rerun
  with current code to regenerate
- `Factor Signal Analyzer` positioning: `turnover` / `breakeven_cost` /
  `net_spread` are idealized proxies (equal-weight, zero slippage) and do
  not represent tradable returns
