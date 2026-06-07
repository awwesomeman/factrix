# Contributing to factrix

Quick-start guide. Full contributing reference: **[docs/development/contributing](https://awwesomeman.github.io/factrix/latest/development/contributing/)**.

## Setup

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync --extra dev
python scripts/setup_dev.py   # activate .githooks/ (per-clone, idempotent)
uv run pytest                 # must be all green before committing
```

## Development cycle

```bash
git checkout -b feat/<short-desc>
# edit → test → commit
git add <specific-files>
cz commit -- -s           # Conventional Commits + Signed-off-by
git push origin feat/<short-desc>
gh pr create
```

## Adding a Metric

When adding or developing a new metric to `factrix`, follow these rules:

1. **Stamping and Registration**:
   - Use the `@metric` decorator from `factrix.metrics` to define a function-based `MetricBase` subclass. This automatically handles registration under the hood.
   - Alternatively, for lower-level or third-party callables, apply the `@metric_spec(MetricSpec(...))` decorator and register the metric explicitly using `factrix.metrics.register(fn)`. Do not use old string annotations or informal tags.
2. **Required Fields**:
   - Define a `cell` defining the scope and density axes (e.g. using `cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE)`).
   - Configure execution metadata including `aggregation`, `test_method`, `se_method`, and any dependency `requires` mapping.

## Rules

- Commit messages via `cz commit`; description < 50 chars; no emoji; no AI co-author.
- Every new metric, result field, or API parameter needs a matching test in the same PR.
- Add WHY narrative to `CHANGELOG.md § [Unreleased]` in each PR; version bumps happen on release-train cadence, not per-PR.
- `uv run pytest` must be green before pushing.

For architecture decisions, submodule workflow, and release process → see the [full contributing guide](https://awwesomeman.github.io/factrix/latest/development/contributing/).
