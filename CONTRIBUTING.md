# Contributing to factrix

Quick-start guide. Full contributing reference: **[docs/development/contributing](https://awwesomeman.github.io/factrix/development/contributing/)**.

## Setup

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync --extra dev
git config core.hooksPath .githooks
uv run pytest    # must be green before committing
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

## Rules

- Commit messages via `cz commit`; description < 50 chars; no emoji; no AI co-author
- Every new metric / Profile field / API parameter needs a matching test in the same PR
- Add WHY narrative to `CHANGELOG.md § [Unreleased]` in each PR; version bumps happen on release-train cadence, not per-PR
- `uv run pytest` must be green before pushing

For architecture decisions, submodule workflow, and release process → see the [full contributing guide](https://awwesomeman.github.io/factrix/development/contributing/).
