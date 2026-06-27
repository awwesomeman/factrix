# Contributing to factrix

GitHub-facing quick start. Canonical policy:
[docs/development/contributing.md](docs/development/contributing.md)
([published](https://awwesomeman.github.io/factrix/latest/development/contributing/)).

## Setup

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync --extra dev
python scripts/setup_dev.py
uv run pytest
```

For release checks, sync the docs toolchain too:

```bash
uv sync --frozen --extra dev --extra docs
```

## Development Cycle

```bash
git checkout -b feat/<short-desc>
# edit, test, commit
git add <specific-files>
cz commit
git push origin feat/<short-desc>
gh pr create
```

## Before Opening a PR

- Keep the change scoped and include tests for new metrics, result fields, or API parameters.
- Run `uv run pytest` locally.
- Use `cz commit` for Conventional Commits.
- Do not append commit signature trailers unless a future DCO policy explicitly
  requires them.
- For metrics, docs, release flow, hooks, changelog policy, and pre-1.0 rules,
  use the full contributing guide.
