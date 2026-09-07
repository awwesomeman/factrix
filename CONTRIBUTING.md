# Contributing to factrix

GitHub-facing quick start. Canonical policy:
[docs/development/contributing.md](docs/development/contributing.md)
([published](https://awwesomeman.github.io/factrix/latest/development/contributing/)).

## Setup

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync --extra dev
python scripts/setup_dev.py   # installs the pre-commit framework hooks
uv run pytest
```

`--extra dev` is enough for `uv run pytest` to pass; the docs-site tests skip
themselves, with a reason, until the `docs` extra is installed. Use
`uv sync --frozen --all-extras` (what CI runs) to make the docs and pandas
checks run rather than skip, and `uv run pytest -rs` to see every skip reason.

## Development cycle

```bash
git checkout -b feat/<short-desc>
# edit, test, commit
git add <specific-files>
cz commit
git push origin feat/<short-desc>
gh pr create
```

## Before opening a PR

- Keep the change scoped and include tests for new metrics, result fields, or API parameters.
- Run `uv run pytest` locally, on an `--all-extras` environment so no check
  silently skips.
- Use `cz commit` for Conventional Commits.
- Do not append commit signature trailers unless a future DCO policy explicitly
  requires them.
- Follow the [full contributing guide](docs/development/contributing.md) for
  testing, documentation, hooks, release flow, and project conventions.
