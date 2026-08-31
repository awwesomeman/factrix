---
title: Contributing to factrix
---

This page covers the development workflow. Use the companion references for
[documentation conventions](documentation.md), the
[release process](release-process.md), and current
[architecture contracts](architecture.md).

## Development setup

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync --extra dev
python scripts/setup_dev.py
uv run pytest
```

`pyproject.toml` and `uv.lock` define the environment. Do not install packages
directly into `.venv`; use `uv add`, update the project metadata, and commit the
lockfile change together.

### Extras

| Command | Purpose |
|---|---|
| `uv sync` | Runtime dependencies only |
| `uv sync --extra dev` | Tests, lint, typing, and commit tooling |
| `uv sync --extra docs` | MkDocs and API-documentation tooling |
| `uv sync --extra pandas` | pandas/pyarrow input adapter |
| `uv sync --extra jupyter` | Notebook environment |
| `uv sync --frozen --all-extras` | Full CI or release verification |

The named `all` extra is the optional runtime bundle; it does not include the
development and docs toolchains. Use `--all-extras` when every declared extra
is required.

On Windows, enable UTF-8 for Markdown and diagnostic tests if the console does
not already do so:

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -m pytest -q -p no:faulthandler
```

`uv sync` installs factrix in editable mode. Restart a Jupyter kernel after
changing module imports, dataclasses, or module-level constants; autoreload is
not sufficient for those changes.

## Git hooks

`python scripts/setup_dev.py` installs the `pre-commit`, `commit-msg`, and
`pre-push` stages declared in `.pre-commit-config.yaml`. Installation is local
to each clone.

| Stage | Checks |
|---|---|
| `pre-commit` | Ruff lint and format checks for staged Python/notebook changes |
| `commit-msg` | Commit-message and contributor-trailer policy |
| `pre-push` | Strict docs build for docs-relevant paths, mypy for package paths, and release-note checks where applicable |

Run the lint stage directly with:

```bash
uv run pre-commit run --all-files --hook-stage pre-commit
```

When updating hook versions, run `uv run pre-commit autoupdate`. Keep the Ruff
hook revision equal to the `ruff==` development pin in `pyproject.toml`; a test
enforces the match.

## Development workflow

```bash
git switch main
git pull --ff-only origin main
git switch -c <type>/<short-description>

# edit and test
uv run pytest tests/test_<area>.py -v
uv run pytest

git add <specific-files>
cz commit
git push -u origin <type>/<short-description>
gh pr create
```

Use lowercase, hyphenated branches under `feat/`, `fix/`, `docs/`,
`refactor/`, or `chore/`. Do not commit directly to `main`.

Commits follow Conventional Commits. Keep the subject concise and use the body
for the reason and behavioural effect. Do not add AI co-author or
`Signed-off-by` trailers unless a future repository policy explicitly requires
them.

Open one pull request for one reviewable concern. The PR body should contain a
short summary, the verification performed, and `Closes #<issue>` when it
resolves an issue. Version bumps and tags follow the independent
[release process](release-process.md), not each merged PR.

## Testing rules

- Use synthetic fixtures from `tests/conftest.py` or `factrix.datasets`; tests
  do not read private or live market data.
- Add tests for new metrics, public result fields, parameters, error paths, and
  documented warnings.
- Characterise statistical size separately from ordinary unit behaviour. A
  reduced-replication guard can pin a measured band, while the larger sweep is
  documented in [Inference calibration](../reference/inference-calibration.md).
- Run focused tests while iterating, then the full suite before opening a PR.
- Do not bypass hooks or tests to make a branch appear green.

The baseline local checks are:

```bash
uv run pre-commit run --all-files --hook-stage pre-commit
uv run mypy factrix
uv run pytest
uv run pytest --doctest-modules factrix
uv run mkdocs build --strict
```

Use `uv sync --frozen --all-extras` before the complete run so optional adapter
tests do not execute against a partially installed pandas/pyarrow environment.

## Python and API changes

- Public functions and classes use type annotations and Google-style
  docstrings.
- `data` names a DataFrame-like input; reserve `df_*` for degrees of freedom.
- A new metric registers one `MetricSpec`; do not add a parallel applicability
  or routing table.
- A formal p-value and `alternative` must be published together through
  `MetricResult.p_value`; metadata must not duplicate the canonical p-value.
- Public metric names describe the statistic. Avoid `_test` suffixes, and add
  a method token when several estimators produce the same concept.

For the complete dispatch, result, guard, and naming invariants, see
[Architecture](architecture.md). For user-visible prose, citations, examples,
and generated pages, see [Documentation conventions](documentation.md).

## Design proposals

`architecture.md` describes the current state, not design history. Use a
GitHub issue for a proposed behavioural or architectural change, record the
decision and definition of done there, and link the implementing PR. Update the
architecture page only when the current contract changes.

The files under `docs/plans/archive/` are frozen historical plans and are not
published. Do not create a new plan file for ordinary development work; use an
issue unless the design genuinely requires a long-lived external artifact.

## Communication and language

Issues, pull requests, commit messages, changelog entries, package docstrings,
and published docs use English. Planning archives may remain bilingual because
they are excluded from the site.

Small decisions belong in the PR rationale. Invariant-level changes must also
update the relevant architecture section. If a decision changes a public
contract, state the migration path explicitly.

## Licensing

factrix is licensed under Apache-2.0. Contributions are licensed to the project
under the same terms unless the contributor states otherwise. Submit only code
you have the right to license, identify third-party sources and licences, and
disclose relevant patent claims. Strong-copyleft code is not accepted into the
main package because it would change downstream obligations.
