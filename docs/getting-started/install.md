---
title: Installation
---

factrix requires Python 3.12+. The core package depends only on `polars`, `numpy` and `scipy`.

## With `pip`

```bash
pip install factrix
```

## With `uv` (recommended)

```bash
uv add factrix
```

## Version pinning

factrix is pre-1.0 (v0.x.x) and the public API may break on MINOR bumps; pin a specific version in long-running projects.

```bash
pip install factrix==X.Y.Z
# or
uv add factrix==X.Y.Z
```

Replace `X.Y.Z` with the [latest release tag](https://github.com/awwesomeman/factrix/releases).

## Local development

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync                  # core dependencies
uv sync --extra dev      # add pytest, ruff, mypy, commitizen
```

### Optional extras

| Extra | Adds |
|-------|------|
| `pandas` | pandas + pyarrow, so `factrix.adapt` accepts pandas input |
| `jupyter` | jupyter / jupyterlab / ipywidgets for the example notebooks |
| `dev` | pytest, ruff, mypy, commitizen, pre-commit (toolchain) |
| `docs` | mkdocs-material, mkdocstrings, mike (build the site) |
| `all` | `jupyter` only; does **not** include `dev` or `docs` |

For every declared extra:

```bash
uv sync --all-extras
```

See [Contributing](../development/contributing.md) for the development setup.
