---
title: Installation
---

factrix requires Python 3.12+. The core module depends only on `polars` and `numpy`.

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
uv sync --extra dev      # add pytest, commitizen, mkdocs
```

### Optional extras

| Extra | Adds |
|-------|------|
| `dev` | pytest, commitizen, mkdocs (toolchain) |
| `charts` | plotly |
| `all` | feature extras (`charts`, …); does **not** include `dev` |

For the full toolchain, combine them:

```bash
uv sync --all-extras --extra dev
```
