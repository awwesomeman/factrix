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

```bash
pip install factrix==0.8.0
# or
uv add factrix==0.8.0
```

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
