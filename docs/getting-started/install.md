# Installation

factrix installs directly from GitHub. The core module depends only on
`polars` and `numpy`.

## With `uv` (recommended)

```bash
uv venv --python 3.12 --clear
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install git+https://github.com/awwesomeman/factrix.git

# Pin to a release tag for production
# uv pip install git+https://github.com/awwesomeman/factrix.git@v0.6.0
```

## With `pip` / `conda`

```bash
conda create -n factrix python=3.12 -y
conda activate factrix
pip install git+https://github.com/awwesomeman/factrix.git
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
