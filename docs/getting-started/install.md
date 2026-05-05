# Installation

`factrix` 支援直接從 GitHub 安裝，Core 模組只依賴 `polars + numpy`。

## 選項 A：使用 `uv`（推薦）

`uv` 是一個極速的 Python 封裝與環境管理工具。

```bash
uv venv --python 3.12 --clear
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install git+https://github.com/awwesomeman/factrix.git

# 指定版本（建議用於正式環境）
# uv pip install git+https://github.com/awwesomeman/factrix.git@v0.6.0
```

## 選項 B：使用 `conda`

```bash
conda create -n factrix python=3.12 -y
conda activate factrix
pip install git+https://github.com/awwesomeman/factrix.git
```

## 開發者安裝

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync                 # core dependencies
uv sync --extra dev     # + pytest, commitizen, mkdocs
```

Dependencies (`--extra`):

| Extra | Contents |
|-------|---------|
| `dev` | pytest, commitizen, mkdocs + plugins |
| `charts` | plotly |
| `all` (= `charts`) | feature extras only — does **not** include `dev` |

For the full toolchain: `uv sync --all-extras --extra dev`.
