# Plan: Extract factrix to Standalone Repo

**Date**: 2026-04-20
**Status**: Proposed — pending execution
**Driver**: factrix 已達 v3.3.0、533 tests 全綠、無未決 refactor——是自然切點。
**Review passes applied**: 2026-04-20 quant + backend senior review（7 必改 + 4 應加 + 4 可考慮）。

---

## 目標

1. 把 `factrix/` 拆成**獨立 repo**，具備套件該有的完整形態（src + tests + pyproject + README + LICENSE + CI + CHANGELOG）。
2. 本 workspace 轉型為 **factor-research consumer**：保留研究資料、下游規格（個股評分等）、研究 notebook，透過 **git submodule + editable install** 使用 factrix。
3. 保留**動態更新能力**：改 factrix 立即生效；同時讓 workspace 的每個 commit 都**記錄當下使用的 factrix SHA**（可重現）。

## 非目標

- 上 PyPI（本階段僅需 git-based install）
- 搬移 `docs/`（全部留在 workspace——詳見下方歸屬表）
- Factorlib 對外公開

## 明確 Out-of-scope（但列為 follow-up）

- **研究資料版本化（Data Versioning）**：`sample_data.parquet`、`tw_stock_daily_*.parquet`、`mlflow.db`、`mlruns/` 目前都在 `.gitignore`，workspace 沒有資料版本。6 個月後要重現某個 2026-04-20 的研究結果：factrix SHA 有解（submodule），**資料沒解**。本計畫不處理，但完成後應盡快評估 DVC / Git LFS / 外部 S3+hash 三選一。（quant 側硬洞）
- **Notebook stored outputs 生命週期**：factrix BC change 會讓 stored output 過時（T8 demo execution 未解、上週 `q1_q5_spread → long_short_spread` 重現）。本計畫遷移後仍存在。workspace 應加 nbstripout pre-commit hook 或明定「notebook 結果以最新 `bump factrix SHA` 重跑為準」的 convention。

---

## 最終架構

```
~/Desktop/dst/code/
├── factrix/                         # 新獨立 repo (git@github.com:<you>/factrix.git)
│   ├── README.md                      # 從 factrix/README.md 拉上來當根 README
│   ├── LICENSE                        # MIT（Phase 3 新增）
│   ├── CHANGELOG.md                   # v3.3.0 初版（Phase 3 新增）
│   ├── ARCHITECTURE.md                # 3 頁 current-state snapshot（Phase 3 新增）
│   ├── pyproject.toml                 # name = "factrix"
│   ├── uv.lock                        # Phase 2 regenerated
│   ├── .gitignore
│   ├── .github/workflows/test.yml     # pytest CI（Phase 3 必須，非可選）
│   ├── factrix/                     # 套件本體（維持 flat layout 不動）
│   ├── tests/                         # 從 workspace 搬進來
│   └── examples/                      # 從 experiments/ 搬進來
│       ├── demo.ipynb
│       ├── benchmark_compact.py
│       ├── benchmark_two_stage_screening.py
│       ├── benchmark_evaluate_batch_cpu.py
│       ├── dogfood_with_extra_columns.py
│       └── _build_demo.py
│
└── factor-analysis/                   # 本 workspace（繼續存在，改用 submodule）
    ├── README.md                      # 重寫為 research workspace 地圖
    ├── pyproject.toml                 # 改 name，把 factrix 設成 local editable dep
    ├── .gitignore
    ├── scripts/check_submodule.sh     # fail-fast 檢查 submodule 已 init
    ├── external/
    │   └── factrix/                 # git submodule，指向 factrix repo 某個 SHA
    ├── docs/                          # 全部保留 + 批改舊 SHA 引用（Phase 4）
    ├── experiments/
    │   └── explore_price_intention.py
    ├── sample_data.parquet            # gitignored
    ├── tw_stock_daily_2017_2025.parquet  # gitignored
    ├── mlflow.db                      # gitignored
    └── mlruns/                        # gitignored
```

---

## 檔案歸屬表

| 類別 | 路徑 | 去向 | 備註 |
|------|------|------|------|
| 套件原始碼 | `factrix/**` | **factrix repo** | 整包搬 |
| 測試 | `tests/**` | **factrix repo** | 包含 profiles/、stats/ 子目錄 |
| 打包 | `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md` | **factrix repo** | pyproject 改 name；uv.lock 在 Phase 2 regenerated |
| Library demo | `experiments/demo.ipynb`, `experiments/_build_demo.py`, `experiments/benchmark_*.py`, `experiments/dogfood_with_extra_columns.py` | **factrix repo** → `examples/` | 示範套件 API 用 |
| 研究探索 | `experiments/explore_price_intention.py` | workspace | 操作真實資料 |
| 設計文件 | `docs/spike_*.md`, `docs/refactor_*.md`, `docs/plan_*.md`, `docs/gate_redesign*.md`, `docs/naming_convention.md`, `docs/individual_stock_scoring_*.md`, `docs/study/` | **workspace** | 設計過程紀錄 = 研究筆記 |
| 待裁決 | `docs/factor_screening*.md`, `docs/factor_screening_gemini.md` | **pending** | Phase 1 必須決策：如果是 factrix 內建 screening feature 規劃 → factrix；如果是 workspace 研究流程 → workspace |
| 研究資料 | `*.parquet`, `mlflow.db`, `mlruns/` | workspace | 已 gitignored；版本化是 follow-up |

**邊界裁決**：
- `docs/naming_convention.md` 屬 API contract 規則，但仍留 workspace。若 factrix 日後公開，將核心規則**抄寫**到 `factrix/README.md` 的 "Naming Rules" 區塊或新的 `ARCHITECTURE.md`，不搬原文件。
- `docs/study/` 目前 untracked——Phase 0 必須決策是否開始追蹤（workspace 既然要定調為研究實驗室，建議追蹤）。

---

## Phase 0：前置（workspace 乾淨化 + 防爆驗證）

**目的**：不要在髒或未知的工作樹上 split；驗證關鍵假設不成立會直接停工。

### P0.1 工作樹乾淨化

1. 處理當前 M 狀態檔：
   - `factrix/_api.py`, `factrix/charts/quantile.py`, `factrix/config.py`, 多個 `factrix/metrics/*.py`, `tests/conftest.py`, `tests/test_factor_session.py`——commit 或 stash 完
2. 處理 D 狀態：`docs/literature_references.md`——確認是留是刪
3. 處理 untracked：`docs/study/`——**決策並執行**：追蹤（`git add docs/study/`）或加入 `.gitignore`

### P0.2 關鍵假設驗證（potential showstopper）

**測試不得綁 workspace 真實資料**——否則 filter-repo 後 tests 全爆：

```bash
grep -rn "sample_data\|tw_stock_daily\|\.parquet" tests/ factrix/ \
  | grep -v "__pycache__\|\.pyc"
```

**預期結果**：零 match，或只有 fixture 內部 synthetic 構造的 parquet（如 `tmp_path / "foo.parquet"`）。如果發現 test load 真實資料檔，**Phase 0 就必須先切 fixture synthetic 化**，不然後面全白做。

### P0.3 Secret scan（filter-repo 會搬歷史，防外洩）

```bash
git log --all -p \
  | grep -iE "api[_-]?key|secret|token|password|bearer|aws_access" \
  | head -20
```

如有真 secret 出現過，評估：rewrite history 移除（BFG / filter-repo `--replace-text`）後再進 Phase 2，或接受風險但記錄 follow-up。

### P0.4 保險點

```bash
git push origin main                   # 把 136 commits 推上去
git tag pre-extraction-backup
git push origin pre-extraction-backup  # 命名錨點
uv run pytest                          # 533 tests 全綠確認
```

**驗證**：`git status` 乾淨、`git log origin/main..main` 空、關鍵 grep 無 match、pre-extraction-backup tag 在 origin。

---

## Phase 1：歸屬決策確認

- [ ] 逐行確認歸屬表
- [ ] **裁決 `docs/factor_screening*.md`**：是 factrix 內建 feature 規劃還是 workspace 研究流程？
- [ ] 確認 `docs/naming_convention.md` 留 workspace（且同意 Phase 3 寫入 factrix README / ARCHITECTURE）
- [ ] 確認 `docs/study/` 追蹤策略

---

## Phase 2：用 git-filter-repo 提取歷史

**工具**：`git-filter-repo`（`pip install git-filter-repo`）。

### P2.1 執行 filter-repo

```bash
cd ~/Desktop/dst/code
git clone factor-analysis/ factrix-extract/
cd factrix-extract/

git filter-repo \
  --path factrix/ \
  --path tests/ \
  --path experiments/demo.ipynb \
  --path experiments/_build_demo.py \
  --path experiments/dogfood_with_extra_columns.py \
  --path-glob 'experiments/benchmark_*.py' \
  --path pyproject.toml \
  --path uv.lock \
  --path .gitignore
```

### P2.2 **保留 SHA 映射**（critical）

filter-repo 會**重寫所有 commit SHA**——記憶與 `docs/` 裡引用的 `d83f67c` / `8f15db8` / `3d7ac0a` 等 SHA 在新 repo 不存在。必須保留對照表：

```bash
cp .git/filter-repo/commit-map ~/Desktop/dst/code/factrix-sha-map.txt
```

格式：每行 `<old_sha> <new_sha>`。Phase 4 用它批改舊引用。

### P2.3 結構調整（在 factrix-extract/ 內）

```bash
# 1. experiments/ → examples/
git mv experiments examples

# 2. factrix/README.md 拉到根當主 README
git mv factrix/README.md README.md

# 3. pyproject.toml 改 name: "factor-analysis" → "factrix"
# （手改 pyproject.toml）

git add -A
git commit -m "chore: rename package factor-analysis→factrix, restructure examples/"
```

### P2.4 **重建 uv.lock**（name 改了，舊 lock 會失效）

```bash
rm uv.lock
uv lock
git add uv.lock
git commit -m "chore: regenerate uv.lock after package rename"
```

### P2.5 驗證

```bash
uv sync
uv run pytest                          # 533 tests 必須全綠
uv run python -c "import factrix as fl; print(fl.__version__)"
```

**退回機制**：Phase 2 全程在 `factrix-extract/` 獨立目錄；失敗 `rm -rf factrix-extract/` 從 `pre-extraction-backup` tag 重 clone 重做。

---

## Phase 3：建立 GitHub repo + 套件衛生

### P3.1 建 repo + 推送

1. GitHub 建立空 repo：`<you>/factrix`
2. 推送：
   ```bash
   cd factrix-extract/
   git remote remove origin
   git remote add origin git@github.com:<you>/factrix.git
   git push -u origin main
   ```

### P3.2 **必要**：加 LICENSE（non-negotiable）

```bash
# 選 MIT 或 Apache-2.0；以 MIT 為例
curl -o LICENSE https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt
# 手改填入年份與 holder
git add LICENSE
git commit -m "chore: add MIT license"
```

### P3.3 **必要**：寫 CHANGELOG.md + 定 SemVer policy

```markdown
# Changelog

All notable changes follow [SemVer](https://semver.org/):
- **MAJOR**: API-breaking changes (field rename, signature change, removed metric)
- **MINOR**: New metrics, new Profile fields, new optional params
- **PATCH**: Bug fixes, docstring/test fixes, internal refactor

## [3.3.0] - 2026-04-20
### Added
- Initial extraction from factor-analysis workspace
- Four Factor session subclasses (Cross-sectional, Event, MacroPanel, MacroCommon)
- ... (parity with current state)
```

factrix 的 BC change 對下游 research 殺傷力大（例：最近 `q1_q5_spread → long_short_spread`），SemVer 與 CHANGELOG 是 extraction 的必要基礎設施。

### P3.4 **必要**：ARCHITECTURE.md（3 頁 current-state snapshot）

factrix repo 若完全沒有 docs/，cold reader（含 AI agent）對設計一頭霧水。寫一份**現況描述**（不是搬歷史）：

- 四種 Factor 類型與對應 Profile dataclass（`CrossSectionalProfile` / `EventProfile` / `MacroPanelProfile` / `MacroCommonProfile`）
- `fl.evaluate` / `fl.preprocess` / `fl.factor()` 契約
- `Factor` session 的 cache key 設計
- Level 2 metrics 接入方式（`return_artifacts=True` → `arts.metric_outputs`）
- Profile verdict 是二元 PASS/FAILED（CAUTION 刻意移除）
- Invariants（從 `project_profile_refactor_status.md` 的 invariants section 翻寫）

### P3.5 **必要**：CI（從「可選」改「必須」）

533 tests 的獨立套件，沒 CI = 隨時回歸 = 等於沒 extract。建立 `.github/workflows/test.yml`：

```yaml
name: test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest
```

### P3.6 Tag + 推送

```bash
git tag v3.3.0 -m "Initial extraction from factor-analysis workspace"
git push origin main
git push --tags
```

### P3.7 移到最終位置

```bash
mv ~/Desktop/dst/code/factrix-extract ~/Desktop/dst/code/factrix
```

---

## Phase 4：Workspace 改造成 consumer（**先加 → 再驗 → 後刪**）

原計畫「先刪後驗」有回退風險。改為保守順序：

### P4.1 加 submodule + 寫新 pyproject（不刪舊 factrix/）

```bash
cd ~/Desktop/dst/code/factor-analysis/

git submodule add git@github.com:<you>/factrix.git external/factrix

# 寫新 pyproject.toml（見下方 snippet）
# 寫新 README.md
```

新的 workspace `pyproject.toml`：

```toml
[project]
name = "factor-research"
version = "0.1.0"
description = "Factor research workspace consuming factrix"
requires-python = ">=3.12"
dependencies = [
    "factrix",
    "mlflow>=3.10",
    "streamlit>=1.55",
    "polars>=1.38",
]

[tool.uv.sources]
factrix = { path = "external/factrix", editable = true }

[dependency-groups]
dev = [
    "jupyter",
    "nbconvert>=7.17",
]
```

新 workspace `scripts/check_submodule.sh`（fail-fast）：

```bash
#!/usr/bin/env bash
if [ ! -f external/factrix/pyproject.toml ]; then
  echo "ERROR: external/factrix is empty. Run: git submodule update --init --recursive" >&2
  exit 1
fi
```

新 workspace `README.md` 骨架：
- 粗體警示 `git clone --recursive` 必須
- 研究專案定位（factor research workspace）
- 資料集說明（台股 daily 2017-2025）
- `external/factrix/` 指向套件本體（link to factrix repo）
- 下游系統入口（個股評分 v3、gate redesign、etc.）

### P4.2 **先驗再刪**：冒煙測試

```bash
uv sync
bash scripts/check_submodule.sh
uv run python -c "import factrix as fl; print(fl.__version__, fl.__file__)"
# 確認 fl.__file__ 指到 external/factrix/factrix/__init__.py
```

挑一個最常用的 research notebook 或 script 跑過——確認 editable import 成功、沒有 import path shadowing（top-level `factrix/` 還在所以可能有衝突，下一步就清掉）。

### P4.3 刪除已遷出的檔案

```bash
git rm -r factrix/ tests/
git rm experiments/demo.ipynb experiments/_build_demo.py \
       experiments/dogfood_with_extra_columns.py \
       experiments/benchmark_*.py

# 清理可能殘留的 egg-info（理論上 gitignored，但以防萬一）
rm -rf factor_analysis.egg-info/ .pytest_cache/
```

再跑一次驗證：

```bash
uv sync
uv run python -c "import factrix as fl; print(fl.__file__)"
# 確認仍指到 external/factrix/，沒有因為頂層 factrix/ 消失而壞掉
```

### P4.4 批改舊 SHA 引用

用 P2.2 保留的 `factrix-sha-map.txt` 批改 workspace 內的舊 SHA：

```bash
# 寫個小 script 讀 commit-map，對每個 <old> <new> 跑 sed
# 目標檔案：docs/*.md、~/.claude/projects/.../memory/*.md
while read old new; do
  grep -rl "$old" docs/ | xargs sed -i.bak "s/$old/$new/g"
done < ~/Desktop/dst/code/factrix-sha-map.txt
find docs/ -name "*.bak" -delete
```

> **注意**：記憶檔（`~/.claude/projects/.../memory/`）路徑不在 workspace 內，手動開啟確認是否也要批改。

### P4.5 Commit workspace 轉型

```bash
git add .
git commit -m "refactor(workspace): extract factrix to submodule, rewrite pyproject/README"
```

---

## Phase 5：端到端驗證

### P5.1 模擬 fresh clone

```bash
cd /tmp
git clone --recursive git@github.com:<you>/factor-research.git test-workspace
cd test-workspace
bash scripts/check_submodule.sh
uv sync
uv run python -c "import factrix as fl; print(fl.__version__)"
```

**也測一次忘記 `--recursive` 的情境**，確認 `check_submodule.sh` 給出可理解的錯誤訊息。

### P5.2 **mlflow 自動 tag factrix SHA**（研究可重現性必備）

以下 snippet 放在 workspace 的 research helper 或 notebook header：

```python
import subprocess
import mlflow

def tag_factrix_sha():
    """Log current factrix commit SHA so runs can be traced to library version."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", "external/factrix", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        mlflow.set_tag("factrix_sha", sha)
        mlflow.set_tag("factrix_version", __import__("factrix").__version__)
    except (subprocess.CalledProcessError, FileNotFoundError):
        mlflow.set_tag("factrix_sha", "unknown")

# 在 research notebook 開始時呼叫
with mlflow.start_run():
    tag_factrix_sha()
    # ... research ...
```

沒這個 tag，`mlruns/` 裡的 experiment 無法對應回 factrix 版本，code 可重現但結果來源不明。

### P5.3 驗證動態更新流程

```bash
# 改 factrix 本體
cd external/factrix
# ... edit factrix/_api.py ...
uv run pytest                          # factrix 自己的 test suite 全綠
git commit -am "feat: ..."
git push

# 回到 workspace，bump submodule pointer
cd ../..
git add external/factrix             # 記錄新的 SHA
git commit -m "chore: bump factrix to <short-sha>: <why>"
```

### P5.4 驗證清單

- [ ] `import factrix` 成功，`fl.__file__` 指到 `external/factrix/factrix/__init__.py`
- [ ] editable install：改 factrix 源碼後 workspace 立即看到（Jupyter 需 restart kernel）
- [ ] `external/factrix` 內 `uv run pytest` 533 tests 全綠
- [ ] factrix repo 的 GitHub Actions CI 綠燈
- [ ] workspace 挑 1-2 個最常用 research notebook/script 能正常執行
- [ ] workspace `git log` 顯示 submodule bump commit 帶有 factrix SHA
- [ ] fresh clone `--recursive` + `uv sync` 成功
- [ ] `scripts/check_submodule.sh` 在未 init 情境給出明確錯誤
- [ ] mlflow run tag 能讀到 factrix_sha

---

## 風險與退回機制

| 風險 | 影響 | 緩解 | 偵測 |
|------|------|------|------|
| filter-repo 漏掉路徑 | 歷史不完整 | Phase 0 push + `pre-extraction-backup` tag——隨時可重做 | Phase 2 驗證 `uv run pytest` |
| **tests 綁真實 parquet**（showstopper） | filter-repo 後 tests 全爆 | **Phase 0.2 grep 驗證**，發現就先切 synthetic | P0.2 |
| pyproject rename 後 import 異常 | 套件 load 失敗 | Phase 2.4 重建 uv.lock | P2.5 `import factrix` |
| **filter-repo 重寫 SHA** | 舊 SHA 引用全失效 | P2.2 保留 commit-map，P4.4 批改舊引用 | P4.4 後 grep 舊 SHA 應為 0 |
| workspace notebook 找不到 factrix | 研究流程中斷 | P4.2 先驗再刪 + Phase 5 fresh clone 驗證 | P5.1 |
| submodule UX 失敗 | 新機器裝不起來 | `scripts/check_submodule.sh` + README 粗體警示 | P5.1 模擬無 `--recursive` |
| factrix BC 打壞 workspace | 研究結果不可重現 | submodule SHA pin + SemVer + CHANGELOG | 正常 dev cycle |
| mlflow run 無法對應 factrix 版本 | 研究可重現性半壞 | P5.2 自動 tag factrix_sha | mlflow UI 看 run tag |
| **資料未版本化** | 6 個月後結果不可重現 | **out-of-scope，follow-up 處理**（DVC / Git LFS / S3） | 本計畫不涵蓋 |
| secret 隨歷史外洩 | 獨立 repo 將來公開風險 | P0.3 pre-scan | P0.3 grep 結果 |
| notebook stored outputs 過時 | 下游讀者看到錯誤結果 | out-of-scope，follow-up 處理（nbstripout） | 本計畫不涵蓋 |

**全案退回**：Phase 2/3 失敗 → 刪 `factrix-extract/`、從 `pre-extraction-backup` tag 重 clone。Phase 4 失敗 → `git reset --hard` 回 Phase 4 前、`git submodule deinit external/factrix`（舊 factrix/ 還在所以可直接用）。

---

## 預估時程

- Phase 0：**1 小時**（grep 驗證 + secret scan + 清未 commit 改動 + push）
- Phase 1：15 分鐘（確認歸屬表 + factor_screening 裁決）
- Phase 2：45 分鐘（filter-repo + SHA map 備份 + 結構調整 + uv lock 重建 + pytest）
- Phase 3：**1 小時**（LICENSE + CHANGELOG + ARCHITECTURE.md + CI workflow + push + tag）
- Phase 4：1 小時（submodule + pyproject + README + 冒煙測試 + 刪檔 + SHA 批改）
- Phase 5：30 分鐘（fresh clone + mlflow snippet + 驗證清單走一遍）

**總計**：約 4.5 小時（Review pass 後從 2 小時翻倍，主因是套件衛生——LICENSE/CHANGELOG/ARCHITECTURE/CI 從「可選」變「必要」）。

---

## 下一步

今天先執行 **Phase 0**（重點是 P0.2 驗證 tests 無真實資料綁定、P0.3 secret scan、P0.4 保險點）。Phase 0 完全清潔後再進 Phase 2。
