# Contributing to factorlib

本文件描述 factorlib 的開發流程。主要讀者是**作者本人 + 未來的 AI agent**
——private repo，所以省略 licensing / DCO / CLA 等 OSS 慣例，聚焦在**實際
開發模式與陷阱**。

---

## 1. 兩種開發模式

### Mode A — Standalone development（推薦多數情況）

直接 clone factorlib repo，獨立 venv，cycle 最快：

```bash
git clone https://github.com/awwesomeman/factorlib.git
cd factorlib
uv sync
uv run pytest        # 驗證 baseline 綠
```

**什麼時候用**：新 feature / 大 refactor / SemVer bump / release。
隔離環境、不會誤動到下游 research、測試 cycle 最短。

### Mode B — In-workspace development（via submodule）

從下游 workspace（`factor-analysis`）的 `external/factorlib/` 編輯——
可以邊改 factorlib、邊在真實 research notebook 看效果：

```bash
cd ~/Desktop/dst/code/factor-analysis
cd external/factorlib
# edit factorlib source
uv run pytest
# 回 workspace 跑 notebook 驗證 end-to-end
cd ../..
uv run jupyter notebook
```

**什麼時候用**：
- Debug 「只在 research 環境出現的 bug」——必須真實資料 context 才重現
- 為某個已知下游需求改 API，想即時驗證
- 小範圍 tweak（< 10 行），不值得 Mode A 的 setup overhead

**注意**：Mode B 有三個關鍵陷阱，見下方 §4。

---

## 2. 環境設置

factorlib 用 **uv** 管理 Python venv 與 lockfile。`pyproject.toml` 與
`uv.lock` 是唯一權威——不要用 `pip install` 直接裝任何東西進 `.venv/`。

```bash
uv sync              # 依 lock 安裝
uv run <cmd>         # 在 venv 內執行（自動 sync 最新 lock）
uv add <pkg>         # 新增 dep，同步更新 pyproject + uv.lock
uv lock --upgrade    # 升 lock 到當前 pyproject constraint 的最新版
```

Python 版本鎖定在 **3.12+**（`pyproject.toml` 的 `requires-python`）。

---

## 3. 開發循環（branch → test → commit → push → PR）

```bash
# 1. 從 main 開 branch（不要直接在 main 上 commit）
git checkout main && git pull
git checkout -b <type>/<short-desc>     # e.g. feat/redundancy-heatmap

# 2. 開發 + 頻繁跑測試
# ...edit...
uv run pytest                            # 必須全綠才 commit
uv run pytest tests/test_<file>.py -v   # 聚焦單一 module 快速迭代

# 3. Commit（conventional commits + signed）
git add <specific-files>                 # 不用 -A
git commit -s -m "feat(metric): add redundancy heatmap aggregation"

# 4. Push + 開 PR
git push origin feat/redundancy-heatmap
gh pr create --title "..." --body "..."

# 5. CI 綠後 merge（目前是 solo → squash-merge 或 rebase-merge）
gh pr merge --squash
```

### Branch 命名

`<type>/<short-desc>`，全小寫連字號：

- `feat/...` — 新 feature
- `fix/...` — bug fix
- `refactor/...` — 重構（行為不變）
- `docs/...` — 文件
- `chore/...` — 打包 / CI / lock 維護

### Commit message

Conventional commits 格式：

```
<type>(<scope>): <imperative description, lowercase, no period>

<optional body — only if context is non-obvious; ≤5 lines>

Signed-off-by: <name> <email>
```

**規則**：
- Subject ≤72 字
- Body 只寫「為什麼做」+「主要做了什麼」——不複述 diff
- 禁用 AI co-author 署名、emoji、句末句號
- 必須 `-s` 附 Signed-off-by

---

## 4. Mode B 的三個關鍵陷阱

### G1. Submodule = detached HEAD（最大的坑）

從 workspace 進 `external/factorlib/` 時，submodule 預設是 **detached HEAD**。
在 detached HEAD 上 commit 的 commits **不屬於任何 branch**——下次
`git submodule update --remote` 會直接覆蓋，commit 消失（reflog 不記）。

```bash
# 丟 commit 的寫法（錯誤示範）
cd external/factorlib
# ...edit...
git commit -am "feat: xxx"              # detached — commit 沒家
git submodule update --remote           # GONE

# 正確：先 branch
cd external/factorlib
git checkout -b feat/xxx                # 建 branch
# ...edit...
git commit -am "feat: xxx"
git push origin feat/xxx
gh pr create                            # 在 factorlib repo 開 PR
```

### G2. Editable install 對 Jupyter 有限制

Workspace 的 `uv sync` 把 factorlib 裝成 editable，所以改 submodule
source 會立刻反映到 `import factorlib`——但 Jupyter kernel 有例外：

- Function body 改動 → `%autoreload 2` 能抓到
- `__init__.py` 的 imports / dataclass 定義 / module-level 常數 → 必須 **restart kernel**

不確定時 restart kernel 是最穩的選擇。

### G3. Workspace 不會自動追蹤 factorlib 更新

factorlib main 合了 PR 之後，workspace 的 submodule 指針**不會自動 bump**。
這是 feature 不是 bug——workspace 的每個 commit 綁定明確的 factorlib SHA，
研究結果可重現。

手動 bump：

```bash
cd external/factorlib && git fetch && git checkout main && git pull
cd ../.. && git add external/factorlib
git commit -m "chore: bump factorlib to <short-sha>: <why>"
```

---

## 5. Submodule sync reference

Mode B 與 consumer workspace 日常操作的指令索引。90% 情境看 cheat sheet
就夠，看不懂再往下讀心智模型與情境說明。

### 5.1 Cheat sheet

| 想做的事 | 指令 |
|---------|------|
| 看 workspace pin 的 SHA | `git submodule status` |
| 看 submodule actual HEAD | `cd external/factorlib && git rev-parse --short HEAD` |
| 拉 factorlib main 最新到 actual | `git submodule update --remote` |
| 切到特定 tag | `cd external/factorlib && git fetch --tags && git checkout vX.Y.Z` |
| 固化 actual 到 pin | `git add external/factorlib && git commit -m "chore: bump..."` |
| 放棄 actual 改動，reset 回 pin | `git submodule update` |
| Fresh clone 初始化 submodule | `git submodule update --init --recursive` |

### 5.2 心智模型

workspace 有 `pin`（記在 workspace commit 裡的 SHA），submodule 自己
有 `actual HEAD`（實際 checkout 的 SHA）。兩者可能不同——`git status`
會顯示 submodule 是 `modified`。**Python editable install 讀的是 actual
HEAD**，所以 import 拿到的版本取決於 actual，不是 pin。

### 5.3 常見情境

- **自己改 factorlib 要 sync 回 workspace**：走 §3 + §4（dev 流程），完成後
  cheat sheet 第 5 列固化
- **別人 push 了 factorlib，我要跟上**：cheat sheet 第 3 列 + 第 5 列
- **Pin 到 tag（推薦，見 §8）**：cheat sheet 第 4 列 + 第 5 列
- **切 branch 後 submodule 亂掉**：cheat sheet 第 6 列 reset

### 5.4 搞不清楚時先做兩件事

1. `git submodule status` 看 pin
2. `cd external/factorlib && git rev-parse HEAD` 看 actual

兩個 SHA 一樣 = 乾淨；不一樣就決定：要 bump pin（第 5 列）還是 reset
actual（第 6 列）。

---

## 6. 測試規範

### Synthetic fixtures only

**factorlib tests 不得 load 真實市場資料**——所有 fixture 必須是
`tests/conftest.py` 或 test module 內 numpy / polars 程式化構造。
這個 invariant 讓 tests 可以在任何環境跑（CI / 新機器 / fresh clone），
也避免 repo 被資料污染。

Pattern（見 `tests/conftest.py`）：

```python
@pytest.fixture
def clean_panel():
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "date": [...],
        "asset_id": [...],
        "factor_value": rng.normal(size=n),
        "forward_return": rng.normal(size=n) * 0.01,
    })
```

### 新 feature 必帶測試

新的 metric / Profile field / API parameter → 對應的 test case 必須同一個
PR 加入。PR reviewer（現在是你 + Claude）應 block PR 如果沒測試。

### CI 必綠

`.github/workflows/test.yml` 在每次 push / PR 跑 full pytest。**CI 紅的
PR 不該 merge**——先 fix 再繼續。

### Metric docstring 風格

`factorlib/metrics/*.py` 的 docstring 是各 metric 「精確公式 / 演算法」的
**authoritative source**。文件整體分工（what goes where）見 README 的
「下一步看哪裡」表 — 單一來源，此處不複述。格式：

1. **第一行 tldr** — 一句話說明這個 metric 量什麼，可含簡短公式
   （例：`IC_IR = mean(IC) / std(IC).`）
2. **公式**：
   - **單行公式** → inline 帶過，格式 `value = <expr>`
   - **多行 algorithm / sandwich SE / 非平凡演算法** → `Formula:` block +
     縮排等式，讓掃 `help()` 的人可以讀 display math
3. **Args / Returns** block 保留（Google-style）
4. **Short-circuit 條件**最後一段說明（哪些 input 會短路成 NaN、
   對應 `metadata["reason"]` 是什麼）
5. 引用論文走 `References:` block（選配；複雜方法才需要，簡單診斷
   指標不用）

範例（inline 公式）：`ts_beta.ts_beta_sign_consistency`
範例（Formula block）：`fama_macbeth.pooled_ols`、`_helpers._sample_non_overlapping`

---

## 7. SemVer + CHANGELOG

目前 factorlib 在 **pre-1.0**（v0.x.x）——公開 API **可能在 MINOR bump 裡
BC 變動**。Consumer（如 `factor-analysis` workspace）應該透過 **git
submodule SHA pin** 而非 version range，直到 1.0.0 穩定為止。

### 版本 bump 規則（在 pre-1.0 內）

- **PATCH** (0.1.0 → 0.1.1)：bug fix、docstring、internal refactor
- **MINOR** (0.1.0 → 0.2.0)：新 feature、**或** BC change（rename、
  signature change、removed metric）
- **MAJOR** (0.x → 1.0.0)：宣告 API 穩定（此後 BC 必須 bump major）

### 每次 bump 必做

1. 更新 `pyproject.toml` 的 `version`
2. 更新 `factorlib/__init__.py` 的 `__version__`
3. 在 `CHANGELOG.md` 加新 section，列 `Added` / `Changed` / `Deprecated`
   / `Removed` / `Fixed`
4. Commit `chore: bump version to <new>`
5. Tag：`git tag v<new> -m "..."` + `git push origin v<new>`

### BC change 特別注意

例：`q1_q5_spread → long_short_spread`（已於 workspace 歷史發生過）
這種 rename 屬 BC change，pre-1.0 走 MINOR bump，但 **CHANGELOG 要
明確列出 migration path**（舊名 → 新名、影響哪些欄位 / field /
cache key），因為下游 workspace 可能有 notebook / script 用到舊名。

---

## 8. Release workflow

Release = 「讓 workspace 能 bump 到一個乾淨、tagged 的 factorlib 版本」。

```bash
# 1. 在 factorlib repo main 分支
git checkout main && git pull

# 2. 驗證 CI 綠 + 本地 pytest 全綠
uv run pytest

# 3. bump version（見上方 §7）
# edit pyproject.toml + factorlib/__init__.py + CHANGELOG.md
git add pyproject.toml factorlib/__init__.py CHANGELOG.md
git commit -s -m "chore: bump version to 0.2.0"

# 4. tag + push
git tag v0.2.0 -m "Release 0.2.0 — <one-line summary>"
git push origin main
git push origin v0.2.0

# 5. 回 workspace bump submodule
cd ~/Desktop/dst/code/factor-analysis
cd external/factorlib && git fetch && git checkout v0.2.0
cd ../.. && git add external/factorlib
git commit -m "chore: bump factorlib to v0.2.0"
git push
```

### Workspace 不跟最新 main，只跟 tag

下游 research workspace 原則上 **pin 到 tag**（而非 main HEAD），這樣
每個 workspace commit 對應一個清楚的 factorlib 版本、可重現。

Main HEAD 只在 Mode B 開發中暫時用（debug 流程），完成後 merge 回
factorlib main、打 tag，再讓 workspace bump 到 tag。

---

## 9. 架構 / 設計決策

新 feature 或大改動前建議先讀：

- `ARCHITECTURE.md` — 套件現況 snapshot（positioning、public API、
  4 種 factor types、Profile contract、Artifacts、invariants）
- `CHANGELOG.md` — 歷史 BC changes + caveats

`ARCHITECTURE.md` 是**當下狀態**描述，不是設計歷程。如果要看「為什麼
設計成這樣」的過程紀錄，去 `awwesomeman/factor-analysis` workspace 的
`docs/spike_*.md` / `docs/refactor_*.md`（pre-extraction 歷史保留於彼處）。

---

## 10. 問問題 / 決策溝通

作者 + AI agent 自用 repo，所以沒有 issue template / discussion board。
決策記錄管道：

- **小改動**：PR description 寫清楚 why + 有無 BC
- **大改動 / 架構決策**：在 workspace repo 的 `docs/` 下新增 `spike_*.md`
  設計文件（跟歷史 spike 一致），factorlib PR 引用它
- **Invariant 級規則變更**：更新 `ARCHITECTURE.md` 的 `Invariants` 區塊
