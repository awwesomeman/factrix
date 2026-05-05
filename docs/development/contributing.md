# Contributing to factrix

本文件描述 factrix 的開發流程。主要讀者是**作者本人 + 未來的 AI agent**
——private repo，所以省略 licensing / DCO / CLA 等 OSS 慣例，聚焦在**實際
開發模式與陷阱**。

---

## 1. 兩種開發模式

### Mode A — Standalone development（推薦多數情況）

直接 clone factrix repo，獨立 venv，cycle 最快：

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync
uv run pytest        # 驗證 baseline 綠
```

**什麼時候用**：新 feature / 大 refactor / SemVer bump / release。
隔離環境、不會誤動到下游 research、測試 cycle 最短。

### Mode B — In-workspace development（via submodule）

從下游 workspace（`factor-analysis`）的 `external/factorlib/` 編輯——
可以邊改 factrix、邊在真實 research notebook 看效果：

> **注意**：submodule 的磁碟資料夾名稱取決於 parent workspace 的
> `.gitmodules` 設定，目前實際路徑仍為 **`external/factorlib/`**。
> 待 parent workspace 將 submodule path rename 後，路徑才會改為
> `external/factrix/`。以下指令以當前磁碟路徑為準。

```bash
cd ~/Desktop/dst/code/factor-analysis
cd external/factorlib
# edit factrix source
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

factrix 用 **uv** 管理 Python venv 與 lockfile。`pyproject.toml` 與
`uv.lock` 是唯一權威——不要用 `pip install` 直接裝任何東西進 `.venv/`。

Python 版本鎖定在 **3.12+**（定義於 `pyproject.toml` 的 `requires-python`）。

### 依賴安裝與 Extras
根據開發需求，你可以使用 `--extra` 增減所需模組：

```bash
uv sync                              # 僅安裝 Core 依賴 (polars, numpy, pandera)
uv sync --extra dev                  # +pytest, commitizen 等開發工具 (寫 Code 必裝)
uv sync --extra charts               # +plotly 圖表繪製
uv sync --all-extras                 # charts + mlflow + jupyter（功能 extras）
```

> **注意**：`dev` extra 不屬於 `all`（工具鏈與功能 extras 刻意分開），
> 因此 `--all-extras` **不會**安裝 pytest / commitizen 等開發工具。
> 開發者請使用：
>
> ```bash
> uv sync --all-extras --extra dev   # 功能 extras + 開發工具，一次裝齊
> ```

### 常用環境指令
```bash
uv run <cmd>         # 在 venv 內執行（例如: uv run pytest）
uv add <pkg>         # 新增 dep，同步更新 pyproject + uv.lock
uv lock --upgrade    # 升 lock 到當前 pyproject constraint 的最新版
```

### Git hooks（pre-push CHANGELOG check）

Repo 根目錄下的 `.githooks/` 收受版控的 hook script。fresh clone 後跑
一次：

```bash
git config core.hooksPath .githooks
```

這條 config 是 per-clone local，不會跨機器自動同步，所以每個 clone /
新機器都要跑一次。

目前唯一的 hook 是 `pre-push`：當 push 含 `chore(release): vX.Y.Z`
（亦即 `cz bump` 產出的 release commit），會檢查 `CHANGELOG.md` 中對
應 `## vX.Y.Z` section 的非空白行數 ≥ 25。低於門檻 → block push，逼
你補 WHY narrative（BREAKING migration、行為方向、動機）再 push。
真要繞過：`git push --no-verify`。

調整門檻（per-shell）：

```bash
CHANGELOG_MIN_LINES=10 git push
```

設計理由見 §7 「Release workflow」對 cz auto-CHANGELOG 的限制說明。

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

# 3. Commit（Conventional Commits + 互動式生成）
git add <specific-files>                 # 不用 -A
cz commit -- -s                          # 使用 Commitizen 產出標準格式並附加簽名檔

# 4. Push + 開 PR
git push origin feat/redundancy-heatmap
gh pr create --title "..." --body "..."

# 5. CI 綠後 merge（目前是 solo → squash-merge 或 rebase-merge）
gh pr merge --squash
```

> **重要**：merge 完 **不** 跑 `cz bump`。版號與 tag 走 release-train
> 節奏（見 §7）——多個 PR 累積後才一次性 release。每個 PR 在自己的
> 改動寫進 `CHANGELOG.md` 的 `## [Unreleased]` 段（含 `### Added` /
> `### Changed` / `### Fixed` / `### Migration` 子段，視適用情況），release
> 時 `cz bump --changelog` 會把該段固化成下一個版本標題。

### Branch 命名

`<type>/<short-desc>`，全小寫連字號：

- `feat/...` — 新 feature
- `fix/...` — bug fix
- `refactor/...` — 重構（行為不變）
- `docs/...` — 文件
- `chore/...` — 打包 / CI / lock 維護

### Commit message

專案採用 **Commitizen** 與 **Conventional Commits** 規範。**請一律使用 `cz commit` 進行互動式提交**，程式會自動檢驗標題長短等規則。

**規則提醒**：
- Description 長度 `< 50 字元`
- Body 採用 `-` 條列式，只寫「為什麼做」+「主要做了什麼」——不複述 diff，每行建議 `< 72 字元`
- 禁用 AI co-author 署名、emoji、句末句號
- 透過 `cz commit -- -s` 中的 `-s` 來附加 Signed-off-by

#### 變更 `Signed-off-by` 簽名名稱與信箱

`git commit -s` 生成的簽名是讀取 Git 環境設定中的 `user.name` 和 `user.email` 欄位。如果你需要更改簽名，請執行以下指令：

```bash
# 僅限當前專案生效
git config user.name "你的新名字"
git config user.email "你的新信箱@example.com"

# 全域設定 (所有專案預設生效)
git config --global user.name "你的新名字"
git config --global user.email "你的新信箱@example.com"
```

---

## 4. Mode B 的三個關鍵陷阱

### G1. Submodule = detached HEAD（最大的坑）

從 workspace 進 `external/factrix/` 時，submodule 預設是 **detached HEAD**。
在 detached HEAD 上 commit 的 commits **不屬於任何 branch**——下次
`git submodule update --remote` 會直接覆蓋，commit 消失（reflog 不記）。

```bash
# 丟 commit 的寫法（錯誤示範）
cd external/factrix
# ...edit...
git commit -am "feat: xxx"              # detached — commit 沒家
git submodule update --remote           # GONE

# 正確：先 branch
cd external/factrix
git checkout -b feat/xxx                # 建 branch
# ...edit...
git commit -am "feat: xxx"
git push origin feat/xxx
gh pr create                            # 在 factrix repo 開 PR
```

### G2. Editable install 對 Jupyter 有限制

Workspace 的 `uv sync` 把 factrix 裝成 editable，所以改 submodule
source 會立刻反映到 `import factrix`——但 Jupyter kernel 有例外：

- Function body 改動 → `%autoreload 2` 能抓到
- `__init__.py` 的 imports / dataclass 定義 / module-level 常數 → 必須 **restart kernel**

不確定時 restart kernel 是最穩的選擇。

### G3. Workspace 不會自動追蹤 factrix 更新

factrix main 合了 PR 之後，workspace 的 submodule 指針**不會自動 bump**。
這是 feature 不是 bug——workspace 的每個 commit 綁定明確的 factrix SHA，
研究結果可重現。

手動 bump：

```bash
cd external/factrix && git fetch && git checkout main && git pull
cd ../.. && git add external/factrix
git commit -m "chore: bump factrix to <short-sha>: <why>"
```

---

## 5. Submodule sync reference

Mode B 與 consumer workspace 日常操作的指令索引。90% 情境看 cheat sheet
就夠，看不懂再往下讀心智模型與情境說明。

### 5.1 Cheat sheet

| 想做的事 | 指令 |
|---------|------|
| 看 workspace pin 的 SHA | `git submodule status` |
| 看 submodule actual HEAD | `cd external/factrix && git rev-parse --short HEAD` |
| 拉 factrix main 最新到 actual | `git submodule update --remote` |
| 切到特定 tag | `cd external/factrix && git fetch --tags && git checkout vX.Y.Z` |
| 固化 actual 到 pin | `git add external/factrix && git commit -m "chore: bump..."` |
| 放棄 actual 改動，reset 回 pin | `git submodule update` |
| Fresh clone 初始化 submodule | `git submodule update --init --recursive` |

### 5.2 心智模型

workspace 有 `pin`（記在 workspace commit 裡的 SHA），submodule 自己
有 `actual HEAD`（實際 checkout 的 SHA）。兩者可能不同——`git status`
會顯示 submodule 是 `modified`。**Python editable install 讀的是 actual
HEAD**，所以 import 拿到的版本取決於 actual，不是 pin。

### 5.3 常見情境

- **自己改 factrix 要 sync 回 workspace**：走 §3 + §4（dev 流程），完成後
  cheat sheet 第 5 列固化
- **別人 push 了 factrix，我要跟上**：cheat sheet 第 3 列 + 第 5 列
- **Pin 到 tag（推薦，見 §8）**：cheat sheet 第 4 列 + 第 5 列
- **切 branch 後 submodule 亂掉**：cheat sheet 第 6 列 reset

### 5.4 搞不清楚時先做兩件事

1. `git submodule status` 看 pin
2. `cd external/factrix && git rev-parse HEAD` 看 actual

兩個 SHA 一樣 = 乾淨；不一樣就決定：要 bump pin（第 5 列）還是 reset
actual（第 6 列）。

---

## 6. 測試規範

### Synthetic fixtures only

**factrix tests 不得 load 真實市場資料**——所有 fixture 必須是
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

`factrix/metrics/*.py` 的 docstring 是各 metric 「精確公式 / 演算法」的
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

## 7. 版本管理與發布 (SemVer & Release)

目前 factrix 在 **pre-1.0**（v0.x.x）——公開 API **可能在 MINOR bump 裡
BC 變動**。Consumer（如 `factor-analysis` workspace）應該透過 **git
submodule SHA pin** 而非 version range，直到 1.0.0 穩定為止。

**專案採用 Commitizen 進行全自動升版與 Changelog 生成，搭配 release-train
節奏：PR 隨時 merge，但 release（bump + tag）獨立排程。**

### Release cadence — release train

PR 與 release 解耦：

- **PR cadence**：高頻、原子。merge 完 **不** bump，**不** tag。
- **Release cadence**：低頻、彙總。當以下任一條件達到才開 release：
  - 累積 ≥ 3 個 user-facing `feat:` / `fix:`
  - 距上一個 tag ≥ 2 週
  - 有對下游 workspace 的緊急 bug fix（單獨 PATCH 隨時可切）
  - 為了讓某人 / 某 demo 拉到具名版本

每個 PR 自己負責把 WHY narrative 寫進 `CHANGELOG.md` 的 `## [Unreleased]`
段（`### Added` / `### Changed` / `### Fixed` / `### Migration` 子段）。
這樣 release 時不用回頭重建敘事——`cz bump --changelog` 把整段固化成下
一個版本標題即可。

### Release workflow

```bash
# 1. 在 factrix repo main 分支確保最新
git checkout main && git pull

# 2. 驗證 CI 綠 + 本地 pytest 全綠
uv run pytest

# 3. 自動改版與打標籤
# cz 根據自上一個 tag 以來的 commits 推導 (feat=MINOR, fix=PATCH)，
# 把 [Unreleased] 段改名為新版本標題，新增空白 [Unreleased] 段，
# 更新 pyproject.toml，自動 commit + tag。
cz bump --changelog

# 4. （選配）手動潤飾 release 段——補 BREAKING migration / 方向 / 動機
#    至 ≥ 25 非空白行，否則 pre-push 會 block（見 §2 hook）。
#    潤完用 git commit --amend 收回 release commit，刪舊 tag 重建：
git commit --amend --no-edit
git tag -d v<X.Y.Z> && git tag v<X.Y.Z>

# 5. Push
git push origin main
git push origin v<X.Y.Z>

# 6. 回 workspace bump submodule
cd ~/Desktop/dst/code/factor-analysis
cd external/factrix && git fetch && git checkout v<X.Y.Z>
cd ../.. && git add external/factrix
git commit -m "chore: bump factrix to v<X.Y.Z>"
git push
```

### BC change 特別注意

例：`q1_q5_spread → long_short_spread`（已於 workspace 歷史發生過）
這種 rename 屬 BC change。開發者在利用 `cz commit` 提交時，必須選擇 Breaking Change，並在 Prompt 中**明確寫出 migration path**（舊名 → 新名、影響哪些欄位）。這樣下游 workspace 才能從自動生成的 CHANGELOG 中找到升級指南。

### Workspace 不跟最新 main，只跟 tag

下游 research workspace 原則上 **pin 到 tag**（而非 main HEAD），這樣
每個 workspace commit 對應一個清楚的 factrix 版本、可重現。

Main HEAD 只在 Mode B 開發中暫時用（debug 流程），完成後 merge 回
factrix main、打 tag，再讓 workspace bump 到 tag。

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
  設計文件（跟歷史 spike 一致），factrix PR 引用它
- **Invariant 級規則變更**：更新 `ARCHITECTURE.md` 的 `Invariants` 區塊

---

## 11. 授權與貢獻條款（Licensing）

本專案採用 [Apache License 2.0](LICENSE)。

**Inbound = Outbound**：依 Apache-2.0 §5，凡你向本 repo 提交（PR、patch、issue 附帶程式碼等）的內容，**預設視為以 Apache-2.0 同等條款授權回本專案**，除非你在提交時明確聲明其他條款。送 PR 前請確認：

- 你對該段程式碼擁有授權權利（自行撰寫，或來源相容於 Apache-2.0）
- 引用第三方程式碼時，於檔案 header 或 PR description 標註原始授權
- 涉及專利的演算法（如：你個人或所屬組織持有專利的方法），請在 PR description 主動揭露

不接受 GPL/AGPL 等具強制 copyleft 的程式碼合併進主程式碼樹（會污染下游使用者的授權義務）。
