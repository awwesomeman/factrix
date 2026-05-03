# factrix

> **Factor Matrix Library** — Polars-native Factor Signal Validator

Factrix 只回答：**「這個因子在統計上真的有效嗎？」**

請帶著 `primary_p` 與 `verdict()` 去下游做回測 / 配置 — `factrix` 不做那些。

---

## 目錄

1. [安裝](#安裝)
2. [30-second smoke test](#30-second-smoke-test)
3. [我的問題對應哪個 factory？](#我的問題對應哪個-factory)
4. [設計核心：三條正交分析軸](#設計核心三條正交分析軸)
5. [5 種支援的分析情境 + 對應檢定方法](#5-種支援的分析情境--對應檢定方法)
6. [PANEL / TIMESERIES：由 N 自動推導](#panel--timeseriesn-自動推導)
7. [樣本守門](#樣本守門)
8. [批次評估與 BHY](#批次評估與-bhy)
9. [profile.diagnose() 與 WarningCode](#profilediagnose-與-warningcode)
10. [Scope & non-goals](#scope--non-goals)
11. [與其他開源套件的差異](#與其他開源套件的差異)
12. [文件導引](#文件導引)

---

## 安裝

`factrix` 支援直接從 GitHub 安裝，Core 模組只依賴 `polars + numpy`。

以下提供兩種常見的環境建置與安裝方式：

### 選項 A：使用 `uv` (推薦)

`uv` 是一個極速的 Python 封裝與環境管理工具。

```bash
# 建立並清理全新的 .venv 虛擬環境 (使用 Python 3.12)
uv venv --python 3.12 --clear

# 啟動虛擬環境 (Linux / macOS)
source .venv/bin/activate
# Windows 使用者請執行: .venv\Scripts\activate

# 安裝 main 分支最新開發版
uv pip install git+https://github.com/awwesomeman/factrix.git

# 若需安裝指定版本 (例如 v0.6.0，建議用於正式環境)
# uv pip install git+https://github.com/awwesomeman/factrix.git@v0.6.0
```

### 選項 B：使用 `conda`

如果您習慣使用 Anaconda / Miniconda，可以依照以下步驟建立環境：

```bash
# 建立名為 factrix 的虛擬環境 (使用 Python 3.12)
conda create -n factrix python=3.12 -y

# 啟動虛擬環境
conda activate factrix

# 安裝 main 分支最新開發版
pip install git+https://github.com/awwesomeman/factrix.git

# 若需安裝指定版本 (例如 v0.6.0，建議用於正式環境)
# pip install git+https://github.com/awwesomeman/factrix.git@v0.6.0
```

> **開發者貢獻指南**
> 若您想進行本地開發與修改源碼，請使用 `git clone`：
> ```bash
> git clone https://github.com/awwesomeman/factrix.git
> cd factrix
> uv sync                  # 安裝 core 依賴
> uv sync --extra dev      # 包含 pytest 等開發工具
> ```

---

## 30-second smoke test

> **`forward_periods` 是 rows，不是 calendar time。**
> factrix 是 **frequency-agnostic** — 不讀 `date` 欄位的單位，只 shift 行數。
> `forward_periods=5` 在 daily panel = 5 個交易日；在 weekly panel = 5 週；在
> 1-min bar = 5 分鐘。**caller 負責確保 panel 已按 asset 排序且時間軸間距規律**。
> 同理 `n_periods` / `MIN_PERIODS_*` 全部以 row 為單位。

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

# n_dates=500 daily bars → forward_periods=5 means 1 trading week
raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

# --- Path A：不確定該選哪個情境 → 讓 factrix 從資料形狀推 ---
result  = fl.suggest_config(panel)            # 回 (suggested, reasoning, warnings)
profile = fl.evaluate(panel, result.suggested)

# --- Path B：知道自己要哪個 → 直接用 factory（type-safe, IDE 擋非法組合）---
cfg     = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)
profile = fl.evaluate(panel, cfg)

print(profile.verdict(), '| primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0

# diagnose() 一次拿完整結果（給人讀也給 AI agent 拿）
print(profile.diagnose())
# {'mode': 'panel', 'n_obs': 494, 'n_assets': 100,
#  'primary_p': 2.13e-40, 'warnings': [], 'info_notes': [],
#  'stats': {'ic_mean': 0.0722, 'ic_t_nw': 14.60, 'ic_p': ..., 'nw_lags_used': 5}}
```

無需外部資料、無需設定檔；`fl.datasets` 產可重現合成 panel。Path A 適合**第一次跑 / 不熟三軸語彙**的 user — `result.reasoning` 會解釋每軸偵測理由，`result.warnings` 列出潛在 risk。下一節提供 use-case → factory 反向查表。

---

## 我的問題對應哪個 factory？

不想學三軸語彙就先用這張反向表 — 從**你想問的研究問題**找出口。每列的 factory call 拿去當 §[30-second smoke test](#30-second-smoke-test) Path B 的 `cfg` 就能跑。

| 你想問的問題                                                                  | Factory                                              |
|-------------------------------------------------------------------------------|------------------------------------------------------|
| 我的 per-asset 因子（P/E、momentum、quality）能否預測 cross-section 排序？     | `individual_continuous(metric=fl.Metric.IC)`         |
| 我的 per-asset 因子每多一單位 exposure 對應多少報酬溢酬？                      | `individual_continuous(metric=fl.Metric.FM)`         |
| 個股事件（earnings / rating / 併購公告）有沒有 abnormal return？               | `individual_sparse()`                                |
| Macro 因子（VIX / DXY / 利率）對 cross-section 有沒有 systematic exposure？    | `common_continuous()`                                |
| Macro 事件（FOMC / index rebalance / 政策公布）有沒有市場效應？                | `common_sparse()`                                    |

> **N=1（單一 asset / series）**：mode 自動切 TIMESERIES。Macro 因子 / 事件兩列照走；個股 SPARSE 也照走（內部 collapse 到同一 TS dummy procedure）。`individual_continuous` 在 N=1 數學上 undefined，`evaluate()` 會 raise `ModeAxisError` 並建議改用 `common_continuous()`。

想了解這 5 個 factory 背後的三軸設計（為什麼是這 5 個、為什麼不能更多）→ 看下節。

---

## 設計核心：三條正交分析軸

```
                            你手上的 panel
                                  │
                ┌─────────────────┴─────────────────┐
       同 date 下每 asset 因子值不同？      所有 asset 共用同一個值？
                │                                   │
            INDIVIDUAL                            COMMON
                │                                   │
        ┌───────┴───────┐                   ┌───────┴───────┐
   連續實數         {-1,0,+1}            連續實數         {-1,0,+1}
        │              │                       │              │
  ┌─────┴─────┐        │                       │              │
  IC          FM       │                       │              │
  │           │        │                       │              │
ind_cont    ind_cont  ind_sparse           comm_cont       comm_sparse
(IC)        (FM)
```

> **N=1 footnote**：mode 自動切 TIMESERIES，**樹結構不變**。COMMON 兩條 leaf 與 SPARSE 任一 leaf 皆可走（兩個 SPARSE leaf 在 N=1 內部路由到同一 TS dummy procedure，並加 `InfoCode.SCOPE_AXIS_COLLAPSED` 留 audit trail）。`INDIVIDUAL × CONTINUOUS` 在 N=1 數學上 undefined（無 cross-section dispersion → IC / per-date OLS 不存在），`evaluate()` 會 raise `ModeAxisError(suggested_fix=common_continuous(...))`。**設計上刻意不把 N=1 做成樹的一級分支** — Mode 不暴露給 user 是 factrix 的設計承諾，footnote 處理足夠。

`AnalysisConfig` 由三條 user-facing 軸構成。三軸**正交**：每個支援的組合對應一個明確的統計檢定。

| Axis     | Values                                | 問的是                                                        |
|----------|---------------------------------------|---------------------------------------------------------------|
| `scope`  | `INDIVIDUAL` / `COMMON`               | 因子值是**每個 asset 自己一套**還是**所有 asset 共用同一個**？ |
| `signal` | `CONTINUOUS` / `SPARSE`               | 訊號是連續實數還是 `{−1, 0, +1}` 觸發？                       |
| `metric` | `IC` / `FM` / *(N/A)*                 | 只在 `(INDIVIDUAL, CONTINUOUS)` 情境細分研究問題              |

### scope 是**因子屬性**，不是**資料結構**

factrix 的輸入是 **panel data** — 同時帶有時序軸（dates, `n_periods` 維度）
與 cross-section 軸（assets, `n_assets` 維度）。`scope` 描述的是**因子值在
cross-section 軸上的形狀**，不是資料的整體結構：

- **`INDIVIDUAL`**：每個 `(date, asset_id)` 有獨立的 factor 值。
  例：P/E、Momentum、Quality — 每檔股票自己的訊號。
- **`COMMON`**：每個 `date` 上所有 `asset_id` 共享同一個 factor 值。
  例：VIX、DXY、FOMC dummy — broadcast 到全 universe 的訊號。

用 polars 一行話判斷：
`COMMON` ⇔ `df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()`。

> **N=1 退化**：cross-section 軸只剩一個 asset 時，「INDIVIDUAL vs
> COMMON」的區分自然消失（沒有 cross-section 可比）。資料退化為純
> 時序，factrix 路由到 `TIMESERIES` 用 NW HAC 時序統計處理 — 詳見下方
> §[PANEL / TIMESERIES](#panel--timeseriesn-自動推導)。

### signal

- **`CONTINUOUS`**：實數值（zscore、percentile、價格動能 …）
- **`SPARSE`**：稀疏觸發 `{−1, 0, +1}`，0 為 no-event；連續零值占比 ≥ 50% 才合理用 SPARSE 路徑

### metric（只在 `(INDIVIDUAL, CONTINUOUS)` 出現）

- **`IC`**（default）：rank-based predictive ordering — Spearman ρ → NW HAC t-test
- **`FM`**：unit-of-exposure premium — Fama-MacBeth λ → NW HAC t-test

`(INDIVIDUAL, SPARSE)`、`(COMMON, *)` 沒有 metric 細分，`metric=None`。

---

## 5 種支援的分析情境 + 對應檢定方法

三軸正交組合出 5 種支援的「分析情境」(每種對應一個標準的統計檢定)，由 `AnalysisConfig` 的 4 個 factory methods 構造（IC / FM 共用同一個 factory，差別在 `metric=` 參數）：

> 5 種情境是 user-facing 數量。底層 `evaluate()` 推導出 `Mode` (PANEL / TIMESERIES) 後再對應到 7 個 procedure — PANEL 走 cross-asset 統計、TIMESERIES 走 NW HAC TS 統計。Mode 推導與 N=1 路徑詳見下方 §[PANEL / TIMESERIES](#panel--timeseriesn-自動推導)；7 個 procedure 的 cell ↔ class 對應見 [`ARCHITECTURE.md` §FactorProcedure protocol](ARCHITECTURE.md#factorprocedure-protocol)。

```python
import factrix as fl

# 1. INDIVIDUAL × CONTINUOUS × IC  — stock-picking IC（default metric=IC）
cfg = fl.AnalysisConfig.individual_continuous(forward_periods=5)

# 2. INDIVIDUAL × CONTINUOUS × FM  — Fama-MacBeth λ（小 N panel 較穩）
cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.FM, forward_periods=5)

# 3. INDIVIDUAL × SPARSE           — 個股事件 (earnings / rating)
cfg = fl.AnalysisConfig.individual_sparse(forward_periods=5)

# 4. COMMON × CONTINUOUS           — broadcast factor (VIX / DXY)
cfg = fl.AnalysisConfig.common_continuous(forward_periods=5)

# 5. COMMON × SPARSE               — broadcast event (FOMC / index rebalance)
cfg = fl.AnalysisConfig.common_sparse(forward_periods=5)

profile = fl.evaluate(panel, cfg)  # panel: (date, asset_id, factor, forward_return)
```

每個 factory 對應的標準檢定方法：

| Factory                                | Procedure                                                                                  | 文獻基礎                                  |
|----------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------|
| `individual_continuous(metric=IC)`      | per-date Spearman ρ → NW HAC t on `E[IC]`                                                  | Grinold (1989); Newey & West (1987)       |
| `individual_continuous(metric=FM)`      | per-date OLS slope `λ_t` → NW HAC t on `E[λ]`                                              | Fama & MacBeth (1973); Petersen (2009)    |
| `individual_sparse()`                   | per-event `AR_{i,τ}` → CAAR → cross-event t                                                | Brown & Warner (1985); MacKinlay (1997)   |
| `common_continuous()`                   | per-asset TS β → cross-asset t on `E[β]`                                                   | Black-Jensen-Scholes (1972)               |
| `common_sparse()`                       | per-asset TS β on dummy → cross-asset t on `E[β]`                                          | TS-β + event-study hybrid                 |

> Factory methods 是 type-safe constructors — 不支援的組合（如對 `SPARSE` 情境傳 `metric=IC`）IDE 直接標紅，不必等 runtime `IncompatibleAxisError`。

### IC 還是 FM？

回答的研究問題不同，**選擇依研究問題決定，不依資料形狀決定**：

- **`IC`**：「factor 對未來報酬有沒有 predictive ordering？」 — rank-based、對 outlier robust
- **`FM`**：「factor 每多一單位 exposure 對應多少報酬溢酬？」 — slope-based、有 economic interpretation

實務參考：`N < 10` 時 IC variance 過高，建議切 `FM`；`N ∈ [20, 50]` 區間兩者皆穩。

---

## PANEL / TIMESERIES：由 N 自動推導

`Mode` **不是 user-facing 軸**，由 raw data 的 `N = panel["asset_id"].n_unique()` 在 evaluate-time 自動決定：

| Mode         | 條件   | 適用 tuple                                        | 統計重心                                  |
|--------------|--------|---------------------------------------------------|-------------------------------------------|
| `PANEL`      | N ≥ 2  | 全部 5 個                                         | cross-sectional / cross-asset aggregation |
| `TIMESERIES` | N = 1  | `(INDIVIDUAL, SPARSE)`、`(COMMON, *)` 三個        | time-series aggregation (NW HAC + ADF)    |

兩 mode **對等 first-class** — `primary_p` 都是真實值，不會壓 1.0。差別只在底層 procedure。

### N=1 的特殊路徑

- **`(INDIVIDUAL, CONTINUOUS, *) × N=1`**：數學上不存在（無 cross-sectional dispersion → IC / per-date OLS undefined）。`evaluate()` 直接 `raise ModeAxisError(suggested_fix=AnalysisConfig.common_continuous(...))`。
- **`(*, SPARSE, None) × N=1`**：scope 軸自然 collapse — `individual_sparse()` 與 `common_sparse()` 在 N=1 路由到同一個 timeseries dummy procedure，`profile.info_notes` 加入 `InfoCode.SCOPE_AXIS_COLLAPSED` 留下 audit trail。

---

## 樣本守門

時序長度 `n_periods` 與資產數 `n_assets` **獨立守門**，不採用 `n_periods × n_assets` 總觀測數作為單一 power 指標 — per-date stat 變異主要由 `n_assets` 決定，time-series aggregation power 主要由 `n_periods` 決定。

| 常數                | 來源                              | 行為                                                                            |
|---------------------|-----------------------------------|---------------------------------------------------------------------------------|
| `MIN_PERIODS_HARD = 20`   | `factrix/_stats/constants.py`    | `n_periods < 20` → raise `InsufficientSampleError(actual_periods, required_periods)`                |
| `MIN_PERIODS_RELIABLE = 30` | 同上                             | `n_periods < 30` → 加 `WarningCode.UNRELIABLE_SE_SHORT_SERIES` 到 `profile.warnings`     |
| `MIN_IC_PERIODS = 10` / `MIN_EVENTS = 10` | `factrix/_types.py` | metric 內部 short-circuit 用                                                     |

`fl.suggest_config(panel)` 可反向給出建議的 factory call + 警報；`fl.describe_analysis_modes()` 列出所有情境及其 procedure / 文獻 / `MIN_PERIODS_*`。

---

## 批次評估與 BHY

BHY 控制的是**同一 statistical family 內**的 FDR：用同一個 procedure 評估多個 candidate factor，再對這批 p-value 做 step-up 校正。**不要混 family** — 跨 IC / FM / TS-β 的 p-value 統計意義不可換算，混批等於放棄 FDR control。

```python
import factrix as fl
import polars as pl

# 10 個 momentum candidate, 同一個分析情境 (IC PANEL) 跑批
candidates = ["mom_5d", "mom_20d", "mom_60d", ...]
cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)

profiles = [
    fl.evaluate(panel.with_columns(pl.col(name).alias("factor")), cfg)
    for name in candidates
]
survivors = fl.multi_factor.bhy(profiles, threshold=0.05)
```

`bhy()` 自動依**分析情境（含 horizon）** 分 family，user **不需手動指定 group key**。`forward_periods` 一定會被切開：每個 horizon 有自己的 null distribution 與 effective sample size，混批會稀釋 step-up 門檻、靜默破壞 FDR 控制。如果任一 family 退化成 `size=1`（典型誤用：一個因子在多個情境各跑一次），會 emit `RuntimeWarning` — 因為這時 BHY 等同於 raw threshold，沒有 FDR 校正力。內部分組規則細節見 [`ARCHITECTURE.md` §BHY family partitioning](ARCHITECTURE.md#bhy-family-partitioning)。

### Horizon-shopping 校正：先壓 horizon、再 BHY

`bhy()` 自動分 family 只解決「同 horizon 內」的 FDR 控制。若你**對每個因子掃多個 horizon、再挑 min-p**，挑 horizon 本身就是隱形多重檢定（K = horizon 數），BHY 不會替你校正這層 — 必須先用 **FWER 程序**把 horizon 維度內縮成單一代表性 p，再餵進 BHY。

選 FWER 程序時依場景抽換：

- **Bonferroni**（`p × K`）：horizon 數小、彼此可視為獨立時最簡單；K 小（≤ 5）時甚至比 BHY 更不保守。
- **Holm**（step-down）：同樣控 FWER 但較不保守，當 horizon 之間 p-value 強度差異大時更實用。
- **不要**用 BHY 當內層程序：(1) 內層只挑單一代表性 p 是 FWER 問題不是 FDR 問題；(2) BHY ∘ BHY 沒有合成定理；(3) K 小時 BHY 因 `c(m)` 因子反而比 Bonferroni 更嚴。

> 別把 K 個因子 × H 個 horizon 共 K×H 個原始 profile 直接 `bhy()`：BHY 會按 horizon 切成 H 個 family of K（這是「在每個 horizon 上獨立挑因子」的正確問題），但這**不**對應「對每個因子挑最佳 horizon」的問題。後者一定要先做 FWER 內縮。

---

## profile.diagnose() 與 WarningCode

`profile.diagnose()` 一次回傳 `dict[str, Any]`，給人讀也給 AI agent 拿：

```python
{
    "mode": "panel",
    "n_obs": 500,
    "primary_p": 0.0001,
    "warnings": ["unreliable_se_short_series"],   # WarningCode.value
    "info_notes": [],                              # InfoCode.value
    "stats": {"ic_mean": 0.082, "ic_t_nw": 4.21, "nw_lags_used": 5},
}
```

`profile.warnings: frozenset[WarningCode]` 是顯式 enum，每個 code 帶一行 `description` gloss：

| WarningCode                    | 觸發條件                                                  |
|--------------------------------|-----------------------------------------------------------|
| `UNRELIABLE_SE_SHORT_SERIES`   | `n_periods < MIN_PERIODS_RELIABLE = 30` → NW HAC SE 不穩定       |
| `PERSISTENT_REGRESSOR`         | `factor_adf_p > 0.10`（CONTINUOUS factor，Stambaugh-style） |
| `EVENT_WINDOW_OVERLAP`         | event windows 重疊（CAAR / sparse 場景）                   |
| `SERIAL_CORRELATION_DETECTED`  | Ljung-Box p < 0.05 on residuals                            |

> **重要**：`warnings` **不**影響 `verdict()`。它是 risk flag，user 自行決定是否在 BHY 之前過濾。`verdict()` 只看 `primary_p < threshold`。

---

## Scope & non-goals

### Scope
- 建議資料頻率：**Daily 到 monthly** bar-based。Sparse signal 模組能適應不定期事件，其餘建議日頻或更低；不支援真正的 HFT (tick-level)。Row-based 語意見頂部 §[30-second smoke test](#30-second-smoke-test) 的 callout。
- 5 種分析情境全覆蓋；`(INDIVIDUAL, CONTINUOUS, *) × N=1` 數學上不存在的組合會 raise `ModeAxisError` 並提示改用 `common_continuous`。
- 統計嚴謹的 PASS/FAIL gate + `profile.diagnose()` warnings + BHY FDR 控制 = 後續 allocation / strategy layer 的可信輸入。

### 明確不做（不只是「沒做」— 是設計上決定不做）

| 不做                                                       | 該用什麼                                                          |
|------------------------------------------------------------|-------------------------------------------------------------------|
| Portfolio optimization (MVO / HRP / Risk Parity)           | `skfolio` / `PyPortfolioOpt` / `riskfolio-lib` / `cvxpy`          |
| ML 信號層 (xgboost / lightgbm + SHAP)                       | `xgboost` + `shap`                                                |
| Regime detection methodology (HMM / threshold)             | `hmmlearn` / 自寫                                                  |
| Structural break detection (Chow / Bai-Perron)             | `ruptures`                                                        |
| GARCH / wild bootstrap SE                                  | `arch`                                                            |
| Persistent-predictor 自動修正 (IVX / Stambaugh)             | `arch` / R `ivx`（factrix 只 flag，不 auto-correct）              |
| Backtest / execution / slippage / margin                    | `vectorbt` / `bt` / Zipline / Backtrader                          |
| Intraday / HFT (tick-level)                                | 另找專用工具                                                       |
| 跨 factor 合成信號 / factor combiner                         | 自寫，或 `scikit-learn`                                            |

這份清單不是 TODO，是**承諾不擴張的 scope 邊界**。擴張請先更新 [`ARCHITECTURE.md`](ARCHITECTURE.md) Invariants。

---

## 與其他開源套件的差異

定位介於**純統計套件**與**回測框架**之間：

- **vs `alphalens`**：`alphalens` 只做連續截面 IC + 分層回報、Pandas-based 已年久失修。`factrix` 是 Polars-native 替代品，且擴展到 sparse signal、broadcast factor、TS β、BHY FDR。
- **vs `vectorbt` / `zipline` / `backtrader`**：那些是回測 / 交易執行框架；`factrix` 不做部位、不處理保證金與滑價。職責止於提供可信的 `primary_p` —「在寫複雜回測之前，先快速淘汰假因子」。
- **vs `skfolio` / `PyPortfolioOpt`**：那些是投組最佳化工具；`factrix` 找出有預測力的 alpha 訊號，不處理權重最佳化。

---

## 文件導引

| 想知道                                                                        | 看                                                                             |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| 怎麼用                                                                        | 上方 §30-second smoke test、`fl.describe_analysis_modes()`、`fl.suggest_config(panel)` |
| **精確公式 / 演算法 / 邊界 case**（authoritative source）                       | 對應 module docstring：`help(factrix.metrics.ic)`、…                            |
| **為什麼選這個方法**（論文依據、deviation）                                     | [`docs/statistical_methods.md`](docs/statistical_methods.md)                   |
| **資料夠不夠用**（N/T 下限、fallback 語意）                                     | [`docs/metric_applicability.md`](docs/metric_applicability.md)                 |
| Profile / Config 介面                                                         | `help(fl.FactorProfile)`、`help(fl.AnalysisConfig)`                            |
| 內部結構（registry SSOT、Procedure protocol、Mode 推導、Invariants）            | [`ARCHITECTURE.md`](ARCHITECTURE.md)                                          |
| 想貢獻                                                                        | [`CONTRIBUTING.md`](CONTRIBUTING.md)                                          |
| 版本變動                                                                      | [`CHANGELOG.md`](CHANGELOG.md)                                                |

---

## License

`factrix` 以 [Apache License 2.0](LICENSE) 釋出。Apache-2.0 同時授予使用者**著作權**與**專利權**，並包含專利反訴終止條款（§3），對下游使用者與貢獻者皆有明確的專利保護。

外部貢獻者送 PR 即視為依 Apache-2.0 §5 將該貢獻以同等授權回饋本專案（inbound = outbound），詳見 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
