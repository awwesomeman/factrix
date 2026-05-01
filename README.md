# factrix

> **Factor Matrix Library** — Polars-native 因子訊號驗證器 (Factor Signal Validator)。

回答的問題只有一個：**「這個因子在統計上真的有效嗎？」**
回答完之後，請帶著 `primary_p` 與 `verdict()` 去下游做回測 / 配置 — `factrix` 不做那些。

---

## 目錄

1. [安裝](#安裝)
2. [30-second smoke test](#30-second-smoke-test)
3. [設計核心：三條正交分析軸](#設計核心三條正交分析軸)
4. [5 個合法 cell + canonical procedure](#5-個合法-cell--canonical-procedure)
5. [Mode A / Mode B：由 N 自動推導](#mode-a--mode-bn-自動推導)
6. [樣本守門](#樣本守門)
7. [批次評估與 BHY](#批次評估與-bhy)
8. [profile.diagnose() 與 WarningCode](#profilediagnose-與-warningcode)
9. [Scope & non-goals](#scope--non-goals)
10. [與其他開源套件的差異](#與其他開源套件的差異)
11. [文件導引](#文件導引)

---

## 安裝

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix

uv sync                  # core (polars + numpy)
uv sync --extra dev      # + pytest
```

或 pip 直裝：

```bash
pip install git+https://<YOUR_GITHUB_TOKEN>@github.com/awwesomeman/factrix.git
```

Core 只依賴 `polars + numpy`。

---

## 30-second smoke test

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

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

無需外部資料、無需設定檔；`fl.datasets` 產可重現合成 panel。

---

## 設計核心：三條正交分析軸

`AnalysisConfig` 由三條 user-facing 軸構成。三軸**正交**：每個合法組合對應一個明確的統計檢定。

| Axis     | Values                                | 問的是                                                        |
|----------|---------------------------------------|---------------------------------------------------------------|
| `scope`  | `INDIVIDUAL` / `COMMON`               | 因子值是**每個 asset 自己一套**還是**所有 asset 共用同一個**？ |
| `signal` | `CONTINUOUS` / `SPARSE`               | 訊號是連續實數還是 `{−1, 0, +1}` 觸發？                       |
| `metric` | `IC` / `FM` / *(N/A)*                 | 只在 `(INDIVIDUAL, CONTINUOUS)` cell 細分研究問題              |

### scope 是**因子屬性**，不是**資料結構**

> 你的 panel 永遠是 cross-sectional 結構（多 asset × 多 date）。`scope` 描述的是**因子值在資產之間的形狀**，不是資料形狀。
>
> - **`INDIVIDUAL`**：每個 `(date, asset_id)` 有獨立的 factor 值。例：P/E、Momentum、Quality — 每檔股票自己的訊號。
> - **`COMMON`**：每個 `date` 上所有 `asset_id` 共享同一個 factor 值。例：VIX、DXY、FOMC dummy — broadcast 到全 universe 的訊號。
>
> 用 polars 一行話判斷：`COMMON` ⇔ `df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()`。

### signal

- **`CONTINUOUS`**：實數值（zscore、percentile、價格動能 …）
- **`SPARSE`**：稀疏觸發 `{−1, 0, +1}`，0 為 no-event；連續零值占比 ≥ 50% 才合理用 SPARSE 路徑

### metric（只在 `(INDIVIDUAL, CONTINUOUS)` 出現）

- **`IC`**（default）：rank-based predictive ordering — Spearman ρ → NW HAC t-test
- **`FM`**：unit-of-exposure premium — Fama-MacBeth λ → NW HAC t-test

`(INDIVIDUAL, SPARSE)`、`(COMMON, *)` 沒有 metric 細分，`metric=None`。

---

## 5 個合法 cell + canonical procedure

合法 `(scope, signal, metric)` 三元組共 5 個，由 `AnalysisConfig` 的 4 個 factory methods 構造：

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

每個 factory 對應的 canonical 統計程序：

| Factory                                | Procedure                                                                                  | 文獻基礎                                  |
|----------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------|
| `individual_continuous(metric=IC)`      | per-date Spearman ρ → NW HAC t on `E[IC]`                                                  | Grinold (1989); Newey & West (1987)       |
| `individual_continuous(metric=FM)`      | per-date OLS slope `λ_t` → NW HAC t on `E[λ]`                                              | Fama & MacBeth (1973); Petersen (2009)    |
| `individual_sparse()`                   | per-event `AR_{i,τ}` → CAAR → cross-event t                                                | Brown & Warner (1985); MacKinlay (1997)   |
| `common_continuous()`                   | per-asset TS β → cross-asset t on `E[β]`                                                   | Black-Jensen-Scholes (1972)               |
| `common_sparse()`                       | per-asset TS β on dummy → cross-asset t on `E[β]`                                          | TS-β + event-study hybrid                 |

> Factory methods 是 type-safe constructors — 違反組合（如對 `SPARSE` cell 傳 `metric=IC`）IDE 直接標紅，不必等 runtime `IncompatibleAxisError`。

### IC 還是 FM？

回答的研究問題不同，**選擇依研究問題決定，不依資料形狀決定**：

- **`IC`**：「factor 對未來報酬有沒有 predictive ordering？」 — rank-based、對 outlier robust
- **`FM`**：「factor 每多一單位 exposure 對應多少報酬溢酬？」 — slope-based、有 economic interpretation

實務參考：`N < 10` 時 IC variance 過高，建議切 `FM`；`N ∈ [20, 50]` 區間兩者皆穩。

---

## Mode A / Mode B：由 N 自動推導

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

時序長度 `T` 與資產數 `N` **獨立守門**，不採用 `T × N` 總觀測數作為單一 power 指標 — per-date stat 變異主要由 `N` 決定，time-series aggregation power 主要由 `T` 決定。

| 常數                | 來源                              | 行為                                                                            |
|---------------------|-----------------------------------|---------------------------------------------------------------------------------|
| `MIN_T_HARD = 20`   | `factrix/_stats/constants.py`    | `T < 20` → raise `InsufficientSampleError(actual_T, required_T)`                |
| `MIN_T_RELIABLE = 30` | 同上                             | `T < 30` → 加 `WarningCode.UNRELIABLE_SE_SHORT_SERIES` 到 `profile.warnings`     |
| `MIN_IC_PERIODS = 10` / `MIN_EVENTS = 10` | `factrix/_types.py` | metric 內部 short-circuit 用                                                     |

`fl.suggest_config(panel)` 可反向給出建議的 factory call + 警報；`fl.describe_analysis_modes()` 列出所有 cell 及其 procedure / 文獻 / `MIN_T_*`。

---

## 批次評估與 BHY

BHY 控制的是**同一 statistical family 內**的 FDR：用同一個 procedure 評估多個 candidate factor，再對這批 p-value 做 step-up 校正。**不要混 family** — 跨 IC / FM / TS-β 的 p-value 統計意義不可換算，混批等於放棄 FDR control。

```python
import factrix as fl
import polars as pl

# 10 個 momentum candidate, 同一個 cell (IC PANEL) 跑批
candidates = ["mom_5d", "mom_20d", "mom_60d", ...]
cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)

profiles = [
    fl.evaluate(panel.with_columns(pl.col(name).alias("factor")), cfg)
    for name in candidates
]
survivors = fl.multi_factor.bhy(profiles, threshold=0.05)
```

`bhy()` 自動依 `(scope, signal, metric)` tuple 分 family，user **不需手動指定 group key**。如果任一 family 退化成 `size=1`（典型誤用：一個 factor 在多個 cell 各跑一次），會 emit `RuntimeWarning` — 因為這時 BHY 等同於 raw threshold，沒有 FDR 校正力。

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
| `UNRELIABLE_SE_SHORT_SERIES`   | `T < MIN_T_RELIABLE = 30` → NW HAC SE 不穩定               |
| `PERSISTENT_REGRESSOR`         | `factor_adf_p > 0.10`（CONTINUOUS factor，Stambaugh-style） |
| `EVENT_WINDOW_OVERLAP`         | event windows 重疊（CAAR / sparse 場景）                   |
| `SERIAL_CORRELATION_DETECTED`  | Ljung-Box p < 0.05 on residuals                            |

> **重要**：`warnings` **不**影響 `verdict()`。它是 risk flag，user 自行決定是否在 BHY 之前過濾。`verdict()` 只看 `primary_p < threshold`。

---

## Scope & non-goals

### Scope
- 建議資料頻率：**Daily 到 monthly** bar-based。Sparse signal 模組能適應不定期事件，其餘建議日頻或更低；不支援真正的 HFT (tick-level)。
- `forward_periods` 是 **rows，不是 calendar time** — 週頻 panel `forward_periods=1` 就是 1-week 前向報酬。
- 5 個 `(scope, signal, metric)` cell 全覆蓋；`(INDIVIDUAL, CONTINUOUS, *) × N=1` 數學上不存在的位置 raise `ModeAxisError` 並提示改 `common_continuous`。
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

- **vs `alphalens`**：alphalens 只做連續截面 IC + 分層回報、Pandas-based 已年久失修。`factrix` 是 Polars-native 替代品，且擴展到 sparse signal、broadcast factor、TS β、BHY FDR。
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
