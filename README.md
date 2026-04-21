# factrix

**Modular factor evaluation toolkit** — 針對**不同訊號幾何 (signal
geometry)** 的因子做穩健性檢測：連續截面（選股）、稀疏事件、小 panel
（跨國配置）、共用時序（共同驅動因子）。全部走同一套 `preprocess →
evaluate → Profile` API，polars-native，每一步有 statistical-discipline
的預設值，對樣本不足 / 資產數少的情境有明確（而非美化）的 fallback。

> **定位：Factor Signal Validator**。回答「這個因子統計上真的有效嗎」，
> **不回答**「該怎麼配置」「該怎麼交易」。驗證後要做配置 / 策略，請把
> Profile 的 `canonical_p` + 效果量帶去 `skfolio` / `PyPortfolioOpt` /
> `vectorbt` 等下游工具。`turnover` / `breakeven_cost` / `net_spread` 是
> screening 用的理想化 proxy（等權、無滑價），不是 tradable P&L。

---

## Install

目前專案尚未公開發佈至公開的 PyPI。針對不同的使用情境，提供以下兩種安裝方式：

### 1. 境內量化研究員直接安裝 (使用套件)
此方法適用於只想要使用套件的研究員，**不需額外架設私有 PyPI**。只需要產生一組擁有讀取權限的 GitHub Token，就能讓 `pip` 直接到 Repo 抓取代碼並自動打包安裝。在這個過程中，Private GitHub 就充當了我們的私有套件庫：

```bash
# 透過 Git Token 進行一鍵安裝
pip install git+https://<YOUR_GITHUB_TOKEN>@github.com/awwesomeman/factrix.git

# 若需安裝畫圖與追蹤等 optional 依賴
pip install "factrix[all] @ git+https://<YOUR_GITHUB_TOKEN>@github.com/awwesomeman/factrix.git"
```

### 2. 框架開發者 (Editable Install)
若你需要開發、修改 `factrix` 的底層源碼或跑測試：

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix

pip install -e .                   # core (polars + numpy + pandera)
pip install -e ".[charts]"         # + plotly 圖表
pip install -e ".[mlflow]"         # + mlflow tracking
pip install -e ".[all]"            # charts + mlflow
pip install -e ".[dev]"            # + pytest / nbformat（跑測試、build demo）
```

Core 只依賴 `polars + numpy + pandera`；plotly / mlflow 屬 optional extras，不裝也能跑完 `evaluate` / `ProfileSet` / BHY 全部核心流程。

---

## 30-second smoke test

確認裝好了、環境跑得起來：

```python
import factrix as fl

raw = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
cfg = fl.CrossSectionalConfig(forward_periods=5)
profile = fl.evaluate(fl.preprocess(raw, config=cfg), "sanity", config=cfg)

print(profile.verdict(), '| ic_mean =', round(profile.ic_mean, 4))
# → PASS | ic_mean = 0.0722
```

三行，不需要任何外部資料（`fl.datasets` 產可重現合成 panel）。通了以後去跑
[`examples/demo.ipynb`](examples/demo.ipynb) 看完整用法。

---

## 為什麼用 factrix

比起自己 rolling 一套 factor-analysis 腳本，主要差在六件事：

1. **一個 `canonical_p` 驅動 `verdict()`** — 每個 factor_type 有**單一** p-value
   當 PASS/FAILED gate（IC t-test / CAAR / Fama-MacBeth / TS β）。不搞「看 IC
   又看 spread 又看 hit rate 最後人工拍板」的 ad-hoc 聚合。其他 signal 質
   / stability / regime 訊息全部走 `profile.diagnose()` 回 structured
   `Diagnostic`，不偷跑進 verdict。

2. **Typed `Profile` dataclass，不是 dict** — `CrossSectionalProfile`、
   `EventProfile` 等 `frozen + slots`，欄位 IDE discoverable，直接餵 polars
   expression 做 filter / rank / BHY。寫 `profile.ic_mean` 打錯 key IDE 就
   會叫，不會 typo 到半夜 debug。

3. **`fl.factor()` session 統一 metric API** — 單因子研究時所有 standalone
   metric 變成 session method（`f.ic()`、`f.quantile_spread()` 等），
   shared Artifacts cache 讓各 method 和 `f.evaluate()` 共用同一份計算；
   per-call `f.quantile_spread(n_groups=3)` 做 sensitivity sweep 不污染
   cache。

4. **Preprocess 和 evaluate 兩步驟 + strict gate** — `fl.preprocess(raw,
   config=cfg)` 把 **`factor_type` + 所有被烙進 prepared 的 preprocess-time
   欄位**（CS 加 `mad_n / return_clip_pct`；MP 加 `demean_cross_section`；
   Event / MC 只有 `forward_periods`）嵌進 prepared 的 `_fl_preprocess_sig`
   marker。`fl.evaluate` / `fl.factor` 逐欄位 diff，任何對不上直接 raise
   並指名哪個欄位、兩邊各是什麼值。擋掉最惡性的那類 bug — 兩邊 config
   silently 對不上、跨 factor_type 誤用（event panel 被 CS 化）、全下游
   metric 無聲無息污染、測出來的 IC 看起來合理但其實量錯 horizon。兩步驟
   保留是為了讓 `prepared` cache 一次、evaluate-time 欄位（`n_groups,
   tie_policy, ortho, regime_labels, …`）可以 sweep。`config=` 在
   `fl.preprocess` 是必填 — 沒有 silent 預設 factor_type。

5. **批次 + BHY 多重檢定一行搞定** — `ps.multiple_testing_correct(p_source=
   "canonical_p", fdr=0.05)` 用 Benjamini-Yekutieli step-up（比 BH 保守、
   容許 dependence）控制 family-wise FDR；`p_source` 白名單只收
   `Profile.P_VALUE_FIELDS`，複合 p 例如 `min(ic_p, spread_p)` 會被拒，不
   讓 user 餵跨 hypothesis 的 p 進 BHY 壞掉 same-test-family 語意。

6. **Short-circuit = NaN, not 0** — metric 算不出來（sample 太小、缺欄位）
   回傳 `NaN` 加 `metadata["reason"]`，BHY 讀 `metadata["p_value"]=1.0`
   保守拒絕。0 是合法的 factor 結果（IC、β 剛好為 0），NaN 才是「跳過」。
   `describe_profile_values` 把 NaN 顯示成 `—`，skips 看得見。

---

## 怎麼選 factor_type

factrix 用兩個軸決定該走哪個 canonical test：**signal geometry**（因子
在 (date, asset) 維度上如何變化）× **data shape**（你手上有多少資產 × 多
長的時序）。當前支援狀況：

| Signal geometry ↓ ／ Data shape → | 寬截面 N ≥ 30（選股典型） | 小 panel 2 ≤ N < 30（跨國配置典型） | 單資產 N = 1（單標的交易典型） |
|---|---|---|---|
| **連續因子**<br/>每 date 每 asset 一個連續值<br/>例：P/E, Momentum, CAPE | `cross_sectional`<br/>IC 非重疊 t-test | `macro_panel`<br/>Fama-MacBeth Newey-West | **[coverage gap]**<br/>詳見下方 Fallback 清單 |
| **稀疏事件**<br/>有號觸發 {−, 0, +}<br/>例：earnings、cross、policy shift | `event_signal`<br/>CAAR 非重疊 t-test | `event_signal`<br/>CAAR + `clustering_hhi` 診斷 | `event_signal` **[degraded]**<br/>`clustering_hhi` 自動停用、CAAR 退化成單資產時間平均（不再是 "cross-sectional average"） |
| **共用時序**<br/>同一 date 所有 asset 共用同一 factor 值<br/>例：VIX, DXY, 黃金 | `macro_common`<br/>跨資產 t-test on per-asset TS β | `macro_common`<br/>同上，但 β 分佈離散度要人工判讀 | `macro_common` **[fallback]**<br/>canonical p 被**保守地壓到 1.0**，verdict **幾乎一定 FAILED**；改看 `ts_beta_tstat` 作為替代指標 |

`fl.describe_factor_types()` / `fl.describe_profile("event_signal")` 印出
對應的 Profile dataclass 欄位與 canonical p 名稱。

### Fallback 行為：誠實清單

- **連續因子 × 單資產**（coverage gap）：你用 `macro_common` 硬塞時，`factor` 會被當時序 z-score，`ts_beta` 轉成自相關測試（不是 cross-sectional 溢酬）— 結果會跑，但語意並非你要的「這個連續 factor 在這支 asset 上有 predictive 力嗎」。這個 cell 是否要補成 `single_asset` 第 5 類，視使用者 friction log 累積信號決定。
- **Event N=1**：CAAR t-test 仍跑，但你失去「跨資產事件一致性」這層證據。`clustering_hhi` 屬於 cross-sectional clustering 的診斷，單資產下無意義故自動停用；若你擔心同一 asset 事件序列自相關（earnings pre-announcement leak 之類），目前沒有內建檢測。
- **Macro common N=1**：底層切到 `ts_beta_single_asset_fallback`，這是**保守 conservative** 的處理 — 它不產生假陽性，但也幾乎不給真陽性。別用 `verdict() == "PASS"` 判斷，改讀 `ts_beta_tstat` / `factor_adf_p` 的實際值。
- **CS / MP 丟 N=1 panel**：會在 `build_artifacts` raise `ValueError` 並指向 `MacroCommonConfig` — 不讓你拿到「silently FAILED」的誤導性結果。
- **樣本 T 不足**（少於 `MIN_IC_PERIODS=10` / `MIN_FM_PERIODS=20` / `MIN_TS_OBS=20` / `MIN_EVENTS=10`）：對應 metric 短路成 `NaN` 並在 `metadata.reason` 標 `insufficient_*`；canonical p 走 conservative default (1.0)，不會產生假陽性。逐欄位 N/T 下限 + fallback 對照見 [`docs/metric_applicability.md`](docs/metric_applicability.md)。

---

## 統計方法與參考文獻

factrix 的每個指標都對應業界 / 學界認可的方法。完整論述（每個 canonical test 的論文依據、實際採用的部分、經評估但刻意未實作的部分、完整參考文獻）放在獨立文件：

- [`docs/statistical_methods.md`](docs/statistical_methods.md) — 逐方法引用、採用細節、完整文獻清單
- [`docs/metric_applicability.md`](docs/metric_applicability.md) — 每個 Profile 欄位的 N/T 下限 + fallback 行為

下面留的「樣本數守門」只是常查閾值的快捷表；其他內容請走上面兩個檔。

### 樣本數守門（統一閾值，見 `_types.py`）

- `MIN_IC_PERIODS = 10`、`MIN_EVENTS = 10`、`MIN_OOS_PERIODS = 5`、`MIN_PORTFOLIO_PERIODS = 5`、`MIN_MONOTONICITY_PERIODS = 5`
- `MIN_FM_PERIODS = 20`（Fama-MacBeth λ 序列，見 `metrics/fama_macbeth.py`）、`MIN_TS_OBS = 20`（per-asset TS regression，見 `metrics/ts_beta.py`）
- 未達門檻時，metric 會 short-circuit 並在 `metadata.reason` 標記 `insufficient_*`；對應 profile 欄位走 conservative default（p=1.0），不會產生假陽性。

---

## Scope & non-goals

### Scope
- **Daily-to-monthly bar-based factor evaluation**。`forward_periods` 是
  **rows，不是 calendar time** — 週頻 panel 寫 `forward_periods=1` 就是
  1-week 前向報酬，同一套 API。
- **訊號幾何 × 資料形狀**覆蓋四種主要 cell（見上方表）；單資產連續因子
  有明確標示的 coverage gap。
- **統計嚴謹的 PASS/FAILED gate** + structured `diagnose()` + BHY FDR 控
  制，作為後續 allocation / strategy layer 的可信輸入。

### 明確不做（不只是「沒做」— 是設計上決定不做）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| Portfolio optimization（MVO / Black-Litterman / HRP / Risk Parity） | 生態已成熟；lib 定位為 validator 不是 optimizer | `skfolio` / `PyPortfolioOpt` / `riskfolio-lib` / `cvxpy` |
| ML 信號層（xgboost / lightgbm + SHAP） | 複雜特徵工程 + 可解釋性不是 validator 職責 | `xgboost` + `shap` |
| Regime detection methodology（HMM / threshold / K-means state） | 方法論學術上不收斂，不內建以免 bake-in 未驗證選擇；`regime_labels` 只提供 input contract | `hmmlearn` / 自寫 threshold rule |
| Structural break detection（Chow / Quandt-Andrews / Bai-Perron） | 屬時序 regime analysis 領域，out of scope | `ruptures` |
| GARCH / wild bootstrap SE | 保持 core 依賴精簡（polars + numpy + pandera）；進階推論外接 | `arch` |
| Predictive regression with persistent predictor 自動修正（IVX / Stambaugh correction） | 透過 `factor_adf_p` 只做 flagging，避免 false confidence | `arch` / R `ivx` |
| Backtest / execution simulation / slippage / margin | 明確非 scope；`turnover` / `breakeven_cost` 是 screening proxy | `vectorbt` / `bt` / Zipline / Backtrader |
| Intraday / HFT（tick-level、sub-second） | per-date CS IC / CAAR / FM λ 的語意在 tick data 上不成立 | 另找專用工具 |
| 跨 factor 合成信號 / factor combiner（產生單一 composite signal）| 屬 signal layer，validator 不碰。注意：factrix 的 `redundancy_matrix` 是 **diagnostic**（量化因子之間的重疊程度），不是 combiner — 它告訴你哪些因子該合、哪些該剃，但不產生合成信號 | 自寫、或用 `scikit-learn` 的 regression |

這份清單不是 TODO，是**承諾不擴張的 scope 邊界**。擴張請先更新
[`ARCHITECTURE.md`](ARCHITECTURE.md) 的 Invariants 區塊，並在
`awwesomeman/factor-analysis` workspace 的 `docs/` 新增 `spike_*.md` 設計文件。

---

## 下一步看哪裡

本專案 metric 相關資訊**刻意分三層**，每層職責不重疊 — 查不同問題走不同入口：

| 想知道 | 看 |
|--------|-----|
| 怎麼用（可執行例子） | [`examples/demo.ipynb`](examples/demo.ipynb) — 四種 factor_type × Level 0–6 的可執行功能索引，`fl.datasets` 產資料、從 fresh clone 可以直接跑 |
| **精確公式 / 演算法 / 邊界 case**（authoritative source）| 對應 module docstring — `help(factrix.metrics.ic)`、`help(factrix.metrics.caar)`、… |
| **為什麼選這個方法**（論文依據、與原式 deviation、經評估但未實作）| [`docs/statistical_methods.md`](docs/statistical_methods.md) |
| **這個指標我的資料夠不夠用**（N/T 下限、fallback 語意）| [`docs/metric_applicability.md`](docs/metric_applicability.md) |
| Profile 欄位、Config 介面 | `fl.describe_profile(<type>)`、`help(fl.CrossSectionalConfig)` 等 |
| 內部結構（module layout、invariants、Profile contract、Artifacts 策略）| [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| 想貢獻（dev workflow、submodule、test 規範、release、docstring 風格）| [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 版本變動 | [`CHANGELOG.md`](CHANGELOG.md) |
