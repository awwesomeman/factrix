# factorlib

**Modular factor evaluation toolkit** — 把 cross-sectional / event-signal /
macro-panel / macro-common 四種因子放在同一套 `preprocess → evaluate →
Profile` API 底下，polars-native，每一步都典型有 statistical-discipline 的
默認值。

> **定位：Factor Signal Analyzer**，不是回測引擎。`turnover` /
> `breakeven_cost` / `net_spread` 是理想化的 proxy（等權、無滑價），screening
> 用；不是 tradable P&L。要 realistic execution 請把 screened 因子餵進
> Zipline / Backtrader / 自家引擎。

---

## Install

```bash
pip install factorlib              # core (polars only)
pip install factorlib[charts]      # + plotly 圖表
pip install factorlib[mlflow]      # + mlflow tracking
pip install factorlib[all]         # 全包
```

---

## 30-second smoke test

確認裝好了、環境跑得起來：

```python
import factorlib as fl

raw = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
cfg = fl.CrossSectionalConfig(forward_periods=5)
profile = fl.evaluate(fl.preprocess(raw, config=cfg), "sanity", config=cfg)

print(profile.verdict(), '| ic_mean =', round(profile.ic_mean, 4))
# → PASS | ic_mean = 0.0722
```

三行，不需要任何外部資料（`fl.datasets` 產可重現合成 panel）。通了以後去跑
[`examples/demo.ipynb`](examples/demo.ipynb) 看完整用法。

---

## 為什麼用 factorlib

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
   config=cfg)` 把**所有被烙進 prepared 的 preprocess-time 欄位**（CS 是
   `forward_periods / mad_n / return_clip_pct`；Event / MC 是
   `forward_periods`；MP 另加 `demean_cross_section`）都嵌進 prepared 的
   `_fl_preprocess_sig` marker。`fl.evaluate` / `fl.factor` 逐欄位 diff，
   任何對不上直接 raise 並指名哪個欄位、兩邊各是什麼值。擋掉最惡性的那
   類 bug — 兩邊 config silently 對不上、全下游 metric 無聲無息污染、
   測出來的 IC 看起來合理但其實量錯 horizon。兩步驟保留是為了讓
   `prepared` cache 一次、evaluate-time 欄位（`n_groups, tie_policy, ortho,
   regime_labels, …`）可以 sweep。

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

## 四種 factor_type

| Type | 訊號型態 | 典型用法 | Canonical test |
|------|---------|----------|----------------|
| `cross_sectional` | 每期每資產連續值 | momentum / value / size | IC 非重疊 t-test |
| `event_signal` | 離散觸發 `{-1, 0, +1}` | 盈餘公告、黃金交叉 | CAAR 非重疊 t-test |
| `macro_panel` | 連續值、小截面 N<30 | 跨國 CPI、利差配置 | Fama-MacBeth Newey-West |
| `macro_common` | 單一時序、全資產共用 | VIX、黃金、USD index | 截面 t-test on per-asset TS β |

```python
fl.describe_factor_types()           # 所有 factor_type 概觀
fl.describe_profile("event_signal")  # 指定 type 的 Profile dataclass 欄位 + canonical p
```

---

## 統計方法與參考文獻

factorlib 的每個指標都對應到一篇業界/學界認可的方法。以下列出所有**實際採用**的統計方法、以及**經評估但刻意未實作**的部分，讓你在把結果帶進研究或報告時能明確交代技術背景。

### Cross-sectional（截面選股）

每篇論文下列兩項子點：**觀點**（論文核心論述）與 **採用**（本專案實際取用的部分）。完整論文題目見節末「完整參考文獻」。

#### Spearman Rank IC — `ic_mean`、per-date 因子-報酬 rank 相關（預設不假設線性）

- **Ambachtsheer (1977)**, *JPM* 4(1)
  - 觀點：提出以預測值與實現報酬間的相關係數作為 skill 度量，IC 概念的最早正式化。
  - 採用：以 per-date IC 作為因子品質的核心量度。
- **Grinold (1989)**, *JPM* 15(3)
  - 觀點：IR ≈ IC·√breadth，將 IC 連結到投資組合 Sharpe-like 績效。
  - 採用：`ic_ir` 作為 IC 時間序列穩定度的標準化量度（signed mean/std）。
- **Grinold & Kahn (2000)**, *Active Portfolio Management* (2nd ed.), McGraw-Hill
  - 觀點：rank-based IC 對 factor outlier 的 robustness 論述；per-date 預處理與 cross-section standardization 的標準流程。
  - 採用：預設 Spearman（非 Pearson）；per-date z-score 預處理管線（`preprocess` step 3-5）。

#### 非重疊 t-test (`ic_p`) — 消除 overlapping forward returns 的序列自相關

- **Hansen & Hodrick (1980)**, *JPE* 88(5)
  - 觀點：K-期前向預測回歸的殘差必然存在 MA(K−1) 結構，未校正的 t-stat 會系統性高估顯著性。
  - 採用：此為 `ic_p` 和 `ic_nw_p` 雙路徑的動機基石。
- **Richardson & Stock (1989)**, *JFE* 25(2)
  - 觀點：多年重疊報酬的漸近推論校正；樣本切割至獨立子集為最保守可行做法。
  - 採用：`_sample_non_overlapping` 以 `forward_periods`-步取樣，正是 Richardson-Stock 的最保守樣本切割形式。

#### Newey-West HAC t-test — `ic_nw_p`、FM λ、time-series 檢定

- **White (1980)**, *Econometrica* 48(4)
  - 觀點：OLS 在異質變異下點估計仍 unbiased，但 SE 錯誤；建議用三明治估計 (X'X)⁻¹(X'ΩX)(X'X)⁻¹。
  - 採用：White 的 HC0 是 Newey-West 的起點；我們的 HAC 實作繼承三明治結構。
- **Newey & West (1987)**, *Econometrica* 55(3)
  - 觀點：Bartlett kernel `w_j = 1 − j/(L+1)` 確保 HAC 共變異數為正半定，同時處理異質變異與自相關。
  - 採用：`_newey_west_se` 與 `_newey_west_t_test` 直接實作 Bartlett kernel；`lags = ⌊T^(1/3)⌋` 為其經典 rule-of-thumb。
- **Newey & West (1994)**, *RES* 61(4)
  - 觀點：提出 `lags = ⌊T^(1/3)⌋` 作為 practical rule-of-thumb，兼顧 bias-variance。
  - 採用：作為 lag 預設公式；**`ic_nw_p` 再疊加 Hansen-Hodrick (1980) 下界** `h − 1`（overlap 週期），取 `lags = max(⌊T^(1/3)⌋, forward_periods − 1)`。
- **Andrews (1991)**, *Econometrica* 59(3)
  - 觀點：data-adaptive MSE-optimal bandwidth；以 AR(1)/VAR pre-whitening 後的解析式選 lag。
  - 採用：未直接實作（需 pre-whitening 步驟）；僅作為 data-adaptive 方向的文獻背景，讓進階使用者知道有更精準的選法可外接。

#### Quantile Spread / Monotonicity / Top Concentration — 輔助診斷

- **Grinold & Kahn (2000)**, ch. 5-6
  - 觀點：quantile-based 分析揭示 IC 無法捕捉的「訊號集中在哪個尾端」資訊。
  - 採用：long-short spread (Q_top − Q_bottom)、quantile 報酬 profile、top-bucket HHI⁻¹。
- **Patton & Timmermann (2010)**, *JFE* 98(3)
  - 觀點：提出 bootstrap-based MR test，嚴格檢驗「所有相鄰 quantile 差」皆同向。
  - 採用：**未實作完整 MR test**。我們的 `monotonicity` 只是 `Spearman(group_index, group_return)` 的趨勢性指標，與 P-T MR test 無統計對應關係；文獻引為 awareness，不作為方法依據。

### Event study（事件驅動）

#### CAAR non-overlapping t-test (`caar_p`)

- **Fama, Fisher, Jensen & Roll (1969)**, *IER* 10(1)
  - 觀點：事件研究方法的原始論文；定義了 AR、CAR、event window、estimation window 等基礎概念。
  - 採用：整個事件研究架構的語彙皆沿用此文。
- **Brown & Warner (1980)**, *JFE* 8(3)
  - 觀點：**月頻**資料下不同 AR 估計模型（mean-adjusted / market-adjusted / market-model）的 size & power 比較；簡單方法在月頻即已足夠。
  - 採用：支撐「CAAR t-test 在合理樣本下 well-specified」的實證基礎。
- **Brown & Warner (1985)**, *JFE* 14(1)
  - 觀點：**日頻**資料下事件研究的推論挑戰（nonsynchronous trading、fatter tails）；parametric t-test 仍 well-specified 只要樣本足。
  - 採用：我們允許日頻輸入 (`forward_periods=1` 即可) 的理論背書。
- **MacKinlay (1997)**, *JEL* 35(1)
  - 觀點：現代事件研究教科書式整合；標準化 event window × estimation window 的切割與推論流程。
  - 採用：`EventConfig` 的 `estimation_window`、`event_window_post` 等欄位語意完全沿用 MacKinlay。
- **Campbell, Lo & MacKinlay (1997)**, *The Econometrics of Financial Markets* (ch. 4), Princeton University Press
  - 觀點：CAAR 推論限制（event-induced variance、clustering 等）的完整論述。
  - 採用：`EventProfile` 將 CAAR 與 BMP / clustering HHI 並列為 canonical + diagnostic 雙軌，正是 Campbell-Lo-MacKinlay 警告的落地。

#### BMP Standardized AR test (`bmp_p`)

- **Patell (1976)**, *JAR* 14(2)
  - 觀點：以估計期 σ 將 AR 標準化，並乘上 prediction-error scaling √(1 + 1/M + (R_m−R̄)²/Σ(R_m−R̄)²) 處理市場模型 OOS 預測誤差。
  - 採用：概念前身；我們**省略 prediction-error scaling**（估計期僅取 mean-adjusted，無 market model），屬 "mean-adjusted SAR" 簡化版。
- **Boehmer, Musumeci & Poulsen (1991)**, *JFE* 30(2)
  - 觀點：事件本身會放大異常報酬的 cross-sectional variance，Patell t 會 over-reject；提出以**事件窗 SAR 的 cross-sectional std** 作分母的修正 t-statistic。
  - 採用：`bmp_p` 實作其跨事件標準化結構；以**常態近似**（`_p_value_from_z`）取 p-value，N≥`MIN_EVENTS`=10 時 z 與 t 誤差在可接受範圍內。

#### Corrado 非參數 rank test (`corrado_rank_test`)

- **Corrado (1989)**, *JFE* 23(2)
  - 觀點：不假設 AR 常態分佈的 rank-based test；論文 eq.(5) 的 SE 以 per-date 平均 rank 的時序 std 估計。
  - 採用：`corrado_rank_test` 採 **pooled-rank 簡化版**（分母為 pooled rank series std），與論文原式略有差異；適合快速 robustness screen，嚴格尺寸檢定建議外接專用套件。
- **Corrado & Zivney (1992)**, *JFQA* 27(3)
  - 觀點：比較 sign test 與 rank test 在日頻 AR 下的功效；rank test 在多數情境下 power 較高。
  - 採用：基於此決定採用 rank test 而非 sign test。

#### Event Clustering HHI

- **Herfindahl (1950)**, Columbia University PhD dissertation；**Hirschman (1945)**, University of California Press
  - 觀點：以 Σs² 量度集中度（產業 / 貿易）；越高越集中於少數主體。
  - 採用：方法論轉借至**時間維度**，計算事件日集中度（高 HHI → 事件聚集在少數交易日）。
- **Kolari & Pynnönen (2010)**, *RFS* 23(11)
  - 觀點：event clustering 違反 BMP 原本假設的跨事件獨立性；提出修正 z-statistic。
  - 採用：`clustering_hhi` diagnostic 用於**偵測**；Kolari-Pynnönen 的修正本身保留為未實作 config 選項（`adjust_clustering='kolari_pynnonen'`）。

#### Event Hit Rate / Profit Factor

- 方向勝率走 binomial test（H₀: p=0.5）、Profit Factor 為總獲利 / 總虧損；屬業界實戰診斷，無單一學術出處。

### Macro panel（跨國 / 跨市場配置）

#### Fama-MacBeth 兩階段回歸 (`fm_beta_p`)

- **Fama & MacBeth (1973)**, *JPE* 81(3)
  - 觀點：兩階段估計：每期 cross-sectional 回歸得到 λ_t，再對 λ 時序做 t-test；相較 pooled 可隔離截面相關性。
  - 採用：`fama_macbeth` 採用其兩階段架構，並於 stage-2 加 Newey-West；**未實作 Shanken (1992) errors-in-variables 校正**（βs 視為已知，適合 pre-computed factor exposures 的情境）。
- **Fama & French (1992)**, *JoF* 47(2) 與 **Fama & French (1993)**, *JFE* 33(1)
  - 觀點：FM 程序在跨 stock panel 上的現代標準用法；建立 size、book-to-market 等 anomaly 的推論模板。
  - 採用：Config 預設、variable 命名、預期使用情境皆以 FF 風格 panel 為基準。
- **Cochrane (2005)**, *Asset Pricing* (Revised ed.), ch. 12, Princeton University Press
  - 觀點：論述 FM 與 GMM、MLE 等估計法的等價性與漸近性質；FM 在 panel 稀疏時計算穩定。
  - 採用：選 FM 而非 GMM 即因其在跨國 / 跨資產這種 panel 不整齊情境下 robust。
- **Petersen (2009)**, *RFS* 22(1)
  - 觀點：系統比較 FM、clustered、White SE；FM 配 Newey-West 對時序相依 panel 是低 bias、計算穩定的組合。
  - 採用：支撐我們「FM λ 時序一律走 Newey-West」的預設。

#### Newey-West 於 λ 時序

- **Newey & West (1987)** — 同前；λ 時序的 lags 從 `forward_periods` 導出以覆蓋重疊週期結構。

#### Pooled OLS 對比

- 非單一論文觀點；FM λ 與 pooled β 正負號不一致時觸發 `macro_panel.fm_pooled_sign_mismatch` veto rule，作為 robustness check。

### Macro common（共用時序因子）

#### Per-asset TS OLS β、Cross-sectional t-test on β 分佈 (`ts_beta_p`)

- **Black, Jensen & Scholes (1972)**, in *Studies in the Theory of Capital Markets* (M. Jensen, ed.), Praeger
  - 觀點：先以時序回歸估 per-asset β（stage 1），再對 β 分佈或其與報酬的關係做推論（stage 2）；適用於**因子本身在截面上沒變化**的情境（例如市場 premium）。
  - 採用：`ts_beta_p` 取其 stage-2 精神，對跨資產 β 分佈做 H₀: mean(β)=0 的 t-test — macro_common 的共用因子在截面上恆定，正符合 BJS 的適用條件。

#### Augmented Dickey-Fuller (`factor_adf_p`) — 因子單位根檢驗

- **Dickey & Fuller (1979)**, *JASA* 74(366)
  - 觀點：OLS β̂ 在單位根下有非標準分佈；提供 tau 統計量的漸近臨界值表。
  - 採用：`_adf` 檢定 H₀: β=0 於 Δy_t = α + β·y_{t−1} + ε 的規格。
- **Said & Dickey (1984)**, *Biometrika* 71(3)
  - 觀點：ADF 可經由加入 Δy 滯後項擴展至 ARMA error 情境。
  - 採用：`_adf(y, lags=k)` 保留 k>0 的擴展 API，預設 k=0 但可手動提升為 Said-Dickey 風格。
- **MacKinnon (1996)**, *JAE* 11(6)
  - 觀點：以 response-surface 回歸給出 ADF 的精確臨界值與 p-value 公式，取代查表。
  - 採用：`_adf_pvalue_interp` 對 MacKinnon constant-only 規格的 7 個臨界點做線性插值，精度約 ±0.03 於決策邊界 (5%/10%)，且 p-value 上限 clip 在 0.95（右尾未完全覆蓋）；足夠定性判斷「是否有單位根」。**不依賴 `statsmodels`**。

#### Stambaugh bias 警示

- **Stambaugh (1999)**, *JFE* 54(3)
  - 觀點：預測回歸 R_{t+1} = α + β·X_t + ε 當 X 高度 persistent 且 X 的 innovation 與 ε 同期相關時，OLS β̂ 有有限樣本偏誤，並給出 bias 的 closed-form。
  - 採用：以 `factor_adf_p > 0.10` 為 trigger、`macro_common.factor_persistent` rule 提示使用者此風險；**刻意不自動修正**，避免 false confidence。
- **Campbell & Yogo (2006)**, *JFE* 81(1)
  - 觀點：對 Stambaugh 情境提供 Bonferroni-type 有效檢定（Q-test），功效高於簡單 t 且 size 正確。
  - 採用：文件指引給進階使用者；本套件不內建 Q-test。
- **Kostakis, Magdalinos & Stamatogiannis (2015)**, *RFS* 28(5)；延伸自 **Phillips & Magdalinos (2009)** 的 IVX 框架
  - 觀點：以工具變數（IVX）消除 persistent predictor 引起的 near-unit-root bias，統計性質不依賴 persistence degree。
  - 採用：IVX 未內建；建議使用者如需 Stambaugh 修正，優先外接 IVX 實作（如 `arch` / R `ivx` package）。

### Multiple testing（多重檢定校正）

#### Benjamini-Yekutieli (BHY)

- **Benjamini & Hochberg (1995)**, *JRSS B* 57(1)
  - 觀點：提出 FDR = E[V/R] 作為比 FWER 寬鬆但仍可控的多重檢定指標；BH step-up 程序要求 p-value 獨立或 PRDS。
  - 採用：BH 是 BHY 的基礎；我們因為不假設因子間獨立而不選 BH。
- **Benjamini & Yekutieli (2001)**, *AoS* 29(4)
  - 觀點：將 BH 的獨立性假設鬆綁至任意相依性，代價是 threshold 除以 c(m) = Σ_{i=1..m}(1/i)。
  - 採用：`bhy_adjust` / `bhy_adjusted_p` 直接實作；per-hypothesis adjusted p 透過 right-cummin 維持 rank 單調性。作為因子池（常高度相關）的預設 FDR 控制。
- **Efron (2010)**, *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*, Cambridge University Press
  - 觀點：1000+ 假設尺度下的 empirical Bayes 處理；locally FDR 與 global FDR 的權衡。
  - 採用：論述支持 FDR 框架在 zoo-scale 的適用性；未直接實作 empirical Bayes 方法。

#### Two-stage screening 支援

- `multiple_testing_correct(n_total=...)`：前篩後仍在原始候選池尺度做 BHY，避免倖存者偏差造成 FDR 低估。前提是前篩條件需與待校正 p-value **邊際獨立**。

#### `verdict()` 預設 t 門檻 2.0

- **Harvey, Liu & Zhu (2016)**, *RFS* 29(1)
  - 觀點：面對 300+ 已發表因子的 data mining 壓力，單因子宣稱應將 t 門檻上調至 3.0；並提出 BHY-based adjusted threshold。
  - 採用：預設門檻 2.0（單因子 95% 經典邊界），將多重檢定的嚴格把關交給 `multiple_testing_correct`（BHY）。
- **Harvey (2017)**, *JoF* 72(4)
  - 觀點：呼籲因子發現的統計誠實性：區分 "exploratory" 與 "confirmatory"，並鼓勵公開失敗研究以減少 publication bias。
  - 採用：「framework detects, user decides」設計哲學與此呼應；`integrations/mlflow.py` 的 `log_failed_run` 提供技術層面的 audit trail（與 Harvey 訴求的 publication 層面不同，但減少 survivorship 偏差的精神一致）。

### Preprocessing（前處理）

#### MAD Winsorize — 每日 median ± k·MAD·1.4826 裁剪

- **Huber (1964)**, *AoMS* 35(1) 與 **Huber (1981)**, *Robust Statistics*, Wiley
  - 觀點：robust estimator 理論；breakdown point 50% 的 MAD 對污染資料比 OLS 標準差穩健得多。
  - 採用：`mad_winsorize` 以 median + MAD 定義裁切邊界；對股票報酬的 fat tail 特別適合。
- **Hampel (1974)**, *JASA* 69(346)
  - 觀點：導出 influence curve 概念並確立 MAD 在常態下的 unbiased consistency constant 1.4826 = 1/Φ⁻¹(0.75)。
  - 採用：`MAD_CONSISTENCY_CONSTANT = 1.4826`（定義於 `_types.py`），直接取自 Hampel 的結果。
- **Rousseeuw & Croux (1993)**, *JASA* 88(424)
  - 觀點：系統比較 MAD、Q_n、S_n 等 robust scale estimator 的效率與 bias；Q_n 效率更高但計算 O(n log n)。
  - 採用：選 MAD 而非 Q_n 是 efficiency-for-speed 取捨；R-C 1993 是此決策的 secondary reference。

#### Forward return 百分位裁切

- 與因子裁切**分開管理**；避免用「未來報酬的分佈」決定「現在因子如何裁切」造成的 look-ahead bias。業界標準做法，無單一學術出處。

#### Cross-section standardization

- **Grinold & Kahn (2000)**, ch. 3-4
  - 觀點：per-date z-score 降低跨時段因子水準漂移對 rank correlation 的干擾。
  - 採用：`preprocess` pipeline step 5 內建；IC 計算前的標準前處理。

### 樣本數守門（統一閾值，見 `_types.py`）

- `MIN_IC_PERIODS = 10`、`MIN_EVENTS = 10`、`MIN_OOS_PERIODS = 5`、`MIN_PORTFOLIO_PERIODS = 5`、`MIN_MONOTONICITY_PERIODS = 5`
- `MIN_FM_PERIODS = 20`（Fama-MacBeth λ 序列，見 `metrics/fama_macbeth.py`）、`MIN_TS_OBS = 20`（per-asset TS regression，見 `metrics/ts_beta.py`）
- 未達門檻時，metric 會 short-circuit 並在 `metadata.reason` 標記 `insufficient_*`；對應 profile 欄位走 conservative default（p=1.0），不會產生假陽性。

### 經評估但刻意未實作

以下方法**在設計討論中有考慮**，但因增量收益有限或會偏離 lean-dependency 原則而未納入。若你的 workflow 真的需要，我們建議使用 `statsmodels` / `arch` / `linearmodels` 等專門套件在 factorlib 輸出之外處理。完整 citation 見節末「完整參考文獻」。

- **Kolari & Pynnönen (2010) clustering adjustment** — `EventConfig.adjust_clustering='kolari_pynnonen'` 目前是**保留 config 選項**，實際調整尚未實作。建議使用者在事件 HHI 過高時改用 BMP 的橫截面 z-score 版本或自行做 calendar block bootstrap。
- **Stationary Block Bootstrap** — 完整 Politis & Romano (1994) 實作未納入；`_stats.py` 的 ADF p-value 採 MacKinnon 臨界值插值，精度 ±0.03 對「是否有單位根」的定性判斷足夠。
- **Hansen SPA / Superior Predictive Ability** — Hansen (2005) 對整個 factor zoo 做 data-snooping 校正；目前以 BHY 為主要 FDR 工具。
- **Stambaugh reverse-regression 自動修正** — 僅以 ADF 診斷 flag 提示，不自動做回歸重估。要完整修正請參考 Stambaugh (1999) 的 Method of Moments 形式或改用 IVX (Phillips & Magdalinos 2009) 預測回歸。
- **HAC SE on per-asset β for macro_common** — `compute_ts_betas` 維持 OLS SE。Stambaugh bias 的根源不是 SE 估計（OLS β̂ 仍然 unbiased），而是 predictor 與 innovation 的同期相關；HAC 只修 SE 不修 bias，我們刻意不加以免 false confidence。

---

## 完整參考文獻

依作者姓氏字母排序。上方「統計方法」正文的簡寫 citation 對應此清單；期刊縮寫採業界慣用（JFE = *Journal of Financial Economics*、JPE = *Journal of Political Economy* 等）。

- Ambachtsheer, K. P. (1977). "Where Are the Customers' Alphas?" *Journal of Portfolio Management* 4(1).
- Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59(3).
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B* 57(1).
- Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing Under Dependency." *Annals of Statistics* 29(4).
- Black, F., Jensen, M. C. & Scholes, M. (1972). "The Capital Asset Pricing Model: Some Empirical Tests." In M. Jensen (ed.), *Studies in the Theory of Capital Markets*. Praeger.
- Boehmer, E., Musumeci, J. & Poulsen, A. B. (1991). "Event-study Methodology Under Conditions of Event-induced Variance." *Journal of Financial Economics* 30(2).
- Brown, S. J. & Warner, J. B. (1980). "Measuring Security Price Performance." *Journal of Financial Economics* 8(3).
- Brown, S. J. & Warner, J. B. (1985). "Using Daily Stock Returns: The Case of Event Studies." *Journal of Financial Economics* 14(1).
- Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
- Campbell, J. Y. & Yogo, M. (2006). "Efficient Tests of Stock Return Predictability." *Journal of Financial Economics* 81(1).
- Cochrane, J. H. (2005). *Asset Pricing* (Revised ed.). Princeton University Press.
- Corrado, C. J. (1989). "A Nonparametric Test for Abnormal Security-price Performance in Event Studies." *Journal of Financial Economics* 23(2).
- Corrado, C. J. & Zivney, T. L. (1992). "The Specification and Power of the Sign Test in Event Study Hypothesis Tests Using Daily Stock Returns." *Journal of Financial and Quantitative Analysis* 27(3).
- Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74(366).
- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press.
- Fama, E. F., Fisher, L., Jensen, M. C. & Roll, R. (1969). "The Adjustment of Stock Prices to New Information." *International Economic Review* 10(1).
- Fama, E. F. & French, K. R. (1992). "The Cross-Section of Expected Stock Returns." *Journal of Finance* 47(2).
- Fama, E. F. & French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics* 33(1).
- Fama, E. F. & MacBeth, J. D. (1973). "Risk, Return, and Equilibrium: Empirical Tests." *Journal of Political Economy* 81(3).
- Grinold, R. C. (1989). "The Fundamental Law of Active Management." *Journal of Portfolio Management* 15(3).
- Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.
- Hampel, F. R. (1974). "The Influence Curve and its Role in Robust Estimation." *Journal of the American Statistical Association* 69(346).
- Hansen, L. P. & Hodrick, R. J. (1980). "Forward Exchange Rates as Optimal Predictors of Future Spot Rates: An Econometric Analysis." *Journal of Political Economy* 88(5).
- Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics* 23(4).
- Harvey, C. R. (2017). "Presidential Address: The Scientific Outlook in Financial Economics." *Journal of Finance* 72(4).
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies* 29(1).
- Herfindahl, O. C. (1950). *Concentration in the U.S. Steel Industry*. PhD dissertation, Columbia University.
- Hirschman, A. O. (1945). *National Power and the Structure of Foreign Trade*. University of California Press.
- Huber, P. J. (1964). "Robust Estimation of a Location Parameter." *Annals of Mathematical Statistics* 35(1).
- Huber, P. J. (1981). *Robust Statistics*. Wiley.
- Kolari, J. W. & Pynnönen, S. (2010). "Event Study Testing with Cross-sectional Correlation of Abnormal Returns." *Review of Financial Studies* 23(11).
- Kostakis, A., Magdalinos, T. & Stamatogiannis, M. P. (2015). "Robust Econometric Inference for Stock Return Predictability." *Review of Financial Studies* 28(5).
- MacKinlay, A. C. (1997). "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1).
- MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11(6).
- Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3).
- Newey, W. K. & West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies* 61(4).
- Patell, J. M. (1976). "Corporate Forecasts of Earnings Per Share and Stock Price Behavior: Empirical Tests." *Journal of Accounting Research* 14(2).
- Patton, A. J. & Timmermann, A. (2010). "Monotonicity in Asset Returns: New Tests with Applications to the Term Structure, the CAPM, and Portfolio Sorts." *Journal of Financial Economics* 98(3).
- Petersen, M. A. (2009). "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches." *Review of Financial Studies* 22(1).
- Phillips, P. C. B. & Magdalinos, T. (2009). "Econometric Inference in the Vicinity of Unity." Working paper, Singapore Management University.
- Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association* 89(428).
- Richardson, M. & Stock, J. H. (1989). "Drawing Inferences from Statistics Based on Multiyear Asset Returns." *Journal of Financial Economics* 25(2).
- Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the Median Absolute Deviation." *Journal of the American Statistical Association* 88(424).
- Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order." *Biometrika* 71(3).
- Shanken, J. (1992). "On the Estimation of Beta-Pricing Models." *Review of Financial Studies* 5(1).
- Stambaugh, R. F. (1999). "Predictive Regressions." *Journal of Financial Economics* 54(3).
- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica* 48(4).

---

## Scope & non-goals

- **Scope**：daily-to-monthly bar-based factor evaluation。`forward_periods`
  是 **rows，不是 calendar time** — 週頻 panel 寫 `forward_periods=1` 就是
  1-week 前向報酬，同一套 API。
- **非 scope**：intraday / high-frequency（tick-level、sub-second）沒測、
  不支援。per-date CS IC / CAAR / FM λ 的語意在 tick data 上不成立。
- **非 scope**：回測、執行成本建模、滑價、margin call、real portfolio
  optimization。Screening 結果自己帶去回測系統。

---

## 下一步看哪裡

| 想知道 | 看 |
|--------|-----|
| 怎麼用（可執行例子） | [`examples/demo.ipynb`](examples/demo.ipynb) — 四種 factor_type × Level 0–6 的可執行功能索引，`fl.datasets` 產資料、從 fresh clone 可以直接跑 |
| 內部結構（module layout、invariants、Profile contract、Artifacts 策略） | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| 想貢獻（dev workflow、submodule、test 規範、release） | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 版本變動 | [`CHANGELOG.md`](CHANGELOG.md) |
| 個別 metric 的數學、interpretation、corner cases | 對應 module 的 docstring：`help(factorlib.metrics.ic)`、`help(factorlib.metrics.caar)`、... |
