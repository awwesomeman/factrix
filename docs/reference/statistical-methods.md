# factrix 統計方法與參考文獻

factrix 的每個指標都對應業界 / 學界認可的方法。本文列出所有**實際採用**的方法、**經評估但刻意未實作**的部分、以及完整文獻清單，方便你把結果帶進研究或報告時能明確交代技術背景。

> 本文職責：metric 的**論文依據**與**採用取捨**（含與論文的 deviation）。**公式細節**不在這裡 — 請讀對應 `.py` module 的 docstring（`help(factrix.metrics.<name>)`）。文件整體分工見 [README.md](https://github.com/awwesomeman/factrix/blob/main/README.md)。

每個方法下列兩項子點：**觀點**（論文核心論述）與 **採用**（本專案實際取用的部分）。完整論文題目見節末「完整參考文獻」。

---

## Cross-sectional（截面選股）

### Spearman Rank IC — `ic_mean`、per-date 因子-報酬 rank 相關（預設不假設線性）

精確實作：`factrix/metrics/ic.py::compute_ic` / `ic_mean`（rank method、per-date N 過濾規則、IR 公式）

- **Ambachtsheer (1977)**, *JPM* 4(1)
  - 觀點：提出以預測值與實現報酬間的相關係數作為 skill 度量，IC 概念的最早正式化。
  - 採用：以 per-date IC 作為因子品質的核心量度。
- **Grinold (1989)**, *JPM* 15(3)
  - 觀點：IR ≈ IC·√breadth，將 IC 連結到投資組合 Sharpe-like 績效。
  - 採用：`ic_ir` 作為 IC 時間序列穩定度的標準化量度（signed mean/std）。
- **Grinold & Kahn (2000)**, *Active Portfolio Management* (2nd ed.), McGraw-Hill
  - 觀點：rank-based IC 對 factor outlier 的 robustness 論述；per-date 預處理與 cross-section standardization 的標準流程。
  - 採用：預設 Spearman（非 Pearson）；per-date z-score 預處理管線（`preprocess` step 3-5）。

### 非重疊 t-test (`ic_p`) — 消除 overlapping forward returns 的序列自相關

精確實作：`factrix/metrics/ic.py::ic` + `factrix/metrics/_helpers.py::_sample_non_overlapping`

- **Hansen & Hodrick (1980)**, *JPE* 88(5)
  - 觀點：K-期前向預測回歸的殘差必然存在 MA(K−1) 結構，未校正的 t-stat 會系統性高估顯著性。
  - 採用：此為 `ic_p` 和 `ic_nw_p` 雙路徑的動機基石。
- **Richardson & Stock (1989)**, *JFE* 25(2)
  - 觀點：提出 `K/T → κ` 的新 asymptotic 框架，給出多年重疊報酬統計量的極限分佈（Wiener 泛函），替代傳統漸近近似。
  - 採用：**本專案不採用 RS 的 K/T 框架**；引用此文僅作為「重疊報酬推論非 standard」的文獻背景。`_sample_non_overlapping` 以 `forward_periods`-步取樣的做法更接近 Hansen-Hodrick (1980) 所示 MA(K−1) 結構下的 conservative sub-sampling 傳統。

### Newey-West HAC t-test — `ic_nw_p`、FM λ、time-series 檢定

精確實作：`factrix/_stats.py::_newey_west_se` / `::_newey_west_t_test`（Bartlett kernel + lag rule）；`factrix/metrics/ic.py::ic_newey_west`（IC 特化 call site）

- **White (1980)**, *Econometrica* 48(4)
  - 觀點：OLS 在異質變異下點估計仍 unbiased，但 SE 錯誤；建議用三明治估計 (X'X)⁻¹(X'ΩX)(X'X)⁻¹。
  - 採用：White 的 HC0 是 Newey-West 的起點；我們的 HAC 實作繼承三明治結構。
- **Newey & West (1987)**, *Econometrica* 55(3)
  - 觀點：Bartlett kernel `w_j = 1 − j/(L+1)` 確保 HAC 共變異數為正半定，同時處理異質變異與自相關。
  - 採用：`_newey_west_se` 與 `_newey_west_t_test` 直接實作 Bartlett kernel。
- **Andrews (1991)**, *Econometrica* 59(3)
  - 觀點：以 minimax MSE 準則導出各 kernel 的最適 bandwidth 成長率；Bartlett kernel 的最適成長率為 `T^(1/3)`，並提供以 AR(1) 近似的 data-adaptive plug-in lag。
  - 採用：本專案取其 Bartlett 最適成長率的粗化版 `⌊T^(1/3)⌋` 作為 lag 預設（非 data-adaptive plug-in 形式）。
- **Newey & West (1994)**, *RES* 61(4)
  - 觀點：以非參數估計 spectral density 的導數，提出 data-adaptive plug-in bandwidth selection，漸近 MSE-optimal。
  - 採用：**未直接實作** plug-in 演算法；本專案 lag 預設採 Andrews 1991 的 Bartlett 成長率 `⌊T^(1/3)⌋`，並於 `ic_nw_p` 疊加 Hansen-Hodrick (1980) 下界 `h − 1`（overlap 週期），取 `lags = max(⌊T^(1/3)⌋, forward_periods − 1)`。此處引 NW 1994 主要作為「自動 lag 選擇方向」的文獻背景。
- **Andrews & Monahan (1992)**, *Econometrica* 60(4)
  - 觀點：以 VAR pre-whitening 後再做 kernel HAC，顯著降低 Bartlett/QS 估計的 bias、改善 t-stat 的 size。
  - 採用：未實作 pre-whitening；僅作為進階路徑的文獻背景，讓使用者知道若 lag 仍不足可外接此類修正。

### Regime IC / Multi-horizon IC — 穩定性診斷

精確實作：`factrix/metrics/ic.py::regime_ic`（per-regime t-test + BHY 跨 regime 校正）、`::multi_horizon_ic`（horizon list 掃 IC，per-horizon 非重疊 t-test）

- **Chen & Zimmermann (2022)**, *Critical Finance Review* 11(2)
  - 觀點：已發表因子的 OOS decay 與 sub-period 穩定性檢驗是 zoo-scale 整理的最低標；建議 report sub-period t-stats 分別檢查而非僅看 full-sample。
  - 採用：`regime_ic` 對每個 regime 跑獨立 t-test（H₀: mean IC = 0 within regime），並用 **BHY** 對 k 個 regime p-value 做相依性校正；預設以「時間 bisection（前半/後半）」作 regime fallback，接受使用者自供 regime 標籤。stat 回傳 **min|t| across regimes**（conservative：最弱 regime 若顯著則全 regime 皆顯著）。
  - 採用：`multi_horizon_ic` 掃預設 horizon list `[1, 5, 10, 20]`，per-horizon 以非重疊採樣（`_sample_non_overlapping`）做 t-test 避免重疊報酬 inflate，並用 `MIN_ASSETS_PER_DATE_IC` 依 horizon 縮放（短期 horizon 要求更多樣本以補償 sub-sampling 之資料耗損）。retention ratio 與 monotonicity 診斷由下游 veto rule (`cs.multi_horizon_decay_fast`) 消費。

### Turnover & Trading-Cost Proxy — `turnover`、`notional_turnover`、`breakeven_cost`、`net_spread`

精確實作：`factrix/metrics/tradability.py::turnover`、`::notional_turnover`、`::breakeven_cost`、`::net_spread`

兩個 turnover 指標**不等價**，各自回答不同問題：

| 指標 | 公式 | 回答什麼 | 可否餵入 cost 公式 |
|------|------|----------|-------------------|
| `turnover` | `1 − mean(Spearman ρ(rank_t, rank_{t+h}))` | 全截面排序在 h 期內的穩定度；中段 rank 翻動也會計入 | **否**（量綱不對；下游會雙重計算未真正下單的 middle-rank churn） |
| `notional_turnover` | 見下方程式碼區塊 | top / bottom 分位集合在 h 期 rebalance 被替換的比例；對應 Novy-Marx & Velikov (2016) τ | **是**（等於 equal-weight Q1/Q_n long-short portfolio 的 notional 替換率） |

```
top_churn_t = 1 − |Q_top_t ∩ Q_top_{t−h}| / |Q_top_t|
bot_churn_t = 1 − |Q_bot_t ∩ Q_bot_{t−h}| / |Q_bot_t|
notional_turnover = mean_t( (top_churn_t + bot_churn_t) / 2 )
```

成本公式（per-period，`c_bps` = 單邊成本 bps）：

```
breakeven_cost = gross_spread / (2 × notional_turnover) × 10000   [bps]
net_spread     = gross_spread − 2 × (c_bps / 10000) × notional_turnover
```

係數「× 2」對應 long 與 short 兩腿都要 rebalance；`notional_turnover` 已在 top / bot 之間取平均，故 full rotation 時 `2 × 1 × c_bps = 2·c_bps`，代數自洽。

- **Novy-Marx & Velikov (2016)**, *Review of Financial Studies* 29(1)
  - 觀點：以實證成本 τ（portfolio 替換比例）估算 anomaly 的 breakeven cost；middle-rank shuffle 不產生 notional 成本，只有分位邊界跨越才算。
  - 採用：`notional_turnover` 定義、`breakeven_cost` 公式、bps 單位對齊。
- **Hansen & Hodrick (1980)**, *JPE* 88(5)
  - 觀點：overlapping forward windows 造成 MA(h−1) 自相關。
  - 採用：`forward_periods > 1` 時兩個指標都先經 `_sample_non_overlapping` 按 stride `h` 取樣後再配對。

設計 caveat：

- **退市 under-count**：`notional_turnover` 在「前一次 rebalance 在 Q_top/Q_bot、但本次不在 panel」的名單上**沉默跳過**（計算只能看到今日 panel 的 asset）。真實 portfolio 會記一筆平倉成本，此處略去；影響量隨 universe churn 放大。
- **`turnover` 的尾部 quantile filter 不可直接入 cost 公式**：`turnover()` 支援 `quantile` 參數把 ρ 限定在 top-q ∪ bot-q 集合上計算，但 tail-union 上的 ρ 本就比全截面 ρ 穩（尾部名單較為 sticky），會系統性**低估** turnover。僅作跨因子 rank-stability 比較用；若要導入 cost 公式應改用 `notional_turnover`。
- **Annualization 由 caller 負責**：兩個指標都回傳 per-rebalance 值。年化成本 = `turnover × (年化期數 / forward_periods)`；`breakeven_cost` / `net_spread` 本身只做 per-period 計算。

---

### Quantile Spread / Monotonicity / Top Concentration — 輔助診斷

精確實作：`factrix/metrics/quantile.py`（spread / VW spread / group returns）、`factrix/metrics/monotonicity.py`、`factrix/metrics/concentration.py`（HHI⁻¹ effective-n）

- **Grinold & Kahn (2000)**, ch. 5-6
  - 觀點：quantile-based 分析揭示 IC 無法捕捉的「訊號集中在哪個尾端」資訊。
  - 採用：long-short spread (Q_top − Q_bottom)、quantile 報酬 profile、top-bucket HHI⁻¹。
- **Patton & Timmermann (2010)**, *JFE* 98(3)
  - 觀點：提出 bootstrap-based MR test，嚴格檢驗「所有相鄰 quantile 差」皆同向。
  - 採用：**未實作完整 MR test**。我們的 `monotonicity` 只是 `Spearman(group_index, group_return)` 的趨勢性指標，與 P-T MR test 無統計對應關係；文獻引為 awareness，不作為方法依據。

---

## Event study（事件驅動）

### CAAR non-overlapping t-test (`caar_p`)

精確實作：`factrix/metrics/caar.py::compute_caar` + `::caar`（signed_car 公式、非重疊採樣、t-stat）

- **Fama, Fisher, Jensen & Roll (1969)**, *IER* 10(1)
  - 觀點：開創事件研究方法（以股票分割為應用），首次使用市場模型殘差觀察價格對資訊的調整；AR、CAR、event window、estimation window 等術語的系統命名為後續文獻（Brown-Warner、MacKinlay）標準化的成果。
  - 採用：整個事件研究架構的基礎方法論承襲此文。
- **Brown & Warner (1980)**, *JFE* 8(3)
  - 觀點：**月頻**資料下不同 AR 估計模型（mean-adjusted / market-adjusted / market-model）的 size & power 比較；簡單方法在月頻即已足夠。
  - 採用：支撐「CAAR t-test 在合理樣本下 well-specified」的實證基礎。
- **Brown & Warner (1985)**, *JFE* 14(1)
  - 觀點：檢視日頻資料的特性（nonsynchronous trading、厚尾、條件異質變異、自相關）對事件研究方法的影響；**結論標準 parametric t-test 在日頻下仍 well-specified**，只需樣本量合理。
  - 採用：我們允許日頻輸入 (`forward_periods=1`) 的理論背書。
- **MacKinlay (1997)**, *JEL* 35(1)
  - 觀點：現代事件研究教科書式整合；標準化 event window × estimation window 的切割與推論流程。
  - 採用：`EventConfig` 的 `estimation_window`、`event_window_post` 等欄位語意完全沿用 MacKinlay。
- **Campbell, Lo & MacKinlay (1997)**, *The Econometrics of Financial Markets* (ch. 4), Princeton University Press
  - 觀點：CAAR 推論限制（event-induced variance、clustering 等）的完整論述。
  - 採用：`EventProfile` 將 CAAR 與 BMP / clustering HHI 並列為 canonical + diagnostic 雙軌，正是 Campbell-Lo-MacKinlay 警告的落地。

### BMP Standardized AR test (`bmp_p`)

精確實作：`factrix/metrics/caar.py::bmp_test`（SAR 標準化、跨事件 std、常態近似 z）

- **Patell (1976)**, *JAR* 14(2)
  - 觀點：以估計期 σ 將 AR 標準化，並乘上 prediction-error scaling √(1 + 1/M + (R_m−R̄)²/Σ(R_m−R̄)²) 處理市場模型 OOS 預測誤差。
  - 採用：概念前身；我們**省略 prediction-error scaling**（估計期僅取 mean-adjusted，無 market model），屬 "mean-adjusted SAR" 簡化版。
- **Boehmer, Musumeci & Poulsen (1991)**, *JFE* 30(2)
  - 觀點：事件本身會放大異常報酬的 cross-sectional variance，Patell t 會 over-reject；提出以**事件窗 SAR 的 cross-sectional std** 作分母的修正 t-statistic。
  - 採用：`bmp_p` 實作其跨事件標準化結構；以**常態近似**（`_p_value_from_z`）取 p-value，N≥`MIN_EVENTS`=10 時 z 與 t 誤差在可接受範圍內。

### Corrado 非參數 rank test (`corrado_rank_test`)

精確實作：`factrix/metrics/corrado.py::corrado_rank_test`（pooled-rank 簡化版，與論文原式的 deviation 見 docstring）

- **Corrado (1989)**, *JFE* 23(2)
  - 觀點：不假設 AR 常態分佈的 rank-based test；論文 eq.(5) 的 SE 以 **combined window（estimation + event period）** 每日跨截面平均 rank 偏差的時序 std 估計。
  - 採用：`corrado_rank_test` 採 **pooled-rank 簡化版**（分母為 pooled rank series std），與論文原式的 "per-date mean of rank deviations across combined window" 略有差異；適合快速 robustness screen，嚴格尺寸檢定建議外接專用套件。
- **Corrado & Zivney (1992)**, *JFQA* 27(3)
  - 觀點：比較 sign test 與 rank test 在日頻 AR 下的功效；rank test 在多數情境下 power 較高。
  - 採用：基於此決定採用 rank test 而非 sign test。

### Event Clustering HHI

精確實作：`factrix/metrics/clustering.py::clustering_diagnostic`（時間維度 HHI + effective-n 歸一化）

- **Hirschman (1945)**, *National Power and the Structure of Foreign Trade*, University of California Press；**Herfindahl (1950)**, Columbia University PhD dissertation
  - 觀點：Hirschman (1945) 最先提出集中度指標（原型為 √Σs²）；Herfindahl (1950) 獨立再發現並改採 Σs² 形式（今日 HHI 通行的平方和形式）。越高越集中於少數主體。
  - 採用：方法論轉借至**時間維度**，計算事件日集中度（高 HHI → 事件聚集在少數交易日）。採 Σs² 形式（Herfindahl 版）。
- **Kolari & Pynnönen (2010)**, *RFS* 23(11)
  - 觀點：ARs 間的 **cross-sectional correlation**（event-date clustering、共同市場衝擊等皆為來源）違反 BMP 原本假設的跨事件獨立性；提出以平均跨期 AR 相關係數修正 BMP t-statistic 的 z-statistic。
  - 採用：`clustering_hhi` diagnostic 用於**偵測事件時間聚集**（cross-sectional correlation 的一個常見來源）；Kolari-Pynnönen 的修正本身保留為未實作 config 選項（`adjust_clustering='kolari_pynnonen'`）。

### Event Hit Rate / Profit Factor

精確實作：`factrix/metrics/event_quality.py::event_hit_rate` / `::profit_factor` / `::event_skewness` / `::event_ic`

- 方向勝率走 binomial test（H₀: p=0.5）、Profit Factor 為總獲利 / 總虧損；屬業界實戰診斷，無單一學術出處。

---

## Macro panel（跨國 / 跨市場配置）

### Fama-MacBeth 兩階段回歸 (`fm_beta_p`)

精確實作：`factrix/metrics/fama_macbeth.py::compute_fm_betas`（stage-1 per-date OLS）+ `::fama_macbeth`（stage-2 NW t-test on λ）；`::pooled_ols` 作為 robustness 對照

- **Fama & MacBeth (1973)**, *JPE* 81(3)
  - 觀點：兩階段估計：每期 cross-sectional 回歸得到 λ_t，再對 λ 時序做 t-test；相較 pooled 可隔離截面相關性。
  - 採用：`fama_macbeth` 採用其兩階段架構，並於 stage-2 加 Newey-West。預設把 Signal 視為觀察值（raw characteristics 如 B/M、momentum、accounting ratios），不做 EIV 校正；需校正時設 `is_estimated_factor=True`（見下條）。
- **Shanken (1992)**, *RFS* 5(1) — 見 §Macro common 與 §參考文獻
  - 觀點：FM stage-2 將第一階段估出的 β 當作已知會低估 SE，需 EIV 校正因子 `1 + λ'Σ_f⁻¹λ`。
  - 採用：`fama_macbeth(is_estimated_factor=True)` 實作 **Kan-Zhang (1999) 單因子簡化形式**：NW SE 乘上 `√(1 + λ̂²/σ²_f)`，**省略 Shanken 原公式中的 `+σ²_f/T` 加性項**，僅對大 T 誠實。適用對象限 **估計出的 signal**（rolling β、PCA score、ML 預測、first-stage 殘差）；raw characteristics 不應啟用，否則 t-stat 會被虛假壓低。
- **Fama & French (1992)**, *JoF* 47(2) 與 **Fama & French (1993)**, *JFE* 33(1)
  - 觀點：FM 程序在跨 stock panel 上的現代標準用法；建立 size、book-to-market 等 anomaly 的推論模板。
  - 採用：Config 預設、variable 命名、預期使用情境皆以 FF 風格 panel 為基準。
- **Cochrane (2005)**, *Asset Pricing* (Revised ed.), ch. 12, Princeton University Press
  - 觀點：論述 FM 與 GMM、MLE 等估計法的等價性與漸近性質；FM 在 panel 稀疏時計算穩定。
  - 採用：選 FM 而非 GMM 即因其在跨國 / 跨資產這種 panel 不整齊情境下 robust。
- **Petersen (2009)**, *RFS* 22(1)
  - 觀點：系統比較 FM、firm-clustered、White、雙向 cluster SE：**firm effect** 存在時只有 firm-clustered SE 不偏；**time effect** 存在時 FM（可配 Newey-West 處理時序相依）才是不偏的組合。
  - 採用：Macro panel 情境以「時序 λ」為主要推論對象（對應 time effect），本專案採 FM + Newey-West。若未來擴展至同時存在 firm effect 的 panel，需改採 clustered SE（未實作）。

### Newey-West 於 λ 時序

- **Newey & West (1987)** — 同前；λ 時序的 lags 從 `forward_periods` 導出以覆蓋重疊週期結構。

### Pooled OLS with clustered SE (`pooled_ols`)

精確實作：`factrix/metrics/fama_macbeth.py::pooled_ols`（sandwich SE、single-way / two-way cluster、non-PSD fallback）

- **Petersen (2009)** — 同前；FM 與 single-way 在平衡 panel 下 point estimate 等價，但 SE 不同。
- **Cameron, Gelbach & Miller (2011)**, *Journal of Business & Economic Statistics* 29(2)
  - 觀點：two-way clustering `V = V_A + V_B − V_A∩B`；在 firm 與 time 兩維度同時存在相依性時必要。
  - 採用：`pooled_ols(two_way_cluster_col=...)` 直接實作 CGM 2011；每個 component 採自己的 finite-sample correction `G/(G−1)`；df 依 Thompson (2011) 取 `min(G_A, G_B) − 1`。
- **Thompson (2011)**, *JFE* 99(1)
  - 觀點：firm × time 雙向 cluster 的簡化實作；推薦 df 採 `min(G_A, G_B) − 1`。
  - 採用：df 計算與建議一致。
- **Cameron & Miller (2015)**, *Journal of Human Resources* 50(2)
  - 觀點：two-way V 在小樣本可能 non-PSD；直接 clip 到 0 會報出 SE=0（偽顯著），應 fallback 至 larger-dimension single-way V。
  - 採用：`non_psd_fallback` 邏輯於 `V[1,1] < 0` 時回退至 `cluster_col` 的 single-way variance，並於 metadata 標註 `variance_non_psd_fallback`。
- **Veto 用法**：FM λ 與 pooled β 正負號不一致時觸發 `macro_panel.fm_pooled_sign_mismatch` veto rule，作為 robustness check；同號但大小差異大時不 veto（視為 SE 差異而非 misspecification）。

---

## Macro common（共用時序因子）

### Per-asset TS OLS β、Cross-sectional t-test on β 分佈 (`ts_beta_p`)

精確實作：`factrix/metrics/ts_beta.py::compute_ts_betas`（per-asset OLS）+ `::ts_beta`（跨資產 t-test）+ `::ts_beta_single_asset_fallback`（N=1 退化路徑）

- **Black, Jensen & Scholes (1972)**, in *Studies in the Theory of Capital Markets* (M. Jensen, ed.), Praeger
  - 觀點：先以時序回歸估 per-asset β（stage 1），再對 β 分佈或其與報酬的關係做推論（stage 2）；適用於**因子本身在截面上沒變化**的情境（例如市場 premium）。
  - 採用：`ts_beta_p` 取其 stage-2 精神，對跨資產 β 分佈做 H₀: mean(β)=0 的 t-test — macro_common 的共用因子在截面上恆定，正符合 BJS 的適用條件。

**N=1 退化路徑（`ts_beta_single_asset_fallback`）**：
- 當 panel 只剩單一資產（`n_assets = 1`），跨資產 t-test 無法進行（需要 N≥3）。此時回傳該資產 stage-1 TS 回歸的 β 與 t-stat，但 **`p_value=1.0`** 以使 BHY 自動排除此列——stage-1 的 per-asset t-stat 是**時序假設**（H₀: β_i=0 for that asset），**不等同於** stage-2 跨資產推論（H₀: mean(β)=0 across assets）。兩者統計意義不同，不應混用。
- 下游 veto rule `macro_common.single_asset` 會以此旗標提示使用者結果僅為 single-asset 時序結論。

### Augmented Dickey-Fuller (`factor_adf_p`) — 因子單位根檢驗

精確實作：`factrix/_stats.py::_adf` + `::_adf_pvalue_interp`（MacKinnon constant-only 插值、p 上限 0.95 clip）

- **Dickey & Fuller (1979)**, *JASA* 74(366)
  - 觀點：OLS β̂ 在單位根下有非標準分佈；提供 tau 統計量的漸近臨界值表。
  - 採用：`_adf` 檢定 H₀: β=0 於 Δy_t = α + β·y_{t−1} + ε 的規格。
- **Said & Dickey (1984)**, *Biometrika* 71(3)
  - 觀點：ADF 可經由加入 Δy 滯後項擴展至 ARMA error 情境。
  - 採用：`_adf(y, lags=k)` 保留 k>0 的擴展 API，預設 k=0 但可手動提升為 Said-Dickey 風格。
- **MacKinnon (1996)**, *JAE* 11(6)
  - 觀點：以 response-surface 回歸給出 ADF 的精確臨界值與 p-value 公式，取代查表。
  - 採用：`_adf_pvalue_interp` 對 MacKinnon constant-only 規格的 7 個臨界點做線性插值，精度約 ±0.03 於決策邊界 (5%/10%)，且 p-value 上限 clip 在 0.95（右尾未完全覆蓋）；足夠定性判斷「是否有單位根」。**不依賴 `statsmodels`**。

### Stambaugh bias 警示

- **Stambaugh (1999)**, *JFE* 54(3)
  - 觀點：預測回歸 R_{t+1} = α + β·X_t + ε 當 X 高度 persistent 且 X 的 innovation 與 ε 同期相關時，OLS β̂ 有有限樣本偏誤，並給出 bias 的 closed-form。
  - 採用：以 `factor_adf_p > 0.10` 為 trigger、`macro_common.factor_persistent` rule 提示使用者此風險；**刻意不自動修正**，避免 false confidence。
- **Campbell & Yogo (2006)**, *JFE* 81(1)
  - 觀點：對 Stambaugh 情境提供 Bonferroni-type 有效檢定（Q-test），功效高於簡單 t 且 size 正確。
  - 採用：文件指引給進階使用者；本套件不內建 Q-test。
- **Kostakis, Magdalinos & Stamatogiannis (2015)**, *RFS* 28(5)；延伸自 **Phillips & Magdalinos (2009)** 的 IVX 框架
  - 觀點：以工具變數（IVX）消除 persistent predictor 引起的 near-unit-root bias，統計性質不依賴 persistence degree。
  - 採用：IVX 未內建；建議使用者如需 Stambaugh 修正，優先外接 IVX 實作（如 `arch` / R `ivx` package）。

---

## OOS persistence（樣本外穩定性）

### Multi-split OOS decay (`multi_split_oos_decay`)

精確實作：`factrix/metrics/oos.py::multi_split_oos_decay`（IS/OOS 切割、per-split `|mean_OOS|/|mean_IS|`、sign-flip 偵測、median survival ratio）

- **McLean & Pontiff (2016)**, *JoF* 71(1)
  - 觀點：已發表因子的 OOS 報酬平均衰減 ~32%（in-sample 高估約一半來自 statistical bias、另一半來自 publication-induced learning）；任何 strategy 若 OOS 報酬不存在即屬 overfitting。
  - 採用：`multi_split_oos_decay` 以 median survival ratio across splits 為 `value`；預設 `survival_threshold=0.5`（OOS 應至少保留 IS 的一半強度）；任一 split 發生 **sign flip** 即 VETOED（IC 翻轉比 decay 更嚴重，代表因子在 OOS 方向錯誤）。
  - 採用：`stat = None` — OOS decay 是描述性統計而非假設檢定，`p_value` 固定 `1.0` 以避免下游 BHY 把它當成顯著項目（conservative by omission）。
- **de Prado (2018)**, *Advances in Financial Machine Learning*, ch. 7
  - 觀點：Combinatorial Purged Cross-Validation (CPCV) 在時序資料上比固定 IS/OOS 切割更 robust。
  - 採用：**未實作 CPCV**；本專案用 3 個固定切點 `[(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]` 並取 median，理由是 CPCV 對外部 dependency 與計算成本較高；median-of-splits 是 lean 版的 robustness check。若需要嚴格 CPCV 建議外接 `mlfinlab`。

**Split 策略 rationale**：
- 固定三個切點而非滑動窗：避免產生隱含多重檢定壓力（滑動窗實質上在「選擇 split 位置」）。
- 取 median 而非 mean：對單一 regime change 剛好落在某 split 點的 robust 更好。
- 0.5 threshold：為嚴格中位數；McLean-Pontiff 觀察到平均 decay 32% 對應 survival ratio ≈ 0.68，因此 0.5 是 conservative 但非極端的門檻。可由 caller 覆蓋。

---

## Factor spanning & selection（因子跨度與選擇）

### Spanning regression (`spanning_alpha`)

精確實作：`factrix/metrics/spanning.py::spanning_alpha`（candidate = α + Σβ·base + ε；`factrix/_ols.py::ols_alpha` 背後 OLS）

- **Barillas & Shanken (2017)**, *RFS* 30(4)
  - 觀點：比較競爭模型 / 因子時，唯一正確方式是檢驗「candidate 是否對既有 base set 提供增量 α」（spanning regression），而非比較 pricing error 本身。
  - 採用：`spanning_alpha` 跑 candidate spread series = α + Σβ·base_spread + ε 的 OLS 回歸；對齊採 inner join 只保留所有 base 與 candidate 共同非 null 的日期（避免以 0 補值污染 β）；t-stat 以 classical OLS SE 計算，未加 Newey-West（base set 通常為 non-overlapping portfolio spread，自相關有限）。

### Greedy forward selection (`greedy_forward_selection`)

精確實作：`factrix/metrics/spanning.py::greedy_forward_selection`（forward pick by |α|、backward elimination by |t|）

- **Feng, Giglio & Xiu (2020)**, *JoF* 75(3)
  - 觀點：面對因子 zoo，提出 double-selection LASSO 框架做有效 pricing factor 挑選；核心觀察：**選擇的順序與 base set 會嚴重影響結論**，應顯式記錄。
  - 採用：本專案採簡化版 **greedy forward + backward elimination**（非 double-selection LASSO）：每步選 |α| 最大且 |t| ≥ 2 的 candidate；每次加入後重跑 backward 檢查已選因子之 |t|，凡 |t| 低於閾值者剔除。Algorithm 透明、可重現，但統計性質不同於 LASSO。

- **Leeb & Pötscher (2005)**, *Econometric Theory* 21(1)；**Berk, Brown, Buja, Zhang & Zhao (2013)**, *AoS* 41(2) — Post-Selection Inference
  - 觀點：stepwise selection 後的 t-stat 分佈**不再是常態**（pre-test bias / PoSI 問題）；報告 selected factor 的 t-stat 作為顯著性宣稱是**統計上無效**的。
  - 採用：`greedy_forward_selection` 僅產生「**候選名單**」而**不輸出顯著性宣稱**。回傳的 `SpanningResult.t_stat` 僅為 selection 過程的 diagnostic，不應當作 post-selection p-value。嚴格推論需另接 PoSI 或 sample-splitting 做 confirmatory test——本專案**刻意不實作**自動 post-selection inference，避免 false confidence。
  - **使用者注意**：若需對 selected set 做發表級的 t-stat claim，請在獨立 holdout 上重跑 `spanning_alpha`，該測試是 honest 的（無 selection bias）。

---

## Multiple testing（多重檢定校正）

### Benjamini-Yekutieli (BHY)

精確實作：`factrix/stats/multiple_testing.py::bhy_adjust` / `::bhy_adjusted_p`（c(m) scaling、right-cummin 單調化、two-stage `n_total` 支援）

- **Benjamini & Hochberg (1995)**, *JRSS B* 57(1)
  - 觀點：提出 FDR = E[V/R] 作為比 FWER 寬鬆但仍可控的多重檢定指標；BH step-up 程序要求 p-value 獨立或 PRDS。
  - 採用：BH 是 BHY 的基礎；我們因為不假設因子間獨立而不選 BH。
- **Benjamini & Yekutieli (2001)**, *AoS* 29(4)
  - 觀點：將 BH 的獨立性假設鬆綁至任意相依性，代價是 threshold 除以 c(m) = Σ_{i=1..m}(1/i)。
  - 採用：`bhy_adjust` / `bhy_adjusted_p` 直接實作；per-hypothesis adjusted p 透過 right-cummin 維持 rank 單調性。作為因子池（常高度相關）的預設 FDR 控制。
- **Efron (2010)**, *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*, Cambridge University Press
  - 觀點：1000+ 假設尺度下的 empirical Bayes 處理；locally FDR 與 global FDR 的權衡。
  - 採用：論述支持 FDR 框架在 zoo-scale 的適用性；未直接實作 empirical Bayes 方法。

### `bhy()` 家族分割設計

`bhy()` 在執行 BHY step-up 前先將 profiles 依 **`(dispatch cell, forward_periods)`** 分成獨立家族，每個家族在自己的 p-value 池中做校正。這是有意識的設計：不同程序（IC vs FM vs TS）的 null distributions 不同、不同 horizon 的有效樣本也不同，混在同一個 BHY 池會稀釋門檻、誤導 FDR 控制。跨家族聚合由使用者自行決定。

**單家族 `RuntimeWarning`**：若大多數家族只包含一個 profile（`singleton_families > 0 and len(families) > 1`），`bhy()` 會發出 `RuntimeWarning` — 單元素家族的 BHY 等同於原始截斷，並不提供 FDR 校正。典型觸發情境是逐個因子呼叫 `bhy()` 而非批次傳入同家族的候選因子。

### Two-stage screening 支援

- `multiple_testing_correct(n_total=...)`：前篩後仍在原始候選池尺度做 BHY，避免倖存者偏差造成 FDR 低估。前提是前篩條件需與待校正 p-value **邊際獨立**。

### `verdict()` 預設 t 門檻 2.0

- **Harvey, Liu & Zhu (2016)**, *RFS* 29(1)
  - 觀點：面對 300+ 已發表因子的 data mining 壓力，單因子宣稱應將 t 門檻上調至 3.0；比較 Bonferroni、Holm、BH、BHY 等多重檢定校正並給出對應 adjusted threshold。
  - 採用：預設門檻 2.0（單因子 95% 經典邊界），將多重檢定的嚴格把關交給 `multiple_testing_correct`（BHY，因子池相依性高時最合適）。
- **Harvey (2017)**, *JoF* 72(4)
  - 觀點：呼籲因子發現的統計誠實性：區分 "exploratory" 與 "confirmatory"，並鼓勵公開失敗研究以減少 publication bias。
  - 採用：「framework detects, user decides」設計哲學與此呼應；`integrations/mlflow.py` 的 `log_failed_run` 提供技術層面的 audit trail（與 Harvey 訴求的 publication 層面不同，但減少 survivorship 偏差的精神一致）。

---

## Preprocessing（前處理）

### MAD Winsorize — 每日 median ± k·MAD·1.4826 裁剪

精確實作：`factrix/preprocess/normalize.py::mad_winsorize`（k 預設值、per-date 套用、consistency constant）

- **Huber (1964)**, *AoMS* 35(1) 與 **Huber (1981)**, *Robust Statistics*, Wiley
  - 觀點：robust estimator 理論；breakdown point 50% 的 MAD 對污染資料比 OLS 標準差穩健得多。
  - 採用：`mad_winsorize` 以 median + MAD 定義裁切邊界；對股票報酬的 fat tail 特別適合。
- **Hampel (1974)**, *JASA* 69(346)
  - 觀點：導出 influence curve 概念，並於 robust statistics 架構下系統化採用 MAD 作為 scale estimator，其常態 Fisher-consistent scaling 為 1/Φ⁻¹(0.75) ≈ 1.4826。
  - 採用：`MAD_CONSISTENCY_CONSTANT = 1.4826`（定義於 `_types.py`）。常數的數學事實本身遠早於 Hampel（可溯至 19 世紀 Gauss 的 probable-error = 0.6745σ 傳統）；Hampel 1974 是現代 robust literature 中將其制度化採用的標準出處。
- **Rousseeuw & Croux (1993)**, *JASA* 88(424)
  - 觀點：系統比較 MAD、Q_n、S_n 等 robust scale estimator 的效率與 bias；Q_n 效率更高但計算 O(n log n)。
  - 採用：選 MAD 而非 Q_n 是 efficiency-for-speed 取捨；R-C 1993 是此決策的 secondary reference。

### Forward return 百分位裁切

- 與因子裁切**分開管理**；避免用「未來報酬的分佈」決定「現在因子如何裁切」造成的 look-ahead bias。業界標準做法，無單一學術出處。

### Cross-section standardization

精確實作：`factrix/preprocess/normalize.py` + `factrix/preprocess/pipeline.py`（step 5 per-date z-score）

- **Grinold & Kahn (2000)**, ch. 3-4
  - 觀點：per-date z-score 降低跨時段因子水準漂移對 rank correlation 的干擾。
  - 採用：`preprocess` pipeline step 5 內建；IC 計算前的標準前處理。

---

## 常數與門檻（Reproducibility reference）

以下常數分散於 `_types.py` 與各模組，集中列表方便 reviewer 核對與重現。

### 數值常數
| 常數 | 值 | 定義位置 | 用途 |
|------|----|---------|------|
| `EPSILON` | `1e-9` | `_types.py` | 除零保護、幾乎為 0 的比較門檻 |
| `DDOF` | `1` | `_types.py` | 樣本標準差自由度校正；全專案統一（Polars `.std()` 預設即為 1） |
| `MAD_CONSISTENCY_CONSTANT` | `1.4826` | `_types.py` | `= 1/Φ⁻¹(0.75)`，使 MAD 成為常態下 σ 的無偏估計 |

### 最低樣本門檻（觸發 short-circuit 回傳）
| 門檻 | 值 | 定義位置 | 對應檢定 |
|------|----|---------|---------|
| `MIN_ASSETS_PER_DATE_IC` | `10` | `_types.py` | IC 時序檢定（per-date IC 數量）；`multi_horizon_ic` 會 scale 以補償 sub-sampling 資料耗損 |
| `MIN_EVENTS` | `10` | `_types.py` | CAAR / BMP 事件 N；BMP 常態近似於 `N ≥ 10` 時 z 與 t 誤差可接受 |
| `MIN_OOS_PERIODS` | `5` | `_types.py` | 每個 IS / OOS 分段至少觀測值；`multi_split_oos_decay` 要求整段 `N ≥ 2·MIN_OOS_PERIODS` |
| `MIN_PORTFOLIO_PERIODS` | `5` | `_types.py` | 非重疊 portfolio spread 最低期數 |
| `MIN_MONOTONICITY_PERIODS` | `5` | `_types.py` | Monotonicity Spearman 檢定最低期數 |
| `MIN_FM_PERIODS` | `20` | `fama_macbeth.py` | λ 時序長度；Newey-West 需要 T 足夠大才能有意義 |
| `MIN_TS_OBS` | `20` | `ts_beta.py` | 每資產 TS 回歸最低觀測值 |
| `_BINOMIAL_EXACT_CUTOFF` | `20` | `_stats.py` | `N < 20` 用精確 binomial CDF，否則用 normal approximation z |
| `TIE_RATIO_WARN_THRESHOLD` | `0.3` | `metrics/_helpers.py` | 若 > 30% 的 factor 值相同會發 warning（tie policy 選擇） |

### 顯著性標記閾值（`_significance_marker`）
| Marker | p 範圍 | 意涵 |
|:------:|-------|------|
| `***` | `p < 0.01` | Highly significant |
| `**`  | `p < 0.05` | Significant |
| `*`   | `p < 0.10` | Weakly significant |
| （空） | `p ≥ 0.10` | Not significant |

此為學界慣例；**單因子宣稱** 的實質門檻仍以 `verdict()` 的 `threshold=2.0`（對應 `p < 0.05`）與 `multiple_testing_correct` 的 BHY 校正後 p 為準，星號只是視覺標示。

---

## 經評估但刻意未實作

以下方法**在設計討論中有考慮**，但因增量收益有限或會偏離 lean-dependency 原則而未納入。若你的 workflow 真的需要，建議使用 `statsmodels` / `arch` / `linearmodels` 等專門套件在 factrix 輸出之外處理。

- **Kolari & Pynnönen (2010) clustering adjustment** — `EventConfig.adjust_clustering='kolari_pynnonen'` 目前是**保留 config 選項**，實際調整尚未實作。建議使用者在事件 HHI 過高時改用 BMP 的橫截面 z-score 版本或自行做 calendar block bootstrap。
- **Stationary Block Bootstrap** — 完整 Politis & Romano (1994) 實作未納入；`_stats.py` 的 ADF p-value 採 MacKinnon 臨界值插值，精度 ±0.03 對「是否有單位根」的定性判斷足夠。
- **Hansen SPA / Superior Predictive Ability** — Hansen (2005) 對整個 factor zoo 做 data-snooping 校正；目前以 BHY 為主要 FDR 工具。
- **Stambaugh reverse-regression 自動修正** — 僅以 ADF 診斷 flag 提示，不自動做回歸重估。要完整修正請參考 Stambaugh (1999) 的 Method of Moments 形式或改用 IVX (Phillips & Magdalinos 2009) 預測回歸。
- **HAC SE on per-asset β for macro_common** — `compute_ts_betas` 維持 OLS SE。Stambaugh bias 的根源不是 SE 估計（OLS β̂ 仍然 unbiased），而是 predictor 與 innovation 的同期相關；HAC 只修 SE 不修 bias，刻意不加以免 false confidence。
- **Bai-Perron / Chow / Quandt-Andrews 結構性斷裂檢定** — 屬 regime analysis 範疇，scope 邊界外。外接 `ruptures`。
- **GARCH / wild bootstrap SE** — 屬條件異變異推論，超出 lean-dep。外接 `arch`。

更廣泛的 scope 邊界（不做的 optimizer / ML / backtest 等）見 [README.md § Scope & non-goals](https://github.com/awwesomeman/factrix/blob/main/README.md#scope--non-goals)。

---

## 完整參考文獻

依作者姓氏字母排序。上方簡寫 citation 對應此清單；期刊縮寫採業界慣用（JFE = *Journal of Financial Economics*、JPE = *Journal of Political Economy* 等）。

- Ambachtsheer, K. P. (1977). "Where Are the Customers' Alphas?" *Journal of Portfolio Management* 4(1).
- Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59(3).
- Andrews, D. W. K. & Monahan, J. C. (1992). "An Improved Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimator." *Econometrica* 60(4).
- Barillas, F. & Shanken, J. (2017). "Which Alpha?" *Review of Financial Studies* 30(4).
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B* 57(1).
- Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing Under Dependency." *Annals of Statistics* 29(4).
- Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013). "Valid Post-Selection Inference." *Annals of Statistics* 41(2).
- Black, F., Jensen, M. C. & Scholes, M. (1972). "The Capital Asset Pricing Model: Some Empirical Tests." In M. Jensen (ed.), *Studies in the Theory of Capital Markets*. Praeger.
- Boehmer, E., Musumeci, J. & Poulsen, A. B. (1991). "Event-study Methodology Under Conditions of Event-induced Variance." *Journal of Financial Economics* 30(2).
- Brown, S. J. & Warner, J. B. (1980). "Measuring Security Price Performance." *Journal of Financial Economics* 8(3).
- Brown, S. J. & Warner, J. B. (1985). "Using Daily Stock Returns: The Case of Event Studies." *Journal of Financial Economics* 14(1).
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). "Robust Inference With Multiway Clustering." *Journal of Business & Economic Statistics* 29(2).
- Cameron, A. C. & Miller, D. L. (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources* 50(2).
- Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
- Campbell, J. Y. & Yogo, M. (2006). "Efficient Tests of Stock Return Predictability." *Journal of Financial Economics* 81(1).
- Chen, A. Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional Asset Pricing." *Critical Finance Review* 11(2).
- Cochrane, J. H. (2005). *Asset Pricing* (Revised ed.). Princeton University Press.
- Corrado, C. J. (1989). "A Nonparametric Test for Abnormal Security-price Performance in Event Studies." *Journal of Financial Economics* 23(2).
- Corrado, C. J. & Zivney, T. L. (1992). "The Specification and Power of the Sign Test in Event Study Hypothesis Tests Using Daily Stock Returns." *Journal of Financial and Quantitative Analysis* 27(3).
- Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74(366).
- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press.
- Fama, E. F., Fisher, L., Jensen, M. C. & Roll, R. (1969). "The Adjustment of Stock Prices to New Information." *International Economic Review* 10(1).
- Fama, E. F. & French, K. R. (1992). "The Cross-Section of Expected Stock Returns." *Journal of Finance* 47(2).
- Fama, E. F. & French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics* 33(1).
- Fama, E. F. & MacBeth, J. D. (1973). "Risk, Return, and Equilibrium: Empirical Tests." *Journal of Political Economy* 81(3).
- Feng, G., Giglio, S. & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance* 75(3).
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
- Kan, R. & Zhang, C. (1999). "Two-Pass Tests of Asset Pricing Models with Useless Factors." *Journal of Finance* 54(1).
- Kolari, J. W. & Pynnönen, S. (2010). "Event Study Testing with Cross-sectional Correlation of Abnormal Returns." *Review of Financial Studies* 23(11).
- Kostakis, A., Magdalinos, T. & Stamatogiannis, M. P. (2015). "Robust Econometric Inference for Stock Return Predictability." *Review of Financial Studies* 28(5).
- Leeb, H. & Pötscher, B. M. (2005). "Model Selection and Inference: Facts and Fiction." *Econometric Theory* 21(1).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- MacKinlay, A. C. (1997). "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1).
- MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11(6).
- McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71(1).
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
- Thompson, S. B. (2011). "Simple Formulas for Standard Errors that Cluster by Both Firm and Time." *Journal of Financial Economics* 99(1).
- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica* 48(4).
