# 截面個股因子評估框架 v3 — Gate-based Pipeline

> **適用範圍：截面個股因子（individual_stock）。**
> 事件訊號（event_signal）與總經/區域訊號（global_macro, group_region）
> 是不同的統計問題，應使用各自適合的方法論獨立評估，不在本文件範圍內。
> 原因見 [§2](#2-為什麼不統一不同因子類型的評估邏輯)。
>
> **設計演進：**
> - v1：4 維度 composite score，hardcoded 權重
> - v2：2 維度 composite score，消除 hardcoded 權重但仍計算分數
> - v3 初版：gate-based，但試圖統一所有因子類型
> - v3 修訂：scope 收窄至截面個股因子；整合批判性檢視回饋（Gate 門檻動態化、sign flip 處理簡化、Profile 擴充 regime/multi-horizon/Long-Short 拆解、Stage 2 greedy forward selection、framework 定位為 evaluation not backtesting）
> - v3 補充：新增 §15 Universe 配置與跨 Universe 評估（N-aware Gate 降級、Pervasiveness Profile）；§14 補充 group_region 與窄 universe individual_stock 的統一性洞察、market timing 歸類
> - v3 統計穩健性檢視：Gate 1 動態門檻簡化（P0 固定 2.0，P1 BHY 校正）；Gate 1 OR 條件標記通過路徑；Gate 2 Multi-split 提升至 P0；Monotonicity n_groups 提升至 10；Step 3 beta-adjusted 選項；IC_Trend 改用 Theil-Sen estimator；新增 Net_Spread profile 指標；Gate Status 新增 CAUTION 層級；Q1_Concentration 取代 Effective_Breadth 命名

## 核心原則

> 1. **因子評估，非策略回測**：本框架回答「這個因子訊號有沒有效」，不回答「用這個因子交易能賺多少錢」。所有計算都是 per-date 截面統計量，不需要 portfolio engine。
> 2. **不壓縮**：因子品質是多維度的，不應壓縮為單一數字。
> 3. **Pass/Fail 優於排序**：順序過濾避免虛假精度（72.3 vs 71.8 無統計意義）。
> 4. **Profile 優於 Score**：呈現所有指標原始值，讓使用者根據自身需求判斷。
> 5. **Context-dependent 排序在 Stage 2**：最終因子篩選取決於已有哪些因子（Barillas & Shanken, "Which Alpha?", 2017），不是 context-free 的分數。
> 6. **一件事做到極致**：不同訊號類型是不同的統計問題，不應為了通用性而稀釋品質。

## 術語定義

| 術語 | 含義 | 範圍 |
|------|------|------|
| **Step** | 資料前處理流水線的步驟（Step 1-6） | §9，在因子評估前執行 |
| **Gate** | 順序過濾的 pass/fail 檢查（Gate 1-2） | §5，因子評估的核心 |
| **Stage** | 評估的獨立階段 | Stage 1 = Gates + Profile（個體品質）；Stage 2 = Incremental Alpha（邊際貢獻） |
| **截面統計量** | 每個日期對所有股票計算一次的統計值（IC、分組均值差等） | 本框架的所有指標都是截面統計量，不涉及 portfolio NAV 追蹤 |

```
原始資料 → Step 1-6 (前處理) → Stage 1 (Gate 1→2 + Profile) → Stage 2 (Incremental Alpha)
```

---

## 目錄

1. [為什麼不用 Composite Score](#1-為什麼不用-composite-score)
2. [為什麼不統一不同因子類型的評估邏輯](#2-為什麼不統一不同因子類型的評估邏輯)
3. [框架定位：因子評估，非策略回測](#3-框架定位因子評估非策略回測)
4. [Rolling vs 非重疊：雙軌計算策略](#4-rolling-vs-非重疊雙軌計算策略)
5. [Gate 設計](#5-gate-設計)
6. [Factor Profile 指標](#6-factor-profile-指標)
7. [集中度問題的三層防線](#7-集中度問題的三層防線)
8. [Stage 2：因子篩選（≥ 20 因子後啟用）](#8-stage-2因子篩選-20-因子後啟用)
9. [資料前處理流水線（Step 1-6）](#9-資料前處理流水線step-1-6)
10. [資料品質警告](#10-資料品質警告)
11. [呈現方式](#11-呈現方式)
12. [v1/v2 → v3 遷移對照](#12-v1v2--v3-遷移對照)
13. [關鍵常數與可配置參數](#13-關鍵常數與可配置參數)
14. [其他因子類型的處理建議](#14-其他因子類型的處理建議)
15. [Universe 配置與跨 Universe 評估](#15-universe-配置與跨-universe-評估)
16. [文獻出處](#16-文獻出處)
17. [實作優先順序](#17-實作優先順序)
18. [待決問題](#18-待決問題)

---

## 1. 為什麼不用 Composite Score

### 1.1 學術文獻中的因子評估做法

無任何 peer-reviewed 論文提出「用單一 composite score 排序因子品質」。主流做法如下：

| 文獻 | 做法 | 是否用 composite score？ |
|------|------|:---:|
| Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns" (2016) | t-stat 單一門檻 (> 3.0)，pass/fail | 否 |
| Chen & Zimmermann, "Open Source Cross-Sectional Asset Pricing" (2022) | 分開報告 t-stat、monotonicity、OOS，不合成 | 否 |
| Hou, Xue & Zhang, "Replicating Anomalies" (2020) | long-short portfolio t-stat，pass/fail | 否 |
| Feng, Giglio & Xiu, "Taming the Factor Zoo" (2020) | Double-selection LASSO，model selection | 否 |
| Kozak, Nagel & Santosh, "Shrinking the Cross-Section" (2020) | Bayesian shrinkage 估計 SDF | 否 |
| Green, Hand & Zhang, "The Characteristics that Provide Independent Information..." (2017) | 聯合回歸，看哪些特徵存活 | 否 |
| Arnott, Harvey & Markowitz, "A Backtesting Protocol in the Era of Machine Learning" (2019) | 七類回測規範 checklist，逐項檢查 | 否 |
| WorldQuant BRAIN | `fitness = sqrt(abs(ret)/max(to,0.125)) × SR` | 是（但非學術文獻，未經 peer review） |

### 1.2 Composite Score 的根本問題

**資訊壓縮不可逆：** 一個 Reliability 指標優秀但 Profitability 差的因子，與一個各方面平庸的因子，可能得到相同的 composite score。壓縮後無法區分兩者。

**Goodhart's Law（Goodhart, "Problems of Monetary Management", 1984）：** "When a measure becomes a target, it ceases to be a good measure." 若 AI 因子生成器以 composite score 為最佳化目標，會產出刷分因子——形式上各指標達標但無真正經濟意義。

**虛假精度（False Precision）：** 每個子指標本身有顯著的估計誤差。composite score 72.3 vs 71.8 的差異在統計上毫無意義，但排序暗示前者「更好」。

**Context-free 的局限：** 「最好的因子」取決於現有因子集、容量限制、風險預算。context-free 的分數無法捕捉這些。Barillas & Shanken, "Which Alpha?" (2017) 的 spanning test 直接問「給定你現有的因子集，這個新因子有沒有增量？」——這比 context-free 打分更有決策價值。

### 1.3 痛點的重新定義

原始痛點：「因子有效性分析橫跨太多維度，仰賴圖表分析，想用一個分數解決。」

重新理解：真正的痛點不是「沒有一個分數」，而是「**資訊太分散、沒有結構化**」。Alphalens 等工具產出大量圖表但缺乏結構化判斷。解法不是壓縮資訊，而是**結構化呈現 + 自動化 pass/fail 判斷**。

---

## 2. 為什麼不統一不同因子類型的評估邏輯

### 2.1 問題：假統一

v3 初版試圖用「相同的 3 Gate pattern，不同的實作」來統一所有因子類型。但這本質上只是一個 `switch` 語句——共享的只有變數名稱和 UI 結構（基礎設施），不是評估邏輯。把基礎設施的共用偽裝成評估邏輯的統一，是 false abstraction。

### 2.2 學術界不統一的原因

| 文獻/工具 | 範圍 | 是否統一不同訊號類型？ |
|-----------|------|:---:|
| Chen & Zimmermann, "Open Source Cross-Sectional Asset Pricing" (2022) | **只做截面股票因子** | 否 |
| MacKinlay, "Event Studies in Economics and Finance" (1997) | 事件研究方法論 | **完全獨立的方法體系**：CAR/BHAR，跟 IC/quantile sort 無關 |
| Cochrane, "Presidential Address: Discount Rates" (2011) | 因子定價 | 明確區分 time-series 和 cross-section 為**不同維度** |
| Macrosynergy, "How to Measure the Quality of a Trading Signal" (2024) | 總經訊號 | 分 `cross_section` / `time_periods` 模式，但都是同類訊號 |

### 2.3 強行統一的問題

- **假等價**：IC_IR t-stat 和 Event_CAAR t-stat 是不同的統計問題，共用 "Gate 1" 名稱誤導可比性
- **最差拖累最好**：為通用性做的妥協讓截面因子（最核心部分）的程式碼變更複雜
- **每種都只能做到「勉強」**：事件訊號有成熟的獨立方法論（MacKinlay, "Event Studies in Economics and Finance", 1997），總經訊號在小截面下需要特殊處理（Giglio, Xiu & Zhang, "Test Assets and Weak Factors", 2025）
- **Profile 指標零重疊**：individual_stock 和 event_signal 沒有任何共同指標

### 2.4 正確做法

共享基礎設施（MLflow、Dashboard UI、engine.py），分開評估邏輯：

```
factrix/
├── scoring/
│   ├── cross_sectional.py    ← 截面因子評估（本文件範圍）
│   ├── event_study.py        ← 事件訊號評估（獨立模組）
│   └── macro_signal.py       ← 總經訊號評估（獨立模組）
├── experiment.py             ← 共享基礎設施
├── dashboard/                ← 共享基礎設施（不同 tab 呈現）
└── engine.py                 ← 共享基礎設施
```

---

## 3. 框架定位：因子評估，非策略回測

| | 因子評估（本框架） | 策略回測 |
|--|--|--|
| 問的問題 | 這個因子訊號有沒有效？ | 用這個因子交易能賺多少錢？ |
| 計算方式 | **per-date 截面統計量**（IC、分組均值差） | 持倉追蹤、NAV 計算、滑價模擬 |
| 複雜度 | `group_by("date").agg(...)` | 完整的 portfolio engine |
| 報酬計算 | 每期重新排序的截面分組均值 | 真實持倉的加權報酬（含 weight drift） |
| 交易成本 | 近似估計（Breakeven Cost） | 精確模擬（滑價、衝擊、借券） |

**本框架的所有指標都是 per-date 截面統計量**，不需要追蹤 portfolio NAV：

```python
# IC：截面 rank correlation（不需分組）
ic[t] = spearman(factor_rank[t], return_rank[t])

# Quantile spread：截面分組均值差（需分組，但不追蹤持倉）
spread[t] = mean(ret_fwd[t] | top 20%) - mean(ret_fwd[t] | bottom 20%)

# 兩者都是 group_by("date").agg(...) 操作，計算複雜度相同
```

Quantile spread（Q1-Q5 均值差）在文獻中常被稱為 "long-short return"，但在本框架中它不是一個 portfolio 的報酬——它是一個**截面分組統計量**。不需要建構持倉、追蹤 NAV、或處理 weight drift。

---

## 4. Rolling vs 非重疊：雙軌計算策略

IC 和 quantile spread 都可以用兩種取樣方式計算。兩者的**均值**（mean）收斂到相同的值，但**統計推論**不同：

### 4.1 兩種取樣方式

| | Rolling（每日，重疊） | 非重疊（每 N 日） |
|--|--|--|
| 取樣 | Day 1, 2, 3, 4, 5, 6, ... | Day 1, 6, 11, 16, ... |
| 觀測數 | 多（每天一個） | 少（每 N 天一個） |
| 相鄰觀測 | **重疊**：共享 N-1 天的報酬 → 高自相關 | **獨立**：無共享 → 無自相關 |
| 均值（Mean） | ≈ 相同（無偏） | ≈ 相同（無偏） |
| 標準差 | **偏低**（自相關壓縮） | 正確 |
| t-stat | **偏高**（被灌水） | 正確 |
| 累積曲線 | 平滑但每天報酬被計入最多 N 次（非真實 P&L） | 階梯狀但每天報酬只計入 1 次 |

### 4.2 本框架的雙軌策略

```
Rolling（每日）：用於視覺化
  → IC 累積曲線、Rolling IC 圖
  → 曲線平滑，看趨勢和轉折，不看絕對數字

非重疊（每 N 日）：用於統計推論
  → Gate 1 的 IC_IR t-stat
  → Gate 2 的 OOS Decay ratio
  → Profile 中所有需要 t-stat 的指標
  → 觀測間獨立，t-stat 正確
```

**實作：** 現有程式碼已正確實作此雙軌策略——IC 曲線用所有日期（`_ic_series`），t-stat 用非重疊取樣（`gather_every(forward_periods)`、`_non_overlapping_dates`）。

---

## 5. Gate 設計

### 架構總覽

```
            截面個股因子
                │
       ┌────────▼────────┐
       │  Gate 1: 顯著性  │  IC_IR t-stat ≥ threshold
       └────────┬────────┘
           PASS  │  FAIL → 標記 FAILED，後續不計算
       ┌────────▼────────┐
       │  Gate 2: 持續性  │  IC_OOS / IC_IS ≥ 0.5
       └────────┬────────┘    sign flip → 直接 VETOED
           PASS  │  FAIL → 標記 VETOED
                │
       ┌────────▼────────┐
       │  PASS 因子池     │  完整 Profile 展示
       │                 │  （含 Breakeven Cost 等
       │                 │   implementation 參考指標）
       └────────┬────────┘
                │ ≥ 20 因子
       ┌────────▼────────┐
       │  Stage 2        │  Incremental Alpha
       └─────────────────┘
```

> **為什麼只有 2 道 Gate，沒有 Economic Viability Gate？** Gates 的定位是「訊號品質的 pass/fail」——純粹回答「這個訊號統計上是否真實且可持續」。經濟可行性（Breakeven Cost、Turnover）屬於 implementation feasibility，取決於市場交易成本、rebalance 頻率、分組方式等 implementation 決策，不是因子本身的固有屬性。這些資訊在 Profile 中展示，由使用者根據自身投資限制判斷。Chen & Zimmermann, "Open Source Cross-Sectional Asset Pricing" (2022) 的 319 因子評估也刻意不評估 turnover/capacity。

### Gate 1: Statistical Significance（統計顯著性）

| 項目 | 內容 |
|------|------|
| 條件 | IC_IR 的 t-stat ≥ `SIGNIFICANCE_THRESHOLD` **OR** Q1-Q5 spread 的 t-stat ≥ `SIGNIFICANCE_THRESHOLD` |
| **預設門檻** | **固定 2.0**（P0）；P1 引入 BHY 連續校正（見下方） |
| t-stat 計算 | 使用非重疊取樣（每 `forward_periods` 日取一個），避免自相關膨脹 |
| 未通過處理 | 標記為 `FAILED`；後續 Gate 和 Profile 指標不計算 |
| 文獻依據 | Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns" (2016)：新因子 t-stat 應 > 3.0 以應對多重檢定 |

**P0：固定門檻 2.0。** 簡單、保守，無邊界不連續問題。使用者可覆蓋為任意固定值（例如 3.0）。

**P1 改進：BHY 連續校正。** 原設計使用離散階梯（< 20 → 2.0, 20-99 → 2.5, ≥ 100 → 3.0），但存在兩個問題：(1) 邊界處不連續——19→20 因子門檻跳變；(2)「當前批次」定義可被操縱——分批小量測試可繞過高門檻。改用 BHY（Benjamini-Hochberg-Yekutieli）校正，控制 FDR = 5%，門檻隨因子數連續提高，且考慮因子間相關性。文獻依據：Harvey & Liu, "Lucky Factors" (2020) 提出 Bayesian multiple testing framework；Chordia, Goyal & Saretto (2020) 確認 BHY 比 Bonferroni 更適當。

```python
# P1: BHY 校正（statsmodels 現成實作）
from statsmodels.stats.multitest import multipletests
rejected, pvals_adj, _, _ = multipletests(raw_pvalues, alpha=0.05, method='fdr_by')
```

> **為什麼允許 Q1-Q5 spread t-stat 作為替代？** IC 衡量的是 Spearman rank correlation（線性單調關係）。部分有效因子有非線性預測力——例如 U 型效應（極端高和低的因子值都預測高報酬）或閾值效應。這類因子的 IC ≈ 0（Gate 1 用 IC_IR 會 FAILED），但 Q1-Q5 spread 可能很大。加入 OR 條件避免系統性漏殺非線性因子。Q1-Q5 spread t-stat 使用與 IC_IR 相同的非重疊取樣計算。P0
>
> **OR 條件的多重檢定代價：** 每增加一個 OR 條件，等效檢定次數翻倍。Gate 1 輸出應**明確標記觸發 PASS 的路徑**（`via IC_IR` / `via Q1-Q5 spread` / `via both`），讓使用者知道因子是線性有效還是僅非線性端有效——這對後續 portfolio construction 有實質影響。P1 可考慮 Romano-Wolf stepdown correction（Romano & Wolf, 2005）處理 OR 帶來的多重檢定膨脹，控制 family-wise error rate。比 Bonferroni 更精確（考慮兩統計量的相關性），但需要 bootstrap 實作。。

### Gate 2: OOS Persistence（樣本外持續性）

| 項目 | 內容 |
|------|------|
| 條件 | `IC_OOS / IC_IS ≥ OOS_DECAY_THRESHOLD`（預設 0.5） |
| 計算 | 80/20 時間切分；IS/OOS IC 均值比 |
| **Sign flip 處理** | IS 與 OOS 的 IC 均值**符號翻轉 → 直接 VETOED**（不做 ×0.5 修正） |
| 未通過處理 | 標記為 `VETOED` |
| 文獻依據 | McLean & Pontiff, "Does Academic Research Destroy Stock Return Predictability?" (2016)：平均 OOS 衰減約 32% |

**Sign flip 直接 VETOED 的理由：** IC 在 OOS 翻轉方向代表因子在 OOS **預測反了**——這比「沒有預測力」更糟（IC = 0），不應允許通過。簡化處理也消除了 ×0.5 這個 hardcoded 參數。

**Multi-split OOS Decay（P0）。** ~~原列為 P1，但~~ 作為一道 Gate，如果 Gate 本身因 cut-point 位置的隨機性而不可靠，其作為篩選機制的 credibility 不足——一個 regime change 恰好落在 80% 處就能把好因子 VETO 掉。使用 3 個切分點（60/40, 70/30, 80/20），分別計算 OOS Decay ratio，取中位數作為 Gate 2 判據。實作成本極低（僅多跑 2 次 `mean(IC)` 計算），但顯著提高 Gate 2 的穩定性。Sign flip 檢查在 3 個切分中**任一個**發生即 VETOED。文獻依據：de Prado, *Advances in Financial Machine Learning* (2018) 提出 Combinatorial Purged Cross-Validation (CPCV) 處理金融資料的 train/test split 問題。

```python
# Multi-split OOS Decay
splits = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
decays = [abs(mean_ic_oos) / abs(mean_ic_is) for s in splits]
sign_flips = [sign(mean_ic_oos) != sign(mean_ic_is) for s in splits]
if any(sign_flips):
    status = "VETOED"  # 任一切分 sign flip → VETOED
else:
    decay = median(decays)  # 取中位數
    status = "PASS" if decay >= OOS_DECAY_THRESHOLD else "VETOED"
```

### Gate Status 總結

| Status | 含義 | 後續處理 |
|--------|------|---------|
| `PASS` | 通過 Gate 1 + Gate 2，無 warning | 進入 PASS 池，顯示完整 Profile |
| `CAUTION` | 通過 Gate 1 + Gate 2，但有 warning | 進入 PASS 池，Profile 附 warning 標記 |
| `VETOED` | 通過 Gate 1 但未通過 Gate 2 | 保留紀錄，顯示已計算的指標 |
| `FAILED` | 未通過 Gate 1 | 僅保留基礎資訊，不計算後續指標 |

> **CAUTION 條件（任一觸發）：** (1) Step 6 正交化未啟用——Gate 1 判斷可能被 beta exposure 灌水；(2) Universe N < 200——統計力受限（見 §15.3 N-aware 降級）；(3) Gate 1 僅透過 Q1-Q5 spread（非 IC_IR）通過——因子可能僅有非線性效果；(4) IC_Trend 顯示衰減中（slope 顯著為負）。CAUTION 讓使用者在 PASS 池中快速識別需要額外注意的因子。

---

## 6. Factor Profile 指標

通過所有 Gates 的因子，計算並呈現以下指標的**原始值**（非 0-100 分數）。分為 Reliability / Profitability 兩組，分組僅用於呈現，不用於加權。

### Reliability 指標（訊號可靠性）

回答：「這個訊號統計上是真的嗎？」

| 指標 | 計算方式 | 顯著性標記 | 良好範圍（參考） |
|------|---------|:---:|---------|
| IC_IR | `∣mean(IC)∣ / std(IC)` | 依 t-stat | > 0.3（Grinold & Kahn, *Active Portfolio Management*, 2000） |
| Monotonicity | `mean(∣Spearman(group_idx, rank(group_ret))∣)`，**n_groups = 10**（台股 N~2000） | 依 t-stat | > 0.6（Patton & Timmermann, "Monotonicity in Asset Returns", 2010） |
| Hit_Rate | 非重疊期間 IC > 0 的比例 | 依二項 t-stat | > 55% |
| OOS_Decay | `∣IC_OOS∣ / ∣IC_IS∣` | — | > 0.5（McLean & Pontiff, "Does Academic Research Destroy Stock Return Predictability?", 2016） |
| **Regime IC** | 依使用者提供的 regime labels 分組計算 IC（fallback：時間二分法） | — | 各 regime 的 IC 方向一致且量級相近 |
| **Multi-horizon IC** | 不同 forward period (1, 5, 10, 20 日) 下的 IC 值 | — | 觀察因子的 time-scale signature 和 IC decay |
| **IC_Trend** | IC 序列對時間的 **Theil-Sen 中位數斜率**（對 outlier 穩健） | — | slope ≈ 0 = 穩定；slope 顯著 < 0 = 衰減中 |

> **Regime IC** 回答「因子是否跨市場環境穩定」——與 OOS_Decay（overfitting 檢定）不同，Regime IC 是 regime robustness 檢定。Chen & Zimmermann, "Open Source Cross-Sectional Asset Pricing" (2022) 分開報告不同子期間的 t-stat 即為此做法。
>
> **Regime 定義由使用者提供，框架不偵測 regime。** 使用者透過 `regime_labels` 參數傳入任意分組標籤（例如 VIX > 20 / ≤ 20、市場報酬正負、HMM 狀態、自訂景氣指標），框架只做 `groupby(regime_label).agg(mean_IC)`。理由：(1) regime 偵測本身是獨立的研究問題（HMM、clustering、閾值選擇），嵌入框架違反「最小複雜度」原則；(2) 與 §15 Universe 設計一致——Universe 是使用者的資料，regime 是使用者的標籤，框架不做定義只做計算；(3) 避免 look-ahead bias 風險（使用者負責確保 regime labels 是 point-in-time）。若未提供 `regime_labels`，fallback 為時間二分法（前半 / 後半）。
>
> **Multi-horizon IC** 回答「因子的有效期間有多長」——一個 5 日有效的因子在 20 日可能無效。這對使用者的持有期決策很重要。
>
> **IC_Trend** 回答「因子正在變好還是變差」——捕捉因子 alpha 隨日曆時間的衰減趨勢。這與 OOS_Decay（IS vs OOS 快照比較）不同：OOS_Decay 檢測 overfitting，IC_Trend 檢測因子被套利掉（crowding decay）。McLean & Pontiff (2016) 發現平均 32% 的 post-publication decay，但個別因子可能衰減 80%+。AI 生成的因子可能有更快的 decay（其他 AI 也在挖同樣的訊號）。Lou & Polk, "Comomentum" (2022) 提供了因子 crowding 的理論框架。
>
> **為什麼用 Theil-Sen 而非 OLS 斜率或比值法？** (1) 比值法（近 3 年 IC_IR / 全期 IC_IR）只比較兩個快照，對中間的 structural break 不敏感；(2) OLS 斜率假設線性衰減，但受 outlier 影響大（如 COVID 期間的 IC 暴跌會扭曲斜率）；(3) Theil-Sen estimator 使用所有相鄰點對的中位數斜率，breakdown point = 29.3%，對 outlier 穩健，且 `scipy.stats.theilslopes` 提供 confidence interval 可直接判斷顯著性。文獻依據：Sen, P. K. (1968). "Estimates of the Regression Coefficient Based on Kendall's Tau." *JASA* 63(324), 1379-1389.
>
> **Monotonicity n_groups = 10 的理由：** 5 組（quintile）下 Spearman 的 sample size 只有 5（df=3），即使完美單調 p ≈ 0.08，無法在 5% 水準拒絕 H₀。10 組（decile）下 df=8，完美單調 p < 0.001，可有效區分 ★/●/○ 三個顯著性層級。台股 N~2000 每組 200 檔，截面統計量穩定。P1 可考慮實作 Patton & Timmermann (2010) 的 MR test（Monotonic Relationship test），利用 portfolio return 的完整分佈而非僅均值，統計力更高。

### Profitability 指標（經濟獲利性）

回答：「扣除成本後能持續賺錢嗎？alpha 來自哪一端？」

| 指標 | 計算方式 | 顯著性標記 | 良好範圍（參考） |
|------|---------|:---:|---------|
| Q1-Q5 Spread | Q1 截面均值 - Q5 截面均值（年化） | 依 t-stat | > 0 |
| **Q1-Q5 Spread (VW)** | 市值加權的 Q1 均值 - Q5 均值（年化） | 依 t-stat | 與等權對比：VW << EW 暗示 alpha 集中在小型股 |
| **Long_Alpha** | Q1 截面均值 - Universe 截面均值（年化） | 依 t-stat | > 0（alpha 來自做多端） |
| **Short_Alpha** | Universe 截面均值 - Q5 截面均值（年化） | 依 t-stat | > 0（alpha 來自做空端） |
| Breakeven_Cost | `Gross_Alpha / (2 × Turnover)` | — | > 預估交易成本 |
| **Net_Spread** | `Q1-Q5 Spread (ann.) - 2 × estimated_cost_bps × Turnover` | 依 t-stat | > 0（扣除成本後仍有 alpha） |
| MDD | Q1-Q5 spread 序列的最大回撤 | — | < 15%（**附同期 benchmark MDD 對照**） |
| Q1_Concentration | `1 / HHI_Q1`，Q1 內因子值佔比的 HHI 倒數 | 依 t-stat | > 0.5 × N_Q1 |
| Turnover | `1 - mean(rank_autocorrelation)` | — | 依策略頻率而異 |

> **Q1-Q5 Spread (VW) 的用途：** 作為等權 spread 的對照。Hou, Xue & Zhang (2020) 發現約 65% 的因子在市值加權後消失。如果 VW spread 遠小於 EW spread，alpha 主要由小型股驅動——Step 6 正交化可能未完全去除 Size 曝露，或 alpha 本質上是 small-cap premium。
>
> **Long_Alpha / Short_Alpha 拆解的重要性：** Q1-Q5 spread 可能主要來自做多端（Long_Alpha 大、Short_Alpha 小）或做空端（反之）。對只能做多的投資者（台股放空受限），做空端的 alpha 實務上不可用。此拆解讓使用者判斷因子對其投資限制是否有用。
>
> **Breakeven Cost 的計算：** `Breakeven = Gross_Alpha / (2 × Turnover)`，其中 Gross_Alpha = Q1-Q5 spread 年化值，Turnover = `1 - mean(rank_autocorrelation)`。意義：單邊交易成本在多少 bps 以下，這個因子的 alpha 還不會被吃掉。文獻依據：Novy-Marx & Velikov, "A Taxonomy of Anomalies and Their Trading Costs" (2016)。
>
> **Breakeven Cost 是 implementation 參考指標，不是 Gate。** 它取決於市場交易成本估計（台股約 30-45 bps）、rebalance 頻率、分組方式——這些是「你怎麼用因子」的問題，不是「因子好不好」的問題。放在 Profile 讓使用者自行判斷。
>
> **Net_Spread 的用途：** Breakeven Cost 告訴你「成本在多少以下因子還賺錢」，Net_Spread 直接告訴你「以預估成本計算，扣完後還剩多少 alpha」。`2 ×` 是因為 long-short 的雙邊交易（買 Q1 + 賣 Q5 各一次），`estimated_cost_bps` 來自 `MARKET_DEFAULTS`。DeMiguel, Martin-Utrera & Nogales (2020) 證明考慮交易成本後因子的排序可能完全改變。Net_Spread 不是 Gate（成本是 implementation 問題），但讓使用者快速判斷「扣完成本還剩多少」。**注意：** Turnover proxy 使用 `1 - mean(rank_autocorrelation)`，這是上界估計——實際 portfolio 可用 buffer rule 降低 turnover。Net_Spread 是截面統計量，不需要 portfolio engine。
>
> **成本模型限制：** Breakeven Cost 假設線性交易成本。實際市場衝擊與 √(order size / daily volume) 成正比（Almgren & Chriss, "Optimal Execution of Portfolio Transactions", 2001 的 square-root law）。大資金量下，Breakeven Cost 會高估因子的經濟可行性。
>
> **Q1-Q5 Spread 和 Breakeven_Cost 共用相同的截面分組計算。** 兩者不是獨立資訊來源——使用者不應 double count。
>
> **MDD 對樣本期間高度敏感。** 應附同期市場或 baseline LS spread 的 MDD 作為 context。孤立的 MDD 數字缺乏解釋力。

### 顯著性標記

t-stat 用於顯示標記，而非加權：

| 標記 | 條件 | 含義 |
|:---:|------|------|
| ★ | t-stat ≥ 3.0 | 高度顯著（Harvey-strict） |
| ● | 2.0 ≤ t-stat < 3.0 | 顯著（95% CI） |
| ○ | t-stat < 2.0 | 不顯著 |

---

## 7. 集中度問題的三層防線

| 集中度類型 | 場景 | 負責機制 | 處理邏輯 |
|-----------|------|---------|---------|
| **類型曝露集中** | Q1 全是半導體股 | Step 6 正交化 | 回歸去除 Industry/Size 曝露後，alpha 若消失 → Gate 1 FAILED |
| **過擬合集中** | AI 因子挑到 IS 特定股票 | Gate 2 (OOS) | IS 好但 OOS 崩 → VETOED |
| **個股集中（容量）** | alpha 只存在少數標的 | Q1_Concentration | Profile 中顯示，使用者自行判斷 |

**Q1_Concentration 的定位：** Profile 參考指標，不是 Gate。低值不代表因子「無效」——只是容量受限。是否因集中度高而放棄因子，是使用者的決策。（原名 Effective_Breadth，為避免與 Grinold & Kahn (2000) 的 Breadth（獨立投注數量）概念混淆而重新命名——HHI 倒數衡量的是「Q1 內個股因子值的集中度」，與 IR = IC × √Breadth 中的 Breadth 定義不同。）

---

## 8. Stage 2：因子篩選（≥ 20 因子後啟用）

Stage 1 評估因子的**個體品質**。Stage 2 評估因子的**邊際貢獻**——取決於你已經有什麼因子。

### 8.1 篩選邏輯：順序門檻，非加權混合

```
Stage 1:  通過所有 Gates → PASS 因子池
Stage 2:  PASS 池內，按 Incremental Alpha 篩選
```

兩者回答不同問題（個體品質 vs 邊際貢獻），不混合、不加權。

### 8.2 Incremental Alpha — Greedy Forward Selection

Stage 2 的 spanning regression 存在**順序依賴（order dependence）**：先加入的因子得到全部功勞，後加入的因子只得到殘差。因此不能簡單「按 α 排序」，需要 greedy forward selection：

```
1. 初始集合 = 已知 base factors（Size, Value, Momentum）
2. 對所有 PASS 因子，計算 spanning regression：
   R_candidate = α + β₁·R_base1 + β₂·R_base2 + ... + ε
3. 選入 α 最大且 t-stat 顯著（≥ SIGNIFICANCE_THRESHOLD）的因子，加入集合
4. 重複步驟 2-3，直到無因子有顯著 α
```

**Backward elimination 驗證（每步執行）：** 每加入一個新因子後，重新測試所有已選因子的 spanning α 是否仍顯著。若某因子的 α 在新因子加入後變得不顯著（t-stat < `SIGNIFICANCE_THRESHOLD`），則將其從集合中剔除。這避免 greedy 演算法的順序依賴問題——先進入集合的因子可能在後續因子加入後變得冗餘。

文獻依據：Barillas & Shanken, "Which Alpha?" (2017)；Feng, Giglio & Xiu, "Taming the Factor Zoo" (2020)。

> **注意：** spanning regression 需要因子的報酬時間序列（非 IC 序列），這是本框架中唯一需要計算 quantile spread 時間序列並用作回歸輸入的地方。Stage 1 的 Gates 和 Profile 都不需要這個序列。

### 8.3 Percentile Rank（Optional）

因子數 ≥ 20 後，可對 Profile 指標加上 percentile rank 標記——這是排名資訊，不是分數。

---

## 9. 資料前處理流水線（Step 1-6）

**來源：** `factrix/engine.py` — `prepare_factor_data()`

所有 Step 在 Stage 1 之前執行。Step 1-5 為必要步驟，Step 6 為 strongly recommended（有 base factor 資料時）。

```
原始資料 → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → [Step 6] → Stage 1
```

### Step 1: Forward Return（前瞻報酬）

| 項目 | 內容 |
|------|------|
| 公式 | `ret_fwd = close[t+N] / close[t] - 1` |
| 預設 | N = 5（5 日前瞻報酬） |
| 目的 | 建立因子預測目標——未來 N 日的報酬 |
| 必要性 | 必要 |

### Step 2: Forward Return Winsorize（前瞻報酬截尾）

| 項目 | 內容 |
|------|------|
| 公式 | 每日截斷至 [1st, 99th] percentile |
| 目的 | 降低極端報酬（如漲跌停、除權息）對截面統計量的影響 |
| 必要性 | 必要 |

### Step 3: Abnormal Return（異常報酬）

| 項目 | 內容 |
|------|------|
| 公式 | `abnormal = ret_fwd - mean(ret_fwd)_date` |
| 目的 | 截面去市場均值，使報酬反映個股相對於市場的超額表現 |
| 必要性 | 必要 |
| **已知限制** | **這是等權市場均值調整，不是 beta 調整。** 高 beta 股票在上漲市場中天然有「正異常報酬」，不是因為 alpha 而是因為高 beta。若未啟用 Step 6，Gate 1 的 IC_IR t-stat 可能被 beta exposure 灌水——**這不只是 Profile 不準的問題，而是 Gate 判斷本身可能錯誤**。 |

**P1 改進：Beta-adjusted abnormal return。** 可選擇性將 Step 3 改為 `abnormal[t] = ret_fwd[t] - beta[t] × market_ret[t]`，beta 使用 trailing 252-day rolling OLS 估計（point-in-time）。這與「最小複雜度」原則不衝突——beta adjustment 是一行 `groupby.apply()` 的計算。Step 3 做 beta adjustment 與 Step 6 完整正交化（Size/Value/Industry）可並存：前者是必要的最低修正，後者是進階的純淨 alpha 估計。

> **未啟用 Step 6 時的 Gate 1 警告：** 如果 Step 6 未啟用且 Step 3 使用預設的等權均值調整，Gate 1 結果應附加顯著警告（非僅 Profile 標記），並將 Gate Status 設為 `CAUTION`（見 §5 Gate Status）。

### Step 4: MAD Winsorize（因子值截尾）

| 項目 | 內容 |
|------|------|
| 公式 | `clip = median ± n_mad × 1.4826 × MAD` |
| 預設 | n_mad = 3.0 |
| 目的 | 防止因子值極端值扭曲後續 z-score。使用 MAD 而非 std，對離群值更穩健 |
| 說明 | 1.4826 是 MAD 的一致性常數，使 MAD 成為常態分佈下 σ 的無偏估計 |
| 必要性 | 必要 |

### Step 5: Cross-sectional Z-score（截面標準化）

| 項目 | 內容 |
|------|------|
| 公式 | `z = (x - median) / (1.4826 × MAD)` |
| 目的 | 將因子值標準化為截面 robust z-score，使不同因子可比 |
| 說明 | 使用 median 和 MAD（非 mean 和 std），對離群值更穩健 |
| 必要性 | 必要 |

### Step 6: Factor Orthogonalization（因子正交化）

| 項目 | 內容 |
|------|------|
| 公式 | `ε = factor_z - (β₁·Size + β₂·Value + β₃·Momentum + Σβ_k·Industry_k)` |
| 計算 | 逐日截面回歸（OLS），殘差 ε 取代 factor_z 進入後續 Stage 1 |
| 目的 | 去除已知因子曝露，確保 Stage 1 的 Gates 和 Profile 反映「純淨 alpha」 |
| Industry 處理 | 使用 dummy variables（one-hot encoding） |
| 必要性 | **Strongly recommended**（有 base factor 資料時）。未啟用時 Step 3 的 abnormal return 無法去除 beta 曝露，可能導致 Gate 和 Profile 結果被系統性風險灌水 |
| 文獻依據 | Hou, Xue & Zhang, "Replicating Anomalies" (2020)：約 65% 因子在市值加權後消失 |

**為什麼 Step 6 在 Step 4-5 之後？** Step 4-5 是 Step 6 的前置條件：

- Step 4（MAD winsorize）截斷極端值 → 避免少數離群值主導 OLS 回歸
- Step 5（z-score）標準化至 mean≈0, std≈1 → 因子值與 base factors 尺度可比，回歸係數穩定
- 若對 raw factor 直接回歸，極端值和量綱差異會產出不可靠的殘差

回歸後不需再 z-score：殘差 ε 已 mean≈0（OLS 性質），後續 Gate/Profile 計算基於 rank，不受尺度影響。

**為什麼 Step 6（因子值正交化）足夠，不需要額外在組合層做 industry-neutral 分組？**

1. **回歸已去均值效應**——殘差在各產業的截面均值為 0（OLS 性質），Q1 不會系統性偏向某產業
2. **非線性殘餘風險由 Q1_Concentration 標記**——如果正交化後 Q1 仍然集中，Profile 中的 Q1_Concentration 會顯示低值
3. **Stage 2 是最終安全網**——即使 Step 6 漏掉了產業曝露，spanning regression 對 Industry factor 做回歸時會抓到
4. **組合中性化增加不必要的複雜度**——需要在每個產業內分別排序、分組、計算，且產業分類本身（如 TSE 大類 vs 細類）又引入新的設計決策

**如果 Step 6 未啟用的 fallback：** Profile 應標記「未正交化」提醒使用者。可選擇性在 quantile spread 計算時使用 industry-neutral 分組作為替代。

---

## 10. 資料品質警告

### 10.1 Survivorship Bias（存活偏誤）

如果股票 universe 只包含目前上市的股票，已下市的股票（可能因基本面惡化而被因子選中）會被遺漏，導致因子報酬被高估。這在 distress-related 因子中尤其嚴重。

**建議：** 底層資料應包含已下市股票（survivorship-bias-free dataset）。若無法取得，應在 Profile 中標注此限制。

### 10.2 Look-ahead Bias（前視偏誤）

Step 1 使用未來價格計算 forward return，這是 by design（建立預測目標）。但因子值必須確保 **point-in-time correctness**：日期 T 的因子值僅使用 T 及之前可取得的資料。

AI 生成的因子可能無意中使用了未來資訊（例如使用未來的財報資料）。框架無法自動偵測此問題——使用者有責任確保因子的 point-in-time 正確性。

---

## 11. 呈現方式

### 11.1 Leaderboard

```
                        Reliability                    Profitability
Factor       Gates  IC_IR  Mono  OOS  Hit%  Regime  Spread  Long_α  Short_α  BE_Cost  MDD
───────────  ─────  ─────  ────  ───  ────  ──────  ──────  ──────  ───────  ───────  ───
Mom_20D      ✓✓✓    0.45★  0.85● 0.78 58%   穩定    15%     12%     3%       67bps    12%
LowVol_20D   ✓✓✓    0.38★  0.91★ 0.65 61%   穩定    10%     8%      2%       89bps    8%
MeanRev_5D   ✓✓✗    0.28●  0.73○ 0.52 54%   不穩    5%      4%      1%       18bps    22%
RSI_14D      ✓✗✗    0.22●  0.68○ 0.41 51%   ───     ───     ───     ───      ───      ───
Overnight    ✗──    0.08○  ───   ───  ───   ───     ───     ───     ───      ───      ───

★ t ≥ 3.0    ● t ≥ 2.0    ○ t < 2.0    ─── 未計算
```

排序：先按 Gates 通過數降序 → 同 gate 數內使用者自選欄位。

### 11.2 Factor Detail

```
┌───────────────────────────────────────────────────────────────┐
│  Momentum_20D                                                 │
│                                                               │
│  Gate 1 (Significance): ✓ IC_IR t = 4.21 (threshold = 3.0)  │
│  Gate 2 (Persistence):  ✓ OOS Decay = 0.78, no sign flip    │
│  Gate 3 (Viability):    ✓ Breakeven = 67 bps (> 30 bps)     │
├───────────────────────────────────────────────────────────────┤
│  Reliability                      Profitability               │
│  ┌─────────────────────────┐     ┌──────────────────────────┐│
│  │ IC_IR       0.45  ★     │     │ Spread      15% ann. ★  ││
│  │ Mono        0.85  ●     │     │  ├ Long_α    12% ann. ★  ││
│  │ Hit Rate    58%   ●     │     │  └ Short_α   3% ann. ○   ││
│  │ OOS Decay   0.78        │     │ BE_Cost     67 bps       ││
│  │ Regime IC   前半0.04     │     │ Breadth     0.62         ││
│  │             後半0.05     │     │ MDD         12% (mkt 18%)││
│  └─────────────────────────┘     │ Turnover    35%           ││
│                                   └──────────────────────────┘│
│  Multi-horizon IC:  1d=0.02  5d=0.05  10d=0.04  20d=0.01    │
│                                                               │
│  [IC 累積圖 (IS/OOS)]       [Quantile Return 長條圖]          │
│  [Rolling IC 圖 (63日)]     [Multi-horizon IC Decay]          │
└───────────────────────────────────────────────────────────────┘
```

**圖表與 Profile 指標的分工：**

| 資訊 | 呈現方式 | 為什麼不做成 Profile 數值指標？ |
|------|---------|------|
| IC 趨勢（累積方向、轉折點） | IC 累積圖 | 曲線形態無法壓縮為單一數字 |
| IC 短期波動 | Rolling IC 圖（63 日滾動平均） | 已被 IC_IR（mean/std）、Hit_Rate（IC>0 比例）、Regime IC（分期間穩定性）覆蓋；再提煉為數字會 double count |
| 各分組報酬差異 | Quantile Return 長條圖 | 已被 Spread、Monotonicity 覆蓋；圖表能看非線性模式（如 U 型），數字無法 |
| IC 隨持有期的衰減 | Multi-horizon IC Decay 圖 | 已有 Multi-horizon IC 數值列在 Profile 中，圖表提供視覺化補充 |

### 11.3 Stage 2 報告（≥ 20 因子，PASS 池）

```
Incremental Alpha — Greedy Forward Selection

Step  Factor         Spanning α     t-stat   加入集合？
────  ─────────────  ────────────   ──────   ─────────
1     LowVol_20D     0.032%/day     3.21★    ✓ 加入
2     Mom_20D        0.018%/day     2.45●    ✓ 加入（相對於 Base + LowVol）
3     MeanRev_5D     0.003%/day     0.82○    ✗ 已被涵蓋（α 不顯著）

Base factors: Size, Value, Momentum
```

---

## 12. v1/v2 → v3 遷移對照

### 指標遷移

```
v1/v2 IC_IR          → v3 Profile (Reliability) + Gate 1 主要指標
v1/v2 Monotonicity   → v3 Profile (Reliability)
v1    Long_Alpha     → v3 Profile (Profitability) Long_Alpha（恢復為獨立指標）
v1/v2 MDD            → v3 Profile (Profitability)，附 benchmark MDD 對照
v1/v2 OOS_Decay      → v3 Gate 2 + Profile (Reliability)
v1    IC_Stability   → 移除（被 IC_IR 覆蓋）
v1/v2 Hit_Rate       → v3 Profile (Reliability)
v1    Turnover (RC)  → v3 Profile (Profitability) Turnover + Breakeven Cost 吸收
v2    Breakeven_Cost  → v3 Profile (Profitability)，從 Gate 降為 Profile（implementation 指標不做 Gate）
v2    Effective_Breadth → v3 Profile (Profitability) Q1_Concentration（重新命名）
v3    新增            → Regime IC, Multi-horizon IC, Short_Alpha
```

### 架構遷移

| 項目 | v1 | v2 | v3 |
|------|----|----|-----|
| 核心輸出 | 4 維度加權分數 | 2 維度 composite | **Gates pass/fail + Profile 原始值** |
| 排序 | 按總分 | 按總分 | **gate 通過數 → 自選欄位** |
| 權重 | 4 routing + sigmoid | 50/50 + sigmoid | **無** |
| 分數映射 | `map_linear` | `map_linear` + 分位數 | **不映射（原始值）** |
| t-stat 用途 | sigmoid 加權 | sigmoid 加權 | **Gate pass/fail + ★●○** |
| 因子類型 | 統一 4 types | 統一 4 types | **截面因子專用** |
| 定位 | 打分框架 | 打分框架 | **評估框架（非回測）** |

---

## 13. 關鍵常數與可配置參數

### Gate 門檻

| 常數 | 預設值 | 說明 | 可配置？ |
|------|--------|------|:---:|
| `SIGNIFICANCE_THRESHOLD` | 動態（2.0/2.5/3.0） | Gate 1 t-stat 門檻，依因子數調整 | v |
| `OOS_DECAY_THRESHOLD` | 0.5 | Gate 2 OOS 衰減門檻 | v |

### 計算參數

| 常數 | 值 | 說明 |
|------|-----|------|
| `EPSILON` | 1e-9 | 避免除以零 |
| `DDOF` | 1 | 樣本標準差 Bessel 修正 |
| `MIN_ASSETS_PER_DATE_IC` | 10 | IC 最小有效期數 |
| `MIN_OOS_PERIODS` | 5 | OOS 最小期數 |
| `MIN_PORTFOLIO_PERIODS` | 5 | Quantile spread 最小期數 |
| `n_groups` | 10 | Monotonicity 分組數（台股 N~2000；N < 200 時降為 5 或 3，見 §15.3） |
| `oos_ratio` | 0.2 | IS/OOS 切分比例（可配置） |
| `q_top` | 0.2 | Quintile 定義（可配置） |
| `MULTI_HORIZON_PERIODS` | [1, 5, 10, 20] | Multi-horizon IC 的前瞻天數（可配置） |

---

## 14. 其他因子類型的處理建議

本框架不處理以下因子類型，但提供方法論方向供未來獨立模組參考：

### event_signal（事件訊號）

**推薦方法論：** Event study（MacKinlay, "Event Studies in Economics and Finance", 1997）。

| 評估步驟 | 方法 |
|---------|------|
| 事件效應是否顯著？ | CAAR t-stat |
| 效應是否持續？ | IS/OOS event decay（依事件日期切分） |
| 報酬結構是否有利？ | Profit Factor + Hit Rate + Skewness |

> 事件訊號沒有「截面排名」概念，IC、Monotonicity 均不適用。

### global_macro / group_region（總經/區域訊號）

**推薦方法論：** Time-series signal evaluation（Macrosynergy, "How to Measure the Quality of a Trading Signal", 2024）。

| 評估步驟 | 方法 |
|---------|------|
| 方向預測是否準確？ | Balanced accuracy / Hit Rate（IC 在 N < 30 時不穩定，Giglio, Xiu & Zhang, "Test Assets and Weak Factors", 2025） |
| 是否跨 regime 穩定？ | Rolling hit rate stability |
| 風險調整報酬？ | Time-series Sharpe |

> **`group_region` 與窄 universe `individual_stock` 的關係：**
> `group_region`（例如產業輪動，N~30）和在窄 universe 上運行的 `individual_stock`（例如台股半導體，N~100）本質上是相同的統計問題——**連續訊號 + 小截面**。兩者面臨的挑戰一致：IC_IR noise 極大（Giglio, Xiu & Zhang, 2025）、quintile sort 每組標的過少、t-stat 因統計力不足而偏低。
>
> 差異僅在截面單位（個股 vs 產業組合）和前處理（個股可做 Step 6 正交化去 Industry；產業輪動不需要也無法去 Industry）。
>
> **實作含義：** 未來 `macro_signal.py` 的開發可直接複用本框架的 Gate pipeline，搭配 N-aware 降級機制（見 [§15](#15-universe-配置與跨-universe-評估)）調整 Gate 門檻和 Profile 參數，不需要從頭設計評估邏輯。

> **Market timing 訊號：**
> 二元型 market timing 訊號（例如殖利率曲線倒掛：是/否 → 後續市場報酬）歸入 `event_signal`，使用 Event_CAAR / Event_Hit_Rate 評估。連續型 market timing 訊號（例如殖利率曲線斜率作為連續預測因子，N=1 無截面）若需處理，應在 `event_signal` 模組中加入「連續值 → 按閾值二元化」的前處理步驟。

---

## 15. Universe 配置與跨 Universe 評估

> 本節定義框架如何在不同分析範圍（universe）上運行，以及如何聚合跨 universe 的評估結果。
> 本框架的 Gate 1-2 + Profile 架構不變——本節描述的是部署參數和聚合層，不是方法論變更。

### 15.1 Universe 定義

**Universe** 是因子評估的分析範圍——即「截面包含哪些標的」。同一個因子可以在不同 universe 上獨立評估。

| Universe 層級 | 範例 | 約略 N | 用途 |
|:---:|------|:------:|------|
| 國家全市場 | 台股全部（上市+上櫃） | ~2000 | **主要評估範圍**——統計力最高 |
| 交易所 | 台股上市 | ~1000 | 反映實際可投資限制（例如基金合約限制） |
| 產業子集 | 台股半導體 | ~100 | 僅用於「因子設計上只對特定產業有意義」的場景 |
| 他國市場 | 美股、港股 | ~2500-5000 | 跨市場有效性檢驗 |

> **經驗法則：Universe 之間的獨立性越高，跨 universe 測試的邊際資訊越大。**
> 嵌套 universe（台股 ⊃ 台股上市 ⊃ 台股半導體）有高度重疊，第二個 universe 的邊際資訊少。
> 平行 universe（台股 ∥ 美股 ∥ 港股）幾乎零重疊，邊際資訊高。
>
> 文獻依據：Fama & French (2012) 在 23 國分 4 個獨立區域測試因子；Jensen, Kelly & Pedersen (2023) 在 93 個獨立國家測試。兩者均選擇**平行而非嵌套**的 universe 結構。
> 完整文獻見 [literature_references.md §12](literature_references.md#12-跨市場因子評估)。

### 15.2 參數設計：市場配置 + N 自動推導

**Universe 不需要預定義配置。** Universe 就是使用者傳入的資料本身——任意篩選條件（市值、產業、指數成分、流動性門檻…）在資料準備階段處理，框架從資料自動推導所有 N-dependent 參數。

真正需要配置的只有**市場層級**的少量參數（與 universe 篩選無關）：

```python
# 市場層級配置（少量、穩定、不常改）
MARKET_DEFAULTS = {
    "tw": {
        "estimated_trading_cost_bps": 30,   # 證交稅 3‰ + 手續費 + 衝擊
        "ortho_factors": ["size", "value", "momentum", "industry_tse30"],
    },
    "us": {
        "estimated_trading_cost_bps": 5,    # 近零手續費，主要是衝擊
        "ortho_factors": ["size", "value", "momentum", "industry_gics"],
    },
}

# Universe = 使用者傳入的資料，框架從資料自動推導
experiment.run(
    factor_data=my_factor_data,   # 已篩選的 DataFrame（任意 universe）
    market="tw",                   # 對應 MARKET_DEFAULTS
    # 自動推導：
    # N = number of unique stocks per date (median)
    # n_groups = auto_n_groups(N)
    # gate_degradation = auto_from_n(N)
)
```

**參數分類：**

| 參數 | 來源 | 說明 |
|------|:---:|------|
| Gate 1 動態門檻 | 因子數量驅動 | 與 universe 和市場均無關 |
| Gate 2 OOS_Decay 門檻 | 方法論常數 | McLean & Pontiff (2016) |
| `n_groups` | **從 N 自動推導** | N≥1000→10, N≥200→5, N≥30→3 |
| N-aware 降級 | **從 N 自動推導** | 見 §15.3 |
| `estimated_trading_cost_bps` | **市場配置** | 市場結構決定，與 universe 篩選無關 |
| `ortho_factors` | **市場配置** | 資料可得性決定 |

### 15.3 N-aware Gate 降級

當 universe N 較小時，Gate 和 Profile 的統計力下降。框架根據 N 自動調整：

| N 範圍 | Gate 1 | Profile 調整 | 文獻依據 |
|:------:|--------|-------------|---------|
| ≥ 200 | 正常模式（IC_IR t-stat） | `n_groups` = 5+；所有 Profile 指標可用 | — |
| 30-199 | 降級：考慮改用 Hit_Rate binomial test 作為替代顯著性檢定 | `n_groups` 降為 5 或 3（tercile）；Q1_Concentration 意義降低 | Giglio, Xiu & Zhang (2025)：因子強度是截面大小的函數 |
| < 30 | 降級：IC_IR 僅供參考，不作為 Gate 判據 | 分組無意義；僅保留 Hit_Rate、OOS_Decay | Giglio, Xiu & Zhang (2025)；Grinold & Kahn (2000)：低 Breadth 下 IR 必然低 |

> **這與 §14 的 `group_region` / `global_macro` 處理建議一致：** 小截面的統計處理方式相同，無論截面單位是個股、產業、還是國家。差異在前處理（Step 6）和成本估計，不在 Gate 邏輯。

### 15.4 跨 Universe Pervasiveness Profile

當同一因子在多個 universe 上運行後，可聚合結果為 **Pervasiveness Profile**。

這是 Gate 1-2 + Profile 之上的附加層（Layer 2），**不混入單一 universe 的 Gate 判斷**：

```
Layer 1:  每個 universe 獨立運行 Gate 1-2 + Profile
Layer 2:  聚合多個 universe 的 Layer 1 結果 → Pervasiveness Profile
```

**Pervasiveness Profile 指標：**

| 指標 | 計算 | 意義 | 文獻依據 |
|------|------|------|---------|
| **Pass_Count** | 因子在多少個 universe 通過 Gate 1-2 | Berkin & Swedroe 「Pervasive」標準的量化版 | Berkin & Swedroe (2016) |
| **Cross-Universe IC_IR** | Random-effects meta-analysis 聚合各 universe 的 IC_IR | 跨 universe 的共識訊號強度 | Griffin (2002)；Fama & French (2012, 2017) 支持 random effects |
| **I² Heterogeneity** | Meta-analysis 的 I² 統計量 | I² < 25% = 跨 universe 行為一致；I² > 75% = 高度異質（本地因子） | Cochrane Q-test |
| **OOS_Decay 一致性** | 各 universe 的 OOS_Decay 變異係數 | 低 CV = 因子穩定性跨 universe 一致 | Jacobs & Müller (2020)：OOS decay 因市場而異 |

> **為什麼不混入 Gate？**
>
> 1. Gate 回答的是「這個因子在**這個 universe**有效嗎」——這是本地問題，跨 universe 資訊不應影響本地判斷
> 2. 一個因子可能只在台股有效（例如台灣特有的制度性因子），在台股內它是「真的」。跨市場失效不代表本地無效
> 3. Griffin (2002)、Fama & French (2012, 2017) 一致證實 local factor models 優於 global models——**本地評估為主，跨 universe 一致性為輔**
>
> Pervasiveness Profile 的價值在於**區分 data-mining 與真因子**：如果一個因子在 5 個獨立 univeｄrse 都通過 Gate，data-mining 的機率極低（Jensen, Kelly & Pedersen, 2023：85% 的因子在 93 國可複製）。

### 15.5 v3 Gate 天然適合跨 Universe 比較

v3 相比 v2 的一個意外優勢：

| | v2 composite score | v3 Gate status |
|---|---|---|
| 跨 universe 可比性 | **低** — 台股的 FQS=70 和美股的 FQS=70 含義不同（score map 的 [min, max] 假設跨 universe 一致，但因子分佈不同） | **高** — PASS/VETOED/FAILED 是 categorical，天然跨 universe 可比 |
| 聚合方式 | 需要標準化才能比較 | 直接數 Pass_Count |

---

## 16. 文獻出處

### 反對 Composite Score

- **Harvey, Liu & Zhu (2016).** "...and the Cross-Section of Expected Returns." *Review of Financial Studies* 29(1), 5-68. — t-stat 門檻 pass/fail。
- **Chen & Zimmermann (2022).** "Open Source Cross-Sectional Asset Pricing." *Critical Finance Review* 11(2), 207-264. — 分開報告，不合成。([網站](https://www.openassetpricing.com/))
- **Arnott, Harvey & Markowitz (2019).** "A Backtesting Protocol in the Era of Machine Learning." *Journal of Financial Data Science* 1(1), 64-74. — checklist 逐項檢查；Model Complexity 建議簡單結構。([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654))
- **Goodhart (1984).** "Problems of Monetary Management: The U.K. Experience." In *Monetary Theory and Practice*. Macmillan, pp. 91-121. — Goodhart's Law.

### 反對統一不同因子類型

- **MacKinlay (1997).** "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1), 13-39.
- **Cochrane (2011).** "Presidential Address: Discount Rates." *Journal of Finance* 66(4), 1047-1108.
- **Giglio, Xiu & Zhang (2025).** "Test Assets and Weak Factors." *Journal of Finance*. ([NBER WP](https://www.nber.org/papers/w29002))
- **Macrosynergy (2024).** "How to Measure the Quality of a Trading Signal." ([連結](https://macrosynergy.com/research/how-to-measure-the-quality-of-a-trading-signal/))

### Gate 設計依據

- **Harvey, Liu & Zhu (2016).** 同上。— Gate 1 固定門檻 2.0（P0）；P1 BHY 校正。
- **Harvey & Liu (2020).** "Lucky Factors." *Journal of Financial Economics* 137(1), 116-142. — Bayesian multiple testing framework；比 Bonferroni/BHY 更適合因子間有相關性的場景。P1 BHY 校正的理論依據。
- **McLean & Pontiff (2016).** "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71(1), 5-32. — OOS 衰減 ~32%。Gate 2。
- **Chordia, Goyal & Saretto (2020).** "Anomalies and False Rejections." *Review of Financial Studies* 33(5), 2134-2179. — BHY 校正。
- **Romano & Wolf (2005).** "Stepwise Multiple Testing as Formalized Data Snooping." *Econometrica* 73(4), 1237-1282. — Stepdown correction；P1 處理 Gate 1 OR 條件多重檢定膨脹的理論依據。
- **de Prado (2018).** *Advances in Financial Machine Learning.* Wiley. — CPCV；Gate 2 multi-split 設計依據。

### Profile 指標依據

- **Grinold & Kahn (2000).** *Active Portfolio Management*（第二版）。— IC_IR；IR = IC × √Breadth。Q1_Concentration 與 Breadth 的區別。
- **Patton & Timmermann (2010).** "Monotonicity in Asset Returns." *Journal of Financial Economics* 98(3), 605-625. — MR test；n_groups = 10 的統計力考量。([PDF](https://public.econ.duke.edu/~ap172/Patton_Timmermann_sorts_JFE_Dec2010.pdf))
- **Novy-Marx & Velikov (2016).** "A Taxonomy of Anomalies and Their Trading Costs." *Review of Financial Studies* 29(1), 104-147. — Breakeven Cost、Net_Spread（Profile implementation 參考指標）。
- **DeMiguel, Martin-Utrera & Nogales (2020).** "Transaction Costs and Trading Volume in the Cross-Section of Stock Returns." *Journal of Financial Economics* 135(1), 271-292. — 考慮交易成本後因子排序可能完全改變；Net_Spread 指標依據。
- **Sen (1968).** "Estimates of the Regression Coefficient Based on Kendall's Tau." *JASA* 63(324), 1379-1389. — IC_Trend 使用 Theil-Sen estimator 的理論依據。
- **Lou & Polk (2022).** "Comomentum: Inferring Arbitrage Activity from Return Correlations." *Review of Financial Studies* 35(7), 3272-3302. — 因子 crowding/decay 的理論框架；IC_Trend 的解釋依據。
- **Berkin & Swedroe (2016).** *Your Complete Guide to Factor-Based Investing.* — 五標準（Persistent, Pervasive, Robust, Investable, Intuitive）。

### 集中度與容量

- **Hou, Xue & Zhang (2020).** "Replicating Anomalies." *Review of Financial Studies* 33(5), 2019-2133.
- **Israel & Moskowitz (2013).** "The Role of Shorting, Firm Size, and Time on Market Anomalies." *Journal of Financial Economics* 108(2), 275-301.

### 因子篩選與增量 Alpha

- **Barillas & Shanken (2017).** "Which Alpha?" *Review of Financial Studies* 30(4), 1316-1338.
- **Feng, Giglio & Xiu (2020).** "Taming the Factor Zoo." *Journal of Finance* 75(3), 1327-1370.
- **Kozak, Nagel & Santosh (2020).** "Shrinking the Cross-Section." *Journal of Financial Economics* 135(2), 271-292.
- **Green, Hand & Zhang (2017).** "The Characteristics that Provide Independent Information..." *Review of Financial Studies* 30(4), 1389-1436.
- **Jensen, Kelly & Pedersen (2023).** "Is There a Replication Crisis in Finance?" *Journal of Finance* 78(5), 2465-2518.

### 業界實務

- **Tulchinsky (2019).** *Finding Alphas.* Wiley. — WorldQuant fitness。
- **Asness (2015).** "The Great Divide." *Institutional Investor*.
- **Harvey (2017).** "Presidential Address: The Scientific Outlook in Financial Economics." *Journal of Finance* 72(4), 1399-1440.

### 跨市場因子評估（§15 依據）

- **Griffin (2002).** "Are the Fama and French Factors Global or Country Specific?" *Review of Financial Studies* 15(3), 783-803. — local model 優於 global model。
- **Fama & French (2012).** "Size, Value, and Momentum in International Stock Returns." *Journal of Financial Economics* 105(3), 457-472. — 拒絕跨區域整合定價假說。
- **Fama & French (2017).** "International Tests of a Five-Factor Model." *Journal of Financial Economics* 123(3), 441-463. — global factors 無法解釋 regional returns。
- **Jacobs & Müller (2020).** "Anomalies Across the Globe." *Journal of Financial Economics* 135(1), 213-230. — 美國是唯一有顯著 post-publication decay 的市場。
- **Jensen, Kelly & Pedersen (2023).** 同上。— 153 因子在 93 國可複製。([JKP Data](https://jkpfactors.com/))
- 完整文獻見 [literature_references.md §12](literature_references.md#12-跨市場因子評估)。

---

## 17. 實作優先順序

| 優先級 | 項目 | 改動範圍 | 說明 |
|:------:|------|---------|------|
| **P0** | Gate 1-2 實作 | `scoring/cross_sectional.py` | Gate 1 固定門檻 2.0；Gate 2 multi-split（3 切分取中位數）；sign flip 直接 VETO |
| **P0** | Gate 2: Multi-split OOS Decay | `scoring/cross_sectional.py` | 3 個切分點（60/40, 70/30, 80/20）取中位數，降低 regime change 敏感度（原 P1 提升） |
| **P0** | Gate Status 輸出 | `experiment.py` | PASS/CAUTION/VETOED/FAILED + MLflow |
| **P0** | Gate 1 OR: 標記通過路徑 | `scoring/cross_sectional.py` | 輸出標記 `via IC_IR` / `via Q1-Q5 spread` / `via both` |
| **P0** | Monotonicity n_groups = 10 | `scoring/cross_sectional.py` | 5 組 Spearman df=3 統計力不足；10 組 df=8 可有效區分顯著性 |
| **P0** | Profile: Q1-Q5 Spread + Long/Short Alpha | `scoring/cross_sectional.py` | 截面分組統計量 |
| **P0** | Profile: Breakeven Cost, Net_Spread, Turnover | `scoring/cross_sectional.py` | Implementation 參考指標；Net_Spread = Spread - 2 × cost × Turnover |
| **P0** | Profile: Q1_Concentration | `scoring/cross_sectional.py` | 容量代理（原 Effective_Breadth 重新命名） |
| **P0** | Step 6: 前處理正交化 | `engine.py` | 未啟用時 Gate 1 可能被 beta exposure 灌水（原 P1 提升） |
| **P1** | Gate 1: BHY 連續校正 | `scoring/cross_sectional.py` | 替代固定門檻，FDR=5%，連續調整；Harvey & Liu (2020) |
| **P1** | Gate 1 OR: Romano-Wolf correction | `scoring/cross_sectional.py` | 處理 OR 條件的多重檢定膨脹 |
| **P1** | Step 3: Beta-adjusted abnormal return | `engine.py` | `ret - beta × market_ret` 替代等權均值調整 |
| **P1** | Profile: IC_Trend (Theil-Sen) | `scoring/cross_sectional.py` | 中位數斜率，對 outlier 穩健；`scipy.stats.theilslopes` |
| **P1** | Profile: Q1-Q5 Spread (VW) | `scoring/cross_sectional.py` | 等權 vs 市值加權對照 |
| **P1** | Profile: Regime IC | `scoring/cross_sectional.py` | 分期間 IC |
| **P1** | Profile: Multi-horizon IC | `scoring/cross_sectional.py` | IC decay curve |
| **P1** | Monotonicity: Patton-Timmermann MR test | `scoring/cross_sectional.py` | 利用 portfolio return 完整分佈，統計力更高 |
| **P1** | Dashboard 重構 | `dashboard/` | Gate-based Leaderboard + Detail（含 CAUTION 標記） |
| **P1** | 移除 composite score 邏輯 | `scorer.py`, `registry.py` | 移除 `map_linear`、sigmoid、routing |
| **P2** | Stage 2: Greedy Forward Selection | 新模組 | ≥ 20 因子；spanning regression（P2+ 可考慮 Double-Selection LASSO） |
| **P3** | event_study.py（獨立模組） | 新模組 | 事件訊號評估（含二元型 market timing） |
| **P3** | macro_signal.py（獨立模組） | 新模組 | 總經/區域訊號評估（複用 Gate pipeline + N-aware 降級） |
| **P3** | N-aware 自動推導 | `config.py` | `MARKET_DEFAULTS` + `auto_n_groups(N)` + Gate 降級 |
| **P4** | 跨 Universe Pervasiveness Profile | 新模組 | Pass_Count + meta-analysis 聚合；需 ≥ 2 universe 資料 |

---

## 18. 待決問題

1. **正交化 base factor 資料源**：Size、Value、Momentum、Industry 的台股資料源與更新頻率。Industry 分類用 TSE 大類（~30）還是細類（~100+）？
2. **Monotonicity n_groups**：台股 ~2000 檔維持 5 組。
3. ~~**Regime IC 的 regime 定義**~~：已決定——由使用者提供 `regime_labels`，框架不偵測 regime。Fallback 為時間二分法。見 §6。
4. **Multi-horizon IC 的天數**：預設 [1, 5, 10, 20] 是否合適？
5. **MDD benchmark**：同期等權市場 Q1-Q5 spread 的 MDD？或特定 benchmark？
6. **event_study.py / macro_signal.py 的開發時程**：是否列入近期計畫，或作為 future work？
7. **N-aware 降級的 N 門檻**：建議 N < 200 開始降級（n_groups 從 5 降為 3），是否合適？還是用更保守的 N < 500？
8. **跨 Universe Pervasiveness Profile 的啟用條件**：≥ 2 個平行（非嵌套）universe 有結果時才啟用？
9. **MARKET_DEFAULTS 的初始範圍**：初期僅配置 `tw`，其他市場（`us`、`hk`）何時加入？
10. **因子定期重新評估頻率**：已 PASS 的因子多久重跑 Gate 1-2 + Profile？已選入 Stage 2 的因子如何監控 alpha decay？常見實務為每季或每半年全量重評。
