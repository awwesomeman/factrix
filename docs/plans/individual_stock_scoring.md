# 個股風格因子績效指標計算方式

> 本文件整理個股風格類（`individual_stock`）因子的績效指標公式、圖表數據計算方式，以及對應的程式碼位置。

---

## 目錄

1. [資料前處理](#1-資料前處理)
2. [四維度評分架構](#2-四維度評分架構)
3. [預測力 (Predictability) — 權重 40%](#3-預測力-predictability--權重-40)
4. [獲利性 (Profitability) — 權重 25%](#4-獲利性-profitability--權重-25)
5. [穩健性 (Robustness) — 權重 25%](#5-穩健性-robustness--權重-25)
6. [可交易性 (Tradability) — 權重 10%](#6-可交易性-tradability--權重-10)
7. [總分計算與 VETO 機制](#7-總分計算與-veto-機制)
8. [圖表數據計算方式](#8-圖表數據計算方式)
9. [關鍵常數](#9-關鍵常數)

---

## 1. 資料前處理

**來源**: `factrix/engine.py` — `prepare_factor_data()`

| 步驟 | 公式 | 說明 |
|------|------|------|
| Forward Return | `ret_fwd = close[t+5] / close[t] - 1` | 預設 5 日前瞻報酬 |
| Forward Return Winsorize | 每日截斷至 [1st, 99th] percentile | 降低極端報酬影響 |
| Abnormal Return | `abnormal = ret_fwd - mean(ret_fwd)_date` | 截面去市場均值 |
| MAD Winsorize | `clip = median ± n_mad × 1.4826 × MAD` | 截面 MAD 截斷，防止極值扭曲 z-score |
| Cross-sectional Z-score | `z = (x - median) / (1.4826 × MAD)` | Robust z-score（使用中位數與 MAD） |

> **1.4826** 是 MAD 的一致性常數，使 MAD 成為常態分佈下 σ 的無偏估計。

---

## 2. 四維度評分架構

**來源**: `factrix/scoring/config.py` — `FACTOR_CONFIGS["individual_stock"]`

| 維度 | 權重 | 包含指標 |
|------|------|---------|
| Predictability (預測力) | 0.40 | IC_IR, Monotonicity |
| Profitability (獲利性) | 0.25 | Long_Alpha, MDD |
| Robustness (穩健性) | 0.25 | OOS_Decay, IC_Stability, Hit_Rate |
| Tradability (可交易性) | 0.10 | Turnover |

**評分公式**（`factrix/scoring/scorer.py` — `FactorScorer.compute()`）：

```
dimension_score = Σ(metric_score × adaptive_weight) / Σ(adaptive_weight)
total_score     = Σ(dimension_score × dimension_weight)
```

**自適應權重**（Adaptive Weighting）：

```
adaptive_weight = w_base × sigmoid(k × (|t_stat| - τ))
```

- `τ = 2.0`（t > 2 ≈ 95% 顯著性門檻）
- `k = 2.0`（sigmoid 斜率）
- 統計顯著性高 → 權重趨近 1；不顯著 → 權重趨近 0

---

## 3. 預測力 (Predictability) — 權重 40%

### 3.1 IC_IR（資訊比率）

**來源**: `factrix/scoring/selection.py` — `calc_ic_ir()`

| 項目 | 內容 |
|------|------|
| IC 計算 | 每日 Spearman rank correlation(factor_rank, forward_return_rank) |
| IC_IR | `IC_IR = \|mean(IC)\| / std(IC)` |
| 分數映射 | `score = linear_map(IC_IR, min=0.1, max=0.6) → [0, 100]` |
| t-stat | 從非重疊 IC 樣本計算（每 `forward_periods` 日取一個），避免自相關膨脹 |

> **非重疊取樣原因**：連續日的 forward return 共享 forward_periods-1 天，導致正自相關，會高估 IC_IR。

### 3.2 Monotonicity（分組單調性）

**來源**: `factrix/scoring/selection.py` — `calc_monotonicity()`

| 項目 | 內容 |
|------|------|
| 分組 | 每日依因子值排序分為 5 組（quintile） |
| 計算 | 每日：`mono = Spearman_corr(group_index ∈ [0,4], rank(group_mean_return))` |
| 彙總 | `avg_mono = mean(\|mono\|)` |
| 分數映射 | `score = linear_map(\|avg_mono\|, min=0.3, max=0.9)` |

---

## 4. 獲利性 (Profitability) — 權重 25%

### 4.1 Long_Alpha（Q1 超額報酬）

**來源**: `factrix/scoring/selection.py` — `calc_long_alpha()`

| 項目 | 內容 |
|------|------|
| Q1 定義 | 因子排名前 20%（`pct_rank >= 0.8`） |
| 超額報酬 | `excess = mean(ret_fwd \| Q1) - mean(ret_fwd \| all)` （每個非重疊期間） |
| 年化 | `ann_excess = (1 + total_excess)^(1/n_years) - 1` |
| 分數映射 | `score = linear_map(ann_excess, min=0.0, max=0.20)` |
| t-stat | `t = mean(excess) / (std(excess) / √n)` |

### 4.2 MDD（最大回撤）

**來源**: `factrix/scoring/timing.py` — `calc_mdd()`

| 項目 | 內容 |
|------|------|
| 組合建構 | Long/Short：Q1 (top 20%) - Q5 (bottom 20%) |
| 報酬序列 | `ls_ret = Q1_mean - Q5_mean`（非重疊） |
| NAV | `NAV = cumprod(1 + ls_ret)` |
| MDD | `MDD = min(NAV / cummax(NAV) - 1)` |
| 分數映射 | `score = linear_map(1 - MDD, min=0.5, max=0.9)` |

> MDD 越小，分數越高。

---

## 5. 穩健性 (Robustness) — 權重 25%

### 5.1 OOS_Decay（樣本外衰減）

**來源**: `factrix/scoring/selection.py` — `calc_oos_decay()`

| 項目 | 內容 |
|------|------|
| 切分 | 80% in-sample / 20% out-of-sample（按日期） |
| Decay | `decay = \|mean_IC_oos\| / \|mean_IC_is\|` |
| 方向懲罰 | 若 IS 與 OOS 的 IC 符號翻轉：`decay × 0.5` |
| 分數映射 | `score = linear_map(decay, min=0.3, max=1.0)` |
| t-stat | OOS IC 的顯著性：`mean_oos / SEM_oos` |

### 5.2 IC_Stability（IC 穩定性）

**來源**: `factrix/scoring/selection.py` — `calc_ic_stability()`

| 項目 | 內容 |
|------|------|
| 取樣 | 每 `forward_periods` 日取一個 IC（非重疊） |
| 窗口 | `window = max(4, min(12, n_ic // 3))` |
| 滾動 IC_IR | 每個 window 計算一次 `\|mean(IC)\| / std(IC)` |
| 彙總 | `mean_rolling_ir = mean(all_window_IC_IR)` |
| 分數映射 | `score = linear_map(mean_rolling_ir, min=0.1, max=0.5)` |

### 5.3 Hit_Rate（勝率）

**來源**: `factrix/scoring/timing.py` — `calc_hit_rate()`

| 項目 | 內容 |
|------|------|
| 計算 | 非重疊 IC 中 IC > 0 的比例 |
| 分數映射 | `score = linear_map(hit_rate, min=0.45, max=0.65)` |
| t-stat | 二項檢定：`t = (hit_rate - 0.5) / √(0.25 / n)` |

---

## 6. 可交易性 (Tradability) — 權重 10%

### 6.1 Turnover（換手率）

**來源**: `factrix/scoring/timing.py` — `calc_turnover()`

| 項目 | 內容 |
|------|------|
| 計算 | 每日 Rank Autocorrelation：`RC = corr(rank_today, rank_yesterday)` |
| 彙總 | `avg_RC = mean(RC)` |
| 分數映射 | `score = linear_map(avg_RC, min=0.5, max=0.95)` |
| 特殊處理 | `avg_RC > 0.99` → 固定 80 分（避免獎勵完全靜態因子） |
| VETO 門檻 | `min_threshold = 20`（分數低於 20 觸發 VETO） |

> RC 高 = 排名穩定 = 低換手 = 低交易成本。

---

## 7. 總分計算與 VETO 機制

**來源**: `factrix/scoring/scorer.py` — `FactorScorer.compute()`

### 總分公式

```
Total = Predictability × 0.40 + Profitability × 0.25 + Robustness × 0.25 + Tradability × 0.10
```

### VETO 機制

- 帶有 `min_threshold` 參數的指標，若 `score < threshold`，則觸發 VETO
- 個股風格配置中，Turnover 設有 `min_threshold = 20`
- VETO 時：`total_score × 0.2`（懲罰乘數，不歸零）
- 狀態標記為 `VETOED`（非 `PASS`）

---

## 8. 圖表數據計算方式

### 8.1 IC 累積圖 & 滾動 IC 圖

**來源**: `factrix/builders.py` — `build_ic_artifact()`

| 數據欄位 | 計算方式 |
|----------|---------|
| `ic` | 每日 Spearman rank correlation（同 IC_IR 的 IC） |
| `cum_ic` | `cumsum(ic)` — 累積 IC，顯示因子有效性趨勢 |
| `rolling_ic` | `rolling_mean(ic, window=63)` — 63 日（約 3 個月）滾動平均 |
| IS/OOS 分界 | 垂直線標記 80/20 切分點 |

**圖表呈現**（`factrix/dashboard/charts.py`）：
- 上方面板：Cumulative IC 折線圖，含 IS/OOS 垂直分界線
- 下方面板：Rolling IC 折線圖，反映不同時期因子效力

### 8.2 NAV 淨值圖

**來源**: `factrix/builders.py` — `build_nav_artifact()`

| 數據欄位 | 計算方式 |
|----------|---------|
| Q1 NAV | Q1（前 20%）組合的非重疊累積淨值 `cumprod(1 + Q1_ret)` |
| Universe NAV | 全市場平均的非重疊累積淨值 `cumprod(1 + univ_ret)` |
| Excess NAV | `Q1_NAV / Universe_NAV` — 超額淨值曲線 |

**圖表呈現**：三條曲線疊加顯示，可觀察 Q1 是否持續優於市場。

### 8.3 維度雷達圖

**來源**: `factrix/dashboard/charts.py`

- 四軸雷達圖，範圍 0–100
- 軸向：Predictability、Profitability、Robustness、Tradability
- 每軸數值為該維度的加權平均分數

### 8.4 指標明細表

**來源**: `factrix/dashboard/app.py`

| 欄位 | 說明 |
|------|------|
| Metric Type | 所屬維度（Predictability / Profitability / Robustness / Tradability） |
| Metric | 指標名稱 |
| Raw | 原始值（格式化為 % 或小數） |
| Score | 映射後分數 (0–100) |
| Threshold | VETO 門檻（若有設定） |
| t-stat | 統計顯著性 |
| Adaptive Weight | 自適應權重（受 t-stat 影響） |

---

## 9. 關鍵常數

**來源**: `factrix/scoring/_utils.py`

| 常數 | 值 | 用途 |
|------|-----|------|
| `EPSILON` | 1e-9 | 避免除以零 |
| `DDOF` | 1 | 樣本標準差（Bessel 修正） |
| `MIN_ASSETS_PER_DATE_IC` | 10 | IC 最小有效期數 |
| `MIN_STABILITY_PERIODS` | 12 | IC_Stability 最小窗口數 |
| `MIN_OOS_PERIODS` | 5 | OOS 最小期數 |
| `MIN_PORTFOLIO_PERIODS` | 5 | Long_Alpha / MDD 最小期數 |
| `DEFAULT_ADAPTIVE_TAU` | 2.0 | 自適應權重 t-stat 門檻 |
| `DEFAULT_ADAPTIVE_K` | 2.0 | 自適應權重 sigmoid 斜率 |

---

## 附錄：分數映射函式 (linear_map)

**來源**: `factrix/scoring/registry.py`

```
linear_map(value, min, max) → [0, 100]

score = 100 × (value - min) / (max - min)
score = clip(score, 0, 100)
```

將原始指標值線性映射到 0–100 分。低於 `min` 為 0 分，高於 `max` 為 100 分。
