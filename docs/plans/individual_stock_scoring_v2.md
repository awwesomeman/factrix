# Factor Quality Score v2 — 評分架構重構計畫

> 本文件為 `individual_stock_scoring.md` (v1) 的優化版本，解決 v1 架構中維度權重主觀、指標冗餘、
> 缺乏因子間比較、以及過度依賴圖表質化分析等痛點。
>
> **v2 修訂紀錄：**
> - 初版：4 → 3 維度，新增 RCQ_Composite
> - 修訂 1：3 → 2 維度，消除 RCQ_Composite 的 hardcoded sub-weight (0.6/0.4)，
>   強化跨因子類型通用性，降低整體複雜度
> - 修訂 2：Stage 2 排名改為順序門檻（消除 0.7/0.3 hardcoded weight）；
>   恢復 optional 前處理正交化（按 factor_type 配置，解決資金集中度/產業曝露問題）

## 目標

以最小複雜度達成統計可靠的因子評分。消除所有 hardcoded 權重參數，使框架無需人工調參即可跨市場、跨因子類型運行。

## 核心原則

> 1. 能用統計決定的不要用人決定，但不要用更多的 overfitting 來解決 overfitting 的問題。
> 2. 簡單的模型在 OOS 通常優於複雜模型（[DeMiguel et al. 2009](literature_references.md#demiguel-garlappi--uppal-2009)；[Arnott, Harvey & Markowitz 2019](literature_references.md#arnott-harvey--markowitz-2019)）。
> 3. 維度應回答「根本問題」，而非描述「計算方法」。

---

## 目錄

1. [為什麼從 v1 的 4 維度和初版 v2 的 3 維度改為 2 維度](#1-為什麼改為-2-維度)
2. [2 維度 FQS 架構](#2-二維度-fqs-架構)
3. [各因子類型的指標路由](#3-各因子類型的指標路由)
4. [維度內加權機制](#4-維度內加權機制)
5. [閘門機制（評分前過濾）](#5-閘門機制評分前過濾)
6. [Stage 2：因子篩選（≥ 20 因子後啟用）](#6-stage-2因子篩選-20-因子後啟用)
7. [分數映射策略](#7-分數映射策略)
8. [v1 → v2 指標遷移對照](#8-v1--v2-指標遷移對照)
9. [資料前處理流水線](#9-資料前處理流水線)
10. [圖表與 Dashboard 更新規格](#10-圖表與-dashboard-更新規格)
11. [關鍵常數](#11-關鍵常數)
12. [參數分類決策表](#12-參數分類決策表)
13. [實作優先順序](#13-實作優先順序)
14. [待決問題](#14-待決問題)

---

## 1. 為什麼改為 2 維度

### 1.1 v1（4 維度）的問題

- 16 個 hardcoded 權重決策（4 factor_type × 4 維度）
- 指標間存在冗餘（IC_IR ⊃ Hit_Rate、LS_IR ⊃ MDD）
- 不同 factor_type 需要完全不同的 routing 配置

### 1.2 初版 v2（3 維度）的問題

- `RCQ_Composite = 0.6 × Monotonicity + 0.4 × IC_IR` 引入新的 hardcoded sub-weight，與「消除 hardcoded weights」的核心原則自我矛盾
- LS_IR 和 Monotonicity 對 event_signal（二元訊號，無法排序分組）和 global_macro（N=10-30，分組每組僅數個資產）不適用——三個維度中兩個對非個股因子失效
- 同時引入熵權法、分位數錨定、MC 敏感度、雙層正交化共 4 個新機制，complexity budget 超支

### 1.3 修訂版（2 維度）的設計邏輯

從 [Cochrane (2011)](literature_references.md#cochrane-2011)、[Berkin & Swedroe (2016)](literature_references.md#berkin--swedroe-2016)、[Harvey (2017)](literature_references.md#harvey-2017)、[Macrosynergy](literature_references.md#macrosynergy-2024) 跨資產訊號評估框架的共同結構提煉，因子評估的根本問題只有兩個：

| 根本問題 | 維度 | 含義 |
|---------|------|------|
| **這個訊號是真的嗎？** | Reliability（訊號可靠性） | 統計上能否區分於雜訊 / 過擬合？ |
| **能拿它賺錢嗎？** | Profitability（經濟獲利性） | 扣除交易成本後能否產生可持續的 alpha？ |

> **為什麼不需要第三個維度「Uniqueness」？** 因為因子的唯一性/增量貢獻是**因子之間**的比較，不是單因子的屬性。這應該是獨立的「因子篩選」階段（Stage 2），不應混入單因子打分公式。[Barillas & Shanken (2017)](literature_references.md#barillas--shanken-2017) 和 [Feng, Giglio, Xiu (2020)](literature_references.md#feng-giglio--xiu-2020) 的 spanning test 和 double-selection LASSO 都是在因子被個別評估之後才執行的。

**2 維度的關鍵優勢：維度權重直接 50/50 固定，零參數。** 兩個維度不需要熵權法（只有 2 個值，熵的鑑別力不足）。等權的理論支持：[DeMiguel et al. (2009)](literature_references.md#demiguel-garlappi--uppal-2009) 證明 1/N 在 OOS 常勝過最佳化權重。

### 1.4 版本演進對比

| 度量 | v1 (4 維度) | v2 初版 (3 維度) | v2 修訂 (2 維度) |
|------|:-----------:|:---------------:|:---------------:|
| hardcoded 權重參數數 | 4 | 3 + 2 (RCQ) = 5 | **0** |
| 對 event_signal 適用度 | 需獨立配置 | 2/3 維度不適用 | **全適用** |
| 對 global_macro 適用度 | 需獨立配置 | 2/3 維度勉強 | **全適用** |
| 新增機制數 | — | 4 個 | **1 個**（Stage 2 incremental alpha） |
| 維度權重方案 | hardcoded | 等權→熵權 | **固定等權 50/50** |

---

## 2. 二維度 FQS 架構

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Factor Quality Score（單因子獨立評估）          │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │   Reliability    │    │  Profitability   │             │
│  │   （訊號可靠性）  │    │  （經濟獲利性）   │             │
│  │                 │    │                 │              │
│  │  指標因          │    │  指標因          │              │
│  │  factor_type    │    │  factor_type    │              │
│  │  而異            │    │  而異            │              │
│  │                 │    │                 │              │
│  │  維度內加權：     │    │  維度內加權：     │              │
│  │  base=1.0       │    │  base=1.0       │              │
│  │  × sigmoid(t)   │    │  × sigmoid(t)   │              │
│  └────────┬────────┘    └────────┬────────┘              │
│           │      50%  /  50%     │                       │
│           └──────────┬───────────┘                       │
│                      │                                   │
│              ┌───────▼───────┐                           │
│              │  FQS (0-100)  │                           │
│              └───────┬───────┘                           │
│                      │                                   │
│              Gates: OOS_Decay VETO, t-stat skip          │
└──────────────────────┼──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Stage 2: Factor Selection（≥ 20 因子後啟用）            │
│                                                         │
│  • FQS ≥ 品質門檻 → 合格因子池                           │
│  • 合格池內按 Incremental Alpha 排序（不與 FQS 混合）     │
│  • Sensitivity Analysis（MC rank stability）             │
│  • 分位數錨定 Score Map（optional 校準）                  │
└─────────────────────────────────────────────────────────┘
```

**評分公式：**

```
FQS = 0.5 × Reliability_Score + 0.5 × Profitability_Score
```

其中：

```
Dimension_Score = Σ(metric_score × adaptive_weight) / Σ(adaptive_weight)
adaptive_weight = 1.0 × sigmoid(k × (|t_stat| - τ))
```

這沿用 v1 已驗證的機制（`factrix/scoring/scorer.py`），不引入新參數。

---

## 3. 各因子類型的指標路由

維度不變，指標根據因子類型切換。每個指標的 base weight 均為 1.0，由 adaptive sigmoid 自動依 t-stat 調權。

### Reliability（訊號可靠性）

回答：「這個訊號統計上是真的嗎？不是雜訊也不是過擬合？」

| 指標 | individual_stock | event_signal | global_macro | group_region | 計算方式 |
|------|:---:|:---:|:---:|:---:|------|
| IC_IR | v | | v* | v* | `∣mean(IC)∣ / std(IC)`，IC = 每日截面 Spearman rank corr |
| Monotonicity | v | | | | 每日分 5 組，`Spearman(group_idx, rank(group_ret))` 取均值 |
| Hit_Rate | v | | v | v | 非重疊期間 IC > 0 的比例；t-stat 用二項檢定 |
| OOS_Decay | v | | v | v | `∣IC_OOS∣ / ∣IC_IS∣`，80/20 時間切分；異號 ×0.5 懲罰 |
| Event_CAAR | | v | | | `signed_car = ret_fwd × sign(signal)`，跨事件日均值 |
| Event_Decay | | v | | | IS/OOS signed CAAR 比較 |
| Event_Stability | | v | | | 月度 signed CAAR 的 `1 - CV` |
| Event_Hit_Rate | | v | | | `signed_car > 0` 的比例 |

> \* IC_IR 在截面 N < 30 時 noise 極大（[Giglio, Xiu & Zhang 2025](literature_references.md#giglio-xiu--zhang-2025)），global_macro 和 group_region 應以 Hit_Rate 和 OOS_Decay 為主，IC_IR 作為輔助參考，adaptive sigmoid 會因其 t-stat 較低而自動降權。

### Profitability（經濟獲利性）

回答：「扣除交易成本後能持續賺錢嗎？」

| 指標 | individual_stock | event_signal | global_macro | group_region | 計算方式 |
|------|:---:|:---:|:---:|:---:|------|
| LS_IR | v | | v* | v* | `mean(R_LS) / std(R_LS) × √252`，R_LS = Q1均值 - Q5均值 |
| Breakeven_Cost | v | | v | v | `Gross_Alpha / (2 × Turnover)`（[Novy-Marx & Velikov 2016](literature_references.md#novy-marx--velikov-2016)） |
| MDD | v | | v | v | L/S 組合最大回撤，`score = map(1 - MDD)` |
| Effective_Breadth | v | | v* | v* | `1 / HHI_Q1`，HHI 由 Q1 內因子值佔比計算；衡量訊號分散度 |
| Profit_Factor | | v | | | `sum(signed_car > 0) / ∣sum(signed_car < 0)∣` |
| Event_Skewness | | v | | | signed CAR 分佈偏態；正偏 = 偶有大賺 |

> \* LS_IR 在 N < 20 時每組僅數個資產，集中度極高且 noise 大。adaptive sigmoid 會因 t-stat 低而自動降權。
>
> 保留 MDD 作為獨立指標的理由（與初版 v2 不同）：LS_IR 的波動度分母是二階矩統計量，反映平均風險水準；MDD 捕捉的是**路徑依賴的極端風險**（最差連續回撤），兩者資訊面不完全重疊。在 2 維度架構下，Profitability 維度有空間容納 MDD 而不造成維度間的冗餘，由 adaptive sigmoid 根據 t-stat 自動決定其權重。

### 資金集中度問題的分析與處理

因子的 alpha 可能過度集中在少數標的或特定類型標的，導致評分虛高但實務不可用。集中度問題可分為三種類型，每種由不同機制處理：

#### 三種集中度類型與對應防線

| 集中度類型 | 場景例子 | 本質 | 負責處理的機制 |
|-----------|---------|------|--------------|
| **類型曝露集中** | Q1 全是半導體股、全是小型股 | 因子與已知因子（Industry/Size）高度相關 | **Step 6 正交化**：回歸去除已知因子曝露後，alpha 若消失 → Stage 1 低分自動過濾 |
| **過擬合集中** | AI 因子意外挑到 IS 期間的特定一批股票 | 對歷史資料過度擬合 | **OOS_Decay VETO**：IS 表現好但 OOS 崩壞 → 觸發 VETO |
| **個股集中（容量問題）** | 因子的 alpha 確實只存在於少數標的，OOS 也持續有效 | 因子是「真的」但可部署資金量極低 | **Effective_Breadth 代理指標**（見下方） |

#### 前兩種類型的處理邏輯

**類型曝露集中**是最常見的假因子來源。[Hou, Xue & Zhang (2020)](literature_references.md#hou-xue--zhang-2020) 發現約 65% 的因子在市值加權後消失，主因就是因子集中於小型股或特定產業。Step 6 前處理正交化通過截面回歸去除 Size/Value/Momentum/Industry 曝露，直接治病因（factor exposure）而非症狀（concentration）：

- 回歸後 alpha 消失 → 因子只是已知因子的偽裝，Stage 1 低分自動過濾
- 回歸後 alpha 仍存在 → 因子有超越已知因子的純淨選股能力

**過擬合集中**由 OOS_Decay 閘門處理。如果因子在 IS 期間挑到的標的無法延續到 OOS，IC_OOS/IC_IS 會遠低於 0.5 門檻，觸發 VETO。

#### 第三種類型：個股集中與容量問題

前兩種類型處理後仍可能存在的情境：因子正交化後仍有 alpha，OOS 也持續有效，但 alpha 集中在少數特定標的。此時：

- Q1 等權設計提供部分保護：Q1 = 前 20% ≈ 400 檔（台股 2000 檔），即使 10 檔有極端訊號也只佔 Q1 的 2.5%
- 但如果因子的有效訊號只覆蓋少數標的，**可部署的資金量（capacity）極低**，對需要部署資金的使用者來說實務價值有限

這屬於因子品質（quality）與因子容量（capacity）的區分：
- 因子品質：「這個訊號是否真實且能獲利？」→ Reliability + Profitability 已覆蓋
- 因子容量：「能放多少資金進去？」→ 需要額外的代理指標

**Effective_Breadth** 作為容量代理指標加入 Profitability 維度：

#### Effective_Breadth（有效廣度）

| 項目 | 內容 |
|------|------|
| 定義 | Q1 組合中有效貢獻 alpha 的標的數，使用 Herfindahl 指數的倒數計算 |
| 公式 | `Eff_Breadth = 1 / HHI_Q1`，其中 `HHI_Q1 = Σ(w_i²)`，`w_i = ∣factor_z_i∣ / Σ∣factor_z∣` 為 Q1 內各標的的因子值佔比 |
| 直覺 | 若 Q1 的 400 檔因子值差異不大 → HHI 低 → Eff_Breadth 高（接近 400）→ 分散良好。若只有 10 檔有極端值 → HHI 高 → Eff_Breadth 低（接近 10）→ 高度集中 |
| t-stat | 跨非重疊期間的 Eff_Breadth 序列，`t = mean / SEM` |
| 分數映射 | `score = map_linear(Eff_Breadth / N_Q1, min=0.1, max=0.7)`，其中 `N_Q1` 為 Q1 標的數 |
| 適用性 | `individual_stock`：v；`event_signal`：不適用（無連續排名）；`global_macro` / `group_region`：v（但因 N 小，自然 Eff_Breadth 低，adaptive sigmoid 會因 t-stat 低而降權） |

> **為什麼不用獨立的集中度指標取代正交化？** 集中度是症狀，正交化治的是病因（factor exposure）。但正交化無法處理個股層級的集中——正交化後殘差仍可能集中在少數標的。Effective_Breadth 補充了正交化的盲點，衡量的是**殘差化後的訊號分散度**，兩者互補。
>
> **為什麼歸入 Profitability 而非 Reliability？** Effective_Breadth 衡量的是「能否大規模部署」——這是經濟獲利性的問題，不是統計可靠性的問題。一個集中但穩定的訊號在統計上是「可靠的」，只是容量受限。

### 為什麼不需要 RCQ_Composite？

在 2 維度架構下，Monotonicity 和 IC_IR 是 Reliability 維度內的**獨立指標**，各自有 base weight = 1.0 和獨立的 t-stat → adaptive sigmoid。系統自動根據統計顯著性調權，不需要人為設定 0.6/0.4。

```
v2 初版:  RCQ_Composite = 0.6 × score(Monotonicity) + 0.4 × score(IC_IR)  ← hardcoded
v2 修訂:  Reliability = adaptive_avg(Monotonicity, IC_IR, Hit_Rate, OOS_Decay)  ← 自適應
```

---

## 4. 維度內加權機制

沿用 v1 已驗證的 adaptive sigmoid，不引入新參數：

```
adaptive_weight(metric) = 1.0 × sigmoid(k × (|t_stat| - τ))
```

| 參數 | 值 | 來源 |
|------|-----|------|
| base weight | 1.0 | 等權基底——所有指標起跑點相同 |
| τ (tau) | 2.0 | 95% 信賴區間門檻；可配置為 3.0（Harvey-strict 模式，[Harvey et al. 2016](literature_references.md#harvey-liu--zhu-2016)） |
| k | 2.0 | sigmoid 斜率控制 |

**效果：**
- t-stat 高的指標 → 權重趨近 1.0（統計可靠，充分計入）
- t-stat 低的指標 → 權重趨近 0（統計不可靠，自動降權）
- t-stat = None（資料不足無法計算）→ 維持 base weight

> 這個機制的優勢在於它是**自適應的但不是 data-fitted 的**——sigmoid 的形狀由 τ 和 k 兩個有理論基礎的常數決定，不需要從資料中估計參數，因此不存在 meta-overfitting 風險。

---

## 5. 閘門機制（評分前過濾）

| 閘門 | 條件 | 動作 | 文獻依據 |
|------|------|------|---------|
| OOS 衰減 | `IC_OOS / IC_IS < 0.5` | VETO（分數 × 0.2） | [McLean & Pontiff (2016)](literature_references.md#mclean--pontiff-2016)：平均 OOS 衰減約 32%；0.5 門檻僅過濾嚴重 overfit |
| 統計顯著性 | 主要指標 t-stat < 1.5 | 跳過完整評分 | 節省計算資源；t < 1.5 連傳統 90% CI 都達不到 |

> 使用 VETO（乘以懲罰係數 0.2）而非硬性 0/1 剔除，避免邊界效應：`IC_OOS/IC_IS = 0.49` 與 `0.51` 本質差異不大。

---

## 6. Stage 2：因子篩選（≥ 20 因子後啟用）

Stage 1 評估因子的**個體品質**，Stage 2 評估因子的**邊際貢獻**。兩者分開是因為邊際貢獻取決於已選入的因子集合，會隨集合變化——它不是因子本身的固有屬性。

### 6.1 增量 Alpha（Incremental Alpha）

- **篩選邏輯：順序門檻，非加權混合。** FQS 和 Incremental Alpha 回答不同問題（個體品質 vs 邊際貢獻），不應混合成單一加權分數。
- Stage 1 的 FQS 作為**品質門檻**：`FQS ≥ threshold` → 進入 Stage 2
- Stage 2 內按 Incremental Alpha **獨立排序**，不與 FQS 混合
- 對每個通過門檻的因子，執行 spanning regression：
  `R_new = α + β₁·R_factor1 + β₂·R_factor2 + ... + ε`
- α ≠ 0 → 該因子有增量貢獻（[Barillas & Shanken 2017](literature_references.md#barillas--shanken-2017)）

```
Stage 1:  FQS ≥ 品質門檻 → 合格因子池
Stage 2:  合格因子池內，按 Incremental Alpha 排序
結果:     零權重參數
```

### 6.2 敏感度分析（Sensitivity Analysis）

- 從 Dirichlet(α=1) 抽取 1000 組隨機維度權重
- 計算每組權重下的因子排名
- 回報每因子的 rank_std
- rank_std 低 → 排名穩健；rank_std 高 (> 5) → 排名依賴權重假設

### 6.3 分位數錨定 Score Map（Optional）

在因子數 ≥ 20 後，可選擇性啟用：

```python
def map_quantile(value, observed_values, p_floor=0.10, p_ceil=0.90,
                 fallback_min, fallback_max):
    if len(observed_values) < MIN_CALIBRATION_SAMPLE:
        return map_linear(value, fallback_min, fallback_max)
    min_val = np.percentile(observed_values, p_floor * 100)
    max_val = np.percentile(observed_values, p_ceil * 100)
    return map_linear(value, min_val, max_val)
```

> 此機制列為 optional 而非 Stage 1 核心，因為：(1) 分數映射不影響維度內排名（monotonic transform）；(2) 增加 MLflow 校準狀態的實作成本；(3) 核心架構應先穩定再優化。

---

## 7. 分數映射策略

### Stage 1（核心，立即實作）

維持 v1 的 `map_linear` + 文獻預設值作為 fallback：

| 指標 | min_val | max_val | 分數範圍 | 文獻/經驗來源 |
|------|---------|---------|---------|-------------|
| IC_IR | 0.1 | 0.6 | 0–100 | [Grinold & Kahn (2000)](literature_references.md#grinold--kahn-2000)：ICIR > 0.5 為「有趣」 |
| Monotonicity | 0.3 | 0.9 | 0–100 | [Patton & Timmermann (2010)](literature_references.md#patton--timmermann-2010) |
| Hit_Rate | 0.45 | 0.65 | 0–100 | 二項檢定 H0: p=0.5 |
| OOS_Decay | 0.3 | 1.0 | 0–100 | [McLean & Pontiff (2016)](literature_references.md#mclean--pontiff-2016)：平均衰減 ~32% |
| LS_IR (ann.) | 0.0 | 2.0 | 0–100 | 業界標準：Sharpe > 1.0 為良好 |
| Breakeven_Cost | 10 bps | 100 bps | 0–100 | [Novy-Marx & Velikov (2016)](literature_references.md#novy-marx--velikov-2016) |
| MDD (1-MDD) | 0.5 | 0.9 | 0–100 | 業界經驗：MDD < 10% 為優 |
| Effective_Breadth | 0.1 | 0.7 | 0–100 | 以 Eff_Breadth/N_Q1 比值映射；0.7 = Q1 中 70% 標的有效貢獻 |
| Event_CAAR | 0.5% | 5% | 0–100 | — |
| Profit_Factor | 0.8 | 2.0 | 0–100 | — |

### Stage 2（Optional，≥ 20 因子後）

切換為分位數錨定（P10/P90），文獻預設值降為 fallback。

---

## 8. v1 → v2 指標遷移對照

```
v1 IC_IR          → v2 Reliability 維度（獨立指標，base=1.0，adaptive sigmoid）
v1 Monotonicity   → v2 Reliability 維度（獨立指標，base=1.0，adaptive sigmoid）
v1 Long_Alpha     → v2 LS_IR（吸收至 Profitability 維度的多空組合 IR）
v1 MDD            → v2 Profitability 維度（保留為獨立指標，捕捉路徑依賴的極端風險）
v1 OOS_Decay      → v2 Reliability 維度（同時作為 VETO 閘門，門檻 0.5）
v1 IC_Stability   → 移除（被 IC_IR 在 Reliability 維度覆蓋，double counting）
v1 Hit_Rate       → v2 Reliability 維度（保留——對 global_macro/group_region 是主要指標）
v1 Turnover (RC)  → v2 Breakeven Cost（升級：加入 alpha 資訊；[Novy-Marx & Velikov 2016](literature_references.md#novy-marx--velikov-2016)）

v2 初版 RCQ_Composite (0.6/0.4) → 消除（Monotonicity 和 IC_IR 各自獨立評分）
v2 新增 Effective_Breadth  → Profitability 維度（容量代理指標，衡量訊號分散度）
```

### 與 v2 初版的關鍵差異

| 項目 | v2 初版 | v2 修訂 | 理由 |
|------|---------|---------|------|
| 維度數 | 3 | **2** | 通用性：event_signal 和 global_macro 完全適用 |
| RCQ_Composite (0.6/0.4) | 存在 | **消除** | 自我矛盾的 hardcoded weight |
| MDD | 移除 | **保留** | 與 LS_IR 資訊面不完全重疊；2 維度下有空間容納 |
| Hit_Rate | 移除 | **保留** | 對小截面 factor_type 是主要可靠性指標 |
| 維度權重 | 等權→熵權 | **固定等權** | 2 維度下熵權無統計意義 |
| 熵權法 | Stage 1 | **移除** | 簡化——2 維度不需要 |
| 正交化 | 前處理+後處理雙層 | **前處理 optional + Stage 2** | 前處理按 factor_type 配置；Stage 2 做 incremental alpha |
| 分位數錨定 | Stage 1 核心 | **Stage 2 optional** | 核心先穩定再優化 |
| 集中度/容量 | 無 | **Effective_Breadth 新增** | 補充正交化無法處理的個股集中盲點 |

---

## 9. 資料前處理流水線

**來源：** `factrix/engine.py` — `prepare_factor_data()`

| 步驟 | 公式 | 說明 | 必要性 |
|------|------|------|--------|
| 1. Forward Return | `ret_fwd = close[t+N] / close[t] - 1` | 前瞻報酬（預設 N=5） | 必要 |
| 2. Forward Return Winsorize | 每日截斷至 [1st, 99th] percentile | 降低極端報酬影響 | 必要 |
| 3. Abnormal Return | `abnormal = ret_fwd - mean(ret_fwd)_date` | 截面去市場均值 | 必要 |
| 4. MAD Winsorize | `clip = median ± n_mad × 1.4826 × MAD` | 截面 MAD 截斷 | 必要 |
| 5. Cross-sectional Z-score | `z = (x - median) / (1.4826 × MAD)` | Robust z-score | 必要 |
| **6. Factor Orthogonalization** | `ε = factor_z - (β₁·Size + β₂·Value + β₃·Momentum + Σβ_k·Industry_k)` | **截面回歸取殘差，移除已知因子曝露** | **Optional** |

### 9.1 Step 6 正交化的設計

**目的：** 解決因子的 alpha 可能來自產業曝露或市值集中度（而非真正的選股能力）的問題。[Hou, Xue & Zhang (2020)](literature_references.md#hou-xue--zhang-2020) 發現約 65% 的因子在市值加權後消失，主因就是因子集中於特定產業或小型股。前處理回歸能區分「因子就是產業曝露」與「因子在各產業內都有效但某產業效果特別強」這兩種情境。

**為什麼不用獨立的集中度指標取代？** 集中度是症狀，產業/市值曝露是病因。回歸治的是病因——如果殘差化後 alpha 消失，因子會被 Stage 1 低分自動過濾；如果 alpha 仍存在，說明因子有純淨的選股能力。加集中度指標只增加複雜度但不增加資訊。

**按 factor_type 配置啟停：**

```python
# config.py
FACTOR_CONFIGS = {
    "individual_stock": {
        "orthogonalize": True,       # 預設啟用：個股選股應去除已知因子曝露
        "ortho_factors": ["size", "value", "momentum", "industry"],
    },
    "event_signal": {
        "orthogonalize": False,      # 二元訊號 (0/±1) 不適用截面回歸
    },
    "group_region": {
        "orthogonalize": False,      # 產業/區域曝露就是訊號本身
    },
    "global_macro": {
        "orthogonalize": False,      # 資產層級曝露就是訊號本身
    },
}
```

**計算方式：**
- 逐日截面回歸：`factor_z_i = β₁·Size_i + β₂·Value_i + β₃·Mom_i + Σβ_k·Ind_ik + ε_i`
- 殘差 ε 取代 factor_z 進入後續評分
- Industry 使用 dummy variables（one-hot encoding）
- 需要 base factor 資料（Size, Value, Momentum, Industry classification）

**與 Stage 2 Incremental Alpha 的關係：**
- Step 6 正交化：對已知大類因子做前處理，確保 Stage 1 分數反映「純淨 alpha」
- Stage 2 Incremental Alpha：對已入選因子做後置篩選，避免冗餘
- 兩者目的不同、處理時機不同，不衝突

---

## 10. 圖表與 Dashboard 更新規格

### 10.1 Leaderboard 排行榜

| 項目 | v1 | v2 修訂 |
|------|----|----|
| 維度欄位 | Predictability, Profitability, Robustness, Tradability | **Reliability, Profitability** |
| 維度數 | 4 | **2** |
| 新增欄位 | — | Rank Stability（≥ 20 因子後顯示） |

### 10.2 維度雷達圖

- v1：4 軸
- v2 修訂：**2 軸 → 改為水平長條對比圖**（2 軸的雷達圖視覺效果差）
- 每軸範圍 0–100

### 10.3 LS NAV 圖

| 數據欄位 | 計算方式 |
|----------|---------|
| `ls_nav` | `cumprod(1 + R_LS)`，R_LS = Q1 均值 - Q5 均值（非重疊） |
| `long_nav` | `cumprod(1 + Q1_ret)`（做多端淨值） |
| `short_nav` | `cumprod(1 + Q5_ret)`（做空端淨值） |
| IS/OOS 分界 | 垂直線標記 80/20 切分點 |

### 10.4 IC 累積圖 & 滾動 IC 圖

與 v1 相同，維持不變。

### 10.5 指標明細表

| 欄位 | 說明 |
|------|------|
| Dimension | **Reliability** 或 **Profitability** |
| Metric | 指標名稱 |
| Raw | 原始值 |
| Score | 映射後分數 (0–100) |
| t-stat | 統計顯著性 |
| Adaptive Weight | sigmoid 調整後權重 |

### 10.6 Stage 2 報告（≥ 20 因子後顯示）

| 圖表 | 說明 |
|------|------|
| 排名穩定性長條圖 | 每個因子的 rank_std，由低到高排列 |
| Incremental Alpha 矩陣 | 每個因子相對於已選因子的 spanning regression α |

---

## 11. 關鍵常數

### 延用 v1 的常數

| 常數 | 值 | 用途 |
|------|-----|------|
| `EPSILON` | 1e-9 | 避免除以零 |
| `DDOF` | 1 | 樣本標準差（Bessel 修正） |
| `MIN_ASSETS_PER_DATE_IC` | 10 | IC 最小有效期數 |
| `MIN_OOS_PERIODS` | 5 | OOS 最小期數 |
| `MIN_PORTFOLIO_PERIODS` | 5 | LS_IR / Breakeven Cost / MDD 最小期數 |
| `DEFAULT_ADAPTIVE_TAU` | 2.0 | 自適應權重 t-stat 門檻 |
| `DEFAULT_ADAPTIVE_K` | 2.0 | 自適應權重 sigmoid 斜率 |
| `DEFAULT_VETO_PENALTY` | 0.2 | VETO 懲罰乘數 |

### v2 新增的常數

| 常數 | 值 | 用途 |
|------|-----|------|
| `MIN_CALIBRATION_SAMPLE` | 20 | 啟用 Stage 2 各機制的最小因子數 |
| `OOS_DECAY_VETO_THRESHOLD` | 0.5 | OOS 衰減閘門：IC_OOS/IC_IS < 此值觸發 VETO |
| `TSTAT_SKIP_THRESHOLD` | 1.5 | 主要指標 t-stat 低於此值跳過完整評分 |
| `SENSITIVITY_N_SIMULATIONS` | 1000 | Monte Carlo 權重抽樣次數（Stage 2） |

---

## 12. 參數分類決策表

| 參數類型 | 參數 | 值 | 決策方式 | 理由 |
|---------|------|-----|---------|------|
| **固定（理論）** | 維度權重 | 50/50 | 等權固定 | 2 維度下無需最佳化；[DeMiguel (2009)](literature_references.md#demiguel-garlappi--uppal-2009) |
| **固定（理論）** | 維度內 base weight | 1.0 | 等權固定 | 讓 adaptive sigmoid 處理差異化 |
| **固定（理論）** | adaptive tau | 2.0 | 95% CI | [Harvey et al. (2016)](literature_references.md#harvey-liu--zhu-2016)；可選 3.0 |
| **固定（理論）** | adaptive k | 2.0 | sigmoid 斜率 | 經驗校準 |
| **固定（設計）** | VETO penalty | 0.2 | 設計決策 | 懲罰但不歸零 |
| **固定（文獻）** | Score map min/max | 各指標不同 | 文獻預設 | 見 §7 分數映射表 |
| **可配置** | n_groups | 5 | 方法論 | N < 3000 用 5 組 |
| **可配置** | oos_ratio | 0.2 | 方法論 | 80/20 切分 |
| **可配置** | q_top | 0.2 | 方法論 | Quintile 定義 |
| **自適應** | 指標加權 | sigmoid(t) | 統計驅動 | t-stat 越高 → 權重越大 |
| **Stage 2 optional** | Score map 校準 | P10/P90 | 分位數錨定 | ≥ 20 因子後啟用 |

> **零 hardcoded 權重：** 維度權重 50/50 是「恰好兩個維度的等權」，不是主觀選擇。維度內指標權重由 adaptive sigmoid 決定，也非 hardcoded。系統中不存在任何需要人工調整的權重參數。

---

## 13. 實作優先順序

| 優先級 | 項目 | 改動範圍 | 說明 |
|:------:|------|---------|------|
| **P0** | 維度重構 4 → 2 | `config.py`, `scorer.py` | routing 改為 `{"reliability": 0.5, "profitability": 0.5}`；指標重新歸類 |
| **P0** | 新增 LS_IR 指標 | `timing.py` | 多空組合年化 IR 計算 |
| **P0** | 新增 Breakeven_Cost 指標 | `timing.py` | `Gross_Alpha / (2 × Turnover)` |
| **P0** | 新增 Effective_Breadth 指標 | `selection.py` | `1 / HHI_Q1`，容量代理指標 |
| **P1** | 閘門機制 | `scorer.py` | OOS_Decay VETO + t-stat skip |
| **P1** | Optional 前處理正交化 | `engine.py`, `config.py` | 截面回歸取殘差；按 factor_type 配置啟停 |
| **P1** | Dashboard 更新 | `dashboard/` | 2 維度 Leaderboard + 水平長條對比圖 |
| **P2** | Stage 2: Incremental Alpha | 新模組 | Spanning regression；≥ 20 因子啟用 |
| **P2** | Stage 2: Sensitivity Analysis | 新模組 | MC rank stability；≥ 20 因子啟用 |
| **P3** | Stage 2: 分位數錨定 Score Map | `registry.py` | Optional 校準機制 |

---

## 14. 待決問題

1. **Breakeven Cost 的交易成本估計**：台股成本結構（證交稅 3‰ + 手續費 1.425‰ × 2 + 衝擊成本）需明確定義 bps 基準，以判斷 Breakeven Cost 的 score map 是否合理。
2. **global_macro / group_region 的 LS_IR 適用門檻**：N < 20 時 LS_IR 每組僅數個資產，建議設定 `MIN_LS_ASSETS = 20`，低於此值該指標回傳 None 由 adaptive sigmoid 自動跳過。
3. **Monotonicity n_groups**：台股 ~2000 檔維持 5 組（quintile）；[Patton & Timmermann (2010)](literature_references.md#patton--timmermann-2010) 支持最多 15 組，但小 N 建議用 all-pairs 版本（power 更高）。
4. **正交化 base factor 資料源**：Step 6 和 Stage 2 的 spanning regression 均需要 base factor 資料。Size（市值）、Value（PB/PE）、Momentum（過去 12 個月報酬）、Industry（產業分類 dummy）四個？需確認台股可用的資料源與更新頻率。
