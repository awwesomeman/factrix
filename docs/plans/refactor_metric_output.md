# Future: MetricOutput 重構計劃

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> 狀態：全部完成（P1 + P2）
> 日期：2026-04-15
> 背景：explore_price_intention notebook 討論 + quant review

## 優先順序

| # | 提案 | 理由 | 複雜度 | 狀態 |
|---|------|------|--------|------|
| P1 | 年化重構 → Alphalens normalization | 影響最廣，但改動最小（只動 preprocess） | 低 | **已完成** |
| P1 | 合併 Long_Short_Alpha | 減冗餘，恆等式拆兩個指標沒意義 | 低 | **已完成** |
| P1 | 因子歸因（expose beta/R²） | 低成本高價值，矩陣已算好只差回傳 | 中 | **已完成** |
| P2 | 統計量泛化（t_stat → stat + p_value） | 正確但不急，目前不影響判斷 | 中 | **已完成** |
| P2 | 複合指標整合數值設計 | 保留 MetricOutput，設計 mean value + min stat 摘要 | 低 | **已完成** |

---

## P1-1: 年化重構 — Alphalens normalization

### 問題

目前年化邏輯有三個問題：

1. **複利年化概念不對：** `_annualize_return()` 用複利年化，但因子 spread 每期重新建構（獨立橫截面交易），不是 buy-and-hold。導致 `ann(long) + ann(short) != ann(spread)`，加法性不成立。

2. **年化寫死在計算層：** `quantile_spread()`、`long_short_alpha()`、`quantile_spread_vw()` 的 `value` 直接存年化值，年化倍數從日曆天推算（`days / 365.25`）。

3. **不同頻率算錯：** 日曆天推算隱含日頻假設，週頻/月頻資料會不準確。

### 解法：forward_return / forward_periods

參考 Alphalens（Quantopian）的做法，在 preprocess 階段將 n-period return 除以 n：

```python
per_period_return = forward_return / forward_periods
```

**核心優勢：只依賴 `forward_periods`，不需要使用者額外指定資料頻率。**

```
日頻, forward_periods=5  → /5  → per-day return
週頻, forward_periods=4  → /4  → per-week return
月頻, forward_periods=1  → /1  → per-month return
```

不同 forward_periods 在同一頻率內自動可比。不同頻率之間本來就不該直接比。

**IC 不受影響：** rank correlation 對線性縮放不變，`corr(rank(factor), rank(return/n)) = corr(rank(factor), rank(return))`。normalization 只影響 spread / alpha 類的量值指標。

### 三層職責

| 層 | 改什麼 | 需要什麼資訊 |
|----|--------|-------------|
| preprocess | `forward_return /= forward_periods` | `forward_periods`（已有） |
| metric 層 | **不動**——value 自然是 per-period basis | — |
| 呈現層 | 想年化時 `× inferred_periods_per_year` | 從 date column 自動推 |

### 學術參考

Fama-French、AQR 論文通常報 per-period 值（如 monthly spread = 0.5%），頂多附註年化。Alphalens 也是除以 n 後報 per-day 值。

### 影響範圍

- `factorlib/preprocess.py`（或 `compute_forward_return`）— 加 `/= forward_periods`
- `quantile_spread()`, `quantile_spread_vw()` — 移除 `_annualize_return()` 呼叫，value 改為 per-period
- `factorlib/tools/_helpers.py` — `_annualize_return()` 移至 dashboard 呈現層或刪除
- Dashboard — 呈現時從 date interval 自動推 `periods_per_year`，`× periods_per_year` 顯示年化值
- 測試 — 數值 assertion 更新

---

## P1-2: 合併 Q1_Q5_Spread 和 Long_Short_Alpha

### 問題

`quantile_spread()` 和 `long_short_alpha()` 是兩個獨立指標，但 `spread = long_alpha + short_alpha` 是恆等式。Profile 裡放兩個指標是冗餘。

### 解法

只保留 `q1_q5_spread`，long/short 拆解放 metadata：

```python
MetricOutput(
    name="q1_q5_spread",
    value=mean_spread,
    stat=t_spread,
    significance="***",
    metadata={
        "mean_per_period": ...,
        "n_periods": ...,
        "long_alpha": ...,      # Q1 - Universe
        "short_alpha": ...,     # Universe - Q5
        "long_stat": ...,
        "short_stat": ...,
    },
)
```

AQR、Fama-French 論文也是報 spread，long/short decomposition 作為 supplementary table。

### 影響範圍

- 刪除 `long_short_alpha()` 函式，邏輯併入 `quantile_spread()`
- `factorlib/gates/profile.py` — profitability 從 6 個指標減為 5 個
- 如果 scoring layer 用固定維度數加權，需同步更新
- 測試、notebook、dashboard 同步更新

---

## P1-3: 因子歸因 — expose 現有計算的 beta / R²

### 問題

目前無法直接回答「這個因子有多少來自 size 暴露？」。`orthogonalize_factor` 和 `spanning_alpha` 內部已經算了回歸係數，但只取殘差/alpha，丟掉 beta 和 R²。

### 兩個視角（互補，都應做）

**1. 因子值層面（Barra 風險模型做法）**

`orthogonalize_factor` 已跑 `factor ~ size + mom_60d + industry_dummies`，保留回傳即可：

- 每個 base factor 的平均 beta（暴露方向和大小）
- Partial R²（size 單獨解釋了因子多少 cross-sectional 變異）
- Total R²（所有 base factors 合計）

**2. 報酬層面（Barillas & Shanken 2018 做法）**

`spanning_alpha` 已跑 `factor_spread ~ base_spreads`，把 beta 也報出來即可：

- 每個 base factor spread 的 beta
- R²（base factors 合計解釋了多少 spread 變異）

**關鍵洞察：兩個視角可以給出相反結論。** 因子值和 size 高度相關（R² 高），但 spanning alpha 仍顯著 → 因子雖然和 size 長得像，但捕捉到了 size 沒有的報酬來源。

### 三種「控制 base factor」方法比較

| | OLS 取殘差 | 組內去均值（demean） | Spanning test |
|---|---|---|---|
| 操作層面 | 因子值（cross-sectional） | 因子值（cross-sectional） | 因子報酬（time-series） |
| 適用 base factor | 連續 + 類別皆可 | 僅類別型 | 任何有 spread 序列的因子 |
| 輸出 | 殘差因子值 + beta + R² | 殘差因子值 | alpha + beta + R² |
| 回答的問題 | 去除暴露後排序能力還在嗎？ | 去除類別效應後排序能力還在嗎？ | 控制報酬後有增量 alpha 嗎？ |
| 目前實作 | `orthogonalize_factor()`（只取殘差） | 未獨立實作（OLS with dummies 涵蓋） | `spanning_alpha()`（只取 alpha） |

### 影響範圍

- `factorlib/tools/regression/orthogonalize.py` — 加 beta + R² 回傳
- `factorlib/tools/regression/spanning.py` — `spanning_alpha` 加 beta + R² 回傳
- Profile 新增歸因 section（非獨立 gate，而是 profile 的一部分）
- notebook 新增歸因 section

---

## P2: 統計量泛化 — t_stat → stat + p_value

### 問題

`MetricOutput` 寫死 `t_stat: float | None`，暗示所有指標都用 t 檢定。但實際上不同指標用不同檢定：

| 指標 | 目前做法 | 實際統計量 |
|------|---------|-----------|
| IC, Spread, Monotonicity | t-test: mean/(std/sqrt(n)) | t 統計量 |
| Hit Rate | binomial score test | z 統計量（但欄位叫 t_stat） |
| Q1 Concentration | 單邊 t-test (H₀: ratio >= 0.5) | t 統計量 |
| IC Trend | Theil-Sen CI → 近似 t | 不是真正的 t 統計量 |
| IC_IR | 描述性統計 | 無統計量 |

### 解法

**`t_stat` 改名 `stat`，保留為頂層欄位**（幾乎每個指標都有，放 metadata 降低可讀性）。新增 p_value。檢定細節放 metadata。

```python
@dataclass
class MetricOutput:
    name: str
    value: float
    stat: float | None = None           # 統計量（t, z, W, chi2, ...）
    significance: str | None = None     # 由 p_value 導出
    metadata: dict[str, object] = field(default_factory=dict)
    # metadata 標準 key：
    #   "p_value": float          — significance 由此導出
    #   "stat_type": str          — "t" | "z" | "wilcoxon" | "bootstrap" | ...
    #   "h0": str                 — "mu=0" | "p=0.5" | "ratio>=0.5" | ...
    #   "method": str             — "non-overlapping t-test" | "binomial score" | ...
```

### significance_marker 改由 p_value 驅動

```python
def _significance_marker(p_value: float | None) -> str:
    if p_value is None: return ""
    if p_value < 0.01: return "***"
    if p_value < 0.05: return "**"
    if p_value < 0.10: return "*"
    return ""
```

不同檢定（t / Wilcoxon / bootstrap / permutation）只要能算出 p_value 就能統一判定顯著性。這比硬套 |t| 閾值正確——因為每個指標測的假設不同，統計量的分佈也不同，只有 p_value 是可比的。

### 遷移策略

1. **Phase 1（向後相容）：** `t_stat` 改名 `stat`，保留 `t_stat` 為 `@property` alias
2. **Phase 2：** 所有函式新增 p_value 計算，significance_marker 改吃 p_value
3. **Phase 3：** 移除 `t_stat` property

### 影響範圍

- `factorlib/tools/_typing.py` — 欄位改名 + 新增 metadata 標準 key 文件
- `factorlib/tools/series/significance.py` — `_significance_marker()` 改吃 p_value
- 所有回傳 MetricOutput 的函式 — 新增 p_value 計算
- gates、dashboard、測試同步更新

---

## P2-2: 複合指標整合數值設計

### 問題

`multi_horizon_ic` 和 `regime_ic` 本質上是多個原始值的集合，硬塞進 `MetricOutput` 的單一 `value` 欄位不合適。

### 結論：不新增型別，設計有意義的 summary

引入 `CompositeMetricOutput` 的代價過高（混合型別、type-check、渲染分裂）。保留 `MetricOutput`，用 **mean value + min |stat|** 的摘要策略：

| 指標 | value | stat | 語意 |
|------|-------|------|------|
| `regime_ic` | mean(all regime ICs) | min \|t\| across regimes | 最弱 regime 仍顯著 → 全部顯著 |
| `multi_horizon_ic` | mean(all horizon ICs) | min \|t\| across horizons | 最弱 horizon 仍顯著 → 全部顯著 |

min |stat| 的邏輯：**如果最弱的 regime/horizon 都通過門檻，那所有都通過。** 比 Bonferroni 修正簡單，語意直觀。

per-regime / per-horizon 的完整細節放 metadata，不丟資訊：

```python
# regime_ic
MetricOutput(
    name="regime_ic",
    value=mean_all_regime_ic,
    stat=min_abs_t_across_regimes,
    significance=...,                   # 由 min regime 的 p_value 導出
    metadata={
        "per_regime": {
            "bull": {"mean_ic": ..., "stat": ..., "p_value": ...},
            "bear": {"mean_ic": ..., "stat": ..., "p_value": ...},
        },
        "direction_consistent": True,
        "aggregation": "mean_value_min_stat",
    },
)

# multi_horizon_ic
MetricOutput(
    name="multi_horizon_ic",
    value=mean_all_horizon_ic,
    stat=min_abs_t_across_horizons,
    significance=...,
    metadata={
        "per_horizon": {
            "1D": {"mean_ic": ..., "stat": ..., "p_value": ...},
            "5D": {"mean_ic": ..., "stat": ..., "p_value": ...},
            "10D": {"mean_ic": ..., "stat": ..., "p_value": ...},
            "20D": {"mean_ic": ..., "stat": ..., "p_value": ...},
        },
        "aggregation": "mean_value_min_stat",
    },
)
```

### 影響範圍

- `factorlib/tools/panel/ic.py` — `regime_ic()` 和 `multi_horizon_ic()` 的 value/stat 計算調整
- 不影響回傳型別、不影響 FactorProfile 結構、不影響遍歷邏輯
- Dashboard 可從 metadata 展開 per-regime / per-horizon 表格

---

## 附錄：未來可能的替代統計方法

目前所有指標使用 t-test 或其變體。以下是未來可能引入的替代方法，供 P2 實作時參考：

| 方法 | 適用場景 | 優勢 |
|------|---------|------|
| Wilcoxon signed-rank | IC 分佈非常態時 | 不依賴常態假設，breakdown point 更高 |
| Bootstrap confidence interval | IC_IR 等 ratio statistic | 直接對 ratio 建 CI，不需推導解析分佈 |
| Permutation test | 小樣本（N < 30） | 精確 p-value，不依賴漸近分佈 |
| Chi-squared / Fisher | Regime IC 聯合檢定 | 同時檢驗多個 regime 是否全部顯著 |

實作時只需讓各指標函式回傳 `metadata["p_value"]` 和 `metadata["stat_type"]`，`_significance_marker()` 統一由 p_value 導出。
