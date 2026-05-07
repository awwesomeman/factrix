# 命名一致性計劃

> 狀態：已實作（Phase 1-3 完成，Phase 4 skip — 無 alias）
> 日期：2026-04-15
>
> **SUPERSEDED 2026-04-20 (commits `8f15db8` → Phase 2a, BC break):** 原表格中的
> `q1_q5_spread` / `q1_q5_spread_vw` / `q1_concentration` 不再適用。
> `n_groups` 為可配置（CS 預設 10、MP 預設 3），硬編碼 "Q1"/"Q5"
> 會誤導。Rename 歷程：
>   - `q1_q5_spread` → `long_short_spread` (`8f15db8`) → `quantile_spread` (Phase 2a)
>   - `q1_q5_spread_vw` → `long_short_spread_vw` (`8f15db8`) → `quantile_spread_vw` (Phase 2a)
>   - `q1_concentration` → `top_concentration`（配對 intermediate 欄位 `top_return` /
>     `bottom_return`）
>
> Phase 2a 統一方向：method / primitive `MetricOutput.name` / Profile field /
> cache key / `describe_profile_values` header 均用 `quantile_spread`。
> 下方表格保留原樣作為歷史紀錄——**請勿**以本文件作為當前 canonical
> source，實際欄位以 Profile dataclass 為準。

## 設計原則

1. **Python 標準：** 全小寫 + underscore（PEP 8），適用於函式名、變數名、MetricOutput name
2. **可讀性優先：** 命名應能直接表達「這個函式做什麼、回傳什麼」
3. **一致性：** 同類函式用同樣的 pattern，降低記憶成本
4. **Quant 慣例：** 業界常見縮寫保持辨識度（ic, ir, oos, vw, hhi）

---

## 一、MetricOutput name — 統一為 snake_case

### 現狀

混合 Title_Case + 大寫縮寫，風格不統一：

```
IC, IC_IR, IC_Trend, Hit_Rate, Monotonicity, OOS_Decay,
Q1_Q5_Spread, Q1_Q5_Spread_VW, Long_Short_Alpha,
Turnover, Breakeven_Cost, Net_Spread, Q1_Concentration,
Multi_Horizon_IC, Regime_IC, Spanning_Alpha
```

### 規則

全部改為 **snake_case**，縮寫不特殊處理（一律小寫）：

| 現在 | 改為 | 說明 |
|------|------|------|
| `IC` | `ic` | |
| `IC_IR` | `ic_ir` | |
| `IC_Trend` | `ic_trend` | |
| `Hit_Rate` | `hit_rate` | |
| `Monotonicity` | `monotonicity` | |
| `OOS_Decay` | `oos_decay` | |
| `Q1_Q5_Spread` | `q1_q5_spread` | |
| `Q1_Q5_Spread_VW` | `q1_q5_spread_vw` | |
| `Long_Short_Alpha` | `long_short_alpha` | future: 合併入 q1_q5_spread metadata |
| `Turnover` | `turnover` | |
| `Breakeven_Cost` | `breakeven_cost` | |
| `Net_Spread` | `net_spread` | |
| `Q1_Concentration` | `q1_concentration` | |
| `Multi_Horizon_IC` | `multi_horizon_ic` | |
| `Regime_IC` | `regime_ic` | |
| `Spanning_Alpha` | `spanning_alpha` | |

### Dashboard 影響

Dashboard 需要 display name mapping（snake_case → 人類可讀）：

```python
DISPLAY_NAMES = {
    "ic": "IC",
    "ic_ir": "IC IR",
    "q1_q5_spread": "Q1-Q5 Spread",
    ...
}
```

---

## 二、函式命名 — 按回傳型別分類

### 規則

| 回傳型別 | 命名 pattern | 說明 |
|---------|-------------|------|
| `pl.DataFrame` | `compute_*` | 產生中間資料，供下游消費 |
| `MetricOutput` | 指標名詞（snake_case，無前綴） | 最終指標，和 MetricOutput.name 對齊 |
| `float` / 內部工具 | `_private` | 不暴露給使用者 |
| 特殊 dataclass | 動作 + 名詞 | 描述做了什麼 |

### 改動對照表

#### `compute_*` → `DataFrame`（保持不變）

| 函式 | 回傳 | 狀態 |
|------|------|------|
| `compute_ic()` | `DataFrame(date, ic)` | OK |
| `compute_forward_return()` | `DataFrame` | OK |
| `compute_abnormal_return()` | `DataFrame` | OK |
| `compute_market_return()` | `DataFrame` | OK |
| `compute_profile()` | `FactorProfile`（特殊 dataclass） | 例外，保留 |

#### 指標名詞 → `MetricOutput`（需移除 `compute_` 前綴）

| 現在 | 改為 | MetricOutput.name |
|------|------|-------------------|
| `compute_monotonicity()` | `monotonicity()` | `monotonicity` |
| `compute_hit_rate()` | `hit_rate()` | `hit_rate` |
| `compute_turnover()` | `turnover()` | `turnover` |
| `ic()` | `ic()` | `ic` | OK |
| `ic_ir()` | `ic_ir()` | `ic_ir` | OK |
| `regime_ic()` | `regime_ic()` | `regime_ic` | OK |
| `multi_horizon_ic()` | `multi_horizon_ic()` | `multi_horizon_ic` | OK |
| `quantile_spread()` | `quantile_spread()` | `q1_q5_spread` | OK |
| `quantile_spread_vw()` | `quantile_spread_vw()` | `q1_q5_spread_vw` | OK |
| `long_short_alpha()` | `long_short_alpha()` | future: 合併入 quantile_spread |
| `q1_concentration()` | `q1_concentration()` | `q1_concentration` | OK |
| `breakeven_cost()` | `breakeven_cost()` | `breakeven_cost` | OK |
| `net_spread()` | `net_spread()` | `net_spread` | OK |
| `theil_sen_slope()` | `ic_trend()` | `ic_trend` | 函式名對齊 metric name |
| `spanning_test()` | `spanning_alpha()` | `spanning_alpha` | 函式名對齊 metric name |

#### `compute_*` → `DataFrame`（需加 `compute_` 前綴）

| 現在 | 改為 | 說明 |
|------|------|------|
| `quantile_spread_series()` | `compute_spread_series()` | 回傳 DataFrame，加 compute_ |
| `quantile_group_returns()` | `compute_group_returns()` | 回傳 DataFrame，加 compute_ |

#### 內部工具 → `_private`

| 現在 | 改為 | 說明 |
|------|------|------|
| `_non_overlapping_ic_tstat()` | `_non_overlapping_ic_tstat()` | 已是 private，OK |
| `sample_non_overlapping()` | `_sample_non_overlapping()` | helpers 不對外暴露 |
| `assign_quantile_groups()` | `_assign_quantile_groups()` | 同上 |
| `median_universe_size()` | `_median_universe_size()` | 同上 |
| `annualize_return()` | `_annualize_return()` | 同上 |

---

## 三、significance.py — 統一為內部工具

### 現狀

4 個函式風格各異：`calc_t_stat`, `t_stat_from_array`, `significance_marker`, `bhy_threshold`

### 改動

| 現在 | 改為 | 說明 |
|------|------|------|
| `calc_t_stat()` | `_calc_t_stat()` | 內部工具，只在其他 tools 裡呼叫 |
| `t_stat_from_array()` | `_t_stat_from_array()` | 同上 |
| `significance_marker()` | `_significance_marker()` | 同上（future: 改吃 p_value） |
| `bhy_threshold()` | `bhy_threshold()` | 保持 public，使用者可能直接呼叫做多重檢定 |

---

## 四、factors/ — generate_* pattern（已一致）

| 函式 | 狀態 |
|------|------|
| `generate_momentum()` | OK |
| `generate_momentum_60d()` | OK |
| `generate_volatility()` | OK |
| `generate_idiosyncratic_vol()` | OK |
| `generate_amihud()` | OK |
| `generate_mean_reversion()` | OK |
| `generate_52w_high_ratio()` | OK |
| `generate_overnight_return()` | OK |
| `generate_intraday_range()` | OK |
| `generate_rsi()` | OK |
| `generate_volume_price_trend()` | OK |
| `generate_market_beta()` | OK |
| `generate_price_intention()` | OK |
| `generate_max_effect()` | OK |
| `generate_size_factor()` | `generate_size()` — 去掉多餘的 `_factor` 後綴 |
| `generate_event_signal_mock()` | OK（mock 函式） |
| `encode_industry_dummies()` | 不是 generate_*，但語意不同（編碼，非生成因子），保留 |

---

## 五、charts/ — *_chart pattern（已一致）

| 函式 | 狀態 |
|------|------|
| `cumulative_ic_chart()` | OK |
| `rolling_ic_chart()` | OK |
| `ic_distribution_chart()` | OK |
| `quantile_return_chart()` | OK |
| `spread_time_series_chart()` | OK |
| `multi_horizon_ic_chart()` | OK |
| `regime_ic_chart()` | OK |
| `compare_line_chart()` | OK |
| `compare_bar_chart()` | OK |

---

## 六、gates/ — *_gate pattern（已一致）

| 函式 | 狀態 |
|------|------|
| `significance_gate()` | OK |
| `oos_persistence_gate()` | OK |
| `evaluate_factor()` | OK（orchestrator，不是 gate） |
| `build_artifacts()` | OK（factory） |
| `compute_profile()` | OK |

---

## 七、class / dataclass 命名（已一致，PascalCase）

| 類別 | 狀態 |
|------|------|
| `MetricOutput` | OK |
| `PipelineConfig` | OK |
| `GateResult` | OK |
| `Artifacts` | OK |
| `FactorProfile` | OK |
| `EvaluationResult` | OK |
| `FactorTracker` | OK |
| `OOSResult` | OK |
| `SplitDetail` | OK |
| `SpanningResult` | future: 考慮統一為 MetricOutput 或 CompositeMetricOutput |
| `ForwardSelectionResult` | 同上 |

---

## 遷移策略

1. **Phase 1（MetricOutput name）：** snake_case rename + dashboard display mapping
2. **Phase 2（函式名）：** rename + 舊名 alias（`compute_monotonicity = monotonicity`）維持向後相容一個版本
3. **Phase 3（internal tools）：** 加 `_` prefix，更新所有 import
4. **Phase 4：** 移除 alias

### 影響範圍

- `factrix/tools/**/*.py` — 所有函式 rename
- `factrix/tools/_typing.py` — MetricOutput name 常量或文件化約定
- `factrix/gates/` — profile, pipeline, significance 讀取 metric name 的地方
- `factrix/dashboard/` — display name mapping
- `tests/` — 所有 assert name == "..." 更新
- `experiments/` — notebook + .py 的 import 和顯示
