# macro_panel 因子類型實作計畫

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> 狀態：已實作
> 日期：2026-04-16
> 前置：P0–P3.5 routing 重構已完成（`refactor_factrix_routing.md` v8）
> 優先序：Future A（已完成）
> 設計依據：`refactor_factrix_routing.md` §1–§3（指標對照表 + 統計備註）

---

## 1. 定位

macro_panel 處理**小截面（N=10–30）連續值因子**：各國 CPI、利差、相對
value、行業 ETF 輪動、大宗商品橫截面。

核心差異：IC-based metrics 在此 N 下 SE 過大（0.19–0.33），不回傳。
Reliability 改以 **Fama-MacBeth β + Pooled OLS** 雙估計量為主。

---

## 2. 已就位的基礎設施

routing 重構已提供：

| 元件 | 位置 | 狀態 |
|------|------|------|
| `FactorType.MACRO_PANEL` | `_types.py` | ✅ |
| `MacroPanelConfig` | `config.py` | ✅（`demean_cross_section`, `min_cross_section`） |
| `evaluate()` match/case | `evaluation/pipeline.py` | ✅（catch-all → `NotImplementedError`） |
| `build_artifacts()` match/case | 同上 | ✅ |
| `compute_profile()` match/case | `evaluation/profile.py` | ✅ |
| `default_gates_for()` | `evaluation/presets.py` | ✅（`_DEFAULT_GATES` dict） |
| `check_caution()` match/case | `evaluation/_caution.py` | ✅ |
| `preprocess()` match/case | `preprocess/pipeline.py` | ✅ |
| `Artifacts.intermediates` dict | `evaluation/_protocol.py` | ✅ |
| `FactorProfile.metrics` flat list | 同上 | ✅ |

**需要新增的：** metrics、gates、profile、preprocess、artifacts build。

---

## 3. 新增檔案

```
factrix/
├── metrics/
│   └── fama_macbeth.py        # FM β + Pooled OLS + Newey-West SE
├── evaluation/
│   └── gates/
│       └── fm_significance.py # FM β 顯著性 gate
```

不新增 `preprocess/macro.py`——`preprocess/` 按步驟分模組（returns、
normalize、orthogonalize），macro_panel 只是跳過部分步驟。orchestrator
`preprocess_macro_panel()` 放在既有的 `preprocess/pipeline.py` 中，
複用 `compute_forward_return` + `cross_sectional_zscore`。

---

## 4. 指標設計（`metrics/fama_macbeth.py`）

### 4.1 compute_fm_betas（raw computation，對稱 `compute_ic`）

```python
def compute_fm_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-date cross-sectional OLS，回傳 (date, beta) DataFrame。

    用於 build_artifacts，和 compute_ic() 同層級。
    """
```

### 4.2 fama_macbeth（significance testing，對稱 `ic()`）

```python
def fama_macbeth(
    beta_df: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
) -> MetricOutput:
    """對 FM β 序列做 Newey-West t 檢定。

    Returns:
        MetricOutput(name="fm_beta", value=mean_β, stat=t_nw, ...)
    """
```

**Newey-West lag 選擇：** 預設 `floor(T^(1/3))`，使用者可覆寫。
Newey-West SE 手寫在 `_stats.py`（Bartlett kernel），維持零 optional dep。

### 4.2 Pooled OLS + clustered SE

```python
def pooled_ols(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    cluster_col: str = "date",
) -> MetricOutput:
    """Pooled OLS with clustered standard errors.

    R_{i,t+1} = α + β · Signal_{i,t} + ε_{i,t}
    SE clustered by date（處理截面相關性）。

    Returns:
        MetricOutput(name="pooled_beta", value=β, stat=t_clustered, ...)
    """
```

### 4.3 β sign consistency

```python
def beta_sign_consistency(
    beta_series: pl.DataFrame,
    *,
    expected_sign: int = 1,
) -> MetricOutput:
    """FM β > 0（或 < 0）的期數佔比。

    macro_panel 版本的 hit_rate。
    """
```

### 4.4 Long-short 1/3 portfolio

```python
def long_short_tercile(
    df: pl.DataFrame,
    *,
    forward_periods: int = 1,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Top 1/3 - Bottom 1/3 portfolio return.

    N=10 時分 3 組（各 3-4 檔），比 quantile spread 更穩定。

    metadata:
        long_return, short_return, portfolio_sharpe, portfolio_sortino,
        max_drawdown, return_skewness, annualized_return, annualized_vol
    """
```

### 4.5 FM spanning alpha

```python
def fm_spanning_alpha(
    df: pl.DataFrame,
    base_factors: dict[str, pl.DataFrame],
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Fama-MacBeth spanning: 控制 global macro factors 後的 alpha。

    R_{i,t+1} = α_t + β_1t · Signal + β_2t · Base1 + ... + ε
    回報 α 的 Newey-West t 統計量。
    """
```

---

## 5. Profile 設計

macro_panel 的 `_macro_panel_profile()` 回傳 flat `FactorProfile.metrics`：

| 指標 | 來源 | 角色 |
|------|------|------|
| `fm_beta` | `fama_macbeth()` | 訊號可靠性（主要） |
| `pooled_beta` | `pooled_ols()` | 訊號可靠性（robustness baseline） |
| `beta_sign_consistency` | `beta_sign_consistency()` | 訊號穩定性 |
| `oos_decay` | 既有 `multi_split_oos_decay()` 作用在 β 序列 | 樣本外穩定性 |
| `beta_trend` | 既有 `ic_trend()` 作用在 β 序列 | 衰減檢測 |
| `long_short_tercile` | `long_short_tercile()` | 經濟顯著性 |
| `portfolio_sharpe` / `portfolio_sortino` | 從 tercile metadata 取 | 經濟顯著性與下行風險 |
| `max_drawdown` | 從 tercile metadata 取 | 尾部風險管理 (Macro 核心) |
| `return_skewness` | 從 tercile metadata 取 | 觀察尾部回報非對稱性 |
| `turnover` | 既有 `turnover()` | 可交易性（參考） |

所有指標放在同一個 `metrics` list 中，不分 reliability/profitability
子類別（v8 FactorProfile 為 flat 設計）。

**不計算：** IC、IC_IR、hit_rate(IC>0)、quantile_spread(5-10 組)、
monotonicity、q1_concentration。

**Robustness 判讀規則：** FM β 和 Pooled β 方向一致且都顯著才算可靠。
Profile 的 `get("fm_beta")` 和 `get("pooled_beta")` 讓使用者自行比對。

---

## 6. Artifacts 設計

`_build_macro_panel_artifacts()` 填充 intermediates：

```python
intermediates = {
    "beta_series": fm_result.metadata["beta_series"],  # date, beta
    "beta_values": beta_series.rename({"beta": "value"}),  # for OOS/trend
    "tercile_series": tercile_spread_series,  # date, spread, long, short
}
```

既有的 `oos_persistence_gate` 和 `trend` 工具可直接作用在 `beta_values`
（和 CS 用 `ic_values` 同樣的 date+value schema）。

---

## 7. Gate 設計

### 7.1 FM significance gate

```python
def fm_significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """FM β t-stat OR Pooled β t-stat >= threshold.

    兩個估計量任一通過即 PASS（類似 CS 的 IC or spread 邏輯）。
    """
```

### 7.2 OOS persistence gate（複用）

既有 `oos_persistence_gate` 直接作用在 `artifacts.get("beta_values")`。
Gate 函式本身不需改動——它讀 `"ic_values"` 或 `"beta_values"` 都是
`(date, value)` schema。

**需要調整：** gate 目前 hardcode `artifacts.get("ic_values")`。
改為讓 `_DEFAULT_GATES` 使用 `functools.partial` 綁定 key：

```python
_DEFAULT_GATES[FactorType.MACRO_PANEL] = [
    fm_significance_gate,
    partial(oos_persistence_gate_generic, value_key="beta_values"),
]
```

或在 `oos_persistence_gate` 加 `value_key` 參數。

### 7.3 Presets

```python
MACRO_PANEL_GATES: list[GateFn] = [
    fm_significance_gate,
    oos_persistence_gate,  # 作用在 beta_values
]

_DEFAULT_GATES[FactorType.MACRO_PANEL] = MACRO_PANEL_GATES
```

---

## 8. Preprocess 設計

macro_panel 的前處理放在 `preprocess/pipeline.py`（和 CS 同一個檔案），
複用既有步驟模組，跳過小 N 不適用的步驟：

```python
# preprocess/pipeline.py 中新增
def preprocess_macro_panel(
    df: pl.DataFrame,
    *,
    config: MacroPanelConfig,
) -> pl.DataFrame:
    """macro_panel 前處理。

    Steps:
        1. Forward return（複用 returns.compute_forward_return）
        2. Return winsorize（複用 returns.winsorize_forward_return）
        3. Optional: cross-section demean signal（config.demean_cross_section）
        4. Factor z-score（複用 normalize.cross_sectional_zscore）

    不做：MAD winsorize（小 N 下 MAD 不穩定）、abnormal return。
    """
```

`preprocess()` dispatcher 的 match/case 加入：

```python
case MacroPanelConfig():
    return preprocess_macro_panel(df, config=config)
```

---

## 9. Caution 規則

`_caution.py` 加入 `_macro_panel_caution()`：

| 條件 | 原因 |
|------|------|
| N < `config.min_cross_section` | 截面太小，FM β noise 極大 |
| FM β 和 Pooled β 方向不一致 | Robustness 不足 |
| `demean_cross_section=True` 且做了 spanning test | 機械負相關警告（§3.7） |

### N-awareness

所有因子類型共用設計原則：pipeline 不因 N 分支，但 profile 和 caution
感知 N。macro_panel 在 N=1 時 FM β 截面回歸退化（只有一個觀測點），
整體無意義但不 crash —— `min_cross_section` caution 會觸發。
cross_sectional 同理（warn_small_n warning）。

---

## 10. 實作步驟

| Step | 內容 | 依賴 |
|------|------|------|
| A1 | `metrics/fama_macbeth.py`：`fama_macbeth()` + `pooled_ols()` + `beta_sign_consistency()` | `_ols.py`, `_stats.py` |
| A2 | `metrics/fama_macbeth.py`：`long_short_tercile()` + `fm_spanning_alpha()` | A1 |
| A3 | `preprocess/pipeline.py`：新增 `preprocess_macro_panel()` orchestrator | 既有 `returns.py`, `normalize.py` |
| A4 | `evaluation/gates/fm_significance.py` | A1 |
| A5 | `evaluation/pipeline.py`：`_build_macro_panel_artifacts()` + match branch | A1, A3 |
| A6 | `evaluation/profile.py`：`_macro_panel_profile()` + match branch | A1, A2 |
| A7 | `evaluation/_caution.py`：`_macro_panel_caution()` + match branch | A5, A6 |
| A8 | `evaluation/presets.py`：`MACRO_PANEL_GATES` + `_DEFAULT_GATES` 更新 | A4 |
| A9 | `metrics/__init__.py`：re-export FM metrics | A1, A2 |
| A10 | Tests：`test_fama_macbeth.py`、`test_macro_panel_pipeline.py` | all |

**預估工作量：** 2–3 天。

---

## 11. 測試資料

合成 N=15 國家 × T=120 月的 panel：

```python
def _make_macro_panel(n_countries=15, n_months=120, signal_strength=0.3):
    """合成 macro panel 測試資料。

    signal = 標準化 CPI 差異
    return = signal_strength * signal + noise
    """
```

額外 edge case：
- N=5（極小截面，應觸發 caution）
- 純噪音（FM β 不顯著，gate FAILED）
- FM/Pooled 方向不一致（caution）

---

## 12. 驗證清單

1. `fl.evaluate(df, "CPI_spread", config=fl.MacroPanelConfig())` 回傳 EvaluationResult
2. `result.profile.get("fm_beta")` 回傳 MetricOutput
3. `result.profile.get("pooled_beta")` 回傳 MetricOutput
4. FM 和 Pooled 方向一致時無額外 caution
5. FM 和 Pooled 方向不一致時觸發 caution
6. N < `min_cross_section` 觸發 caution
7. `fl.quick_check(df, "X", factor_type="macro_panel")` 正常運作
8. `fl.batch_evaluate(factors, factor_type="macro_panel")` 正常運作
9. `fl.compare(results)` 可混合 CS 和 macro_panel 結果（缺失欄位填 None）
10. OOS decay gate 作用在 β 序列上
11. IC 類指標不出現在 macro_panel profile 中
12. `repr(result)` 正確顯示 FM/Pooled β

---

## 13. 外部依賴

| 依賴 | 用途 | 是否必須 |
|------|------|----------|
| `numpy` | OLS、Newey-West | 是（已有） |
| `scipy` | t-distribution | 是（已有） |
| `statsmodels` | Newey-West SE（`sm.OLS` + `cov_type='HAC'`）| 建議但非必須 |

**不使用 statsmodels 的替代方案：** 手寫 Newey-West。公式簡單
（Bartlett kernel），但 edge case 多（自由度修正、singular XTX）。
建議用 `statsmodels`——它已是 quant Python 生態系的標準件。

若堅持零 optional dep，將 Newey-West 實作放在 `_ols.py` 中。

---

## 14. 參考文獻

- Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests"
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
- Petersen (2009), "Estimating Standard Errors in Finance Panel Data Sets"
- Newey & West (1987), "A Simple, Positive Semi-definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix"
