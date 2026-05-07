# event_signal 因子類型實作計畫

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> 狀態：已完成（2026-04-16）
> 日期：2026-04-16
> 前置：P0–P3.5 routing 重構 + macro_panel / macro_common 已完成
> 優先序：P4
> 設計依據：`refactor_factrix_routing.md` §2–§3（指標對照表 + 統計備註 §3.4–§3.6）

---

## 1. 定位

event_signal 處理**離散觸發、稀疏事件**的因子：營收驚喜、技術突破、
黃金交叉、法說會、VIX 閾值突破。

核心差異：因子值是離散 {-1, 0, +1}（不是連續 z-score），分析單位是
每個事件（不是每個日期的截面）。IC / Monotonicity / Quantile spread
在此不適用。核心指標是 **CAAR t-test + BMP test**（處理事件引發的波動膨脹）。

**邊界：** Per-event 統計量（CAAR、MFE/MAE、Profit Factor、Skewness）
屬於因子有效性分析，在 factrix 範圍。Portfolio-level 策略指標
（Sortino、NAV curve MDD、equity curve）需要假設持倉規則，屬於回測
引擎（Zipline / vectorbt）範圍。

---

## 2. 已就位的基礎設施

| 元件 | 位置 | 狀態 |
|------|------|------|
| `FactorType.EVENT_SIGNAL` | `_types.py` | ✅ |
| `EventConfig` | `config.py` | ✅（`event_window_pre/post`, `cluster_window`, `adjust_clustering`） |
| `evaluate()` match/case | `evaluation/pipeline.py` | ✅（catch-all → `NotImplementedError`） |
| `compute_profile()` match/case | `evaluation/profile.py` | ✅ |
| `check_caution()` match/case | `evaluation/_caution.py` | ✅ |
| `preprocess()` match/case | `preprocess/pipeline.py` | ✅ |
| `_DEFAULT_GATES` dict | `evaluation/presets.py` | ✅ |
| `Artifacts.intermediates` dict | `evaluation/_protocol.py` | ✅ |
| 既有可複用：`multi_split_oos_decay` | `metrics/oos.py` | ✅ |
| 既有可複用：`ic_trend` | `metrics/trend.py` | ✅ |
| 既有可複用：`_oos_decay_metric` / `_beta_trend_metric` | `evaluation/profile.py` | ✅ |
| 既有可複用：`oos_persistence_gate(value_key=)` | `evaluation/gates/oos_persistence.py` | ✅ |

---

## 3. 新增檔案

```
factrix/
├── metrics/
│   ├── caar.py              # CAAR, BMP test (Standardized AR)
│   ├── corrado.py           # Corrado rank test（非參數備援）
│   └── clustering.py        # 事件 clustering 診斷（Herfindahl index）
│   └── mfe_mae.py             # MFE, MAE, Bars_to_MFE, Profit Factor, Skewness
├── evaluation/
│   └── gates/
│       └── event_significance.py  # CAAR/BMP OR hit_rate gate
```

---

## 4. 指標設計

### 4.1 compute_caar（raw computation，對稱 `compute_ic` / `compute_fm_betas`）

```python
def compute_caar(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-event-date signed abnormal return → (date, caar) DataFrame.

    signed_car = forward_return × sign(factor)
    caar = per-date mean of signed_car across events
    """
```

### 4.2 caar（significance testing，對稱 `ic()` / `fama_macbeth()`）

```python
def caar(
    caar_df: pl.DataFrame,
    *,
    forward_periods: int = 5,
) -> MetricOutput:
    """CAAR t-test with non-overlapping sampling. H₀: mean(CAAR) = 0."""
```

### 4.3 bmp_test（BMP Standardized AR test）

```python
def bmp_test(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    estimation_window: int = 60,
) -> MetricOutput:
    """Boehmer-Musumeci-Poulsen test: standardize AR by per-asset
    residual volatility to handle event-induced variance inflation.

    Returns MetricOutput(name="bmp_sar", value=mean_SAR, stat=t_bmp, ...)
    """
```

### 4.4 event_hit_rate

```python
def event_hit_rate(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Fraction of events where signed_car > 0.

    Uses binomial score test: H₀: p = 0.5.
    """
```

### 4.5 clustering_diagnostic

```python
def clustering_diagnostic(
    df: pl.DataFrame,
    *,
    cluster_window: int = 3,
) -> MetricOutput:
    """Event clustering Herfindahl index on event dates.

    High HHI = events concentrate in few dates → independence
    assumption violated → CAAR t-stat may be inflated.
    """
```

### 4.6 corrado_rank_test

```python
def corrado_rank_test(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Non-parametric alternative to CAAR t-test.

    Robust to extreme returns and non-normal distributions.
    """
```

### 4.7 MFE/MAE 指標（`metrics/mfe_mae.py`）

per-event 路徑品質分析。需要事件窗口內逐 bar 價格（不只端點 forward_return）。

```python
def compute_mfe_mae(
    df: pl.DataFrame,
    *,
    window: int = 24,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    """Per-event MFE/MAE/Bars_to_MFE/Bars_to_MAE.

    Returns (date, asset_id, mfe, mae, bars_to_mfe, bars_to_mae).
    Requires bar-by-bar price data within the event window.
    """
```

```python
def mfe_mae_summary(mfe_mae_df: pl.DataFrame) -> MetricOutput:
    """Aggregate: MFE p50, MAE p75, MFE/MAE ratio, Bars_to_MFE mean.

    metadata: mfe_p50, mae_p75, mae_p95, mfe_mae_ratio,
              bars_to_mfe_mean, bars_to_mae_mean
    """
```

```python
def profit_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """sum(positive signed_car) / sum(negative signed_car).

    Per-event aggregate — no strategy assumptions.
    """
```

```python
def event_skewness(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Skewness of signed_car distribution.

    Positive skew = occasional large gains, frequent small losses.
    """
```

---

## 5. Profile 設計

flat `FactorProfile.metrics`，和 CS / macro_panel / macro_common 一致：

| 指標 | 來源 | 角色 |
|------|------|------|
| `caar` | `caar()` | 核心效果量 + 顯著性 |
| `bmp_sar` | `bmp_test()` | 波動膨脹修正後的顯著性 |
| `event_hit_rate` | `event_hit_rate()` | 方向正確率（per-event） |
| `oos_decay` | 既有 `_oos_decay_metric()` 作用在 CAAR 序列 | 樣本外穩定性 |
| `caar_trend` | 既有 `_beta_trend_metric()` 作用在 CAAR 序列（rename） | 衰減檢測 |
| `clustering_hhi` | `clustering_diagnostic()` | 獨立性診斷 |
| `mfe_mae` | `mfe_mae_summary()` | 路徑品質（MFE p50 / MAE p75 ratio） |
| `profit_factor` | `profit_factor()` | 盈虧比（per-event 聚合） |
| `event_skewness` | `event_skewness()` | signed_car 分佈偏態 |

**不計算：** IC、IC_IR、quantile_spread、monotonicity、q1_concentration、
turnover、breakeven_cost、net_spread。
**不計算（策略層）：** Sortino、portfolio-level MDD、equity curve。

**Standalone metrics（不在預設 profile，進階使用者按需呼叫）：**
- `corrado_rank_test` — 非參數備援
- `compute_mfe_mae` — per-event 明細 DataFrame

---

## 6. Artifacts 設計

`_build_event_artifacts()` 填充 intermediates：

```python
intermediates = {
    "caar_series": caar_df,                    # date, caar
    "caar_values": caar_df.rename({"caar": "value"}),  # for OOS/trend
    "mfe_mae": mfe_mae_df,                    # date, asset_id, mfe, mae, bars_to_mfe, bars_to_mae
}
```

`mfe_mae` 需要逐 bar 價格資料（`price` 欄位），在 `build_artifacts`
階段從 `prepared` 計算。若 `price` 不存在則跳過（profile 中 mfe_mae
相關指標回傳 None）。

和 CS 的 `ic_values`、macro_panel 的 `beta_values` 對稱——
`(date, value)` schema 確保 OOS gate / trend 工具直接複用。

---

## 7. Gate 設計

### 7.1 Event significance gate

```python
def event_significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """CAAR t-stat (or BMP SAR t-stat) OR event hit_rate t-stat >= threshold.

    Either path passing is sufficient — parallels CS (IC or spread)
    and macro_panel (FM β or Pooled β).
    """
```

### 7.2 OOS persistence gate（完全複用）

```python
# presets.py
EVENT_SIGNAL_GATES: list[GateFn] = [
    event_significance_gate,
    partial(oos_persistence_gate, value_key="caar_values"),
]
```

和 macro_panel / macro_common 完全相同的複用模式。

---

## 8. Preprocess 設計

放在 `preprocess/pipeline.py`（和其他型別一致）：

```python
def preprocess_event_signal(
    df: pl.DataFrame,
    *,
    config: EventConfig,
) -> pl.DataFrame:
    """事件訊號前處理。

    Steps:
        1. Forward return（複用 compute_forward_return）
        2. Return winsorize（複用 winsorize_forward_return）
        3. Abnormal return（複用 compute_abnormal_return，多資產時截面去均值）

    不做：MAD winsorize、z-score（因子值已是離散 {-1, 0, +1}）。
    保留 factor 原始值（不 rename 為 factor_raw + factor_zscore）。
    """
```

輸出欄位：`date, asset_id, factor, forward_return, abnormal_return`

---

## 9. Caution 規則

`_caution.py` 加入 `_event_signal_caution()`：

| 條件 | 原因 |
|------|------|
| 事件總數 < 30 | 統計力不足 |
| Gate 僅透過 hit_rate（非 CAAR）通過 | 方向正確但經濟上可能不顯著 |
| clustering HHI > 閾值 且未設 `adjust_clustering` | 事件集中破壞獨立性假設（僅 N>1） |
| CAAR 和 BMP 方向不一致 | CAAR 可能被波動膨脹偽陽性 |

---

## 9.5 單資產 vs 多資產行為（N-awareness）

所有四種因子類型共用的設計原則：**pipeline 不因 N 分支，但 profile
和 caution 感知 N，自動跳過無意義的指標。**

### 跨因子類型的 N-awareness 行為

| 類型 | N=1 時跳過 | N=1 時仍有意義 | 處理方式 |
|------|-----------|---------------|---------|
| cross_sectional | IC（單筆無 rank）、quantile_spread、monotonicity、q1_concentration | — (整體無意義) | `warn_small_n` warning |
| macro_panel | FM β（單筆截面回歸退化）、quantile_spread | — (整體無意義) | `min_cross_section` caution |
| macro_common | cross-asset aggregation（`ts_beta` 的 t-test 只有 1 個觀測） | 單資產 TS β + R²（該資產對共同因子的敏感度和穩定性） | profile 內條件處理 |
| event_signal | `clustering_diagnostic`（N=1 無截面聚集問題） | CAAR、hit_rate、MFE/MAE、profit_factor、skewness | profile 內條件處理 |

### event_signal 的 N-awareness 實作

```python
def _event_signal_profile(artifacts, config):
    n_assets = artifacts.prepared["asset_id"].n_unique()

    # 核心指標：所有 N 都計算
    metrics = [caar_metric, bmp_metric, hit_rate, oos, caar_trn,
               mfe_mae, profit_fac, skew]

    # N > 1 才有意義的指標
    if n_assets > 1:
        metrics.append(clustering_diagnostic(...))

    return FactorProfile(metrics=metrics)
```

### macro_common 的 N-awareness 實作

N=1 時 `ts_beta()` 的 cross-sectional t-test 只有 1 個觀測（無法算
std），應改為直接回報單資產的 TS β 和 per-asset t-stat：

```python
def _macro_common_profile(artifacts, config):
    ts_betas_df = artifacts.get("beta_series")
    n_assets = len(ts_betas_df)

    if n_assets == 1:
        # 直接回報該資產的 β 和 t-stat（不做 cross-asset aggregation）
        beta_metric = MetricOutput(
            name="ts_beta", value=ts_betas_df["beta"][0],
            stat=ts_betas_df["t_stat"][0], ...)
    else:
        beta_metric = ts_beta(ts_betas_df)  # cross-asset t-test
```

### 測試覆蓋

測試資料需包含 N=1 edge case：
- event_signal: 單一 BTCUSDT 的技術訊號
- macro_common: 單一 ETF 對 VIX 的 TS β

---

## 10. 實作步驟

| Step | 內容 | 依賴 |
|------|------|------|
| E1 | `metrics/caar.py`：`compute_caar()` + `caar()` + `bmp_test()` + `event_hit_rate()` | `_stats.py` |
| E2 | `metrics/mfe_mae.py`：`compute_mfe_mae()` + `mfe_mae_summary()` + `profit_factor()` + `event_skewness()` | — |
| E3 | `metrics/clustering.py`：`clustering_diagnostic()` | — |
| E4 | `metrics/corrado.py`：`corrado_rank_test()` | — |
| E5 | `preprocess/pipeline.py`：`preprocess_event_signal()` + match branch | 既有 `returns.py` |
| E6 | `evaluation/gates/event_significance.py` | E1 |
| E7 | `evaluation/pipeline.py`：`_build_event_artifacts()` + match branch | E1, E2, E5 |
| E8 | `evaluation/profile.py`：`_event_signal_profile()` + match branch | E1, E2, E3 |
| E9 | `evaluation/_caution.py`：`_event_signal_caution()` + match branch | E7, E8 |
| E10 | `evaluation/presets.py`：`EVENT_SIGNAL_GATES` + `_DEFAULT_GATES` 更新 | E6 |
| E11 | `metrics/__init__.py`：re-export event metrics | E1-E4 |
| E12 | `_api.py`：更新 `_PROFILE_METRICS` / `_STANDALONE_METRICS` | E8 |
| E13 | `validation.py`：`_SCHEMAS[EVENT_SIGNAL]` | — |
| E14 | Tests：`test_caar.py`、`test_mfe_mae.py`、`test_event_pipeline.py` | all |

**預估工作量：** 2–3 天。

---

## 11. 測試資料

合成事件訊號測試資料：

```python
def _make_event_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    signal_strength: float = 0.01,
    seed: int = 42,
) -> pl.DataFrame:
    """合成事件訊號資料。

    每天每資產有 event_prob 機率觸發事件（factor = +1 or -1）。
    事件後的 forward_return = signal_strength * sign(factor) + noise。
    """
```

Edge cases：
- 零事件（所有 factor = 0，gate 應 FAILED）
- 高 clustering（同一天 80% 資產觸發事件）
- 純噪音（signal_strength = 0，gate FAILED）
- 單資產 N=1（CAAR / hit_rate / MFE/MAE 正常，clustering_diagnostic 跳過）
- 多資產 N=50（完整 profile 含 clustering_diagnostic）

---

## 12. 驗證清單

1. `fl.evaluate(df, "EarningsSurprise", config=fl.EventConfig())` 回傳 EvaluationResult
2. `result.profile.get("caar")` 回傳 MetricOutput
3. `result.profile.get("bmp_sar")` 回傳 MetricOutput
4. `result.profile.get("event_hit_rate")` 回傳 MetricOutput
5. `result.profile.get("clustering_hhi")` 回傳 MetricOutput
6. `result.profile.get("mfe_mae")` 回傳 MetricOutput（含 mfe_p50, mae_p75, ratio）
7. `result.profile.get("profit_factor")` 回傳 MetricOutput
8. `result.profile.get("event_skewness")` 回傳 MetricOutput
9. IC / quantile_spread 等 CS 指標不出現在 event_signal profile 中
10. `fl.quick_check(df, "X", factor_type="event_signal")` 正常運作
11. OOS decay gate 作用在 CAAR 序列上
12. 高 clustering 時觸發 caution
13. `repr(result)` 正確顯示所有 event 指標
14. `from factrix.metrics import corrado_rank_test, compute_mfe_mae` 可獨立使用
15. `fl.describe_profile("event_signal")` 顯示正確指標列表

---

## 13. 外部依賴

| 依賴 | 用途 | 是否必須 |
|------|------|----------|
| `numpy` | OLS（BMP 估計期殘差）| 是（已有） |
| `scipy` | t-distribution、KS test | 是（已有） |

零新增 optional dep。

---

## 14. 設計決策記錄

### 14.1 MFE/MAE 和 Profit Factor 屬於因子分析

MFE/MAE 是 per-event 的路徑品質統計量，不需要假設持倉權重、再平衡、
成本模型。它回答「因子預測的 alpha 的路徑長什麼樣」——和 CAAR（平均
效果量）互補。Profit Factor 同理：sum(gains)/sum(losses) 是 per-event
聚合，不涉及策略邏輯。

### 14.2 Sortino / portfolio-level MDD 不屬於因子分析

Sortino 需要下行波動率定義（哪個 MAR？）、MDD 需要 NAV 曲線（假設
持倉規則）。這些是策略層指標，由回測框架處理。
factrix 的 per-event MAE p95 已提供尾部風險資訊，不需要 portfolio
層面的 drawdown。

### 14.3 為何不用 KS test

設計文件 §3.4 明確記載：「KS test 測的是整體分布是否相等，對事件
異常報酬的檢驗不對題。」CAAR t-test + BMP test + Corrado rank test
覆蓋了參數和非參數的顯著性需求。

### 14.4 CAAR 序列和 OOS/trend 的複用

CAAR 序列 rename 為 `(date, value)` 後，直接複用既有的
`multi_split_oos_decay` 和 `ic_trend`。和 CS（IC values）、
macro_panel（FM β values）、macro_common（rolling TS β）完全對稱。

---

## 15. 參考文獻

- MacKinlay (1997), "Event Studies in Economics and Finance"
- Boehmer, Musumeci & Poulsen (1991), "Event-study methodology under
  conditions of event-induced variance" — BMP test
- Corrado (1989), "A nonparametric test for abnormal security-price
  performance in event studies"
- Harvey, Liu & Zhu (2016), t-stat threshold — Gate 1
- McLean & Pontiff (2016), OOS decay — Gate 2
