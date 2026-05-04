# 實作計畫：截面個股因子評估框架 v3

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> **對應規格文件：** [individual_stock_scoring_v3.md](individual_stock_scoring_v3.md)
> **架構策略：** 有策略的模組化重構，非推翻重來。保留現有基礎設施（Polars、MLflow、Streamlit、factor generators），重建評分層為解耦工具 + Gate Pipeline。
> **核心原則：** 每個工具獨立可用，Pipeline 只是組合器，Dashboard 只是呈現層。

---

## 1. 架構設計

### 1.1 現有架構的問題

| 模組 | 問題 |
|------|------|
| `scoring/scorer.py` | `FactorScorer` 整個 class 是 composite-score 導向（4 維度加權、sigmoid、`map_linear`），v3 不需要 |
| `scoring/selection.py` | IC_IR、Monotonicity 等計算邏輯可復用，但被 `@register` + `map_linear` 耦合到評分框架 |
| `scoring/config.py` | dimension routing 權重、VETO penalty — v3 不需要 |
| `engine.py` | `prepare_factor_data()` 把 Step 1-5 打包，無法單獨使用某步驟 |
| `builders.py` | build_nav_artifact 綁定 portfolio NAV 追蹤 — v3 不追蹤 NAV |

### 1.2 目標架構

```
factorlib/
├── tools/                          ← Phase 1：獨立工具層
│   ├── _typing.py                  # MetricOutput 統一輸出型別
│   ├── series/                     # 輸入：time-indexed 數值序列
│   │   ├── oos.py                  # IS/OOS multi-split decay, sign flip detection
│   │   ├── trend.py                # Theil-Sen IC trend
│   │   ├── significance.py         # t-stat, BHY 校正, ★●○ markers
│   │   └── hit_rate.py             # proportion > 0 (or custom condition)
│   ├── panel/                      # 輸入：date × asset_id 截面 panel
│   │   ├── ic.py                   # Spearman rank IC (rolling / non-overlapping / multi-horizon)
│   │   ├── quantile.py             # 分組 spread, Long/Short alpha, VW spread
│   │   ├── monotonicity.py         # decile monotonicity test
│   │   └── concentration.py        # Q1 HHI (Q1_Concentration)
│   ├── event/                      # 輸入：event-indexed 事件資料
│   │   ├── caar.py                 # Cumulative Average Abnormal Return
│   │   └── event_stats.py          # Event hit rate, skewness, dispersion
│   ├── regression/                 # 輸入：factor panel + base factor panel
│   │   ├── orthogonalize.py        # 因子正交化（Step 6）
│   │   └── spanning.py             # Spanning regression (Stage 2)
│   └── cost/                       # 輸入：spread 序列 + turnover 估計
│       └── tradability.py          # Breakeven cost, Net spread, Turnover
│
├── preprocess/                  ← Phase 1：重構自 engine.py
│   ├── returns.py                  # Step 1-3 (forward return, winsorize, abnormal return)
│   ├── normalize.py                # Step 4-5 (MAD winsorize, z-score)
│   └── pipeline.py                 # run_all_steps() orchestration
│
├── gates/                          ← Phase 2：Gate Pipeline
│   ├── _protocol.py                # GateFn type alias, GateResult, Artifacts dataclass
│   ├── significance.py             # significance_gate（函式）
│   ├── oos_persistence.py          # oos_persistence_gate（函式）
│   ├── profile.py                  # compute_profile
│   ├── pipeline.py                 # evaluate_factor（接受 list[GateFn]）
│   ├── presets.py                  # CROSS_SECTIONAL_GATES, EVENT_SIGNAL_GATES
│   └── config.py                   # PipelineConfig + MARKET_DEFAULTS
│
├── dashboard/                      ← Phase 3：Dashboard 重構
│   ├── app.py                      # Gate-based leaderboard + detail
│   ├── charts.py                   # IC 累積圖, quantile return, multi-horizon
│   └── data.py                     # MLflow 資料
│
├── engine.py                       # 被 preprocess/ 取代
├── validation.py                   # 保留，schema 微調
├── experiment.py                   # 保留，新增 Gate Status logging
├── builders.py                     # 保留，IC artifact 邏輯遷移至 tools/
├── factors/                        # 完全保留不動
├── scoring/                        # 標記 deprecated，Phase 2 後移除
└── reporting.py                    # Phase 3 更新
```

### 1.3 工具按「資料形狀」分類的設計原則

工具不知道「因子類別」（cross-sectional / event / macro），只知道自己需要什麼形狀的資料。Pipeline 層負責組裝。

```
series/ 工具  →  吃任何 date + value 序列（IC 序列、CAAR 序列、spread 序列）
panel/ 工具   →  吃 date × asset_id × factor × return 截面資料
event/ 工具   →  吃 event-indexed 事件資料
regression/   →  吃 factor panel + base factor panel
```

好處：
- `series/oos.py` 既可分析 IC 的 OOS decay（截面因子），也可分析 CAAR 的 OOS decay（事件訊號）— 零程式碼修改
- 未來新增 event_signal / macro_signal pipeline 時，series/ 下的工具全部直接復用
- 多標的事件訊號可同時使用 event/ 和 panel/ 兩套工具（資料形狀同時滿足兩者）

### 1.4 統一輸出型別

Input 因資料形狀而異，但 Output 統一：

```python
# tools/_typing.py

@dataclass
class MetricOutput:
    name: str                          # "IC_IR", "CAAR", "OOS_Decay" ...
    value: float                       # 原始值（不壓縮為分數）
    t_stat: float | None = None
    significance: str | None = None    # "★" / "●" / "○"
    metadata: dict = field(default_factory=dict)
    # metadata 範例：
    #   {"via": "IC_IR", "per_split": [0.82, 0.75, 0.68]}
    #   {"long_alpha": 0.12, "short_alpha": 0.03}
```

所有工具回傳 `MetricOutput`，Pipeline 和 Dashboard 用統一介面消費。

---

## 2. Phase 1：獨立工具層

> **目標：** 每個模組可 `from factorlib.tools.xxx import ...` 獨立使用，附帶可獨立跑的 example/test。
> **改動策略：** 從現有程式碼抽取可復用邏輯 + 新建缺少的工具。

### 2.1 Preprocessing 拆分

從 `engine.py` 的 `prepare_factor_data()` 拆出：

| 模組 | 來源 | 包含的函式 |
|------|------|-----------|
| `preprocess/returns.py` | `engine.py` | `compute_forward_return()` (Step 1), `winsorize_forward_return()` (Step 2), `compute_abnormal_return()` (Step 3) |
| `preprocess/normalize.py` | `engine.py` | `mad_winsorize()` (Step 4), `cross_sectional_zscore()` (Step 5) |
| `preprocess/pipeline.py` | 新建 | `run_preprocessing()` — 串接 Step 1-5，呼叫上面兩個模組 |

新架構不做 backward compat wrapper，由 `preprocess/` 直接取代 `engine.py`。

### 2.2 Series 工具

| 模組 | 來源 | 關鍵函式 | v3 對應 |
|------|------|---------|---------|
| `series/significance.py` | 從 `_utils.py` 抽取 `calc_t_stat`；新增 BHY、★●○ | `calc_t_stat()`, `significance_marker()`, `bhy_threshold()` (P1) | Gate 1 門檻、Profile 顯著性標記 |
| `series/oos.py` | 從 `selection.py` `calc_oos_decay` 重構 | `multi_split_oos_decay()`, `detect_sign_flip()` | Gate 2 |
| `series/trend.py` | 全新 | `theil_sen_slope()` | Profile: IC_Trend (P1) |
| `series/hit_rate.py` | 從 `timing.py` 抽取 | `compute_hit_rate()` | Profile: Hit_Rate |

**`series/oos.py` 設計（v3 核心：multi-split）：**

```python
@dataclass
class OOSResult:
    decay_ratio: float               # median of per-split ratios
    sign_flipped: bool               # 任一 split sign flip → True
    per_split: list[SplitDetail]     # 每個 split 的 detail
    status: str                      # "PASS" / "VETOED"

def multi_split_oos_decay(
    series: pl.DataFrame,            # date + value（IC 序列或 CAAR 序列皆可）
    splits: list[tuple[float, float]] = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)],
    decay_threshold: float = 0.5,
) -> OOSResult: ...
```

### 2.3 Panel 工具

| 模組 | 來源 | 關鍵函式 | v3 對應 |
|------|------|---------|---------|
| `panel/ic.py` | 從 `_utils.py` 抽取 `_ic_series`, `_non_overlapping_ic_tstat`；新增 multi-horizon | `compute_ic()`, `non_overlapping_ic_tstat()`, `multi_horizon_ic()` (P1) | Gate 1、Profile: IC_IR, Multi-horizon IC |
| `panel/quantile.py` | 部分從 `selection.py` `calc_long_alpha`；大幅擴充 | `quantile_spread()`, `long_short_alpha()`, `quantile_spread_vw()` (P1) | Profile: Q1-Q5 Spread, Long/Short Alpha, VW Spread |
| `panel/monotonicity.py` | 從 `selection.py` `calc_monotonicity` 重構 | `compute_monotonicity()` | Profile: Monotonicity (n_groups=10) |
| `panel/concentration.py` | 全新 | `q1_concentration()` | Profile: Q1_Concentration |

**`panel/ic.py` 設計：**

```python
def compute_ic(
    df: pl.DataFrame,                 # date, asset_id, factor, forward_return
    method: str = "spearman",
) -> pl.DataFrame:                    # date + ic（time-indexed 序列）
    """截面 Rank IC。回傳值可直接餵入 series/ 工具。"""
    ...

def non_overlapping_ic_tstat(
    ic_series: pl.DataFrame,          # compute_ic() 的輸出
    forward_periods: int = 5,
) -> float: ...

def multi_horizon_ic(
    df: pl.DataFrame,                 # date, asset_id, factor, close
    periods: list[int] = [1, 5, 10, 20],
) -> dict[int, float]:               # {period: mean_ic}
    ...
```

### 2.4 Regression 工具

| 模組 | 來源 | 關鍵函式 | v3 對應 |
|------|------|---------|---------|
| `regression/orthogonalize.py` | 全新 | `orthogonalize_factor()` | Step 6 (P0) |
| `regression/spanning.py` | 全新 | `greedy_forward_selection()` | Stage 2 (P2) |

**`regression/orthogonalize.py` 設計：**

```python
def orthogonalize_factor(
    factor_df: pl.DataFrame,          # date, asset_id, factor（已 z-score）
    base_factors: pl.DataFrame,       # date, asset_id, size, value, momentum, industry_*
) -> pl.DataFrame:                    # date, asset_id, factor（殘差取代原值）
    """
    逐日截面 OLS：factor = β₁·Size + β₂·Value + ... + Σβ_k·Industry_k + ε
    回傳 ε 取代 factor 欄位。
    
    獨立使用場景：任何需要「去除已知因子曝露後看殘差」的分析。
    """
    ...
```

### 2.5 Event 工具

| 模組 | 來源 | 關鍵函式 | v3 對應 |
|------|------|---------|---------|
| `event/caar.py` | 從 `timing.py` 重構 | `compute_caar()` | 事件訊號評估 (P3) |
| `event/event_stats.py` | 從 `timing.py` 重構 | `event_hit_rate()`, `event_skewness()`, `event_dispersion()` | 事件訊號 Profile (P3) |

事件工具歸 P3，但架構在 Phase 1 就預留目錄結構。

### 2.6 Cost 工具

| 模組 | 來源 | 關鍵函式 | v3 對應 |
|------|------|---------|---------|
| `cost/tradability.py` | 從 `timing.py` `calc_turnover` 重構 + 新建 | `compute_turnover()`, `breakeven_cost()`, `net_spread()` | Profile: Turnover, Breakeven Cost, Net_Spread |

### 2.7 共用工具

`tools/_helpers.py` 提供跨模組共用的內部工具（非 public API）：

| 函式 | 用途 | 使用者 |
|------|------|--------|
| `sample_non_overlapping` | 非重疊日期取樣 | ic, quantile, monotonicity, concentration, hit_rate |
| `assign_quantile_groups` | 截面分位數分組 | quantile, monotonicity |
| `annualize_return` | 複合年化報酬 | quantile (spread + long/short alpha) |

`tools/_typing.py` 提供統一型別和常數：
- `MetricOutput` — 所有工具的統一輸出
- `EPSILON`, `DDOF`, `MAD_CONSISTENCY_CONSTANT` — 數值常數
- `MIN_ASSETS_PER_DATE_IC`, `MIN_OOS_PERIODS`, `MIN_PORTFOLIO_PERIODS`, `MIN_MONOTONICITY_PERIODS` — 最小樣本門檻

### 2.8 實作順序

Phase 1 內部的順序（有依賴關係）：

```
Wave 1（無依賴，可平行）：
├── _typing.py（MetricOutput + 常數）
├── preprocess/returns.py（Step 1-3 搬移）
├── preprocess/normalize.py（Step 4-5 搬移）
└── series/significance.py（calc_t_stat + markers）

Wave 2（依賴 Wave 1 的型別定義）：
├── panel/ic.py（依賴 significance.py）
├── panel/quantile.py
├── panel/monotonicity.py
├── panel/concentration.py
├── series/oos.py（依賴 significance.py）
├── series/hit_rate.py
├── series/trend.py
└── cost/tradability.py

Wave 3：
├── regression/orthogonalize.py（Step 6）
└── preprocess/pipeline.py（組合 returns + normalize）
```

---

## 3. Phase 2：Gate Pipeline

> **目標：** 組裝 Phase 1 工具為可自由組裝的 Gate Pipeline。
> **設計原則：** Gate 是函式、Artifacts 是 typed dataclass、Pipeline 是 runner。
> **前置條件：** Phase 1 Wave 1-2 完成。

### 3.1 核心型別

```python
# gates/_protocol.py
from typing import Literal

GateStatus = Literal["PASS", "FAILED", "VETOED"]
GateFn = Callable[[Artifacts], GateResult]

@dataclass
class GateResult:
    name: str
    passed: bool
    status: GateStatus
    detail: dict[str, Any] = field(default_factory=dict)

@dataclass
class Artifacts:
    """Pipeline 預算的中間產物。Gate 只讀，不寫。

    所有 artifacts 在 Gate 執行前一次算好，避免：
    - Gate 之間透過 mutable dict 產生隱式依賴
    - 重複計算（IC series 只算一次）
    - 調換 Gate 順序導致 KeyError
    """
    prepared: pl.DataFrame
    ic_series: pl.DataFrame            # compute_ic() 的輸出
    spread_series: pl.DataFrame        # quantile_spread_series() 的輸出
    config: PipelineConfig

@dataclass
class FactorProfile:
    reliability: list[MetricOutput]
    profitability: list[MetricOutput]

@dataclass
class EvaluationResult:
    factor_name: str
    status: str                        # "PASS" / "CAUTION" / "VETOED" / "FAILED"
    gate_results: list[GateResult]
    profile: FactorProfile | None
    caution_reasons: list[str] = field(default_factory=list)
```

### 3.2 Gate 作為函式

每道 Gate 是一個簽名為 `(Artifacts) -> GateResult` 的普通函式。
使用 `functools.partial` 綁定參數即可自訂門檻，不需要寫 class。

```python
# gates/significance.py
def significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """Gate 1: IC_IR t-stat OR Q1-Q5 spread t-stat ≥ threshold."""
    ic_tstat = ic.non_overlapping_ic_tstat(
        artifacts.ic_series, artifacts.config.forward_periods
    )
    spread_arr = artifacts.spread_series["spread"].drop_nulls().to_numpy()
    spread_tstat = significance.t_stat_from_array(spread_arr)

    via = []
    if abs(ic_tstat) >= threshold:
        via.append("IC_IR")
    if abs(spread_tstat) >= threshold:
        via.append("Q1-Q5_spread")

    passed = len(via) > 0
    return GateResult(
        name="significance",
        passed=passed,
        status="PASS" if passed else "FAILED",
        detail={"ic_tstat": ic_tstat, "spread_tstat": spread_tstat, "via": via},
    )


# gates/oos_persistence.py
def oos_persistence_gate(
    artifacts: Artifacts,
    *,
    decay_threshold: float = 0.5,
) -> GateResult:
    """Gate 2: multi-split OOS decay ≥ threshold, no sign flip."""
    ic_values = artifacts.ic_series.rename({"ic": "value"})
    oos_result = oos.multi_split_oos_decay(ic_values, decay_threshold=decay_threshold)
    return GateResult(
        name="oos_persistence",
        passed=oos_result.status == "PASS",
        status=oos_result.status,
        detail={"decay_ratio": oos_result.decay_ratio, "sign_flipped": oos_result.sign_flipped},
    )
```

### 3.3 Pipeline 作為 Runner

```python
# gates/pipeline.py
def evaluate_factor(
    df: pl.DataFrame,
    factor_name: str,
    gates: list[GateFn],               # ← 使用者傳入，非 hardcoded
    config: PipelineConfig,
) -> EvaluationResult:
    # 1. 預算所有中間產物（一次，共享，Gate 只讀）
    artifacts = _build_artifacts(df, config)

    # 2. 依序跑 Gate（短路：FAILED/VETOED 後停止）
    gate_results: list[GateResult] = []
    for gate_fn in gates:
        result = gate_fn(artifacts)
        gate_results.append(result)
        if not result.passed:
            return EvaluationResult(
                factor_name=factor_name,
                status=result.status,
                gate_results=gate_results,
                profile=None,
            )

    # 3. 全部通過 → Profile + CAUTION 檢查
    profile = compute_profile(artifacts)
    caution_reasons = _check_caution(artifacts, gate_results)
    return EvaluationResult(
        factor_name=factor_name,
        status="CAUTION" if caution_reasons else "PASS",
        gate_results=gate_results,
        profile=profile,
        caution_reasons=caution_reasons,
    )


def _build_artifacts(df: pl.DataFrame, config: PipelineConfig) -> Artifacts:
    """預算常用中間產物。所有 Gate 和 Profile 共享，避免重複計算。"""
    ic_series = ic.compute_ic(df)
    spread_series = quantile.quantile_spread_series(
        df, config.forward_periods, config.n_groups,
    )
    return Artifacts(
        prepared=df,
        ic_series=ic_series,
        spread_series=spread_series,
        config=config,
    )
```

### 3.4 預設 Gate 組合 + 使用者自訂

```python
# gates/presets.py
CROSS_SECTIONAL_GATES: list[GateFn] = [significance_gate, oos_persistence_gate]
EVENT_SIGNAL_GATES: list[GateFn] = [caar_significance_gate, event_oos_gate]  # P3
```

```python
# 使用者用法

from functools import partial
from factorlib.gates.presets import CROSS_SECTIONAL_GATES
from factorlib.gates.significance import significance_gate
from factorlib.gates.oos_persistence import oos_persistence_gate
from factorlib.gates.pipeline import evaluate_factor

# 標準用法
result = evaluate_factor(df, "Mom_20D", gates=CROSS_SECTIONAL_GATES, config=config)

# 改門檻
strict = partial(significance_gate, threshold=3.0)
result = evaluate_factor(df, "Mom_20D", gates=[strict, oos_persistence_gate], config=config)

# 跳過 OOS
result = evaluate_factor(df, "Mom_20D", gates=[significance_gate], config=config)

# 加自訂 Gate（就是一個函式）
def my_monotonicity_gate(artifacts: Artifacts) -> GateResult:
    mono = compute_monotonicity(artifacts.prepared)
    passed = mono.value >= 0.3
    return GateResult(name="min_monotonicity", passed=passed, status="PASS" if passed else "VETOED")

result = evaluate_factor(
    df, "Mom_20D",
    gates=[significance_gate, my_monotonicity_gate, oos_persistence_gate],
    config=config,
)
```

### 3.5 為什麼 Gate 是函式而非 Protocol class

| 考量 | Protocol class | 函式 |
|------|---------------|------|
| 定義一道 Gate | class + name property + run method | 一個函式 |
| 自訂 Gate | 使用者需理解 Protocol 繼承 | 使用者寫一個普通函式 |
| 改門檻 | 改 constructor 參數 | `partial(gate_fn, threshold=3.0)` |
| 共享計算 | Gate 互相透過 mutable dict 讀寫（隱式依賴） | Pipeline 預算 Artifacts → Gate 只讀（顯式） |
| 型別安全 | `ctx.artifacts["ic_series"]` — string key | `artifacts.ic_series` — IDE 補全 + type check |

### 3.6 CAUTION 條件

根據 v3 spec §5：

| 條件 | 檢查方式 |
|------|---------|
| Step 6 未啟用 | `config.ortho is None` |
| Universe N < 200 | `df.select("asset_id").n_unique() < 200` per date (median) |
| Gate 1 僅透過 Q1-Q5 spread | 任一 gate_result 的 `detail.get("via") == ["Q1-Q5_spread"]` |
| IC_Trend 顯示衰減 | profile 中 IC_Trend slope 顯著 < 0（P1 才有） |

### 3.7 config.py

```python
# gates/config.py
@dataclass
class PipelineConfig:
    # Preprocessing
    forward_periods: int = 5
    return_clip_pct: tuple[float, float] = (0.01, 0.99)
    mad_n: float = 3.0
    orthogonalize: bool = False
    base_factors: pl.DataFrame | None = None

    # Profile
    n_groups: int = 10
    q_top: float = 0.2
    multi_horizon_periods: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    regime_labels: pl.DataFrame | None = None

    # Market
    estimated_cost_bps: float = 30.0

MARKET_DEFAULTS = {
    "tw": {"estimated_cost_bps": 30, "ortho_factors": ["size", "value", "momentum", "industry_tse30"]},
    "us": {"estimated_cost_bps": 5, "ortho_factors": ["size", "value", "momentum", "industry_gics"]},
}
```

> **注意：** Gate 門檻（`significance_threshold`, `oos_decay_threshold`）不在 PipelineConfig 裡。
> 它們是 Gate 函式自己的參數，透過 `functools.partial` 綁定。
> 這讓使用者可以對不同 Gate 設不同門檻，不需要所有門檻都塞在同一個 config 裡。

### 3.8 MLflow 整合

更新 `experiment.py` 的 `FactorTracker`：

| 現有 | 更新 |
|------|------|
| log 4 dimension scores | log Gate Status (PASS/CAUTION/VETOED/FAILED) |
| log per-metric scores | log Profile MetricOutput 原始值 + t-stat + significance |
| log total_score | 移除 |
| log routing weights | 移除 |
| — | log 每道 Gate 的 name + status + detail |

### 3.9 不同因子類別的 Pipeline 差異

Gate 本身不區分因子類別。差異在**組裝配置**：

| 因子類別 | 預設 Gate 組合 | Artifacts 差異 | 說明 |
|---------|---------------|---------------|------|
| cross_sectional | `[significance_gate, oos_persistence_gate]` | IC series + spread series | 本文件範圍 |
| event_signal | `[caar_significance_gate, event_oos_gate]` | CAAR series（替代 IC） | P3 |
| macro_signal | `[hit_rate_gate, oos_persistence_gate]` | hit rate series（N 小，IC 不穩定） | P3 |

`_build_artifacts` 可能需要根據因子類別計算不同的中間產物。
P3 實作時可以擴充 `Artifacts` dataclass 加入 optional 欄位（如 `caar_series`），
或設計為 `Artifacts` 的子類別。屆時再決定。

---

## 4. Phase 3：Dashboard 重構

> **前置條件：** Phase 2 完成。

### 4.1 Leaderboard

```
Factor       Status   Gate1        Gate2         IC_IR  Mono   Spread  Long_α  BE_Cost
───────────  ──────   ──────────   ──────────    ─────  ────   ──────  ──────  ───────
Mom_20D      PASS     ✓(via both)  ✓(0.78)       0.45★  0.85●  15%★    12%★    67bps
LowVol_20D   CAUTION  ✓(via IC)   ✓(0.65)       0.38★  0.91★  10%●    8%●     89bps
MeanRev_5D   VETOED   ✓(via IC)   ✗(0.42)       0.28●  ───    ───     ───     ───
RSI_14D      FAILED   ✗(t=1.3)    ───           ───    ───    ───     ───     ───
```

排序：PASS > CAUTION > VETOED > FAILED → 同 status 內依使用者自選欄位。

### 4.2 Factor Detail

Gate 結果 + Profile 兩欄呈現（Reliability / Profitability）+ ★●○ 標記。
圖表：IC 累積圖 (IS/OOS)、Rolling IC、Quantile Return 長條圖、Multi-horizon IC Decay。

### 4.3 移除項目

- Radar chart（composite score 視覺化）
- Dimension bar chart
- Total score 排序

---

## 5. 現有程式碼處置

| 模組 | 處置 | 時機 |
|------|------|------|
| `engine.py` | 新架構不做 backward compat wrapper，直接由 `preprocess/` 取代 | Phase 1 |
| `validation.py` | 保留，schema 微調（新增 Step 6 相關欄位） | Phase 1 |
| `scoring/` 整個目錄 | 被 `tools/` + `gates/` 取代，Phase 2 後移除 | Phase 2 |
| `builders.py` | IC artifact 遷移至 `tools/panel/ic.py`；NAV artifact 移除（v3 不追蹤 NAV） | Phase 2 |
| `experiment.py` | 保留，新增 Gate Status logging | Phase 2 |
| `factors/` | 完全保留不動 | — |
| `dashboard/` | Phase 3 重構 | Phase 3 |
| `reporting.py` | Phase 3 更新 | Phase 3 |

---

## 6. 優先級與里程碑

### P0-a：工具層骨架 + Preprocessing 拆分 ✅ 已完成

| 項目 | 說明 | 狀態 |
|------|------|:----:|
| `tools/_typing.py` | MetricOutput + 常數定義 | ✅ |
| `tools/_helpers.py` | 共用工具（sample_non_overlapping, assign_quantile_groups, annualize_return） | ✅ |
| `preprocess/returns.py` | Step 1-3 | ✅ |
| `preprocess/normalize.py` | Step 4-5 | ✅ |
| `preprocess/pipeline.py` | orchestration | ✅ |

交付物：`from factorlib.preprocess.returns import compute_forward_return` 可獨立使用。

### P0-b：核心 Panel + Series 工具 ✅ 已完成

| 項目 | 說明 | 狀態 |
|------|------|:----:|
| `panel/ic.py` | IC 計算（含 multi-horizon，單次掃描所有 horizon） | ✅ |
| `panel/quantile.py` | Spread + Long/Short Alpha（支援 _precomputed_series 避免重複計算） | ✅ |
| `panel/monotonicity.py` | n_groups=10（復用 _helpers.assign_quantile_groups） | ✅ |
| `panel/concentration.py` | Q1_Concentration (HHI) | ✅ |
| `series/significance.py` | t-stat + `***`/`**`/`*` markers | ✅ |
| `series/oos.py` | Multi-split OOS decay + sign flip（Literal 型別 status） | ✅ |
| `series/hit_rate.py` | Hit rate | ✅ |
| `series/trend.py` | Theil-Sen IC trend | ✅ |
| `cost/tradability.py` | Turnover + Breakeven + Net Spread | ✅ |
| `regression/orthogonalize.py` | Step 6 因子正交化 | ✅ |

交付物：每個工具可獨立 import 使用，已通過 sample_data.parquet 端對端驗證。

### P0-c：Gate Pipeline ✅ 已完成

| 項目 | 說明 | 狀態 |
|------|------|:----:|
| `gates/_protocol.py` | GateFn type, GateResult（passed 為 property）, Artifacts（含 ic_values）, EvaluationStatus Literal | ✅ |
| `gates/significance.py` | significance_gate 函式 | ✅ |
| `gates/oos_persistence.py` | oos_persistence_gate 函式 | ✅ |
| `gates/profile.py` | compute_profile（組裝所有 tools → FactorProfile，復用 artifacts 避免重複計算） | ✅ |
| `gates/pipeline.py` | evaluate_factor（接受 `list[GateFn]`，預算 Artifacts，依序跑 Gate，CAUTION 含 IC trend decay 檢查） | ✅ |
| `gates/presets.py` | CROSS_SECTIONAL_GATES 預設組合 | ✅ |
| `gates/config.py` | PipelineConfig + MARKET_DEFAULTS | ✅ |
| `experiment.py` 更新 | `log_evaluation()` — Gate Status + Profile metrics MLflow logging | ✅ |

交付物：`evaluate_factor(df, "Mom_20D", gates=CROSS_SECTIONAL_GATES, config=config)` 回傳完整 `EvaluationResult`。

### P1：Profile 擴充 ✅ 工具已完成（Dashboard 延後）

| 項目 | 說明 | 狀態 |
|------|------|:----:|
| `series/significance.py` 改進 | 學術慣例 markers（`***`/`**`/`*`，含 p<0.10 層級）；BHY 連續校正（正態近似，vectorized） | ✅ |
| `panel/quantile.py` 擴充 | VW Spread（`quantile_spread_vw`） | ✅ |
| `panel/ic.py` 擴充 | Regime IC（group_by 聚合，zero-IC 方向修正） | ✅ |
| Dashboard 重構 | Gate-based leaderboard + detail | 延後 — 待實際探索模組後決定 |
| 移除 `scoring/` 模組 | deprecated composite score | 延後 — dashboard 仍依賴 |

### P2：Stage 2 ✅ 已完成

| 項目 | 說明 | 狀態 |
|------|------|:----:|
| `regression/spanning.py` | Greedy Forward Selection + Backward Elimination | ✅ |

### P3：其他因子類別

| 項目 | 說明 |
|------|------|
| `event/caar.py` | 事件訊號 CAAR |
| `event/event_stats.py` | 事件 hit rate, skewness, dispersion |
| `gates/event_signal.py` | 事件訊號 pipeline（組裝 event/ + series/ 工具） |
| `gates/macro_signal.py` | 總經/區域訊號 pipeline（複用 Gate + N-aware 降級） |

### P4：跨 Universe

| 項目 | 說明 |
|------|------|
| Pervasiveness Profile | Pass_Count + Random-effects meta-analysis + I² |

---

## 7. 訊號因子的可擴展性

Phase 1 工具的設計天然支援未來的訊號因子評估，因為工具按資料形狀分類而非因子類別。

**多標的事件訊號可同時使用 event/ 和 panel/ 兩套工具：**

```
多標的事件訊號（營收創高，截面多檔觸發）
├─ 當作 event → event/caar.py → CAAR t-stat
├─ 當作 panel → panel/ic.py → 截面 IC（觸發強度 vs return）
│              → panel/quantile.py → 按觸發強度分組 spread
└─ 兩者輸出都是 series → series/oos.py, series/trend.py 通吃

單標的事件訊號（台積電法說會 → 買/賣）
├─ 當作 event → event/caar.py → CAAR t-stat
└─ panel/ 工具不適用（沒有截面）→ Pipeline 層不呼叫
```

Pipeline 層（`gates/event_signal.py`）負責判斷單/多標的，決定組裝哪些工具。工具本身不做任何判斷。
