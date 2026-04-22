# Factor Type 路由架構設計

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> 狀態：已實作（P0–P3.5 + Future A + Future B；event_signal 待 P4）
> 日期：2026-04-16
> 版本：v8（整合 user/dev 審查回饋 + optional deps + integrations）
> 背景：現有 factorlib 僅支援截面風格因子（個股選股）。擴展到事件訊號、宏觀 panel、
> 宏觀共用因子，需要路由機制讓使用者拿到正確的工具。
>
> 定位：集成所有因子類別的有效性分析工具——獨立可調用的因子評價模組（類似
> alphalens 但涵蓋所有因子類型）、可控 gate 機制、批次篩選、多因子比較。
> 目前以開發方式做探索，最終目標是成為正式套件（開源或非開源）。
>
> 設計原則：
> - `tools/` → `metrics/`（只放 MetricOutput 工具）；`gates/` → `evaluation/`
> - 跨模組共用型別/統計原語提升到 top-level `_types.py` / `_stats.py`
> - 套件思維：函式命名精簡（`fl.evaluate` 而非 `fl.evaluate_factor`）
> - Clean break：探索階段是「允許 break 的窗口」，趁使用者少時把 API 做對
> - Public/Private：`_` prefix = internal；每個目錄一句話能描述
> - 回傳值自帶 `__repr__`；錯誤訊息對不看 source code 的人友善
> - 核心零 optional 依賴（polars only）；charts / mlflow / dashboard 走 extras
>
> 範圍：四種因子類型覆蓋 ~95% 因子研究。
> HF signals、配對交易、策略級規則不在範圍。

---

## 1. 四種因子類型

### 1.1 分類與目的

| 類型 | 範例 | signal 特徵 | 核心問題 | 目的 |
|------|------|------------|---------|------|
| `cross_sectional` | momentum, value, size | 連續值，每期每資產都有 | 排序能預測截面報酬差異嗎？ | 選股 |
| `event_signal` | 營收發佈、黃金交叉、法說會 | 離散觸發，只有事件日有 signal | 事件發生後報酬有異常嗎？ | 事件交易 |
| `macro_panel` | 各國 CPI、利差、相對 value | 連續值，小截面（N=10-30） | 宏觀指標能預測跨國配置嗎？ | 跨國配置 |
| `macro_common` | VIX、黃金、比特幣、美元指數 | 單一時序，所有資產共用 | 資產對共同因子的 exposure 是否穩定？ | 風險歸因（非選股） |

### 1.2 關鍵差異

```
                    截面有 signal？    每期都有？    N 夠大？
cross_sectional         ✓               ✓            ✓
event_signal            ✓               ✗            ✓
macro_panel             ✓               ✓            ✗
macro_common            ✗ (共用)         ✓            ✗
```

### 1.3 如何選擇 factor_type

```
你的 signal 每個資產各有一個值嗎？
├── 否（所有資產共用同一個值）→ macro_common
└── 是
    ├── 每期都有嗎？
    │   ├── 否（只有事件日才有）→ event_signal
    │   └── 是
    │       ├── 截面 N ≥ 30？→ cross_sectional
    │       └── 截面 N < 30？→ macro_panel
    └── 不確定 → 先用 cross_sectional 跑看看，IC SE 會告訴你 N 夠不夠
```

---

## 2. 每種類型的評估工具對照

### 2.1 Reliability

| 指標 | cross_sectional | event_signal | macro_panel | macro_common |
|------|:-:|:-:|:-:|:-:|
| IC (Spearman) | 主要 | — | ✗（註 1） | — |
| IC_IR | 主要 | — | ✗ | — |
| FM beta (Fama-MacBeth) | — | — | 主要（註 2） | — |
| Pooled OLS + clustered SE (by date) | — | — | 主要（註 2） | — |
| TS beta (time-series) | — | — | — | 主要 |
| CAAR t-test | — | 主要（註 3） | — | — |
| BMP test (Standardized AR) | — | 主要（處理波動率膨脹，註 3） | — | — |
| Corrado rank test | — | 備用（非參數，註 3） | — | — |
| Hit rate (signal→return) | IC > 0 比例 | event direction | — | — |
| Precision / F1 Score | — | 主要 (二元訊號評估) | — | — |
| β sign consistency | — | — | FM β > 0 比例 | TS β sign 比例 |
| OOS decay | IC 時序 | CAAR 時序 | β 時序 | β 時序 |
| Trend (Theil-Sen) | IC trend | CAAR decay | β trend | β trend |

### 2.2 Profitability

| 指標 | cross_sectional | event_signal | macro_panel | macro_common |
|------|:-:|:-:|:-:|:-:|
| Quantile spread (Q1-Q5, 5-10 組) | 主要 | — | — | — |
| Long-short 1/3 | — | — | 主要 | — |
| Event L/S | — | 主要（Post-event drift）| — | — |
| Factor-mimicking portfolio (FMP) | — | — | — | 可選（僅 tradable，註 4） |
| Portfolio Sharpe / Sortino | 可算 | 主要（Sortino 治下行風險）| 主要 | 可選（同 FMP） |
| Max Drawdown (MDD) | 可算 | 主要 | 可算 | 可算 |
| Monotonicity | 主要 | — | — | — |
| Concentration | 主要 | — | — | — |
| Turnover / Breakeven | 主要 | — | 可參考 | — |

### 2.3 Attribution

| 指標 | cross_sectional | event_signal | macro_panel | macro_common |
|------|:-:|:-:|:-:|:-:|
| Spanning alpha (vs CS style factors) | 主要 | 主要（註 5） | — | — |
| Spanning alpha (vs global macro factors) | — | — | 主要（註 2） | — |
| Orthogonalize R² | 主要 | — | — | — |
| TS regression R² | — | — | — | 主要 |

---

## 3. 統計嚴謹度備註

### 3.1 macro_panel：IC 類工具不回傳（註 1）
N=10-30 時 Spearman IC 的 SE ≈ 0.19–0.33，每期幾乎是噪音。IC / IC_IR /
hit_rate(IC>0) 在 macro_panel 的 profile 中不計算；reliability 以 FM β 與
pooled baseline 為主。

### 3.2 macro_panel：FM 需搭配 pooled baseline（註 2）
FM 在 N=10 時 cross-sectional regression 的 coefficient noise 很大。MVP profile
同時回傳：
- Fama-MacBeth β + Newey-West SE
- Pooled OLS + clustered SE (by date)

兩個估計量一致性當 robustness signal。單獨 FM 不足以下結論。

同樣的小 N 警告適用於 spanning alpha：macro_panel 的 spanning 測試（對 global
macro factors）其 SE 在 N=10 時同樣不穩，結論應以 FM / pooled spanning 兩種
一致性判讀，不可單看其一。

Panel random effects + Driscoll-Kraay SE 為 Future+ 擴充項（corporate finance
實證常用，asset pricing 實務較少必要）。

### 3.3 macro_common：角色是風險歸因，不是選股訊號（註 4）
VIX / 黃金 / 美元指數等單一時序因子，其 β 穩定性測的是「資產暴露是否穩定」，
不是「這個因子能選出未來贏家」。macro_common 的輸出**不進 factor scoring**，
只進 orthogonalization 的 base factor 或風險模型。

若想把這類因子轉成預測訊號（例如 VIX↑ 時 low-vol 贏），那是 cross_sectional
條件因子（`signal = asset's conditional β × factor innovation`），走
cross_sectional type，不屬於 macro_common。

FMP（factor-mimicking portfolio）只對 tradable 因子（黃金、美元 index、ETF）
有意義；不可交易的狀態變數（VIX、Breakeven 通膨、經濟政策不確定性指數）不應
計算 FMP，否則會誤導為「可交易 alpha」。以 `PipelineConfig.tradable: bool` 控制。

### 3.4 event_signal：顯著性檢驗與事件變異數膨脹（註 3）
事件研究的標準顯著性工具通常是 CAAR t-test。然而，事件發生當下往往伴隨市場波動率的大幅飆升（Event-induced variance），若直接用傳統 t-test 非常容易產生高估顯著性的問題（False Positives）。

因此實務標準必須整合：
1. **BMP test (Boehmer, Musumeci, and Poulsen)**：將每檔資產的 Abnormal Return (AR) 針對自身的估計期殘差進行標準化（Standardized Abnormal Return, SAR）後再行檢驗，這是量化界處理事件波動膨脹的標配。
2. **Corrado rank test**：作為非參數備援，對付極端報酬跳空與非常態分配。
3. **KS test**：由於 KS test 測的是整體分布是否相等，對事件異常報酬的檢驗不對題，在此架構下不予採用。

### 3.5 event_signal：事件 clustering 破壞獨立性假設
CAAR t-test 與 Corrado rank test 預設事件獨立。實務常見的非獨立來源：
- 日曆 clustering：財報季同週多家公司釋出、總統大選、FOMC
- 產業 clustering：同產業對同一外生事件反應

未修正會使 SE 低估、t-stat 膨脹、p-value 偏低。MVP 處理方案：
- 診斷：Reliability 中回報 `event_date` Herfindahl index 與同一週內
  事件家數分布，讓使用者判斷 clustering 嚴重程度
- 修正（optional）：`PipelineConfig.adjust_clustering`，預設 `"none"`
- 若診斷偵測到高度 clustering 且未設修正，dashboard 標記 CAUTION

### 3.6 event_signal：spanning 歸因到 CS 風格因子（註 5）
事件異常報酬常被 CS 風格因子部分解釋——post-earnings drift 被 momentum 吸收、
M&A target 異常報酬被 size / value 解釋。Attribution 層對 CAAR 序列跑 spanning
回歸（對 size / value / momentum / profitability），回報：
- 事件 alpha after controlling for style exposure
- 各風格因子的 beta 與 t-stat

若 spanning alpha 不顯著但 raw CAAR 顯著，意味著事件因子可被風格因子複製，
沒有獨立配置價值。

### 3.7 macro_panel：相對因子的截面 demean 副作用
`signal_i = x_i - mean(x)` 在小 N 時會引入機械負相關（加總為 0）：
- Long-short portfolio return 不受影響
- 單邊組合（只做 long）的 β 會被改寫
- Spanning test 前需確認 base factor 也經過對應處理

以 `PipelineConfig.demean_cross_section: bool` 顯式控制，預設 `False`。

---

## 4. 架構設計

### 4.1 最終專案結構

```
factorlib/
├── __init__.py              # 頂層 exports（不含 optional deps）
├── _api.py                  # quick_check, compare, batch_evaluate, split_by_group
├── _types.py                # MetricOutput, OOSResult, FactorType(StrEnum), constants
├── _stats.py                # t-stat, p-value, significance marker, bhy_threshold
├── _ols.py                  # 共用 OLS helpers（spanning + orthogonalize 共用）
├── config.py                # BaseConfig, CrossSectionalConfig, EventConfig, ...
├── adapt.py                 # column name mapping
├── validation.py            # pandera schema check（per-type schema）
│
├── preprocess/              # 把原始資料變成可評估格式
│   ├── pipeline.py          # preprocess() dispatcher
│   ├── returns.py           # forward return, winsorize, abnormal return
│   ├── normalize.py         # MAD winsorize, z-score
│   └── orthogonalize.py     # 資料轉換（非 metric）
│
├── evaluation/              # 跑 pipeline：artifacts → gates → profile → caution
│   ├── _protocol.py         # Artifacts, FactorProfile, EvaluationResult（含 __repr__）
│   ├── pipeline.py          # evaluate(), build_artifacts()
│   ├── profile.py           # compute_profile()
│   ├── presets.py            # GATES_BY_TYPE, default_gates_for()
│   ├── _caution.py          # _check_caution rules
│   └── gates/               # 實際 gate functions
│       ├── significance.py
│       └── oos_persistence.py
│
├── metrics/                 # 獨立可調用的指標，每個回傳 MetricOutput
│   ├── __init__.py          # re-exports + __all__ + docstring 分類
│   ├── _helpers.py          # metrics-only helpers（sampling, quantile groups）
│   ├── ic.py
│   ├── quantile.py
│   ├── monotonicity.py
│   ├── concentration.py
│   ├── hit_rate.py
│   ├── trend.py
│   ├── oos.py
│   ├── spanning.py
│   ├── tradability.py
│   ├── caar.py              # P4（event_signal, 包含 CAAR 與 BMP test）
│   ├── corrado.py           # P4（event_signal）
│   ├── classification.py    # P4（event_signal, Precision, Recall, F1 Score）
│   ├── clustering.py        # P4（event_signal）
│   ├── fama_macbeth.py      # Future A（macro_panel）
│   └── ts_beta.py           # Future B（macro_common）
│
├── factors/                 # 因子生成器
│
├── charts/                  # optional: requires plotly
│   ├── __init__.py          # lazy import plotly + report_charts()
│   └── ...
│
└── integrations/            # optional extras
    ├── __init__.py
    ├── mlflow.py            # FactorTracker（原 experiment.py）
    └── streamlit/           # dashboard app（原 dashboard/）
        ├── app.py
        ├── charts.py
        └── data.py
```

### 4.2 每個位置一句話

| 位置 | 職責 |
|------|------|
| `_types.py` | 全 lib 共用型別（`MetricOutput`, `FactorType` StrEnum） |
| `_stats.py` | 全 lib 共用統計原語（t-stat, p-value, bhy_threshold） |
| `_ols.py` | 共用 OLS helpers（spanning + orthogonalize 共用，避免循環依賴） |
| `config.py` | 使用者面向設定（`CrossSectionalConfig`, `EventConfig`） |
| `_api.py` | high-level convenience（quick_check, compare, batch_evaluate） |
| `preprocess/` | 把原始資料變成可評估格式（含 orthogonalize） |
| `evaluation/` | 跑 pipeline：artifacts → gates → profile → caution |
| `evaluation/gates/` | 實際的 gate functions |
| `metrics/` | 獨立可調用的指標，每個回傳 `MetricOutput` |
| `factors/` | 因子生成器 |
| `charts/` | 圖表（optional: `pip install factorlib[charts]`） |
| `integrations/mlflow.py` | MLflow tracking（optional: `pip install factorlib[mlflow]`） |
| `integrations/streamlit/` | Dashboard app（optional: `pip install factorlib[dashboard]`） |

### 4.3 Optional dependencies

核心套件只依賴 `polars`。其他依賴走 extras：

```toml
# pyproject.toml
[project.optional-dependencies]
charts = ["plotly>=5.0"]
mlflow = ["mlflow>=2.0"]
dashboard = ["streamlit>=1.0", "plotly>=5.0"]
all = ["plotly>=5.0", "mlflow>=2.0", "streamlit>=1.0"]
```

```bash
pip install factorlib              # 核心
pip install factorlib[charts]      # + plotly
pip install factorlib[mlflow]      # + mlflow tracking
pip install factorlib[all]         # 全裝
```

`charts/__init__.py` 使用 lazy import：
```python
try:
    import plotly
except ImportError:
    raise ImportError(
        "Charts require plotly. Install with: pip install factorlib[charts]"
    )
```

### 4.4 `_` prefix 慣例

- `_` prefix = internal，不在任何 `__init__.py` export
- 無 `_` = public
- `metrics/` 底下只有 `_helpers.py` 有 `_`，其餘全是 public metric tools
- `evaluation/` 底下 `pipeline.py` / `profile.py` / `presets.py` = public（可被 power user import）；`_protocol.py` / `_caution.py` = internal

### 4.5 依賴方向（單向，無循環）

```
_types.py, _stats.py, _ols.py    ← 零依賴
       ↑
config.py                        ← 依賴 _types
       ↑
metrics/                         ← 依賴 _types, _stats
preprocess/                      ← 依賴 _types, _ols
       ↑
evaluation/                      ← 依賴 _types, _stats, metrics, config
       ↑
_api.py                          ← 依賴 evaluation, metrics, preprocess, config
       ↑
__init__.py                      ← re-export

integrations/, charts/           ← 依賴 evaluation（EvaluationResult），不被核心依賴
```

`_ols.py` 解決 `orthogonalize`（preprocess/）和 `spanning`（metrics/）共用 OLS
邏輯的問題——兩者都 import top-level `_ols.py`，不產生 preprocess → metrics 循環。

### 4.6 `metrics/` 結構

Flat，分類在 docstring 和 `__all__`：

```python
# metrics/__init__.py
"""factorlib.metrics — Independent factor evaluation metrics.

All metrics return MetricOutput and can be used standalone.

Signal Quality:
    compute_ic, ic, ic_ir, hit_rate, ic_trend,
    multi_split_oos_decay

Portfolio Performance:
    quantile_spread, monotonicity, q1_concentration,
    turnover, breakeven_cost, net_spread

Factor Attribution:
    spanning_alpha, greedy_forward_selection
"""

__all__ = [
    "compute_ic", "ic", "ic_ir", "regime_ic", "multi_horizon_ic",
    "hit_rate", "ic_trend", "multi_split_oos_decay",
    "compute_spread_series", "quantile_spread", "quantile_spread_vw",
    "compute_group_returns", "monotonicity", "q1_concentration",
    "turnover", "breakeven_cost", "net_spread",
    "spanning_alpha", "greedy_forward_selection",
]
```

### 4.7 Config：Per-type 子類

```python
# factorlib/config.py
from factorlib._types import FactorType

@dataclass(kw_only=True)
class BaseConfig:
    forward_periods: int = 5
    estimated_cost_bps: float = 30.0
    multi_horizon_periods: list[int] = field(
        default_factory=lambda: [1, 5, 10, 20],
    )

@dataclass(kw_only=True)
class CrossSectionalConfig(BaseConfig):
    factor_type: ClassVar[FactorType] = FactorType.CROSS_SECTIONAL
    n_groups: int = 10
    q_top: float = 0.2
    orthogonalize: bool = False
    mad_n: float = 3.0
    return_clip_pct: tuple[float, float] = (0.01, 0.99)

@dataclass(kw_only=True)
class EventConfig(BaseConfig):
    factor_type: ClassVar[FactorType] = FactorType.EVENT_SIGNAL
    event_window_pre: int = 5
    event_window_post: int = 20
    cluster_window: int = 3
    adjust_clustering: Literal["none", "calendar_block_bootstrap",
                                "kolari_pynnonen"] = "none"

@dataclass(kw_only=True)
class MacroPanelConfig(BaseConfig):
    factor_type: ClassVar[FactorType] = FactorType.MACRO_PANEL
    demean_cross_section: bool = False
    min_cross_section: int = 10

@dataclass(kw_only=True)
class MacroCommonConfig(BaseConfig):
    factor_type: ClassVar[FactorType] = FactorType.MACRO_COMMON
    ts_window: int = 60
    tradable: bool = False
```

v7 → v8 改動：
- `CSConfig` → `CrossSectionalConfig`（不縮寫，與 MacroPanelConfig 風格一致）
- `factor_type` 從 `ClassVar[str]` → `ClassVar[FactorType]`（StrEnum，打錯字 compile-time 報錯）

```python
# _types.py
class FactorType(enum.StrEnum):
    CROSS_SECTIONAL = "cross_sectional"
    EVENT_SIGNAL = "event_signal"
    MACRO_PANEL = "macro_panel"
    MACRO_COMMON = "macro_common"
```

### 4.8 Artifacts

```python
@dataclass
class Artifacts:
    prepared: pl.DataFrame
    config: BaseConfig
    intermediates: dict[str, pl.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pl.DataFrame:
        if key not in self.intermediates:
            ft = type(self.config).factor_type
            raise KeyError(
                f"Artifacts has no '{key}'. "
                f"Available for {ft}: {list(self.intermediates.keys())}"
            )
        return self.intermediates[key]
```

> 注意：`intermediates: dict` 犧牲了型別安全（IDE 無法補全 key）。在 contributor
> 數增加後若 bug 頻繁，可回退到 per-type typed dataclass（`CrossSectionalArtifacts`
> 等）搭配 Union dispatch。目前階段 dict + `.get()` 的明確 KeyError 足夠。

### 4.9 FactorProfile

```python
@dataclass
class FactorProfile:
    metrics: list[MetricOutput]
    attribution: list[MetricOutput] = field(default_factory=list)

    def get(self, name: str) -> MetricOutput | None:
        """按 name 查找 metric，避免線性搜尋的 name typo 成為隱性 bug。"""
        for m in self.metrics + self.attribution:
            if m.name == name:
                return m
        return None
```

### 4.10 EvaluationResult

```python
@dataclass
class EvaluationResult:
    factor_name: str
    status: EvaluationStatus
    gate_results: list[GateResult] = field(default_factory=list)
    profile: FactorProfile | None = None
    artifacts: Artifacts | None = None      # ← v8 新增：charts 需要
    caution_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict: ...          # ← v8 新增
    def to_dataframe(self) -> pl.DataFrame: ...  # ← v8 新增
    def __repr__(self) -> str: ...          # 格式化表格
```

v7 → v8 改動：
- **`artifacts` 欄位**：`evaluate()` 回傳時附帶 artifacts，使用者不需另呼叫
  `build_artifacts()` 就能用 `report_charts(result)`（解決 user 審查 #5）
- **`to_dict()` / `to_dataframe()`**：批次跑完後序列化到自己的 DataFrame（user 審查 #11）

### 4.11 Factor type discovery

```python
fl.FACTOR_TYPES
# {FactorType.CROSS_SECTIONAL: CrossSectionalConfig, ...}

# 每型附帶一句話描述（user 審查 #8）
fl.describe_factor_types()
# cross_sectional : 截面因子（每期每資產有 signal，N ≥ 30）→ 選股
# event_signal    : 事件訊號（離散觸發）→ 事件交易
# macro_panel     : 宏觀 panel（小截面 N < 30）→ 跨國配置
# macro_common    : 宏觀共用（單一時序）→ 風險歸因
```

### 4.12 build_artifacts / profile / caution 分型

均以 `type(config).factor_type` dispatch，使用 `match/case`。
CS 走現有邏輯；其他型暫時 raise `NotImplementedError`。

> 注意：三處 match/case 是已知的維護風險（dev 審查 #2）。目前 4 型 × 3 處 = 12 個
> branch 是可管理的。若後續 type 超過 6，考慮 registry pattern 收束。

---

## 5. 使用者面向 API 設計（套件導向，clean break）

### 5.1 命名原則

| 現有 | 套件版 | 理由 |
|---|---|---|
| `evaluate_factor()` | `evaluate()` | `fl.evaluate` 已清楚 |
| `preprocess_cs_factor()` | `preprocess()` | 統一入口 |
| `compare_factors()` | `compare()` | 同上 |
| `CSConfig` | `CrossSectionalConfig` | 不縮寫，與其他 Config 風格一致 |
| `CROSS_SECTIONAL_GATES` | 不 export 到頂層 | gates 是 internal concern |
| `from factorlib.tools import ...` | `from factorlib.metrics import ...` | metrics 更精確 |
| `FactorTracker` | `from factorlib.integrations.mlflow import FactorTracker` | optional dep |

### 5.2 Top-level exports

```python
__all__ = [
    # Core workflow
    "adapt", "preprocess", "evaluate", "quick_check",
    # Batch & comparison
    "batch_evaluate", "compare",
    # Configuration
    "CrossSectionalConfig", "EventConfig",
    "MacroPanelConfig", "MacroCommonConfig",
    "FACTOR_TYPES", "describe_factor_types",
    # Artifacts（進階）
    "build_artifacts",
    # Validation
    "validate_factor_data",
]
```

不 export `BaseConfig`（使用者不該直接用）、不 export `FactorTracker`（在 integrations/）。

### 5.3 Import ergonomics

| Persona | 使用場景 | 程式碼 |
|---------|---------|--------|
| Quick Screener | 快速篩選 | `import factorlib as fl` → `fl.quick_check(df, "X")` |
| Quick Screener | 批次 | `fl.batch_evaluate(factors)` → `fl.compare(results)` |
| Factor Developer | 完整 pipeline | `fl.adapt()` → `fl.preprocess()` → `fl.evaluate()` |
| Factor Developer | 單一指標 | `from factorlib.metrics import compute_ic, ic_ir` |
| Factor Developer | 圖表 | `from factorlib.charts import report_charts`（需 `[charts]`） |
| Factor Developer | 追蹤 | `from factorlib.integrations.mlflow import FactorTracker`（需 `[mlflow]`） |
| Portfolio Builder | 因子篩選 | `fl.batch_evaluate()` → `fl.compare()` + `from factorlib.metrics import spanning_alpha` |

### 5.4 Level 0 — quick_check

```python
import factorlib as fl

result = fl.quick_check(factor_df, "Mom_20D")
print(result)

# 切換因子類型
result = fl.quick_check(event_df, "Earnings", factor_type="event_signal")

# 傳入自訂 config
result = fl.quick_check(df, "X", config=fl.CrossSectionalConfig(n_groups=5))
```

`preprocess=True`（預設）。

### 5.5 Level 1 — evaluate

```python
cfg = fl.CrossSectionalConfig(forward_periods=5)
result = fl.evaluate(prepared, "Mom_20D", config=cfg)

# 自訂 gates
result = fl.evaluate(prepared, "X", config=cfg, gates=custom_gates)
```

`preprocess=False`（預設）。

> user 審查 #1 的對策：`evaluate(raw_df, ...)` 在 `preprocess=False`（預設）時，
> 若偵測到 df 缺少 `forward_return` 欄位，error message 明確提示：
> `"Missing 'forward_return'. Did you forget fl.preprocess(df)? Or set preprocess=True."`

### 5.6 Level 2 — metrics (modular)

```python
from factorlib.metrics import compute_ic, quantile_spread, spanning_alpha
```

### 5.7 Level 3 — compare

```python
table = fl.compare(results, sort_by="ic_ir", bhy_correction=True)
```

跨型混合時：warn + 以 `None` 填充缺失欄位。

### 5.8 Level 4 — preprocess

```python
prepared = fl.preprocess(factor_df, forward_periods=5)
```

### 5.9 Level 5 — charts

```python
from factorlib.charts import report_charts

# result.artifacts 已內建，不需另外 build
figs = report_charts(result)
```

### 5.10 Level 6 — batch_evaluate

```python
results = fl.batch_evaluate(factors, forward_periods=5)
table = fl.compare(results, bhy_correction=True)

# With MLflow tracking（optional）
from factorlib.integrations.mlflow import FactorTracker
tracker = FactorTracker("Factor_Zoo")
results = fl.batch_evaluate(
    factors, forward_periods=5,
    on_result=tracker.log,
)
```

`batch_evaluate` 不再有 `tracker` 參數，改為通用 `on_result: Callable`：

```python
def batch_evaluate(
    factors: list[tuple[str, pl.DataFrame]] | dict[str, pl.DataFrame],
    *,
    factor_type: str = "cross_sectional",
    config: BaseConfig | None = None,
    gates: list[GateFn] | None = None,
    preprocess: bool = True,
    on_result: Callable[[str, EvaluationResult], None] | None = None,
    stop_on_error: bool = False,
    **config_overrides,
) -> dict[str, EvaluationResult]:
```

### 5.11 EvaluationResult `__repr__`

```
Factor: Mom_20D | Status: PASS
┌──────────────┬────────┬────────┬─────┐
│ metric       │ value  │ t_stat │ sig │
├──────────────┼────────┼────────┼─────┤
│ ic           │ 0.0512 │  3.42  │ *** │
│ ic_ir        │ 0.285  │        │     │
│ hit_rate     │ 0.620  │  2.68  │ **  │
│ oos_decay    │ 0.72   │        │     │
│ q1_q5_spread │ 0.0034 │  2.91  │ **  │
│ turnover     │ 0.42   │        │     │
│ net_spread   │ 0.0009 │        │     │
└──────────────┴────────┴────────┴─────┘

Attribution (vs size, value, momentum):
┌──────────────┬────────┬────────┬─────┐
│ spanning_α   │ 0.0021 │  2.14  │ *   │
└──────────────┴────────┴────────┴─────┘
```

使用 Python f-string + box-drawing chars 手刻（零外部依賴）。

### 5.12 Error Messages

```python
# 缺欄位（evaluate preprocess=False 時）
fl.evaluate(df_without_fwd_ret, "X", config=cfg)
# ValueError: cross_sectional requires columns {date, asset_id, factor,
# forward_return}. Missing: {forward_return}.
# Hint: call fl.preprocess(df) first, or set preprocess=True.

# 錯 factor_type
fl.quick_check(df, "X", factor_type="magic")
# ValueError: unknown factor_type 'magic'.
# Supported: cross_sectional, event_signal, macro_panel, macro_common.
# Use fl.describe_factor_types() for details.

# 小 N 自動偵測（user 審查 #4）
fl.quick_check(country_df, "CPI")  # N=10, factor_type 預設 CS
# UserWarning: Median cross-section size = 10 (< 30).
# Consider using MacroPanelConfig instead of CrossSectionalConfig.
# IC-based metrics may be unreliable at this N.
```

---

## 6. 實作優先順序

| 優先 | 內容 | Breaking | 工作量 |
|------|------|:---:|--------|
| P0 | 結構搬家（見 §6.1 搬遷表）| ✓ | 0.5 d |
| P1 | Config per-type 子類 + FactorType StrEnum + Artifacts intermediates dict + FactorProfile（`.get()` + 2-tier）+ EvaluationResult（`__repr__` + `to_dict` + `artifacts` 欄位）| ✓ | 1 d |
| P2 | `evaluate` rename + keyword-only + routing + schema validation + `_DEFAULT_GATES` + 小 N warning | ✓ | 1 d |
| P3 | API 便捷層：`_api.py`（quick_check, compare, batch_evaluate）+ `metrics/__init__.py` re-exports + `preprocess()` dispatcher + `charts/__init__.py` lazy import + `__init__.py` clean break | ✓ | 1 d |
| P3.5 | 遷移：`experiment.py` → `integrations/mlflow.py`；`dashboard/` → `integrations/streamlit/`；`experiments/*.py` + `tests/*.py` 更新 import | ✓ | 0.5 d |
| P4 | event_signal：`metrics/caar.py` + `corrado.py` + `clustering.py` + `mfe_mae.py` + event gates + profile | ✗ | — |
| Future A | `metrics/fama_macbeth.py` + pooled baseline；macro_panel 全套 | ✓ 已實作 | — |
| Future B | `metrics/ts_beta.py`；macro_common 全套 | ✓ 已實作 | — |

P0–P3.5 合計 ~4 天，一次 clean break。

### 6.1 搬遷對照表

| 來源 | 目標 |
|------|------|
| `tools/panel/ic.py` | `metrics/ic.py` |
| `tools/panel/quantile.py` | `metrics/quantile.py` |
| `tools/panel/monotonicity.py` | `metrics/monotonicity.py` |
| `tools/panel/concentration.py` | `metrics/concentration.py` |
| `tools/series/hit_rate.py` | `metrics/hit_rate.py` |
| `tools/series/trend.py` | `metrics/trend.py` |
| `tools/series/oos.py` | `metrics/oos.py` |
| `tools/series/significance.py` | `factorlib/_stats.py` |
| `tools/regression/spanning.py` | `metrics/spanning.py` |
| `tools/regression/orthogonalize.py` | `preprocess/orthogonalize.py` |
| `tools/regression/` 共用 OLS helpers | `factorlib/_ols.py` |
| `tools/cost/tradability.py` | `metrics/tradability.py` |
| `tools/_typing.py` | `factorlib/_types.py`（+ FactorType StrEnum） |
| `tools/_helpers.py` | `metrics/_helpers.py` |
| `tools/comparison.py` | split → `_stats.py`（bhy_threshold）+ `_api.py`（split_by_group）|
| `gates/_protocol.py` | `evaluation/_protocol.py` |
| `gates/pipeline.py` | `evaluation/pipeline.py` |
| `gates/profile.py` | `evaluation/profile.py` |
| `gates/presets.py` | `evaluation/presets.py` |
| `gates/significance.py` | `evaluation/gates/significance.py` |
| `gates/oos_persistence.py` | `evaluation/gates/oos_persistence.py` |
| `gates/config.py` | `factorlib/config.py` |
| `experiment.py` | `integrations/mlflow.py` |
| `dashboard/` | `integrations/streamlit/` |

---

## 7. Verification

1. `pytest tests/ -v` 全綠
2. `from factorlib.metrics import compute_ic, quantile_spread` works
3. `fl.quick_check(df, "Mom_20D")` returns EvaluationResult with repr
4. `fl.FACTOR_TYPES` returns config registry（StrEnum keys）
5. `fl.describe_factor_types()` 印出四型描述
6. CS evaluation produces flat `profile.metrics` list
7. Non-CS types raise `NotImplementedError`
8. `artifacts.get("wrong_key")` raises KeyError with helpful message
9. `profile.get("ic")` returns MetricOutput or None
10. `result.to_dict()` / `result.to_dataframe()` 可用
11. `result.artifacts` is not None after `evaluate()`
12. `from factorlib._types import MetricOutput` works
13. `from factorlib._stats import bhy_threshold` works
14. `from factorlib.charts import report_charts` 未裝 plotly 時報明確 ImportError
15. `from factorlib.integrations.mlflow import FactorTracker` 未裝 mlflow 時報明確 ImportError
16. No circular imports
17. 小 N 時 evaluate/quick_check 發出 UserWarning

---

## 8. 遺留議題

1. **外部統計套件選擇**：`statsmodels` 足夠 MVP；Future A 做 panel RE 時
   `linearmodels` 幾乎必選。
2. **Clustering 診斷的 CAUTION 閾值**：P4 實作時根據實際資料定。
3. **`tradable` 的判定責任**：由使用者宣告。
4. **四型範圍界定**：HF / 配對交易 / 策略級不在範圍。
5. **`compare` 跨型欄位差異**：跨型混合時 warn + None 填充。
6. **`fl.example_data()`**：內建範例資料加速上手（user 審查 #15）。
7. **NaN 處理策略**：preprocess 的 drop / warn / 門檻需文件化（user 審查 #16）。
8. **`batch_evaluate` parallel**：`n_jobs` 參數或 async support（user 審查 #17）。
9. **`__repr__` 實作**：使用 f-string + box-drawing chars，零外部依賴。
10. **multi-horizon 使用方式**：一次 evaluate 回傳多 horizon 的結果？還是需要多次
    呼叫？需在 API 文件中明確（user 審查 #14）。
11. **`GateStatus` 定義重複**：`_protocol.py` vs `oos.py` 有兩份不同的 Literal，
    統一到 `_types.py`（dev 審查 #9）。
12. **metrics 函式命名統一**：`compute_ic` 帶 prefix、`ic_ir` 不帶——
    確認 canonical 名稱（dev 審查 #11）。
13. **`intermediates` 型別安全升級**：目前 `dict[str, DataFrame]` + `.get()` 在
    探索階段足夠。當 ≥ 2 型的 intermediates 穩定且 KeyError 頻繁出現時，改回
    per-type typed dataclass（`CrossSectionalIntermediates` 等）+ Union dispatch，
    恢復 IDE autocomplete。
14. **Evaluation vs Backtesting 邊界**：Turnover / Breakeven / Net Spread 是
    理想化的 proxy（等權、無滑價、無市場衝擊），不是真實交易回測。套件定位是
    **Factor Signal Analyzer**，不是 Order-matching Engine（Zipline / Backtrader）。
    README / API docstring 需明確聲明此邊界，避免使用者誤讀為可交易收益。
15. **CS IC t-test vs Newey-West 決策**：CS 的 IC / spread t-stat 使用簡單
    t-test + non-overlapping sampling，macro_panel FM β 使用 Newey-West。
    Non-overlapping sampling 已大幅消除 return overlap 造成的自相關，因此
    CS 不需 NW。若未來發現 non-overlapping 不夠（e.g. 因子 persistence
    引入的間接自相關），可將 CS 也切換到 NW。
16. **OOS decay threshold 跨型適用性**：`decay_threshold=0.5` 對 IC（bounded
    [-1, 1]）和 β（unbounded）有不同統計意義。目前共用同一 threshold 是
    MVP 簡化；未來可依 factor_type 設不同預設值。

---

## 9. 新增 factor type 指引（開發者友善）

```
如何新增一個 factor type：

1.  _types.py: 在 FactorType StrEnum 加新值
2.  config.py: 新增 XxxConfig(BaseConfig) 子類，設 factor_type ClassVar
3.  __init__.py: export 新 Config
4.  _api.py: 更新 FACTOR_TYPES registry + _DESCRIPTIONS
5.  validation.py: 在 _SCHEMAS dict 加 per-type schema
6.  preprocess/pipeline.py preprocess(): 加 case branch
7.  evaluation/pipeline.py build_artifacts: 加 case branch → 填充 intermediates
8.  evaluation/profile.py compute_profile: 加 case branch → 呼叫對應 metrics
9.  evaluation/_caution.py check_caution: 加 case branch
10. evaluation/presets.py: 加 gate list + 更新 _DEFAULT_GATES
11. metrics/: 新增 metric tool（如 caar.py, fama_macbeth.py）
12. metrics/__init__.py: 新增 re-exports + 更新 __all__ + docstring
13. tests/: 新增對應測試

不需改動：Artifacts class, BaseConfig, 其他 type 的任何檔案
```
