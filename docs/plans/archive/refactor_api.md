# Plan: v0.5 API Refactor — Scope Clarification + `AnalysisConfig` 三軸正交 + Mode A/B 對等 + `canonical_p` → `primary_p`

> Status: **Draft v3.3 — sign-off pending**
> Date: 2026-05-01
> Driven by: README §怎麼選分析模式 設計討論 + senior backend review (gemini) + senior quant user review (subagent) + 「無既有用戶、保持簡潔」directive + senior quant pre-implementation reviews #1, #2 (2026-05-01)
>
> **v3.3 changes from v3.2**（extensibility & customization review）：
> 1. **`StatCode` enum + `FactorProfile.stats: Mapping[StatCode, float]`**（B3 後續）— cell-specific 統計改走 typed dict，central dataclass schema 不再隨 metric 線性膨脹
> 2. **Verdict policy 分層**（C1）— `profile.verdict(threshold=, gate=)` 介面：`primary_p` 仍為 procedure-canonical SSOT，gate 為 user-overrideable policy（恢復 v0.4 `p_source` override 能力但 type-safe）
> 3. **參數命名**：使用 `threshold`（非 `alpha`）— alpha 在統計文獻特指 Type I error rate，gate 換成非 p 統計量時誤導
> 4. **Custom metric 列入「明確不做」**（C3）— user-level Metric enum runtime extension 不開放；power user 走 procedure registration
> 5. **Registry 內部使用，v0.6+ 候選對外**（C4）— v0.5 不公開 `factrix.extension` namespace，但設計就緒，underscore 前綴
>
> **v3.2 changes from v3.1**（second backend review — SSOT + over-engineering）：
> 1. **Mode B sparse collapse**（B1）— `(*, SPARSE, Mode.TIMESERIES)` 不論 user 傳哪個 scope 都路由到同一 registry entry / 同一 BHY family；scope 軸真退化，非 cosmetic
> 2. **Registry as SSOT**（A1）— `_validate_axis_compat` / `describe_analysis_modes` / `suggest_config` 全部反查 registry，不另寫一份 5-tuple table
> 3. **移除 `PRIMARY_P_FIELD: ClassVar[str]`**（A2）— stringly-typed 雙重來源；統一走 `Profile.primary_p` property
> 4. **單一 `FactorProfile` dataclass + `FactorProcedure` callable**（B3）— 取代 N×M Profile class proliferation
> 5. **Exception 階層收斂**（B2）— `MissingMetricError` / `RedundantMetricError` 併入 `IncompatibleAxisError`
> 6. **Stats 常數集中**（A3）— `MIN_T_HARD` / `MIN_T_RELIABLE` / Bartlett lag 抽出 `factrix._stats.constants`
> 7. **`_FALLBACK_MAP` 取代散在 raise 點的 `suggested_fix`**（A4）
> 8. **移除 Python `Warning` class 體系**（B6）— 統一走 `WarningCode` enum + `Profile.warnings` frozenset；不 `warnings.warn()`
> 9. **`event_temporal_hhi` 降級到 `diagnose()` dict**（B5）— 不污染 Profile top-level schema
>
> **v3.1 changes from v3**（pre-implementation review blockers）：
> 1. **驗證責任歸屬統一到 `__post_init__`**（B1）— factories 退化為純便利建構子，`from_dict` round-trip 安全
> 2. **BHY family key 加入 `mode` 維度**（B2）— Mode A 與 Mode B 不混 family，避免 step-up FDR 假設破壞
> 3. **`(*, SPARSE, N=1)` procedure 共用層**（B3）— registry 1 tuple 1 Profile，但 procedure function 抽成共用，避免複製
> 4. **Mode B `(INDIVIDUAL/COMMON, SPARSE, N=1)` `primary_p` 來源明定**（B4）— TS dummy regression β / NW HAC SE，CAAR 路徑只在 Mode A
> 5. **`degraded_modes: frozenset[DegradedMode]`**（B5）— 多軸退化可同時記錄
> 6. **stringly-typed 收斂**（I2）— `info_notes` / `warnings` 改用 `InfoCode` / `WarningCode` enum
> 7. **T 樣本長度分層**（I5）— T < 20 raise，20 ≤ T < 30 warning
> 8. **ADF 限定 CONTINUOUS**（I6）— SPARSE dummy 不做 persistence test
> 9. **factory kwargs 顯式化**（I1）— 移除 `**kwargs`，列出 forwarded fields
> 10. **`suggest_config` reasoning 補 `metric` 鍵**（I4）
> 11. **`describe_analysis_modes(format=...)` Pythonic 介面**（I3）
>
> **v3 changes from v2**：
> 1. **版本號修正**：v0.4 → **v0.5**（無中間階段，不跳號到 v0.6）
> 2. **移除所有 backward compat 機制**（無既有用戶要遷移）：
>    - 移除 alias / `DeprecationWarning` / 雙名並存
>    - 移除 `statistical_version` global flag
>    - 移除 `fl.compare_bhy_versions()` migration tool
>    - 移除 `factrix-codemod` 自動 rewrite（內部 50+ 檔人工 + sed/grep 解決）
> 3. **Phase 收斂為單一 rip-and-replace**（v0.5 一次到位）
> 4. **部分 power features 移到 §11 Deferred**（保持 v0.5 surface 精簡）：
>    - `regime_labels` hook
>    - `benchmark_factors` 進 AnalysisConfig（spanning test 仍在 multi_factor namespace 但獨立 API）
>    - `(COMMON, CONTINUOUS, FM)` 第 6 tuple — v0.5 維持 5 tuple，後續 friction log 累積再加

---

## 0. Scope Clarification（核心定位）

> **factrix 是 single-factor predictive significance validator，配套 multi-factor collection-level diagnostic**。一句話 framing：**對單一 factor 答「significant 嗎」，對多 factor 答「overlap 嗎、誰被誰 span 嗎、哪些假陽性要剔嗎」；不答「最佳 weights 是什麼、composite signal 怎麼建」。**

### 0.1 兩個正交的 scope 軸

|  | **Single-factor**（per-factor verdict） | **Multi-factor (collection-level diagnostic)** |
|---|---|---|
| **Mode A — Panel** (N ≥ 2) | alphalens-style 選股驗證、FM-λ risk premium、event study CAAR | BHY 跨 factor FDR、`redundancy_matrix`、spanning test |
| **Mode B — Single-asset** (N = 1) | market timing TS β、single-name event response、單 ETF 策略訊號驗證 | 多 signal 對單資產：BHY 控 FDR、`redundancy_matrix` 看 RSI/MACD/momentum 冗餘度、spanning vs naïve benchmark |

**四個 cell 全是 first-class** — 沒有「primary mode」與「fallback mode」之分。

### 0.2 Single-factor 為 computation primitive

- 每次 `evaluate()` 接受一個 factor，回一個 Profile，產一個 `primary_p` + verdict
- Multi-factor 操作（BHY、redundancy、spanning、orthogonalization）為**對 Profile collection 的 post-hoc diagnostic**
- 拒絕重新定位為 multi-factor framework：joint regression / composite signal 屬不同 paradigm，已有成熟 lib（`linearmodels`, `statsmodels`），factrix 切入會 reinvent inferior wheel

### 0.3 Mode A / Mode B 對等不互為 fallback

| 維度 | Mode A (Panel) | Mode B (Timeseries) |
|---|---|---|
| 研究問題 | 多資產截面 / 廣播訊號下，factor 有 predictive power 嗎？ | 對這支特定資產，factor 是否能預測未來報酬，足以建構交易策略？ |
| 聚合方式 | cross-sectional / cross-asset / cross-event aggregation | time-series aggregation (NW HAC, blocked OOS, rolling β stability) |
| 主要統計產出 | rank IC、FM-λ、CAAR、cross-asset β distribution | TS β + NW HAC t-stat、OOS Sharpe、rolling β 穩定性、event response per asset |
| 典型用戶 | 選股 quant、跨國配置、event study、macro risk premium | market timing、single-name swing、ETF 策略、retail quant |
| `primary_p` 行為 | 真實值，標準 verdict | 真實值，標準 verdict |

兩 mode 共用設計哲學（structured `primary_p` + diagnose、BHY 同 family group），統計機制適配各自情境。

---

## 1. 動機

### 1.1 v0.4 設計債

當前 `factor_type` enum 將三件正交資訊塞進單一字串：

| 真正的維度 | 應有 | v0.4 實際 |
|---|---|---|
| Factor scope (cross-sectional / common factor) | 2 值 | 部分編碼（混進 type 名） |
| Signal type (continuous / sparse) | 2 值 | 部分編碼 |
| Metric (IC / FM-λ) | 在 (INDIVIDUAL, CONTINUOUS) 內 2 值 | **被偷渡為兩個 type**（`cross_sectional` vs `macro_panel`） |

後果：命名不一致、`(COMMON, SPARSE)` coverage hole、`macro_*` 字串誤導、`canonical_p` 命名語意模糊。

### 1.2 Mode B 設計過度防禦

當前 `(COMMON, *, N=1)` 強制 `primary_p = 1.0`。實務後果：
- 大盤擇時研究員（Mode B 典型用戶）永遠拿不到 PASS
- 必須自己讀 `ts_beta_tstat` → 違反「`verdict()` 是 single source of truth」
- AI agent 看到 FAILED 會剔除真實有效訊號（誤殺）

為防 Type I error 引入巨量 Type II error，且建立在「Mode B 是 fallback」的錯誤前提上。

`(INDIVIDUAL, SPARSE, N=1)` 同樣有問題：HHI 自動停用、無 serial-corr 檢測 → 拔掉警報器，event window 重疊時嚴重低估 SE。

### 1.3 多/單因子定位張力

當前架構暗示性地涵蓋多因子能力（BHY、redundancy、spanning、orthogonalization），但 README/docstring 從未明說 factrix 是「single-factor primary + multi-factor diagnostic」。使用者讀起來像 multi-factor framework，期待錯位（期望 joint regression、composite signal — factrix 都不做）。§0 解決此張力。

### 1.4 其他發現（UX / AI agent 視角）

- `canonical_p` 命名語意模糊（見 §3）
- 缺非線性 / joint regression / composite signal / portfolio attribution 的明確 non-goal 聲明
- `metadata.reason = insufficient_*` 是 stringly-typed，AI agent 難以 pattern-match
- AI agent schema 應是 single-config + flat enum fields

---

## 2. 設計變更總覽（v0.4 → v0.5）

| 維度 | v0.4 (current) | **v0.5 (target)** |
|---|---|---|
| 主要抽象 | `factor_type` enum (4 strings) | `AnalysisConfig` factory methods（4 個 constructor） |
| Type strings | `cross_sectional` / `macro_panel` / `event_signal` / `macro_common` | _(廢除字串，改為 factory methods)_ |
| Config 類別 | 4 個 (`CrossSectionalConfig` etc.) | 1 個 (`AnalysisConfig` with `FactorScope` / `Signal` / `Metric` enum) |
| Verdict gate p | `canonical_p` | `primary_p` |
| BHY group key | `factor_type` | `(scope, signal, metric)` tuple — automatic |
| Coverage holes | (COMMON, SPARSE) 無歸屬 | `(COMMON, SPARSE)` 為合法 cell |
| Mode B (N=1) verdict | `primary_p` 強制壓 `1.0` | 真實值 + `mode="timeseries"` 中性標示 |
| Single-asset 定位 | 「fallback / 邊界 / best-effort」 | first-class Mode B（與 Mode A 對等） |
| (INDIVIDUAL, SPARSE, N=1) 警報 | HHI 停用、無 serial-corr 檢測 | NW HAC SE + Ljung-Box + event-overlap warning + actionable hint |
| Exception 階層 | 單一 `ValueError` / `ConfigError` | `IncompatibleAxisError` / `MissingMetricError` / `RedundantMetricError` / `ModeAxisError` 含 `suggested_fix` 屬性 |
| Multi-factor diagnostic API | 散落 | 集中於 `multi_factor` namespace |
| Non-goals | 部分聲明 | 補：joint multi-factor regression、composite signal、portfolio attribution、非線性 & feature interaction |
| **Backward compat** | — | **無**（單一 rip-and-replace；無 alias、無 deprecation cycle） |

---

## 3. `canonical_p` → `primary_p`

### 3.1 改名動機

- "canonical" 在統計文獻中指「規範的、典型的」（如 canonical correlation），factrix 用法是「**verdict gate 的主 p**」— 語意 mismatch
- "primary" 直接傳達「主 p / 次 p」（其他 p 走 `diagnose()` 為 secondary）
- `secondary diagnostic` 形成自然對稱命名

> **替代方案討論**（採納 quant review #3）：考慮過 `gate_p` / `headline_p` / `verdict_p`。`gate_p` 最精準，但 `primary_p` 與 `secondary diagnostic` 的對稱命名仍最易理解。最終採 `primary_p`，不再 bikeshed。

### 3.2 影響範圍（單次 rename）

| 識別字 | rename to |
|---|---|
| `Profile.canonical_p` (property) | `Profile.primary_p`（unified `FactorProfile` 的具名欄位，§4.4.2） |
| `CANONICAL_P_FIELD: ClassVar[str]` | **刪除**（A2：stringly-typed 雙重來源；統一走 `profile.primary_p`） |
| `BHY.p_source="canonical_p"` (string) | 改為 `bhy(gate: StatCode \| None = None)`（C1：type-safe、與 `verdict(gate=)` 命名一致） |
| `describe_profile_values` header | `primary_p` |
| README、CHANGELOG、ARCHITECTURE.md、docs/ | 全文 rename |
| Test fixture / assertion | 全文 rename |

無 alias、無 deprecation warning — 一次切完。

**`primary_p` 與 verdict gate 的關係（C1）**：
- `primary_p`：每 procedure 寫死的 canonical gate p；不可被 user 覆寫；BHY 預設用此值跨 family 比較
- verdict gate policy：`verdict(threshold=, gate=)` / `bhy(threshold=, gate=)` 的 `gate` 參數可指向任一 `StatCode`；`primary_p` 仍是 SSOT，gate 只是 user 選擇 read 哪個 stat 做 cutoff 判斷

---

## 4. `AnalysisConfig` 三軸正交設計

### 4.1 軸定義

```python
class FactorScope(StrEnum):
    INDIVIDUAL = "individual"             # each asset has own value; aka asset-specific / idiosyncratic / cross-sectional
    COMMON     = "common"                 # one value common to all assets; aka common factor / shared / broadcast

class Signal(StrEnum):
    CONTINUOUS = "continuous"
    SPARSE     = "sparse"                 # {-1, 0, +1} triggers; aka event-based

class Metric(StrEnum):
    IC = "ic"                            # rank-based (Spearman) — robust to outliers
    FM = "fm"                            # OLS slope (Fama-MacBeth) — unit-of-exposure premium

class Mode(StrEnum):                     # derived from N at evaluate-time, not user-set
    PANEL      = "panel"                 # N ≥ 2 (panel data, multi-asset)
    TIMESERIES = "timeseries"             # N = 1 (single-asset time series)
```

`Mode` **不是 user-facing 軸** — 由 raw data 的 N 自動推導，暴露為 `profile.mode` 唯讀屬性。

### 4.2 Factory Methods（接受 backend review #1；I1 顯式 kwargs / B1 驗證下移）

不採用 `metric: Metric | None` runtime validation。改用 4 個 type-safe factory methods，所有 axis 驗證統一在 `__post_init__`（factories 為純便利建構子）：

```python
@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    scope: FactorScope
    signal: Signal
    metric: Metric | None
    forward_periods: int = 5
    # 其他 evaluate-time fields（preprocess flags 等）依現有 v0.4 欄位 1:1 搬遷，
    # 不在本 plan 重新設計；factory signature 同步加上對應 keyword-only 參數

    def __post_init__(self) -> None:
        """Single source of truth for axis validation.
        Reachable via factories, direct construction, AND from_dict."""
        _validate_axis_compat(self.scope, self.signal, self.metric)

    @classmethod
    def individual_continuous(
        cls, *, metric: Metric = Metric.IC, forward_periods: int = 5,
    ) -> Self:
        """Per-(date, asset) continuous factor.
        metric=IC for rank predictive ordering;
        metric=FM for unit-of-exposure premium (Fama-MacBeth)."""
        return cls(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, metric,
                   forward_periods=forward_periods)

    @classmethod
    def individual_sparse(cls, *, forward_periods: int = 5) -> Self:
        """Per-(date, asset) sparse trigger ({-1, 0, +1}).
        Mode A canonical: CAAR cross-event t-test.
        Mode B (N=1): TS dummy regression + NW HAC SE (see §5.2)."""
        return cls(FactorScope.INDIVIDUAL, Signal.SPARSE, None,
                   forward_periods=forward_periods)

    @classmethod
    def common_continuous(cls, *, forward_periods: int = 5) -> Self:
        """Broadcast continuous factor (e.g. VIX).
        Canonical: per-asset β → cross-asset t-test on E[β]."""
        return cls(FactorScope.COMMON, Signal.CONTINUOUS, None,
                   forward_periods=forward_periods)

    @classmethod
    def common_sparse(cls, *, forward_periods: int = 5) -> Self:
        """Broadcast sparse trigger (FOMC, policy, index rebalance).
        Mode A canonical: per-asset β on dummy + cross-asset t-test.
        Mode B (N=1): TS dummy regression + NW HAC SE (see §5.2)."""
        return cls(FactorScope.COMMON, Signal.SPARSE, None,
                   forward_periods=forward_periods)
```

> **factory signature 範圍說明**：v0.5 重構不重新設計 preprocess / windowing fields；以現有 v0.4 `AnalysisConfig` 同名欄位 1:1 搬遷為原則。實作 §8 時，先盤點現有欄位再決定是否補進 factory keyword args（每個欄位一個 keyword-only parameter，不再用 `**kwargs`）。

> **Note**：v3 維持 5 tuple（含 `(INDIVIDUAL, CONTINUOUS, IC)` 與 `(INDIVIDUAL, CONTINUOUS, FM)` 兩 metric）；`(COMMON, CONTINUOUS, FM)` 移到 §11 Deferred — quant review #1 指出跨國 ETF FM-λ 是真實情境，但 v0.5 先穩 5 tuple，friction log 累積再加。

### 4.3 5 個合法 `(scope, signal, metric)` tuple

| Factory call | Tuple | Canonical procedure |
|---|---|---|
| `AnalysisConfig.individual_continuous(metric=IC)` (default) | `(INDIVIDUAL, CONTINUOUS, IC)` | per-date Spearman ρ → NW HAC t-test on `E[IC]` |
| `AnalysisConfig.individual_continuous(metric=FM)` | `(INDIVIDUAL, CONTINUOUS, FM)` | per-date OLS λ → NW HAC t-test on `E[λ]` |
| `AnalysisConfig.individual_sparse()` | `(INDIVIDUAL, SPARSE, None)` | per-event AR → CAAR → cross-event t-test |
| `AnalysisConfig.common_continuous()` | `(COMMON, CONTINUOUS, None)` | per-asset β → cross-asset t-test on `E[β]` |
| `AnalysisConfig.common_sparse()` | `(COMMON, SPARSE, None)` | per-asset β on sparse dummy → cross-asset t-test |

### 4.4 Registry-based dispatch（A1 SSOT；B3 單一 Profile + procedure callable）

> **v0.5 範圍（C4）**：registry / procedure / dispatch key 全部以 underscore 前綴維持**內部使用**（`factrix._registry`、`factrix._procedures`）。`FactorProcedure` Protocol、`register_procedure()`、`FactorProfile` constructor 暫不對外 export，README 不行銷「可擴充」。v0.6+ 累積外部 plug-in 需求後升級為公開 `factrix.extension` namespace（去 underscore 即升級）。設計就緒、API 穩定承諾延後。


#### 4.4.1 Registry 為「合法 cell」的單一來源

v3.1 草案讓 `_DISPATCH_REGISTRY` / `_validate_axis_compat` / 4 個 factory / `describe_analysis_modes` / `suggest_config` 各自編碼一份「5 tuple 是合法的」知識 — 6 處同義重複。v3.2 收斂：**registry 為 SSOT**，其他模組反查。

```python
@dataclass(frozen=True, slots=True)
class _DispatchKey:
    scope: FactorScope | _ScopeCollapsedSentinel  # § 5.4.1 collapse 用
    signal: Signal
    metric: Metric | None
    mode: Mode

@dataclass(frozen=True)
class _RegistryEntry:
    key: _DispatchKey
    procedure: FactorProcedure                     # 見 4.4.2
    input_schema: InputSchema
    canonical_use_case: str                        # describe_analysis_modes 直接讀
    references: tuple[str, ...]                    # 文獻引用

_DISPATCH_REGISTRY: dict[_DispatchKey, _RegistryEntry] = {}

def _validate_axis_compat(scope, signal, metric) -> None:
    """A1: query registry, no parallel rule table."""
    if not any(
        e.key.scope in (scope, _SCOPE_COLLAPSED) and
        e.key.signal == signal and e.key.metric == metric
        for e in _DISPATCH_REGISTRY.values()
    ):
        raise IncompatibleAxisError(...)
```

`describe_analysis_modes()` / `suggest_config()` / 5-tuple round-trip 測試 全部 iterate `_DISPATCH_REGISTRY` — 不再手寫表。新增 cell 只動 registry 一處。

#### 4.4.2 Procedure callable 取代 Profile class proliferation（B3）

v3.1 草案：1 dispatch key → 1 Profile dataclass，最多 8–10 個 class，差異只在「呼叫哪個 procedure」與 `PRIMARY_P_FIELD`。v3.2 改為：

```python
class FactorProcedure(Protocol):
    """Pure compute: raw -> populated FactorProfile."""
    INPUT_SCHEMA: ClassVar[InputSchema]
    def compute(self, raw: RawData, config: AnalysisConfig) -> FactorProfile: ...

class StatCode(StrEnum):
    """Cell-specific scalar stats. Add new metric → add enum value + procedure populates."""
    IC_MEAN            = "ic_mean"
    IC_T_NW            = "ic_t_nw"
    IC_P               = "ic_p"
    FM_LAMBDA_MEAN     = "fm_lambda_mean"
    FM_LAMBDA_T_NW     = "fm_lambda_t_nw"
    FM_LAMBDA_P        = "fm_lambda_p"
    TS_BETA            = "ts_beta"
    TS_BETA_T_NW       = "ts_beta_t_nw"
    TS_BETA_P          = "ts_beta_p"
    CAAR_MEAN          = "caar_mean"
    CAAR_T_NW          = "caar_t_nw"
    CAAR_P             = "caar_p"
    FACTOR_ADF_P       = "factor_adf_p"
    LJUNG_BOX_P        = "ljung_box_p"
    EVENT_TEMPORAL_HHI = "event_temporal_hhi"
    NW_LAGS_USED       = "nw_lags_used"

@dataclass(frozen=True, slots=True)
class FactorProfile:
    """Unified profile — one dataclass; cell-specific stats live in `stats` dict.

    Schema does NOT grow when new metrics/cells are added.
    primary_p is a named field (A2: no PRIMARY_P_FIELD string indirection).
    """
    # Cross-cell common fields
    config:     AnalysisConfig
    mode:       Mode
    primary_p:  float                                 # procedure-canonical gate p
    n_obs:      int                                   # T or n_events depending on cell
    warnings:   frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode]    = frozenset()

    # Cell-specific stats — populated by procedure, keyed by StatCode (typed dict)
    stats:      Mapping[StatCode, float] = field(default_factory=dict)

    def verdict(
        self,
        *,
        threshold: float = 0.05,
        gate: StatCode | None = None,
    ) -> Verdict:
        """Verdict policy (C1).

        threshold: gate cutoff (default 0.05). Generic — not tied to Type I
                   error semantics, since gate may be non-p stat in future.
        gate: which stat to gate on. Default None → use procedure-canonical
              `primary_p`. Override with StatCode for custom policy.
              KeyError if requested gate not populated for this profile.
        """
        p = self.primary_p if gate is None else self.stats[gate]
        return Verdict.PASS if p < threshold else Verdict.FAIL

    def diagnose(self) -> dict[str, Any]:
        """Secondary stats + warnings + info notes for human/AI introspection."""
        ...
```

Procedure 範例：

```python
class _ICContPanelProcedure:
    INPUT_SCHEMA = ...
    def compute(self, raw, config):
        ic_series = ...
        T = len(ic_series)
        nw_lags = auto_bartlett(T)
        ic_mean, ic_t = nw_t(ic_series, lags=nw_lags)
        ic_p = two_sided_t_p(ic_t, df=T - 1)
        return FactorProfile(
            config=config, mode=Mode.PANEL,
            primary_p=ic_p, n_obs=T,
            stats={
                StatCode.IC_MEAN: ic_mean,
                StatCode.IC_T_NW: ic_t,
                StatCode.IC_P: ic_p,
                StatCode.NW_LAGS_USED: nw_lags,
            },
        )

class _TSDummyRegressionProcedure:
    """Shared by (*, SPARSE, None, TIMESERIES) — see §5.4.1."""
    INPUT_SCHEMA = ...
    def compute(self, raw, config): ...
```

Registry 只註冊 procedure instance：

```python
register(_DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL),
         _ICContPanelProcedure(), use_case=..., refs=...)
register(_DispatchKey(_SCOPE_COLLAPSED, Signal.SPARSE, None, Mode.TIMESERIES),
         _TSDummyRegressionProcedure(), use_case=..., refs=...)
```

`evaluate()` 流程：
1. config + raw → 推導 `mode`
2. 若 `signal=SPARSE` 且 `mode=TIMESERIES` → scope 改寫為 `_SCOPE_COLLAPSED`（§5.4.1）
3. 組 `_DispatchKey` → registry 查 entry
4. `entry.procedure.compute(raw, config)` → `FactorProfile`

新增 cell：寫一個 `Procedure` class + 一行 `register()`。Profile dataclass 不變。減少 class explosion，所有 cell stats 集中在單一 schema，BHY / verdict gate 跨 cell 取值零分支。

### 4.5 Exception 階層 + structured fix hint（B2 收斂為 3 子類）

v3.1 草案有 5 個 ConfigError 子類，其中 `MissingMetricError` / `RedundantMetricError` 都只是「axis-metric 不合」的特例 — `IncompatibleAxisError` + `suggested_fix` 已足夠表達。v3.2 收斂：

```python
class FactrixError(Exception):
    """Base for all factrix errors."""

class ConfigError(FactrixError):
    """Base for AnalysisConfig validation errors."""
    suggested_fix: AnalysisConfig | None = None

class IncompatibleAxisError(ConfigError):
    """Any (scope, signal, metric) tuple not in registry.
    Covers: signal=SPARSE w/ metric set; (COMMON, CONTINUOUS) w/ metric set;
    (INDIVIDUAL, CONTINUOUS) w/ metric=None; etc.
    Reachable via direct construction or from_dict (factories never trigger).
    suggested_fix populated from _FALLBACK_MAP when nearest-legal cell exists."""

class ModeAxisError(ConfigError):
    """e.g. (INDIVIDUAL, CONTINUOUS, N=1) — undefined at N=1 (no cross-sectional dispersion).
    Raised at evaluate-time (mode is not part of AnalysisConfig).
    suggested_fix from _FALLBACK_MAP (e.g. common_continuous(...))."""

class InsufficientSampleError(ConfigError):
    """T < MIN_T_HARD for Mode B procedures (§5.2).
    Below floor NW HAC SE too biased for meaningful primary_p.
    Raised at evaluate-time. suggested_fix=None (data limitation, not config error)."""
```

**`_FALLBACK_MAP` 集中規則（A4）**：

```python
_FALLBACK_MAP: dict[tuple[FactorScope, Signal, Mode], Callable[[], AnalysisConfig]] = {
    (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Mode.TIMESERIES):
        lambda: AnalysisConfig.common_continuous(),
    # ... 新增 fallback 改一處
}
```

raise 點查表，不在每個 raise site 重複「該轉哪個 cell」的知識。

**驗證單一入口**：所有 axis-compat 檢查集中於 `_validate_axis_compat()`，由 `AnalysisConfig.__post_init__` 呼叫，內部反查 registry（A1）。Mode-axis 與 sample-size 檢查在 `evaluate()` 流程內（需 raw data 才知 N、T）。

### 4.6 `from_dict()` / `to_dict()` round-trip（接受 backend review #6）

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "scope":  self.scope.value,
        "signal":        self.signal.value,
        "metric":    self.metric.value if self.metric else None,
        "forward_periods": self.forward_periods,
        ...
    }

@classmethod
def from_dict(cls, d: dict[str, Any]) -> Self:
    return cls(
        scope=FactorScope(d["scope"]),
        signal=Signal(d["signal"]),
        metric=Metric(d["metric"]) if d.get("metric") else None,
        ...
    )
```

**測試強制**：
1. 5 個合法 tuple 必過 `assert config == AnalysisConfig.from_dict(config.to_dict())`
2. **invalid tuple 必經 `from_dict` 觸發對應 ConfigError**（驗證 B1：`__post_init__` 為單一 source of truth）。
   例如 `from_dict({"scope":"individual","signal":"sparse","metric":"ic"})` → `IncompatibleAxisError`。

---

## 5. Mode A / Mode B 對等支援

### 5.1 設計原則

- Mode B (single-asset, N=1) 是 first-class 正當 scope，與 Mode A 對等
- 拒絕 `primary_p` 強制壓 1.0、拒絕 "degraded" 用語
- 兩 mode 共用：structured `primary_p` + `verdict()` + `diagnose()` + BHY pool eligibility
- 兩 mode 不共用：底層 procedure（cross-sectional aggregation vs time-series aggregation）

### 5.2 Mode B 完整 procedure 設計

**Stats 常數集中（A3）**：以下常數集中於 `factrix._stats.constants`，禁止 literal 散落：

```python
MIN_T_HARD     = 20    # < this → InsufficientSampleError
MIN_T_RELIABLE = 30    # < this → WarningCode.UNRELIABLE_SE_SHORT_PERIODS
def auto_bartlett(T: int) -> int:
    """Newey-West (1994) automatic lag: floor(4 * (T/100)^(2/9))."""
    return max(1, int(4 * (T / 100) ** (2 / 9)))
```

**T 樣本長度分層（I5）**：所有 Mode B procedure 適用同一 floor：
- `T < MIN_T_HARD` → `InsufficientSampleError`（不出 verdict）
- `MIN_T_HARD ≤ T < MIN_T_RELIABLE` → 出 verdict + `WarningCode.UNRELIABLE_SE_SHORT_PERIODS`
- `T ≥ MIN_T_RELIABLE` → 標準 verdict

#### `(COMMON, CONTINUOUS, N=1)` — Single-asset continuous broadcast factor

```python
# Bartlett 自動 lag: lag = floor(4 * (T/100)^(2/9))
ts_beta, ts_beta_se_nw = ols_with_newey_west(y_t, factor_t, lags=auto_bartlett(T))
ts_beta_tstat = ts_beta / ts_beta_se_nw
primary_p     = two_sided_t_p(ts_beta_tstat, df=T - 2)

profile.primary_p = primary_p           # 真實值
profile.mode      = Mode.TIMESERIES     # 中性標示
```

附 diagnose（**ADF 限定 CONTINUOUS — I6**）：
- `factor_adf_p` 自動納入（Stambaugh-style persistent regressor 警告）
- ADF p > 0.1 → `WarningCode.PERSISTENT_REGRESSOR`

#### `(*, SPARSE, N=1)` — Single-asset sparse trigger（B3 共用 procedure / B4 primary_p 明定）

兩個 tuple `(INDIVIDUAL, SPARSE, None, TIMESERIES)` 與 `(COMMON, SPARSE, None, TIMESERIES)` 共用此 procedure（§4.4 `_ts_dummy_regression`）。

**`primary_p` 來源（B4 明定）**：calendar-time TS dummy regression β 的 NW HAC t-stat — **非** event-time CAAR：

```python
# y_t: T-period return series; d_t: {-1, 0, +1} sparse trigger at calendar time t
beta, beta_se_nw = ols_with_newey_west(
    y_t, d_t,
    lags=auto_bartlett(T),               # 統一 Bartlett rule，與 (COMMON, CONTINUOUS) 一致
)
beta_tstat = beta / beta_se_nw
primary_p  = two_sided_t_p(beta_tstat, df=T - 2)
```

> **理由**：N=1 下 cross-event AR 聚合自由度依賴 `n_events`（事件數，可能很少），而 calendar-time 序列回歸的自由度依賴 `T`（總時序長度），後者更穩。CAAR SE（NW on per-event AR series）保留為 secondary diagnostic 欄位（`caar_mean` / `caar_se_nw`）走 `diagnose()`；不再參與 `primary_p`。Mode A `(INDIVIDUAL, SPARSE, Mode.PANEL)` 仍用 CAAR cross-event t-test（事件數跨資產夠多，原 procedure 適用）。

附四層警報（不再依賴 `n_events` 計算 `primary_p`，但仍作 diagnostic）：
1. **Event window overlap** — `min(dt_between_events) < 2 * window_length` → `WarningCode.EVENT_WINDOW_OVERLAP`
2. **Ljung-Box on residual ε_t** — `lags=min(10, T//10)`；p < 0.05 → `WarningCode.SERIAL_CORRELATION_DETECTED`
3. **`event_temporal_hhi`**（放 `profile.stats[StatCode.EVENT_TEMPORAL_HHI]`，由 `diagnose()` 暴露；不污染 top-level schema）— 衡量同一 asset 事件在時間軸上的集中程度
4. **T < MIN_T_RELIABLE** → `WarningCode.UNRELIABLE_SE_SHORT_PERIODS`（T < MIN_T_HARD 已在 InsufficientSampleError 攔截）

### 5.3 Mode B 適用情境（明確 IN scope）

| 情境 | factrix 角色 | 不負責的部分 |
|---|---|---|
| Market timing（SPY × VIX、TAIEX × USD/TWD） | TS β + NW HAC verdict、ADF persistence、rolling β stability | 實際 backtest（用 `vectorbt`） |
| Single-name swing trading（個股 × technical signal） | per-signal verdict、跨 signal redundancy、BHY 跨 signal FDR | execution / slippage / 部位管理 |
| Single ETF strategy（QQQ × momentum、TLT × yield curve） | TS β、event response（FOMC 等共用 dummy） | 投組組成 |
| Retail quant 多 signal 篩選（單股 N=1，多個 indicator） | 跨 signal collection-level diagnostic | composite signal generation |

### 5.4 Mode 路由表

| Tuple | Mode A (N≥2) 行為 | **Mode B (N=1) 行為** |
|---|---|---|
| `(INDIVIDUAL, CONTINUOUS, IC)` | 標準 IC + NW | **raise `ModeAxisError`**，`suggested_fix=common_continuous(...)` |
| `(INDIVIDUAL, CONTINUOUS, FM)` | 標準 FM-λ + NW | 同上 |
| `(INDIVIDUAL, SPARSE)` | 標準 CAAR + cross-event t-test (`primary_p` from CAAR) | **collapsed** → `(_, SPARSE, None, TIMESERIES)` 共用 entry（§5.4.1） |
| `(COMMON, CONTINUOUS)` | 標準 cross-asset t-test on E[β] | NW HAC TS β t-test + persistence diagnose |
| `(COMMON, SPARSE)` | 標準 cross-asset on β with dummy | **collapsed** → `(_, SPARSE, None, TIMESERIES)` 共用 entry（§5.4.1） |

#### 5.4.1 Mode B sparse collapse（B1 真退化）

N=1 下 `INDIVIDUAL` 與 `COMMON` 在 SPARSE 訊號上 **數學完全等價**（無 cross-section → 兩者皆退化為 §5.2 `_ts_dummy_regression`）。v3.1 草案保留兩 registry entry 共用 procedure，但 BHY family key 仍以 scope 區分 — 這會把同 procedure 產出的 p 值人為切兩 family，破壞 step-up FDR power。

**v3.2 修正**：`evaluate()` 在偵測 `signal=SPARSE` 且 `N=1` 時：
1. registry 路由 key 改寫為 `(_SCOPE_COLLAPSED_, SPARSE, None, TIMESERIES)`（單一 entry，sentinel 取代 user 傳入的 scope）
2. emit `InfoCode.SCOPE_AXIS_COLLAPSED`（告知 user 其 scope 選擇在此 cell 已被 collapse）
3. BHY family key 同樣使用該 sentinel — 兩個 user-facing factory 產出的 profiles 進**同一 family**

訊息：「`scope` axis is degenerate under Mode B sparse; both `INDIVIDUAL` and `COMMON` route to the unified single-asset sparse procedure with identical statistics and BHY family.」

> **設計動機**：避免 hidden coupling — user 選擇 `individual_sparse` vs `common_sparse` 在 N=1 下對統計輸出零影響、對 BHY 行為也零影響。scope 軸是真退化，不是 cosmetic 標籤。

### 5.5 Mode 不可路由情境

`(INDIVIDUAL, CONTINUOUS, N=1)` 數學上不存在（無 cross-sectional dispersion → IC undefined、per-date OLS undefined）。raise `ModeAxisError`，附 `suggested_fix=common_continuous(...)`（在 N=1 下為唯一合法 CONTINUOUS 路徑）。

> **不採用「自動轉 (INDIVIDUAL, CONTINUOUS) → (COMMON, CONTINUOUS) at N=1」**：silent scope 改寫違反「strict gate, no silent fallback」設計哲學。raise + suggested_fix 是 explicit user-correctable 的 happy path。
>
> **對比 SPARSE 路徑**：`(INDIVIDUAL, SPARSE, N=1)` 與 `(COMMON, SPARSE, N=1)` 皆合法路由（§5.4），因 sparse dummy 在 N=1 下 procedure 良定義；只是兩者統計等價，故 emit info 而非 raise。

### 5.6 BHY family key（B2 — 不混 Mode A / Mode B；B1 — Mode B sparse 不分 scope）

BHY step-up FDR 假設 family 內 p 值由可比的 procedure 產生。Mode A（cross-sectional / cross-event aggregation）與 Mode B（time-series aggregation）p 值機制不同（不同 null distribution、不同 effective sample），混在同一 family 會破壞 step-up 的 dependence 控制。

**Family key = registry key** — 直接重用 §4.4 dispatch key，保證「同 procedure 同 family」：
- Mode A：`(scope, signal, metric, Mode.PANEL)` — 4 種 panel cell
- Mode B continuous：`(COMMON, CONTINUOUS, None, Mode.TIMESERIES)`
- Mode B sparse：`(_SCOPE_COLLAPSED_, SPARSE, None, Mode.TIMESERIES)`（§5.4.1）

`fl.multi_factor.bhy()` 自動依此 key 分群，每群獨立做 step-up；不同 key 的 profiles **不**進同一 BHY pool。

範例：對單一 N=1 標的跑 10 個 sparse triggers（混 individual / common factory）+ 5 個 continuous timing factors → 自動分為兩 family：
- `(_SCOPE_COLLAPSED_, SPARSE, None, TIMESERIES)` — 10 個 sparse 全進此 family（不再依 user 的 scope 選擇切兩半）
- `(COMMON, CONTINUOUS, None, TIMESERIES)` — 5 個 continuous

> 跨 family 比較交給 user：plan 不在 `bhy()` 內做 cross-family aggregation，避免 hidden assumption。

---

## 6. 「明確不做」更新

### 6.1 非線性 & 特徵交互作用（NEW）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **非線性與特徵交互作用 (Non-linearity & Feature Interactions)** | factrix 底層為 Spearman 秩相關 (IC) / OLS 迴歸 (FM, TS) / t-test，皆線性。條件式（"VIX>20 時 momentum 才有效"）、threshold effect、跨 factor 交互作用無法直接表達。 | `xgboost` / `lightgbm` + `shap` 做特徵重要性與交互作用；`causalml` 做條件處理效應；自寫 regime-conditioned slice 切資料後再餵 factrix |

### 6.2 Joint multi-factor regression（NEW）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **Joint multi-factor regression**（一個 model 多個 x，給聯合 β / 共線性 / regularization） | factrix 是 single-factor primary（§0.2）— 每 `evaluate()` 一個 factor 一個 verdict。多 factor 操作走 collection-level diagnostic（BHY、redundancy、spanning），不做聯合估計。 | `linearmodels` (PanelOLS, IV, FM)、`statsmodels`、`pyfinance`、`scikit-learn` (Lasso/Ridge) |

### 6.3 Composite signal generation（重申）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **跨 factor 合成信號 / factor combiner**（產生單一 composite signal） | 屬 signal layer，validator 不碰。`redundancy_matrix` 是 **diagnostic**，不是 combiner。 | 自寫、或 `scikit-learn` regression / GBM |

### 6.4 Portfolio attribution / risk decomposition（NEW）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **Portfolio attribution / risk decomposition** | 屬 portfolio analytics，與「factor 是否 significant」無關 | `riskfolio-lib` / Bloomberg PORT / Barra factor model |

### 6.5 Return-process modeling（NEW，取代原 v1 「single-asset framework 不擴張」錯誤聲明）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **Return-process modeling**（ARIMA / state-space / VAR / GARCH on return itself） | factrix 測「factor → return」relationship；return process 自身建模屬 TS forecasting paradigm。Mode B 仍 first-class，但只在「factor predicts return」框架內運作。 | `statsmodels` (ARIMA / state-space / VAR)、`arch` (GARCH / MIDAS)、`prophet` |

關鍵差別：factrix 完整負責 Mode B 的 factor → return 驗證；statsmodels 等用於 factrix 不做的 return process modeling，**不是** N=1 的替代。

### 6.6 User-level custom Metric / runtime Metric enum extension（C3 — NEW）

| 不做 | 理由 | 該用什麼 |
|---|---|---|
| **User runtime 擴展 `Metric` enum / 自訂 procedure** | `Metric` 是 procedure-canonical 集合；每個值對應一個有文獻引用、SE 性質明確的 procedure。允許 user runtime 擴展等於開放 untrusted procedure 進 verdict gate，統計性質無保證。 | (a) **stat variant**（weighted IC、bootstrapped CAAR）：將來走 `EvalConfig(se_method=...)` 正交軸；(b) **procedure variant**：見 §11 Deferred「Procedure registration as public extension API」（v0.6+ 候選） |

**辨識**：絕大多數「自訂 metric」需求其實是同 procedure 的 SE 變體（stat variant），不是新 procedure。要新 procedure 的場景由 developer 走 §11 deferred 的 extension API。

**Verdict gate override 不算 custom metric**：user 透過 `profile.verdict(gate=StatCode.X)` 換 gate stat 是 verdict policy 變更，不是新增 procedure；procedure-canonical `primary_p` 不變（§3.2）。

---

## 7. 其他 UX / AI Agent 友善優化

### 7.1 `fl.describe_analysis_modes()` 取代 `describe_factor_types()`

- 印出全部 5 個合法 tuple，每行附：canonical procedure、文獻引用、典型 use case、Mode A/B 行為差異
- `format: Literal["text", "json"] = "text"` 參數輸出 machine-readable schema（I3：Pythonic 介面，非 CLI flag）
- 收斂為單一查詢介面（取代 `describe_factor_types()` / `describe_profile()` 雙 entry）

### 7.2 `fl.suggest_config(raw)` auto-detect helper

```python
suggestion = fl.suggest_config(raw)
# Returns:
SuggestConfigResult(
    suggested = AnalysisConfig.common_continuous(forward_periods=5),
    reasoning = {
        "scope":  "factor varies across assets at given date: NO → COMMON",
        "signal": "sparsity ratio = 0.02 (≥ 0.5 threshold): NO → CONTINUOUS",
        "metric": "scope=COMMON: metric axis collapsed (no IC/FM choice)",   # I4
        "mode":   "n_assets = 1 detected → Mode B (timeseries)",
    },
    warnings  = [WarningCode.UNRELIABLE_SE_SHORT_PERIODS],   # I2 — enum, not str
)
```

- 不自動套用 — 印建議 + structured reasoning dict（不是自由文字 list）
- `reasoning` 必含 4 鍵：`scope` / `signal` / `metric` / `mode`（I4）；`metric` 在 collapse cell 給予 explicit 說明
- `warnings` 為 `list[WarningCode]`，AI agent 可 pattern-match
- AI agent 可作 first-pass tool call

### 7.3 `Mode` 與 `DegradedMode` 顯式分離（B5 frozenset / I2 enum-typed info）

三個正交 enum + 兩個 collection 欄位：

```python
class Mode(StrEnum):                     # sample regime — 中性
    """Sample regime, derived from N at evaluate-time.

    PANEL      = Mode A: N >= 2 (multi-asset / multi-event panel)
    TIMESERIES = Mode B: N == 1 (single-asset time series)

    A5: docstring is the SSOT for the 'Mode A/B' marketing label
    used in README, docs, and describe_analysis_modes() output.
    """
    PANEL      = "panel"
    TIMESERIES = "timeseries"

class WarningCode(StrEnum):              # procedure 退化警示（取代 v3 草案的 DegradedMode）
    INSUFFICIENT_EVENTS         = "insufficient_events"            # events < MIN_EVENTS
    INSUFFICIENT_ASSETS         = "insufficient_assets"            # N too small for cross-asset agg power
    UNRELIABLE_SE_SHORT_PERIODS  = "unreliable_se_short_periods"    # 20 ≤ n_periods < 30 → NW HAC SE 偏誤
    EVENT_WINDOW_OVERLAP        = "event_window_overlap"           # min(dt_between_events) < 2 * window_length
    PERSISTENT_REGRESSOR        = "persistent_regressor"           # ADF p > 0.1（CONTINUOUS only — I6）
    SERIAL_CORRELATION_DETECTED = "serial_correlation_detected"    # Ljung-Box p < 0.05

class InfoCode(StrEnum):                 # 中性事實提示（非 degradation、非 warning）
    SCOPE_AXIS_COLLAPSED = "scope_axis_collapsed"                  # §5.4 (*, SPARSE, N=1)
```

> **v3.1 變更**：`DegradedMode` 命名混淆 `Mode`（兩者不同層）— 改名 `WarningCode`，並從 enum-as-single-value 改為 `frozenset[WarningCode]` 收集（B5）。

Profile 欄位：

```python
profile.mode:       Mode                            # 單值 — 由 N 推導
profile.warnings:   frozenset[WarningCode]          # 可同時多軸退化
profile.info_notes: frozenset[InfoCode]             # 中性事實提示
```

AI agent pattern-match 範例：
```python
if WarningCode.PERSISTENT_REGRESSOR in profile.warnings: ...
if profile.warnings & {WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
                       WarningCode.SERIAL_CORRELATION_DETECTED}: ...
```

T < `MIN_T_HARD` 不入 `WarningCode` — 它是 `InsufficientSampleError`（不出 verdict）。`WarningCode` 只覆蓋「verdict 仍可信但要警惕」情境。

**B6：v0.5 不使用 Python `warnings.warn()` 體系** — 移除 v0.4 的 `UnreliableSESizeWarning` / `PersistentRegressorWarning` / `EventOverlapWarning` 等 `Warning` subclass。所有 procedure-level 警示統一走 `Profile.warnings: frozenset[WarningCode]`，user 顯示由 `Profile.summary()` 負責。動機：
- batch run / AI agent 場景，stdout warning 噪音是負擔
- frozenset 比 stderr 字串更便於 pattern-match 與聚合
- 保留單一通道，避免「有些走 warnings、有些走 frozenset」的雙軌歧義

### 7.4 `multi_factor` namespace API（明確規格）

| API | 輸入 | 輸出 | 責任邊界 |
|---|---|---|---|
| `fl.multi_factor.bhy(profileset, *, threshold=0.05, gate=None)` | `ProfileSet` | `ProfileSet` (subset) | FDR 校正後通過的 factors。`threshold` = FDR 上限；`gate=None` 用 `primary_p`（procedure-canonical），`gate=StatCode.X` 須在所有 family member 都有值 |
| `fl.multi_factor.redundancy_matrix(profileset)` | `ProfileSet` | DataFrame (N×N) | pairwise 重疊度量；diagnostic only |
| `fl.multi_factor.spanning_test(target, benchmarks)` | `Profile` + `[Profile]` | `SpanningResult` | target 控制 benchmarks 後是否仍有 alpha；採 Newey-West-adjusted SE |
| `fl.multi_factor.orthogonalize(target, against)` | factor data + benchmark factors | residualized factor data | 殘差化以餵入新一輪 `evaluate()` |

**命名一致性**：`threshold` / `gate` 兩個參數名跨 `Profile.verdict()` 與 `multi_factor.bhy()` 統一；`gate=None` 一致代表「用 procedure-canonical `primary_p`」。

> 既有 v0.4 的 `redundancy_matrix` / `ortho` config 機制移到此 namespace 下；spanning test 在 v0.5 新增為獨立 API（不走 AnalysisConfig.benchmark_factors，保持 evaluate 單純）。

### 7.5 命名一致性 invariants（v3.3）

實作 / review 強制檢查：

| 概念 | 統一名稱 | 禁止 |
|---|---|---|
| Verdict gate cutoff | `threshold: float`（不寫 `alpha`） | `alpha`、`level`、`cutoff`、`fdr`（FDR 仍走 `threshold`，函數名 `bhy` 已表達 FDR 語意） |
| Verdict gate stat selector | `gate: StatCode \| None`（`None` = procedure-canonical `primary_p`） | `p_source`（v0.4 字串）、`stat`、`metric`（與 axis enum 撞名） |
| Cross-cell common p | `primary_p` (named field) | `canonical_p`、`headline_p`、字串 `PRIMARY_P_FIELD` indirection |
| Cell-specific stats | `profile.stats[StatCode.X]` | top-level optional fields、`extras: dict[str, Any]` |
| Sample-size 欄位 | `n_obs: int`（dataclass 欄位）/ `T`（公式內局部變數） | `n`、`T_obs`、`length` 混用 |
| Time-series HAC lag | `nw_lags: int`（kwarg）/ `auto_bartlett(T)`（自動推導） | `lag`、`hac_lag`、`bandwidth` 混用 |
| Procedure 介面方法 | `compute(self, raw, config) -> FactorProfile` | `__call__`、`run`、`evaluate` |
| Code suffix | `WarningCode` / `InfoCode` / `StatCode`（一律 `*Code`） | `Warning`、`Info`、`Stat` 不帶 `Code` |
| Sentinel | `_SCOPE_COLLAPSED`（leading underscore + UPPER_SNAKE） | `ScopeCollapsed`、`COLLAPSED_SCOPE` |
| 內部 module / private | `factrix._registry` / `_procedures` / `_stats.constants` 全 underscore（C4） | 公開 `factrix.registry` 直到 v0.6+ |

新增 PR 若違反上述命名 → review block。

---

## 8. Implementation 計畫（單 phase rip-and-replace）

無既有用戶要遷移 → 不需 deprecation cycle、不需 alias、不需 codemod。一個 feature branch 內完成：

### 8.1 程式碼變動

- [ ] 新增 `FactorScope` / `Signal` / `Metric` / `Mode` / `WarningCode` / `InfoCode` / `StatCode` / `Verdict` enums（Mode docstring 帶 A/B 標籤 — A5）
- [ ] `factrix._stats.constants`：`MIN_T_HARD` / `MIN_T_RELIABLE` / `auto_bartlett` 集中（A3）
- [ ] 新增 `AnalysisConfig` dataclass + `__post_init__` axis 驗證 + 4 factory methods（顯式 kwargs，無 `**kwargs`）
- [ ] 新增 Exception 階層（`FactrixError` / `ConfigError` + 3 子類：`IncompatibleAxisError` / `ModeAxisError` / `InsufficientSampleError`）— B2 收斂
- [ ] `_FALLBACK_MAP` 集中 `suggested_fix` 規則（A4）
- [ ] **單一 `FactorProfile` dataclass**（B3：取代 N×M Profile class），cell-specific stats 統一走 `stats: Mapping[StatCode, float]`（v3.3：central schema 不隨 metric 線性膨脹），`primary_p` 為具名欄位（**無 `PRIMARY_P_FIELD` ClassVar** — A2）
- [ ] `Profile.verdict(threshold=0.05, gate=None)` / `multi_factor.bhy(threshold=0.05, gate=None)`（C1：分離 procedure-canonical `primary_p` 與 user-overrideable verdict policy；參數命名一致）
- [ ] **`FactorProcedure` Protocol + 每 cell 一個 procedure class**（B3）
- [ ] Registry SSOT（A1）：
  - `_DispatchKey(scope, signal, metric, mode)` + `_RegistryEntry`
  - `_validate_axis_compat()` 反查 registry，不另寫規則
  - `describe_analysis_modes()` / `suggest_config()` iterate registry
- [ ] **Mode B sparse collapse**（B1）：`evaluate()` 在 `(*, SPARSE, N=1)` 路徑將 scope 改寫為 `_SCOPE_COLLAPSED` sentinel；單一 registry entry；單一 BHY family
- [ ] Mode B procedure 完整實作：
  - NW HAC SE + auto-lag (Bartlett rule，A3 共用)
  - T 樣本長度分層：T<MIN_T_HARD raise / 中段 warn / 標準（I5）
  - `(*, SPARSE, N=1)` 走 calendar-time TS dummy regression（**非** CAAR）
  - Ljung-Box on residual ε_t
  - Event overlap detection + actionable hints
  - `event_temporal_hhi` → `profile.extras` + `diagnose()`（B5，**非** top-level 欄位）
  - ADF persistence diagnose（**僅 CONTINUOUS** — I6）
- [ ] `Profile.warnings: frozenset[WarningCode]` / `info_notes: frozenset[InfoCode]`（B5、I2）
- [ ] **移除 Python `Warning` subclass 體系**（B6）— 不再 `warnings.warn()`
- [ ] BHY family key = registry key — Mode A / Mode B 不混 family；Mode B sparse 不分 scope（B1 / B2）
- [ ] **移除 `BHY.p_source` 參數**（A2）— 一律取 `profile.primary_p`
- [ ] `multi_factor` namespace（migrate `redundancy_matrix` / `ortho`，新增 `spanning_test`）
- [ ] `from_dict` / `to_dict` + 5-tuple round-trip 測試 + invalid-tuple `from_dict` 觸發 `IncompatibleAxisError` 測試
- [ ] `describe_analysis_modes(format=...)` / `suggest_config()` helpers（reasoning 含 4 鍵；warnings 為 enum）— I3 / I4
- [ ] **每個 factory 產生的 config 必能在 registry 找到 entry** 之 invariant test（A1 SSOT 強制）

### 8.2 刪除（無 alias）

- [ ] `FactorType` enum 與全部字串
- [ ] `CrossSectionalConfig` / `MacroPanelConfig` / `EventSignalConfig` / `MacroCommonConfig`
- [ ] `Profile.canonical_p` property（改 `primary_p`）
- [ ] `CANONICAL_P_FIELD: ClassVar[str]`（A2 — 不再有 stringly-typed indirection）
- [ ] 各 v0.4 Profile dataclass（B3 — collapse 為單一 `FactorProfile`）
- [ ] `BHY.p_source` 參數（A2）
- [ ] `UnreliableSESizeWarning` / `PersistentRegressorWarning` / `EventOverlapWarning` 等 Python `Warning` subclass（B6）
- [ ] `describe_factor_types()` / `describe_profile()` 舊雙 entry
- [ ] N=1 時 `primary_p = 1.0` 強制壓邏輯

### 8.3 文件 / 測試 / examples 更新（手工 + sed/grep）

- [ ] README §怎麼選分析模式 與 v0.5 API 對齊（移除 v0.6 transitional 註記）
- [ ] README smoke test 換用 `AnalysisConfig.individual_continuous(...)`
- [ ] ARCHITECTURE.md 更新
- [ ] CHANGELOG.md 顯著標 BREAKING change（v0.5 入口）
- [ ] `docs/statistical_methods.md` Mode B procedure 補述
- [ ] `docs/metric_applicability.md` 五 tuple 對照
- [ ] 既有 ~50 個 test 檔案 rename：`factor_type="cross_sectional"` → `AnalysisConfig.individual_continuous(...)`、`canonical_p` → `primary_p`
- [ ] `examples/demo.ipynb` 改寫為 5 tuple × Mode A/B 範例

### 8.4 Release

- [ ] `cz bump` → v0.5.0
- [ ] CHANGELOG 採 hybrid workflow（`cz bump --changelog` scaffold + 手工 polish 為 Keep a Changelog 風格）

---

## 9. 風險

無既有用戶 → 無 migration risk。剩餘技術風險：

### 9.1 Newey-West HAC lag 自動選擇可能不穩

- §5.2 / §5.3 預設 Bartlett rule `lag = floor(4 * (T/100)^(2/9))`
- 短 T 時可能 over/under-smooth；high-freq autocorr 場景可能 under-smooth
- **緩解**：
  - 提供 `EvalConfig(nw_lags=...)` override
  - 預設 lag 公式在 docstring 明列（Newey & West 1994 automatic）
  - `diagnose()` 印實際使用的 lag 值
  - T < 30 時自動加 `UnreliableSESizeWarning`

### 9.2 Mode B 多因子場景下的 BHY 統計純度

- 單資產跑多 signal + BHY → 同一 Y 序列被多次測試，內部相關性需 BHY 涵蓋
- **緩解**：BHY 採 step-up Benjamini-Yekutieli 已對 dependence 保守；`statistical_methods.md` 文件化

### 9.3 Test surface rename 範圍

- ~50 檔需手工 rename
- **緩解**：分批 atomic commit（依 commit convention：≤50 char、無 AI sig、sign-off）；rename 機械操作，CI 全 pass 為 acceptance criteria

---

## 10. 決議事項（需 sign-off）

| # | 議題 | 我的推薦 |
|---|---|---|
| 1 | factrix 是 single-factor primary + multi-factor diagnostic？（§0） | YES |
| 2 | Mode A / Mode B 對等 first-class，移除 N=1 fallback 病理化框架？（§0.3 / §1.2） | YES |
| 3 | 版本號 v0.4 → **v0.5**（單一 rip-and-replace，無 v0.5 → v0.6 兩階段）？ | YES |
| 4 | `canonical_p` → `primary_p` rename（無 alias、無 deprecation）？ | YES |
| 5 | Factory Methods (4 個 constructor) 取代 `metric: Metric \| None` runtime check？ | YES |
| 6 | Registry-based dispatch 取代 if-else？ | YES |
| 7 | Mode B `(COMMON, *, N=1)` 不再壓 `primary_p=1.0`？ | YES |
| 8 | Mode B `(INDIVIDUAL, SPARSE, N=1)` 加 NW HAC + Ljung-Box + actionable hint？ | YES |
| 9 | `WarningCode`（前 DegradedMode 改名）與 `Mode` 拆兩 enum；Profile 用 `frozenset[WarningCode]`？ | YES |
| 9b | `(*, SPARSE, N=1)` `primary_p` 來自 calendar-time TS dummy regression，**非** CAAR？ | YES |
| 9c | T < 20 raise `InsufficientSampleError`；20 ≤ T < 30 仍出 verdict + warning？ | YES |
| 9d | BHY family key 加 `mode`，Mode A / Mode B 不混 family？ | YES |
| 9e | `__post_init__` 為 axis 驗證單一入口（factories + from_dict 共用）？ | YES |
| 10a | Mode B sparse 真退化：`(*, SPARSE, N=1)` 路由到單一 registry entry / 單一 BHY family？（B1） | YES |
| 10b | Registry 為「合法 cell」SSOT，validator/describe/suggest 全部反查？（A1） | YES |
| 10c | 刪除 `PRIMARY_P_FIELD: ClassVar[str]`，BHY 統一取 `profile.primary_p`？（A2） | YES |
| 10d | 單一 `FactorProfile` dataclass + `FactorProcedure` callable，取代 N×M Profile class？（B3） | YES |
| 10e | Exception 子類收斂為 3 個（IncompatibleAxis / ModeAxis / InsufficientSample）？（B2） | YES |
| 10f | 移除 Python `Warning` 體系，統一走 `frozenset[WarningCode]`？（B6） | YES |
| 10g | `event_temporal_hhi` 走 `profile.extras` + `diagnose()`，不入 top-level schema？（B5） | YES |
| 10h | `_FALLBACK_MAP` 集中 `suggested_fix` 規則？（A4） | YES |
| 11a | `StatCode` enum + `FactorProfile.stats: Mapping[StatCode, float]` 取代 optional 欄位 + `extras` dict？（B3 後續，extensibility） | YES |
| 11b | `verdict(threshold=, gate=)` / `bhy(threshold=, gate=)`：`primary_p` SSOT 不變，`gate` 是 user policy override？（C1） | YES |
| 11c | 參數命名 `threshold`（非 `alpha`）跨所有 verdict / FDR 介面一致？ | YES |
| 11d | User-level custom Metric 列入「明確不做」（§6.6）？（C3） | YES |
| 11e | Registry / procedure 在 v0.5 underscore 前綴內部使用，v0.6+ 候選對外公開？（C4） | YES |
| 11f | §7.5 命名一致性 invariants 表為 PR review checklist？ | YES |
| 10 | Exception 階層細分 + `suggested_fix: AnalysisConfig` 屬性？ | YES |
| 11 | `from_dict` / `to_dict` round-trip 強制測試？ | YES |
| 12 | `multi_factor` namespace（BHY / redundancy / spanning / orthogonalize 集中）？ | YES |
| 13 | 「明確不做」補 5 條目（非線性、joint regression、composite signal、portfolio attribution、return-process modeling）？ | YES |
| 14 | `describe_analysis_modes()` 取代 `describe_factor_types()`？ | YES |
| 15 | `suggest_config(raw)` with structured reasoning dict？ | YES |
| 16 | v0.5 維持 5 tuple，`(COMMON, CONTINUOUS, FM)` deferred？ | YES |
| 17 | `regime_labels` hook、`benchmark_factors` 進 Config — deferred？ | YES |

---

## 11. Out of scope / Deferred

### Permanently out of scope（明確不做）

- ML signal layer（XGBoost / LightGBM / SHAP / deep models）— §6.1
- 非線性與特徵交互作用 — §6.1
- Joint multi-factor regression — §6.2
- Composite signal generation — §6.3
- Portfolio attribution / risk decomposition — §6.4
- Return-process modeling（ARIMA / state-space / GARCH / VAR）— §6.5
- IVX / Stambaugh correction（仍走 flagging via `factor_adf_p` + `PersistentRegressorWarning`）
- Wild bootstrap / jackknife SE
- Regime detection methodology（HMM / threshold / structural break）
- Structural break test（Chow / Quandt-Andrews / Bai-Perron）
- Backtest / execution simulation / slippage / margin
- Trading metrics（Sharpe / Calmar / drawdown）full computation — 僅 `breakeven_cost` proxy
- FF3 / Carhart4 benchmark 便利取得

### Deferred to post-v0.5（friction log 累積後評估）

| Feature | 來源 | 為何 deferred |
|---|---|---|
| `(COMMON, CONTINUOUS, FM)` 第 6 tuple | quant review #1 | 跨國 ETF FM-λ 是真實情境但無立即用例；維持 v0.5 surface 精簡 |
| `AnalysisConfig.regime_labels` hook | quant review #5 | 「VIX>20 時 momentum 才有效」powerful but adds per-regime dispatch + BHY group key 擴展 |
| `AnalysisConfig.benchmark_factors`（spanning 進 evaluate verdict） | gemini review #7 邏輯延伸 | 維持 `evaluate()` 單純（單因子 verdict）+ `multi_factor.spanning_test` 已可用 |
| Multi-horizon `forward_periods` sweep 包裝 | quant review #9 | user-side loop 已可達成 |
| Sub-period rolling verdict 包裝 | quant review #9 | building block 已有，包裝可後續 |
| Industry / sector neutralization | quant review #9 | 待 `regime_labels` 一起做 |
| Factor crowding / regime change detection | quant review #9 | 屬 regime methodology，需先想清楚 |
| `redundancy_matrix` 三軸化 | — | 後續 spike |
| `factrix-codemod` 自動化 | gemini review #6 | 無外部用戶，不需自動 codemod；內部一次性手工夠用 |
| **Procedure registration as public extension API**（C4） | extensibility review | v3.2 registry / procedure callable / dispatch key 設計已 ready 對外；v0.5 仍以 underscore 前綴維持內部使用，避免 API 穩定承諾過早。v0.6+ 真有 plug-in 需求時去 underscore + 公開 `factrix.extension` namespace（含 `FactorProcedure` Protocol、`register_procedure`、`FactorProfile` constructor helper）|
| `EvalConfig(se_method="bootstrap")` 等 stat variant 軸 | C3 引申 | bootstrap / wild bootstrap / jackknife 等 SE 變體屬正交軸，非新 metric；累積實際需求再加 |

---

## 12. 參考

### Project 內部
- README §怎麼選分析模式 (Analysis Mode) — v0.5 target 設計（README 已對齊 v0.5 命名後仍適用）
- `docs/plans/plan_gate_redesign.md` — Profile 架構遷移歷史（canonical_p 起源）
- `docs/plans/naming_convention.md` — 既有 rename precedent
- `docs/statistical_methods.md` — 統計方法詳述（v0.5 release 同步更新）

### 統計文獻
- Newey & West (1987, 1994) — HAC SE
- Brown & Warner (1985); MacKinlay (1997) — event study SE
- Ljung & Box (1978) — Q test
- Stambaugh (1999); Campbell & Yogo (2006) — persistent regressor correction (referenced, not implemented)
- Fama & MacBeth (1973); Petersen (2009) — FM-λ + clustered SE
- Black, Jensen & Scholes (1972); Fama & French (1993) — TS-β factor model
- Grinold (1989) — fundamental law of active management (IC)
- Benjamini & Yekutieli (2001) — FDR under dependence (BHY)
- Barillas & Shanken (2017); GRS (Gibbons-Ross-Shanken 1989) — spanning test framework

### Reviews 整合（v3 反饋來源）
- Senior backend review (gemini, 2026-05-01) — Factory Methods、Registry pattern、Exception 結構化、from_dict round-trip
- Senior quant user review (subagent, 2026-05-01) — metric on (COMMON, CONTINUOUS)（deferred）、跳過 v0.5 中間階段、regime_labels hook（deferred）、DegradedMode enum 補充、out-of-scope 明列
