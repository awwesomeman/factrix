# ADR: Profile-Centric Architecture（取代 Gate 模式）

> Status: Accepted — 2026-04-17
> Supersedes: `gate_redesign.md`
> Drives: `plan_gate_redesign.md`、`factor_screening.md` (v4)

---

## 1. Context

factrix 的研究 UX 目前以 **Gate**（pass/fail 決策鏈）為核心抽象。`gate_redesign.md` 的討論揭露了這個抽象的多重張力：

- gate 內部的 OR 邏輯（IC OR spread）汙染 primary_p 的統計解釋（`min(p)` 違反 BHY 均勻零分佈假設）。
- AND chain short-circuit 讓使用者看不到後續 gate 的資訊（研究情境的資訊損失）。
- `GateResult.detail: dict[str, Any]` 是 untyped bag，欄位可拼錯、IDE 零幫助。
- 一套 `significance_gate` 無法表達不同因子類型（cross-sectional / time-series / Fama-MacBeth / event）各自的 canonical test。
- 要塞進 `primary_p` + `suggestion` 時，發現 gate 抽象同時承擔了「決策」、「推論 p 的生產」、「AI 反饋」三個不同責任。

對齊業界 / 主流研究後的觀察：

- 學術（Harvey-Liu-Zhu 2016, Lopez de Prado DSR/PBO）→ 每因子**一個主要檢定 p**，統一跑 multiple-testing correction。
- 機構（AQR / DFA / BlackRock SAE）→ **factor dossier / scorecard**（多維剖面，不壓縮為單分），研究員讀剖面做判斷。
- gate chain 是 MLOps / production readiness 的 pattern，**不是 factor research 的 pattern**。

---

## 2. Decision

**以 `FactorProfile`（per-factor-type typed dataclass）為核心抽象，完全退場 Gate 系統；不留 fallback 相容層。**

因為沒有相容性顧慮，設計可以做激進的 collapse：

- **無 `EvaluationResult` 包裝**：`status` / `verdict` / `diagnose` / `artifacts` 全部住在 profile 上，top-level API 直接回 `FactorProfile`。
- **Evaluator = `Profile.from_artifacts()` classmethod**：不需要獨立 evaluator function 或 registry dispatch function；profile class 自帶建構子。
- **ProfileSet = polars-native**：內部就是 polars DataFrame + profile class reference，沒有 list↔DataFrame round-trip。
- **API 2 個 entry point**：`fl.evaluate(df, name, ...)`（單因子）+ `fl.evaluate_batch(factors, ...)`（批次）。`quick_check` / `compare` 不再存在。

架構分層：

```
Layer 0  Artifacts
   Artifacts.from_raw(df, name, config) -> Artifacts
   # 既有 pipeline 前半段收斂成 Artifacts 的 classmethod

Layer 1  FactorProfile（per-type typed schema）
   CrossSectionalProfile.from_artifacts(artifacts) -> CrossSectionalProfile
   EventProfile.from_artifacts(artifacts)           -> EventProfile
   MacroPanelProfile.from_artifacts(artifacts)      -> MacroPanelProfile
   MacroCommonProfile.from_artifacts(artifacts)     -> MacroCommonProfile
   # 每類因子一個 dataclass，from_artifacts classmethod 作為唯一建構路徑

Layer 2  ProfileSet (polars-native)
   ProfileSet[P].filter(pl.Expr).rank_by(field).top(n)
   # 內部為 polars DataFrame + ProfileT；iter_profiles() 反序列化成 dataclass

Layer 3  multiple_testing_correct(profile_set, p_source="canonical_p")
   # 回傳新 ProfileSet（含 bhy_significant / p_adjusted 欄位）

Layer 4  Deep-dive
   plot_ic_timeseries(profile, artifacts) / event_window_decomp(...) / ...
```

頂層 facade：

```python
fl.evaluate(df, name, *, factor_type="cross_sectional", **cfg) -> FactorProfile
fl.evaluate_batch(factors, *, factor_type=..., **kw) -> ProfileSet[FactorProfile]
```

---

## 3. 設計抉擇（A–F 定案）

### A. Profile 欄位：**Flat + 命名前綴**

```python
@dataclass
class CrossSectionalProfile:
    factor_name: str
    n_periods: int
    # IC family
    ic_mean: float
    ic_tstat: float
    ic_p: float
    ic_ir: float
    # Spread family
    spread_mean: float
    spread_tstat: float
    spread_p: float
    # Stability
    oos_decay: float
    sign_stability: float
    # Implementation
    turnover: float
    breakeven_cost: float
    net_spread: float
```

前綴（`ic_*`, `spread_*`, `oos_*`）提供邏輯分組而不犧牲 filter 語法；polars/pandas 互通零摩擦。

### B. ProfileSet 底層：**Polars-native（內部即 DataFrame）**

```python
class ProfileSet(Generic[P]):
    _df: pl.DataFrame              # source of truth
    _profile_cls: type[P]           # 型別 reference（for iter_profiles reconstruction）

    def filter(self, predicate: pl.Expr | Callable[[P], bool]) -> ProfileSet[P]: ...
    def rank_by(self, field: str, descending: bool = True) -> ProfileSet[P]: ...
    def top(self, n: int) -> ProfileSet[P]: ...
    def iter_profiles(self) -> Iterator[P]: ...   # 需要 typed 物件時反序列化
    def to_polars(self) -> pl.DataFrame: ...
```

- **Polars DataFrame 是 source of truth**，profile dataclass 是 view。避免原 list+DataFrame round-trip 的語義歧異。
- Constructor 檢查 homogeneity：`ProfileSet(profiles)` 若 profiles 類型不同 → `TypeError`。
- `filter(pl.Expr)` 必須是 row-wise boolean；執行時驗證 `predicate` 結果 dtype == Boolean 且 row 數不變，否則 raise。
- `filter(Callable)` 走 `iter_profiles` lambda path（複雜條件 escape hatch）。

### C. Filter 語法：**Polars 原生優先，lambda 作為 escape hatch**

```python
# 主推（codebase 一致）
profiles.filter((pl.col("ic_tstat") >= 2) & (pl.col("oos_decay") < 0.5))

# Escape hatch（複雜條件）
profiles.filter(lambda p: custom_logic(p))
```

不採 Django 風 `ic_tstat__gte=2` kwargs 方言。

### D. Discovery API：**三個都做**

- `fl.list_factor_types()` → programmatic enumeration（for scripting）。
- `help(CrossSectionalProfile)` / `__annotations__` → IDE / REPL（for humans）。
- `fl.evaluate(artifacts, factor_type="cross_sectional")` → ergonomic facade；未知 type 給 helpful error 列出合法值。

**關鍵改動**：`describe_profile()` 的資料來源從手寫 `_PROFILE_METRICS` dict 改為**反射 profile dataclass 的 `__annotations__`**。schema 只定義一次，print 展示、filter 合法欄位、BHY p_source 合法值都從同一 source of truth 長出來。

### E. Suggestion / Diagnostic：**Lazy method `profile.diagnose()`**

```python
@dataclass
class CrossSectionalProfile:
    ...

    def diagnose(self) -> list[Diagnostic]:
        """Contextual hints (not stored; recomputed on demand)."""
        ...
```

理由：診斷規則會演化、會加、會依外部上下文調整。存成欄位會 freeze 邏輯，也會污染 `print(profile)` 的可讀性。

### F. Gate 退場：**全部移除，不走 compat shim**

使用者拍板「gate 可以直接全部退場」。不留 deprecation wrapper。

- 刪除：`factrix/evaluation/gates/` 目錄整包。
- 刪除：`GateFn`、`GateResult`、`GateStatus`、`default_gates_for()`。
- **刪除 `EvaluationResult` 整個 wrapper**：status / diagnose / artifacts 全部住在 profile 上；top-level API 直接回 `FactorProfile`。
- 無 fallback 期；詳見 `plan_gate_redesign.md` 的 2-phase 計畫。

### G. P-value 白名單與型別標註

為防止合成 p 旁路餵給 BHY（違反同檢定族假設），profile class 以 `ClassVar` 白名單宣告合法 p-value 欄位：

```python
from typing import NewType, ClassVar

PValue = NewType("PValue", float)   # value metric 用 float；p-value 用 PValue

@register_profile(FactorType.CROSS_SECTIONAL)
@dataclass(frozen=True, slots=True)
class CrossSectionalProfile:
    factor_name: str
    n_periods: int
    ic_p: PValue
    spread_p: PValue
    ic_ir: float
    oos_decay: float
    ...

    CANONICAL_P_FIELD: ClassVar[str] = "ic_p"
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({"ic_p", "spread_p"})

    @property
    def canonical_p(self) -> PValue:
        return getattr(self, self.CANONICAL_P_FIELD)
```

`multiple_testing_correct(p_source=...)` 在執行前驗證 `p_source in profile_cls.P_VALUE_FIELDS` 或 `p_source == "canonical_p"`；不合法 → `ValueError` 列出白名單。

---

## 4. Status / Verdict 的處置

`Verdict` 分離成兩個獨立語義：

1. **`verdict()` — 純 binary**（single canonical test）
   - `PASS`：canonical test 顯著。
   - `FAILED`：canonical test 不顯著。
   - 只看 canonical p（cross-sectional 就是 IC p），**不涉及 spread 等次要證據**——避免重現原 gate 的 OR 邏輯。

2. **`diagnose() -> list[Diagnostic]`** — severity-scaled hints
   - `Diagnostic(severity ∈ {"info", "warn", "veto"}, message)`。
   - 次要證據、OOS 異常、turnover 過高、n_periods 過小、事件 clustering 等所有非 canonical 觀察都走這裡。

```python
class CrossSectionalProfile:
    def verdict(self, threshold: float = 2.0) -> Verdict:
        """Binary verdict on canonical test only. Threshold defaults to 2.0
        for single-test view; use 3.0+ for multi-testing context (Harvey 2016)."""
        return "PASS" if self.ic_p <= _p_from_t(threshold, self.n_periods) else "FAILED"

    def diagnose(self) -> list[Diagnostic]:
        hints: list[Diagnostic] = []
        if abs(self.spread_tstat) >= 2.0 and self.ic_p > 0.05:
            hints.append(Diagnostic("warn",
                "IC not significant but spread t-stat strong; "
                "consider event-driven or TS framing."))
        if self.oos_decay > 0.5:
            hints.append(Diagnostic("veto", "OOS decay > 50%"))
        if self.turnover > 1.0:
            hints.append(Diagnostic("warn", "Turnover > 1.0, check capacity"))
        ...
        return hints
```

**Pipeline 例外 → 不存為 verdict，而是讓 `evaluate()` raise `EvaluationError`**。因為沒有 `EvaluationResult` wrapper，「計算失敗」不能塞進 profile 欄位裡；由 `batch_evaluate` 的 `stop_on_error=False` path 決定是捕捉記錄還是向上拋。

使用者決策樹：

- 快速 binary 篩 → `profile.verdict()`。
- 看警示細節 → `profile.diagnose()`。
- 跨多因子 FDR 控制 → `ProfileSet.multiple_testing_correct()`。
- 完全自訂 → `ProfileSet.filter(pl.Expr)` 直接操作欄位。

---

## 5. BHY / Multiple Testing 的歸屬

從 gate 抽離，成為獨立 module：

```python
# factrix/stats/multiple_testing.py
def bhy_adjust(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray: ...
```

因為 `ProfileSet` 是 polars-native，BHY 結果直接以新增欄位形式返回同型 ProfileSet（不需要獨立 wrapper 物件）：

```python
class ProfileSet(Generic[P]):
    def multiple_testing_correct(
        self,
        p_source: str = "canonical_p",
        method: Literal["bhy"] = "bhy",
        fdr: float = 0.05,
    ) -> ProfileSet[P]:
        """Returns new ProfileSet with bhy_significant, p_adjusted columns added.

        p_source must be "canonical_p" or in profile_cls.P_VALUE_FIELDS.
        Rejects composed / derived columns.
        """
```

**關鍵不變量**：`p_source` 由 profile class 的 `P_VALUE_FIELDS` 白名單強制，執行前驗證；合成 p（`min(ic_p, spread_p)`）會在白名單檢查時 raise。所有因子走同一檢定族，FDR 控制才成立。

**型別層級的跨類防護**：`ProfileSet` 是 single-type generic；跨類在 constructor 就擋下（`TypeError`）。使用者要跨類比較需顯式 per-type 跑 BHY 再併表（見情境 5）。

---

## 6. 新舊 API 對應表

**Top-level API**（collapse 至 2 個 entry point）

| 現有 | 新架構 | 動作 |
|------|--------|------|
| `fl.quick_check(df, name, factor_type=, ...)` | **刪除**（merge 進 `fl.evaluate`） | 刪除 |
| `fl.batch_evaluate(factors, gates=, ...)` | `fl.evaluate_batch(factors, factor_type=, keep_artifacts=, stop_on_error=)` | 改名 + 砍 gates 參數 |
| `fl.compare(results, sort_by=, bhy=)` | **刪除**；由 `ProfileSet.to_polars()` + `ProfileSet.multiple_testing_correct()` 取代 | 刪除 |
| — | **新增** `fl.evaluate(df, name, factor_type=, **cfg) -> FactorProfile` | 新增 |
| `fl.split_by_group()` | 保留；回傳 `dict[str, tuple[DataFrame, Config]]` 不變 | 不動 |

**Discovery API**

| 現有 | 新架構 | 動作 |
|------|--------|------|
| `FACTOR_TYPES: dict[FactorType, type[BaseConfig]]` | 保留 | 不動 |
| `describe_factor_types()` | 保留 | 不動 |
| `describe_profile(type)` | 保留簽名，內部反射 profile dataclass `__annotations__` + ClassVar 資訊 | 換內臟 |
| — | **新增** `fl.list_factor_types() -> list[str]` | 新增 |
| `_PROFILE_METRICS` dict | **刪除**（profile dataclass 即 schema） | 刪除 |
| `_STANDALONE_METRICS` dict | **刪除**（Layer 4 deep-dive module 直接 import） | 刪除 |

**Evaluation / Profile 核心**

| 現有 | 新架構 | 動作 |
|------|--------|------|
| `EvaluationResult` wrapper | **整個刪除**；API 直接回 `FactorProfile` | 刪除 |
| `FactorProfile`（untyped `list[MetricOutput]`）| 完全改寫為 per-type dataclass（`CrossSectionalProfile` / `EventProfile` / `MacroPanelProfile` / `MacroCommonProfile`） | 重寫 |
| `pipeline.evaluate(df, name, ...)` function | 功能搬進 `fl.evaluate`；pipeline module 可能只剩 `Artifacts.from_raw()` | 改寫 |
| `Artifacts.get(key)` | 保留 | 不動 |
| — | **新增** `Artifacts.from_raw(df, name, config)` classmethod | 新增 |
| — | **新增** `FactorProfile.from_artifacts(artifacts)` classmethod per type | 新增 |
| — | **新增** `@register_profile(FactorType.X)` decorator + `_PROFILE_REGISTRY` | 新增 |

**Gate 相關（全刪）**

| 現有 | 動作 |
|------|------|
| `factrix/evaluation/gates/` 目錄 | **刪除** |
| `factrix/evaluation/presets.py` | **刪除** |
| `factrix/evaluation/_caution.py` | **刪除**（邏輯搬進 profile.diagnose()） |
| `GateFn` / `GateResult` / `GateStatus` / `default_gates_for` | **刪除** |
| `EvaluationResult.gate_results` / `.caution_reasons` | **刪除**（隨 wrapper 消失） |

**ProfileSet / Stats**

| 新增 | 動作 |
|------|------|
| `ProfileSet[P]`（polars-native）| **新增** |
| `factrix/stats/multiple_testing.bhy_adjust()` | **新增** |
| `_stats.bhy_threshold()` | **刪除**（by `factor_screening` plan） |

使用者 breaking changes 一覽（release notes）：`EvaluationResult` 不再存在、`.profile.metrics` 改為直接欄位存取、`gate_results` 消失、`quick_check` / `compare` 刪除。遷移對照見情境 7。

---

## 7. 使用者情境範例

以下情境描述設計落地後，使用者程式碼的典型樣貌。對應 factrix 三大價值主張：
- (a) **能力探索**：使用者知道某類因子有哪些工具可用
- (b) **快速篩選**：從 N 個候選因子中挑出潛在有效者
- (c) **深入分析**：對單一因子做細部剖析 / 策略建構

---

### 情境 1：探索因子類型支援哪些指標 — 解決 (a)

```python
import factrix as fl

# 列出支援的因子類型（programmatic）
fl.list_factor_types()
# → ["cross_sectional", "event_signal", "macro_panel", "macro_common"]

# 打印某類因子的完整剖面 schema（from dataclass annotations）
fl.describe_profile("cross_sectional")
#
#   cross_sectional — profile fields:
#   ────────────────────────────────────────
#     factor_name       : str
#     n_periods         : int
#     # IC family
#     ic_mean           : float
#     ic_tstat          : float
#     ic_p              : PValue    (canonical for BHY)
#     ic_ir             : float
#     # Spread family
#     spread_mean       : float
#     spread_tstat      : float
#     spread_p          : PValue
#     # Stability
#     oos_decay         : float
#     sign_stability    : float
#     # Implementation
#     turnover          : float
#     breakeven_cost    : float
#     net_spread        : float
#
#   Methods:
#     .canonical_p()    → ic_p
#     .verdict(t=2.0)   → EvaluationStatus
#     .diagnose()       → list[Diagnostic]
#
#   Deep-dive (from factrix.plots):
#     plot_ic_timeseries(profile, artifacts)
#     plot_quantile_returns(profile, artifacts)

# IDE 路徑：直接 help()
from factrix.evaluation.profiles import CrossSectionalProfile
help(CrossSectionalProfile)
```

---

### 情境 2：研究員手動快篩單一因子 — 解決 (b)、(c)

```python
import factrix as fl

# 單因子評估（前處理 + 算 profile）→ 直接回 FactorProfile（無 wrapper）
profile = fl.evaluate(df, "momentum_12_1", factor_type="cross_sectional")

# print 顯示完整剖面 + diagnostics
print(profile)
# CrossSectionalProfile(factor_name='momentum_12_1', n_periods=252)
# ┌──────────────┬─────────┐
# │ ic_mean      │  0.0234 │
# │ ic_tstat     │   2.81  │
# │ ic_p         │  0.0051 │
# │ ic_ir        │  0.4512 │
# │ spread_tstat │   2.15  │
# │ spread_p     │  0.0316 │
# │ oos_decay    │  0.2341 │
# │ turnover     │  0.8234 │
# │ net_spread   │  0.0041 │
# └──────────────┴─────────┘
# Verdict: PASS
# Diagnostics:
#   [warn]  Turnover > 0.5, check capacity

# 直接存取 typed 欄位（IDE 補全）
profile.ic_ir             # 0.4512
profile.canonical_p       # 0.0051 (property; 內部 return self.ic_p)
profile.verdict()         # "PASS"
profile.diagnose()        # [Diagnostic(severity="warn", message="...")]

# 深入分析（Layer 4）。artifacts 需要時另行取得
artifacts = fl.Artifacts.from_raw(df, "momentum_12_1", config)
from factrix.plots import plot_ic_timeseries
plot_ic_timeseries(profile, artifacts)
```

---

### 情境 3：批次篩 200 個候選因子 + BHY FDR 控制 — 解決 (b)

```python
import factrix as fl
import polars as pl

# 批次評估（不保留 artifacts 省記憶體）→ 直接回 ProfileSet
profiles = fl.evaluate_batch(
    candidate_factors,              # dict[str, pl.DataFrame] or list[(name, df)]
    factor_type="cross_sectional",
    keep_artifacts=False,
    stop_on_error=False,
)
# profiles: ProfileSet[CrossSectionalProfile]

# Phase 1：基本過濾（透明 boolean 條件，無隱藏 OR）
candidates = (
    profiles
    .filter(pl.col("n_periods") >= 60)           # 足夠資料
    .filter(pl.col("ic_ir") >= 0.3)              # 有訊號
    .filter(pl.col("turnover") <= 0.8)           # 成本可控
)

# Phase 2：多重檢定 FDR 控制（獨立顯式步驟）
# 直接回傳新 ProfileSet，含 bhy_significant / p_adjusted 欄位
adjusted = candidates.multiple_testing_correct(
    p_source="canonical_p",     # 白名單內；= ic_p
    method="bhy",
    fdr=0.05,
)

survivors = adjusted.filter(pl.col("bhy_significant"))
print(f"{len(profiles)} → {len(candidates)} → {len(survivors)}")
# 200 → 47 → 12

# Phase 3：排序挑 top
top = survivors.rank_by("ic_ir", descending=True).top(5)

# 輸出到 polars 看整體
adjusted.to_polars().select([
    "factor_name", "ic_ir", "ic_p", "p_adjusted",
    "bhy_significant", "net_spread", "oos_decay",
])
```

**關鍵**：合成 p 的旁路被擋住——
```python
# 錯誤做法（會被擋下）
candidates.multiple_testing_correct(p_source="min_p")
# → ValueError: 'min_p' not in canonical p-value whitelist for
#   CrossSectionalProfile. Valid: {'ic_p', 'spread_p'}.
#   See ADR gate_redesign_v2 §5 for why composed p is rejected.
```

---

### 情境 4：AI Agent 的批次分析 workflow

```python
import factrix as fl

# AI 遍歷候選因子（stop_on_error=False → 失敗的因子不進 ProfileSet）
profiles = fl.evaluate_batch(
    ai_generated_candidates,
    stop_on_error=False,
    on_error=lambda name, exc: ai_log(f"{name}: {exc}"),
)

for profile in profiles.iter_profiles():
    # 結構化資料餵給 AI（typed，不需解析 dict）
    summary = {
        "canonical_p": profile.canonical_p,
        "ic_ir": profile.ic_ir,
        "turnover": profile.turnover,
        "verdict": profile.verdict(),
        "diagnostics": [
            {"severity": d.severity, "message": d.message}
            for d in profile.diagnose()
        ],
    }
    ai_log(summary)

# 跨因子去重（避免 AI 反覆發明同類因子）
matrix = fl.redundancy_matrix(profiles, method="value_series")
```

---

### 情境 5：跨因子類型比較（分開 BHY，不誤混）

```python
# 同時評估一批 cross-sectional 和 event-signal 因子
xs_profiles = fl.evaluate_batch(xs_candidates, factor_type="cross_sectional")
ev_profiles = fl.evaluate_batch(ev_candidates, factor_type="event_signal")
# 型別：ProfileSet[CrossSectionalProfile] 和 ProfileSet[EventProfile]

# 跨類混 ProfileSet 會被 constructor 擋下
# fl.ProfileSet([*xs_profiles.iter_profiles(), *ev_profiles.iter_profiles()])
# → TypeError: ProfileSet is single-type; got
#   {CrossSectionalProfile, EventProfile}. Run multiple_testing per type.

# 正確：每類獨立 FDR 控制（每類有自己的 canonical test family）
xs_adj = xs_profiles.multiple_testing_correct(p_source="ic_p", fdr=0.05)
ev_adj = ev_profiles.multiple_testing_correct(p_source="caar_p", fdr=0.05)

# 併表展示（但 bhy_significant 是 per-type 算的）
combined = pl.concat([
    xs_adj.to_polars().with_columns(pl.lit("cross_sectional").alias("type")),
    ev_adj.to_polars().with_columns(pl.lit("event_signal").alias("type")),
], how="diagonal")
```

---

### 情境 6：研究員自訂篩選邏輯（escape hatch）

```python
# 複雜非 polars 條件用 lambda
profiles.filter(
    lambda p: p.ic_ir > 0.3 and (
        p.oos_decay < 0.2 or p.sign_stability > 0.85
    )
)

# 自訂 verdict threshold（遵循 Harvey 2016 for multi-testing）
strict_survivors = [p for p in profiles if p.verdict(threshold=3.0) == "PASS"]
```

---

### 情境 7：Deprecation — 舊 gate 程式碼要怎麼改寫

```python
# --- 舊程式碼（gate 時代）---
result = fl.quick_check(df, "mom")
for gate in result.gate_results:          # ← 整套機制已移除
    print(gate.name, gate.status, gate.detail.get("via"))
significant = [r for r in results.values()
               if any(g.status == "PASS" for g in r.gate_results)]

# --- 新程式碼（profile 時代）---
# 1. quick_check → evaluate（直接回 FactorProfile，不再有 EvaluationResult wrapper）
profile = fl.evaluate(df, "mom")
print(profile.verdict(), profile.diagnose())

# 2. batch_evaluate → evaluate_batch（回 ProfileSet）
profiles = fl.evaluate_batch(candidates)
significant = profiles.filter(lambda p: p.verdict() == "PASS")

# 3. compare() → ProfileSet.to_polars()（+ optional multiple_testing_correct）
table = profiles.multiple_testing_correct(p_source="canonical_p").to_polars()
```

---

## 8. 取捨與不做的事

### 採納

- **Per-type typed profile**：因子類型歧異性的 first-class 表達。
- **filter + rank 組合器**：透明的 boolean 條件，無隱藏 OR。
- **Canonical p 明確**：每類 profile 在 `verdict()` 和 BHY 使用時都有明確選擇。
- **Gate 全退場**：不留半套，避免長期維護兩套概念。

### 明確不做

- **Composite scorecard（加權單分）**：壓縮資訊內涵，研究情境不適合。
- **`any_of` / `all_of` p-value combinator**：合成 p 丟給 BHY 不合法（違反同檢定族假設）。
- **`threshold()` DSL / Rule Engine**：表達不了 OOS sign flip、事件窗半衰期等時序型檢定。
- **Compat shim for gates**：使用者決定直接退場，減少長期技術債。
- **跨類 BHY 自動聚合**：靜默 merge 不同檢定族會誤導；要求使用者顯式分組。

---

## 9. 風險

| 風險 | 緩解 |
|------|------|
| One-shot big-bang → review 負擔集中 | Phase A 先完整落地可測試 feature branch；Phase B 只刪舊。ADR sign-off 需先解決 Q1（verdict 二態）、Q2（per-type canonical p 精確定義）等 critical review 項 |
| 無 fallback → 使用者 notebook 一次全 break | release notes 列 breaking changes；情境 7 提供改寫對照；版本號 bump major |
| `verdict()` 預設 threshold 與既有 gate 行為微妙不同 | Phase A 測試同時比對既有 gate pass 集合，差異文件化於 migration plan |
| 新舊計算路徑 parity | 不存在 → 無此風險（no fallback 的正面效益） |
| Profile dataclass 欄位未來擴充 | 加欄位不破壞下游 polars filter；改名 / 刪欄位走 DeprecationWarning |
| canonical_p 定義未釘死造成 BHY 結果漂移 | 每類 profile 的 `CANONICAL_P_FIELD` 在 ADR 釘死；改動需 minor version bump 並標 release note |

---

## 10. 參考

- Harvey, Liu, Zhu (2016) — `...and the Cross-Section of Expected Returns`
- Harvey & Liu (2020) — `False (and Missed) Discoveries in Financial Economics`
- Benjamini & Yekutieli (2001) — BHY procedure
- Lopez de Prado (2018) — `Advances in Financial Machine Learning`（DSR / PBO）
- Jensen, Kelly, Pedersen (2023) — `Is There a Replication Crisis in Finance?`

（完整清單見 `docs/literature_references.md`）
