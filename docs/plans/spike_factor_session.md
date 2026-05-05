# Spike — `fl.factor()` session object for unified standalone metric UX

> **IMPLEMENTED 2026-04-20**
> - P0 `CrossSectionalFactor` + `fl.factor()` factory: commit `8f3d800`
> - PR 2/3 `EventFactor` / `MacroPanelFactor` / `MacroCommonFactor`: commit `d83f67c`
> - Post-impl polish (`_short_circuit_if` helper, cache-key assertion,
>   tradability hoist to base, `n_groups` / `estimated_cost_bps` override
>   threading, `Factor` base dropped from `fl.__all__`): commits
>   `8f15db8`, `85b81e3`, `dcbe346`
> - BC rename `q1_q5_spread` → `quantile_spread` (`8f15db8`) →
>   `quantile_spread` (Phase 2a); cache keys and method names here
>   reflect the final `quantile_spread` canonical form.

**狀態**：IMPLEMENTED（見上方）
**Owner**：jason pan
**日期**：2026-04-19
**解鎖的後續工作**：研究者 standalone metric 呼叫統一介面；
`fl.evaluate()` 維持 production 單次入口；`factrix.metrics.*` 降級為
primitive 層。**關聯 spike**：`spike_metric_outputs_retention.md`（已合併，
引入 `Artifacts.metric_outputs` + `from_artifacts` tuple return；本 spike
基於其結果直接繼續）。

---

## 1. 問題陳述

目前 standalone metric 函數的 input contract **不一致**——使用者要記得
誰吃 processed 誰吃 prepared：

| Metric | 吃什麼 | 原因 |
|---|---|---|
| `ic(ic_df)` / `ic_ir(ic_df)` | processed（`compute_ic()` 的輸出） | 與 `hit_rate` / `ic_trend` 共用 `ic_series`，build 一次多 metric 共用 |
| `hit_rate(ic_values)` / `ic_trend(ic_values)` | processed | 同上 |
| `caar(caar_df)` | processed | 與 event-side metric 共用 `caar_series` |
| `quantile_spread(df)` / `monotonicity(df)` / `top_concentration(df)` / `turnover(df)` | prepared panel | 各自的 bucketing / 計算方式不同，沒有有用的共用中間產物 |
| `bmp_test(df)` / `ts_beta(df)` | prepared panel | 同上 |

**根因合理**：performance 驅動的共用 intermediate 設計（`build_artifacts`
算一次 `ic_series` 給 `ic` / `hit_rate` / `ic_trend` / `ic_ir` 共用；算一次
`spread_series` 給 `quantile_spread` 省一次 bucketing）。

**但使用者體感糟**：
- 使用者要背「哪個 metric 吃 df、哪個吃 ic_df」
- 要呼叫 `ic()` 前得先記得 `compute_ic(df)`，得知道 `compute_ic` 存在
- 想連續跑 3 個 IC-family metric 時，不小心三次都傳 raw df 就各自重算 `compute_ic`
- `factor_name` 在連續呼叫中重複傳
- IDE autocomplete 沒法列出「CS 因子可用的 metric」——`from factrix.metrics import ...` 是 flat 命名空間

想要的 API（目標）：

```python
# Research 場景：一次 bind, 多次呼叫
f = fl.factor(df, "Mom_20D", factor_type="cross_sectional", config=cfg)

f.ic()                              # MetricOutput（內部 reuse cached ic_series）
f.quantile_spread(n_groups=10)      # MetricOutput（per-call 蓋過 cfg.n_groups）
f.monotonicity()
f.hit_rate()
f.ic_trend()

# 合成 Profile（reuse 同一份 cache，不會重 build）
profile = f.evaluate()
profile, artifacts = f.evaluate(return_artifacts=True)

# 需要時的 escape hatch
f.artifacts                         # Artifacts
f.artifacts.metric_outputs["ic"]    # 已 run 過的 MetricOutput
```

關鍵：**使用者不需要學 `Artifacts` 或「processed vs prepared」的差別**。
所有 metric 都透過 `f.<metric>()` 呼叫，內部 dispatch 到現有的 primitive。

**Scope boundary**：本 spike **不**改 `fl.evaluate` / `fl.evaluate_batch` /
`factrix.metrics.*` 的 signature；`fl.factor()` 是**純加層**。已存在的
code path 全部 bit-for-bit 不動（迴歸風險接近 0）。

## 2. 現況（source-code 驗證）

- `factrix/_api.py::evaluate` L314 —— 目前唯一的高階入口，一次呼叫回
  `FactorProfile`，內部走 `_evaluate_one` → `build_artifacts` →
  `profile_cls.from_artifacts`
- `factrix/evaluation/pipeline.py::build_artifacts` L27 —— public top-level
  export（`factrix/__init__.py:39`），接受 `(df, config)`，回 `Artifacts`
- `factrix/evaluation/_protocol.py::Artifacts` L67 —— 已經是完整的 cache
  bundle：`prepared` / `intermediates` / `metric_outputs` / `config` /
  `factor_name`
- `factrix/metrics/ic.py::ic` L58 —— 接 `ic_df`（processed）
- `factrix/metrics/quantile.py::quantile_spread` L86 —— 接 `df`（prepared
  panel），含 `_precomputed_series` 私有 kwarg（library-internal 優化）
- `factrix/metrics/__init__.py` —— 23 個 metric function 一次 re-export，
  flat namespace，無 factor_type grouping
- CS Profile 的 metric 呼叫在 `cross_sectional.py::from_artifacts` L126-254
  寫死（10 個 metric 逐一 unpack），**metric 集合跟 factor_type 綁死**

## 3. 待決策的政策問題

### 3.1 命名 / 呼叫風格：`fl.factor()` vs `fl.Factor()` vs `fl.bind()`

**(a) `fl.factor(df, name, ...)` 小寫 function + 回 `Factor` instance**。**建議**。
- Lowercase 讀起來像「動詞 / factory」：`f = fl.factor(...)`
- 跟 `fl.evaluate` / `fl.preprocess` / `fl.adapt` 的小寫慣例一致
- 內部真的有 `Factor` class（大寫 type name 拿來 isinstance 檢查、type
  hint、docs），但使用者大多只用 factory function
- 跟 polars 的 `pl.DataFrame(...)` 大寫混用也 OK——後者是明確的 class
  constructor，factory function 約定是小寫

**(b) `fl.Factor(df, name, ...)` 大寫直接 class constructor**
- 類似 sklearn `LogisticRegression(...)`
- 缺點：factrix 現有 top-level API 沒有 PascalCase constructor 風格，
  `fl.CrossSectionalConfig` 這類是 config class 不是 factory；加一個會讓
  API shape 不統一

**(c) `fl.bind(df, name, ...)` / `fl.session(...)` 另取動詞**
- 「factor」跟「session」這個概念的語義距離大，讀起來模糊
- 否決

### 3.2 Snapshot 語意：`fl.factor()` 何時跑 `build_artifacts`？

**(a) Eager：`fl.factor()` 當下就 `build_artifacts(df, config)`，artifacts
stash 在 instance 上**。**建議**。
- Snapshot 語意清楚：`fl.factor(df, ...)` 之後即便 df 被使用者 mutate，
  Factor 仍持原始計算結果
- 跟 `fl.evaluate` 的行為一致（`evaluate` 內部就是 eager `build_artifacts`）
- Research 場景大多 snapshot 即期望（notebook 裡 df 不太會改）
- 成本：即使使用者只想 call 一個 metric 也付 full `build_artifacts`——但
  `build_artifacts` 的核心其實就是算 `ic_series` / `spread_series` 這類
  intermediate，這些又是絕大多數 metric 會用到的；YAGNI 避開 lazy 的複雜度

**(b) Lazy：`fl.factor()` 只存 `(df, name, config)`，第一次 call metric
才 build**
- 省一次計算（如果使用者只 call 一個不依賴 intermediate 的 metric）
- 增加 cache invalidation 複雜度：per-call config override 時要不要重 build？
  per-call n_groups 蓋過時 spread_series 要 cache 嗎？
- 使用者 mental model 更難：`f.ic()` 第一次是「expensive」、第二次是「cheap」
- 否決

### 3.3 Factor class 架構：per-type subclass vs 單一 class + 動態 attr

**(a) Per-factor-type subclass，類似 Profile pattern**。**建議**。
```python
class Factor:                              # base；大部分使用者只看 type hint
    artifacts: Artifacts
    def evaluate(self) -> FactorProfile: ...

class CrossSectionalFactor(Factor):
    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.CROSS_SECTIONAL
    def ic(self, forward_periods: int | None = None) -> MetricOutput: ...
    def ic_ir(self) -> MetricOutput: ...
    def hit_rate(self, forward_periods: int | None = None) -> MetricOutput: ...
    def ic_trend(self) -> MetricOutput: ...
    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput: ...
    def monotonicity(self, n_groups: int | None = None) -> MetricOutput: ...
    def top_concentration(self, q_top: float | None = None) -> MetricOutput: ...
    def turnover(self) -> MetricOutput: ...
    def breakeven_cost(self) -> MetricOutput: ...
    def net_spread(self) -> MetricOutput: ...
    # Level-2 opt-in（config 沒填 → 回 short-circuit MetricOutput）
    def regime_ic(self) -> MetricOutput: ...
    def multi_horizon_ic(self) -> MetricOutput: ...
    def spanning_alpha(self) -> MetricOutput: ...

class EventFactor(Factor):
    EXPECTED_FACTOR_TYPE: ClassVar[FactorType] = FactorType.EVENT_SIGNAL
    def caar(self) -> MetricOutput: ...
    def bmp_test(self) -> MetricOutput: ...
    def event_hit_rate(self) -> MetricOutput: ...
    # ...
```
- IDE autocomplete 完整：`f.` 列出該 factor_type **實際可呼叫**的 metric
- Type hint 直接走 dataclass / method signature，不需 runtime magic
- 新增 factor_type 就新增一個 Factor subclass，跟 Profile 1:1 對應
- Dispatch：`fl.factor(df, name, factor_type='cross_sectional')` 在 factory
  function 裡根據 config class 走 match/case（跟 `build_artifacts` 同 pattern）

**CS Factor method set 明列（P0 範圍，共 14 個）**：
- L1：`ic` / `ic_ir` / `hit_rate` / `ic_trend` / `quantile_spread` /
  `monotonicity` / `top_concentration` / `turnover` / `breakeven_cost` /
  `net_spread` / `oos_decay`
- L2 opt-in：`regime_ic` / `multi_horizon_ic` / `spanning_alpha`

**`oos_decay`** 對應 `multi_split_oos_decay`——經 PR 0a migration 後回
`MetricOutput`（value=`survival_ratio`、metadata 含 `sign_flipped` /
`status` / `per_split`），跟其他 metric 同 contract 納入 Factor。**排除原案
撤回**（原案因 `OOSDecayResult` 非 `MetricOutput` 排除；現走型別統一 migration
解除限制，詳 §3.13）。

**排除**（不屬 single-factor metric、概念邊界）：
- `greedy_forward_selection` / `redundancy_matrix`：**batch / multi-factor
  操作**，屬 `ProfileSet` 不屬 `Factor`。Single-factor facade 加這類方法
  語意錯誤（single → batch 越界）
- `compute_*` DataFrame helpers：**prep helper 不是 metric**，`build_artifacts`
  會做，使用者要手動用走 function 層（`fl.compute_ic(df)`）

`breakeven_cost` / `net_spread` 需要其他 metric 的結果當輸入，Factor method
internally 走 `self.quantile_spread().value` 跟 `self.turnover().value`
（hit cache，不重算）後轉呼叫 primitive——UX 一致、使用者不用自己串。

**(b) 單一 `Factor` class，`__getattr__` 動態 expose metric**
```python
class Factor:
    def __getattr__(self, name):
        if name in _METRIC_REGISTRY[self.factor_type]:
            return lambda **kw: _dispatch(name, self.artifacts, **kw)
        raise AttributeError
```
- IDE 不會 autocomplete 動態 attr，**主要 UX 目標破功**
- 新增 metric 只動 registry 但驅動不了 autocomplete
- Runtime error 比靜態 method 更容易遲發現
- 否決

**(c) Builder pattern `f.metric("ic").run()`**
- 多一層無用的 indirection
- 否決

### 3.4 Per-call kwargs：是否允許蓋過 bound config？

情境：`f = fl.factor(df, name, config=cfg)` 已 bind `cfg.n_groups = 5`，
使用者想 `f.quantile_spread(n_groups=10)` 看敏感度。

**(a) 允許 per-call override（只對本次 call 生效，不改 bound config）**。**建議**。
- 研究直覺：「大多用預設，偶爾改一兩個參數試敏感度」
- 實作上就是 method 接 kwargs、用 `n_groups=kwargs.get("n_groups", self.config.n_groups)` 覆寫
- **注意 cache invalidation**：如果 override 改了會影響 cached intermediate
  的參數（例如 `n_groups` 改了、`spread_series` 就不能用了）——method
  要偵測這個 case 並**繞過** cache（直接對 `artifacts.prepared` 重算），
  **不污染** cached `spread_series`。文件講清楚：「override 生效，但不寫回
  Factor cache」
- 覆寫後的結果**不**寫回 `artifacts.metric_outputs`（避免 `f.evaluate()`
  拿到 user-overridden 結果進 Profile）

**(b) 不允許 override，使用者要重 bind**
- 每次敏感度測試都要新 `fl.factor(df, name, config=replace(cfg, n_groups=10))`
- research ergonomics 明顯倒退
- 否決

### 3.5 `fl.factor().evaluate()` 跟 `fl.evaluate()` 的等價關係

**(a) 兩者**結果等價、cache 共用**，`fl.evaluate` 是 `fl.factor().evaluate()` 的單次 shorthand**。**建議**。
```python
# 以下兩行產出相同 profile（value-equal）
p1 = fl.evaluate(df, "Mom_20D", config=cfg)
p2 = fl.factor(df, "Mom_20D", config=cfg).evaluate()
```
- `fl.evaluate` implementation 維持不動——還是走 `_evaluate_one`
- `Factor.evaluate()` 也走 `_evaluate_one`（傳 `self.artifacts`），或
  refactor 成共享 helper
- **關鍵：如果使用者先 call 過單 metric，`f.evaluate()` 不重算**——因為
  `_evaluate_one` 讀的是 `artifacts.intermediates`，而那些本來就是 eager
  build 出來的。單 metric 只是讀 intermediate 跑統計，**不改 cache**
- 同理，`f.evaluate()` 跑完後、使用者再 call `f.ic()`，`f.ic()` 也**不重算**
  ——讀 `artifacts.metric_outputs["ic"]` 直接回（`from_artifacts` 存過了）

**(b) 單 metric 呼叫 → evaluate 會重跑**
- 實質重複工作
- 否決（違反 Factor 存在的初衷：shared cache）

### 3.6 Metric method 的實作 = 直接呼叫 primitive + wire up artifacts

**(a) 每個 Factor method 是薄 adapter，內部拉對應 intermediate / prepared
然後 call 既有 `factrix.metrics.*` primitive**。**建議**。
```python
class CrossSectionalFactor(Factor):
    def ic(self, forward_periods: int | None = None) -> MetricOutput:
        from factrix.metrics.ic import ic as _ic
        fp = forward_periods if forward_periods is not None else self.config.forward_periods
        cached = self.artifacts.metric_outputs.get("ic")
        if cached is not None and forward_periods is None:
            return cached                           # §3.5 cache hit
        out = _ic(self.artifacts.get("ic_series"), forward_periods=fp)
        if forward_periods is None:
            self.artifacts.metric_outputs["ic"] = out  # stash for evaluate()
        return out

    def quantile_spread(self, n_groups: int | None = None) -> MetricOutput:
        from factrix.metrics.quantile import quantile_spread as _qs
        ng = n_groups if n_groups is not None else self.config.n_groups
        cached = self.artifacts.metric_outputs.get("quantile_spread")
        if cached is not None and n_groups is None:
            return cached
        # n_groups override → 不用 cached spread_series（參數不同）
        precomp = self.artifacts.get("spread_series") if n_groups is None else None
        out = _qs(
            self.artifacts.prepared,
            forward_periods=self.config.forward_periods,
            n_groups=ng,
            _precomputed_series=precomp,
        )
        if n_groups is None:
            self.artifacts.metric_outputs["quantile_spread"] = out
        return out
```
- primitive 層 contract 不動（維持給 library authors / power users 用）
- Factor method 只負責「從 artifacts 拉對的東西、把對的 kwargs 餵進去、
  cache 結果」——每個 method 5-15 行
- 跟現有 `from_artifacts` 的 dispatch 邏輯**邏輯等價**但拆到 method 上，
  research 可單點呼叫

**(b) Factor method 直接吃 raw df、內部自己 `compute_ic` → 放棄 cache**
- 「連 call 三個 IC-family metric 默默算三次 IC」的反 pattern
- 否決（這正是使用者原 pain point）

### 3.7 `Factor` 跟 `Artifacts` 的職責分工

兩個類保留各自身分：
- **`Artifacts`**：data-only bundle（`prepared` / `intermediates` /
  `metric_outputs` / `config` / `factor_name` / `compact`）。被
  `build_artifacts` 產出、被 `from_artifacts` / `evaluate_batch` / user-defined
  metric 讀取。**保留 dataclass、不加 method**（避免膨脹成 god class）
- **`Factor`**：research facade，**hold 一份 `Artifacts`** + 提供 metric
  methods + `evaluate()`。使用者主要看到的對象

```python
@dataclass
class Factor:
    artifacts: Artifacts

    @property
    def config(self) -> BaseConfig: return self.artifacts.config
    @property
    def factor_name(self) -> str: return self.artifacts.factor_name

    def evaluate(self, *, return_artifacts: bool = False): ...
```

這樣 **power user 仍可 `f.artifacts` 走既有路徑**（所有下游工具吃
`Artifacts` 的都不受影響）。

**`__post_init__` 驗證**：`Factor` 是 `@dataclass`，使用者**可以**直接
`CrossSectionalFactor(artifacts=some_arts)` 繞過 `fl.factor()` factory。
為防兩個 subtle bug：
1. `artifacts.factor_name == ""`（factory 裡才 set）
2. `factor_type` 跟 subclass 不匹配（`CrossSectionalFactor(artifacts=event_arts)`）

`__post_init__` 做 hard check：
```python
def __post_init__(self) -> None:
    actual_ft = type(self.artifacts.config).factor_type
    if actual_ft != self.EXPECTED_FACTOR_TYPE:
        raise TypeError(
            f"{type(self).__name__} expects factor_type="
            f"{self.EXPECTED_FACTOR_TYPE.value!r}; got {actual_ft.value!r}. "
            f"Use fl.factor(df, name, factor_type=...) instead of "
            f"direct instantiation to pick the right Factor subclass."
        )
    if not self.artifacts.factor_name:
        raise ValueError(
            "Factor.artifacts.factor_name is empty. "
            "Use fl.factor(df, name=...) instead of direct instantiation."
        )
```
Docstring 明講：**「優先走 `fl.factor(df, name)`；直接 instantiate 是
advanced path，需自行確保 artifacts 狀態合法」**。

### 3.8 Factory function `fl.factor()` 的 signature

**(a) 鏡像 `fl.evaluate` 的簽名**。**建議**。
```python
def factor(
    df: pl.DataFrame,
    factor_name: str,
    *,
    factor_type: str | FactorType = "cross_sectional",
    config: BaseConfig | None = None,
    **config_overrides: Any,
) -> Factor: ...
```
- 使用者 `fl.evaluate(...)` 換成 `fl.factor(...)` 不用 re-learn
- 一樣走 `_PREPROCESSED_MARKER` 檢查（`forward_return` 必須存在；保持 strict gate）
- 一樣走 `_config_for_type` 路徑
- 產出時間：跟 `fl.evaluate` 的 eager `build_artifacts` 同級成本（相同操作，
  差別只是回 `Factor` 而非 `Profile`）

內部：
```python
def factor(df, factor_name, *, factor_type="cross_sectional", config=None, **overrides):
    # 重用 _evaluate_one 的 config 組裝 + strict gate 邏輯
    if config is None:
        config = _config_for_type(factor_type, **overrides)
    elif overrides:
        raise TypeError(...)

    if _PREPROCESSED_MARKER not in df.columns:
        raise ValueError(...)

    artifacts = build_artifacts(df, config)
    artifacts.factor_name = factor_name

    # 根據 factor_type 選 Factor subclass
    factor_cls = _FACTOR_REGISTRY[type(config).factor_type]
    return factor_cls(artifacts=artifacts)
```

### 3.9 Metric method 的 MetricOutput.metadata 是否 read-only？

**(a) 不包 `MappingProxyType`——直接回 primitive 產出的 MetricOutput**。**建議**。
- Primitive 產出的 `MetricOutput.metadata` 本就是 mutable dict
- 只有**存進 `artifacts.metric_outputs`** 的 copy 才包 proxy（跟前一個 spike 規則一致）
- Factor method 回給使用者的是「本次 call 的產出」——使用者拿到自己的 dict，
  可以愛怎麼改怎麼改；**污染風險在「存進 Artifacts」那一步就擋住了**
- 不加額外 overhead

### 3.10 錯誤處理：factor_type 跟 metric 不匹配怎麼辦？

情境：`f = fl.factor(df, ..., factor_type='event_signal')`，使用者 call
`f.ic()`——IC 是 CS 專屬 metric。

**(a) `AttributeError`——method 根本不存在於 `EventFactor`**。**建議**。
- 直接來自 §3.3 的 per-type subclass 設計
- 靜態檢查（mypy / IDE）就會 underline red
- Python 慣例錯誤
- 訊息建議：custom `__getattribute__` 在 AttributeError 時加 hint「CS 因子
  才有 `.ic()`；event_signal 可用 `.caar()` / `.bmp_test()` / ...」

**(b) 允許 cross-type call、internal raise `TypeError`**
- 違反 §3.3 per-type subclass 的初衷；不否決但不採用

### 3.12 Cache 架構：`from_artifacts` 跟 Factor 共用單一 source of truth

**背景問題**：`cross_sectional.py::from_artifacts` L154-181 目前的 body 形如：
```python
outputs = dict(artifacts.metric_outputs)                              # 複製 L2 pre-populate
ic_m = _stash(outputs, ic_metric(ic_series, forward_periods=fp))       # L1：無條件重算
spread_m = _stash(outputs, quantile_spread(artifacts.prepared, ...))   # L1：無條件重算
```
L1 metric 無條件重算並 `_stash` 覆寫——只有 L2 metric（`regime_ic` /
`multi_horizon_ic` / `spanning_alpha`）走 `_augment_level2_intermediates`
的 pre-populate pattern，from_artifacts 才會保留。

**影響**：若 Factor method 把單次 call 結果寫回
`artifacts.metric_outputs["ic"]`，之後的 `f.evaluate()` 在 `from_artifacts`
裡會**被覆蓋掉**——Factor cache 對 evaluate 無效、cache「共用」合約破功
（§3.5）。

**(a) 改 `from_artifacts` 的 L1 也走 pre-populate pattern，跟 L2 對齊**。**建議**。
```python
# 新 pattern（每個 L1 metric 加 guard）
if "ic" in outputs:
    ic_m = outputs["ic"]                                  # cache hit
else:
    ic_m = _stash(outputs, ic_metric(ic_series, forward_periods=fp))
```
- 單一 source of truth：`artifacts.metric_outputs` 是 Factor 跟 Profile 共用的 cache
- Factor method cache hit / miss 邏輯簡單：讀寫 `artifacts.metric_outputs` 一處
- Config 一致性保證：Factor 跟 `from_artifacts` 都用 `artifacts.config`，
  Factor 寫入的 cache 在 `from_artifacts` 讀出時參數必然相同（§3.4 per-call
  override 路徑刻意不寫回 cache，避開這個 hazard）
- 成本：4 個 `from_artifacts`（CS / Event / MacroPanel / MacroCommon）約
  10 處 `_stash(outputs, fn(...))` 各包一層 `if k in outputs` guard——機械
  改動、不影響語意
- **副產品好處**：future 若有人在 `build_artifacts` 階段 pre-compute L1
  metric（e.g. 對 regime split 時一次把 base IC 算好）也直接相容

**(b) Factor 自己持 `_local_cache: dict`，跟 `artifacts.metric_outputs` 分離**
- 兩份 cache：Factor 本地（research）vs artifacts（Profile 建構）
- 使用者 call `f.ic()` 命中 `_local_cache`；`f.evaluate()` 走 `from_artifacts`
  不讀 `_local_cache` → 實質**重算一次 ic_metric**
- 優點：Factor impl 跟 from_artifacts 邏輯完全解耦
- 缺點：研究者期待的 "call 過再 evaluate 不重算" 破功；§3.5 合約變質
- 否決

**(c) Factor 不寫回 cache，每次 `f.ic()` 都重算**
- 研究者連續 `f.ic(); f.ic(); f.ic()` 會算三次
- 違反 Factor 存在的初衷
- 否決

**選 (a)**。實作時 4 個 Profile 的 `from_artifacts` 加 guard 當作本 spike
附帶的小 refactor（約 0.2d，歸到 §5 PR 1）。

### 3.13 `factrix.metrics.*` 統一回 `MetricOutput`（single return contract）

**背景**：審計 `factrix/metrics/*.py` 顯示 29 個 metric 已回 `MetricOutput`，
但 4 個**偏離 contract**：

| 函式 | 現 return | 問題 |
|---|---|---|
| `multi_split_oos_decay` | `OOSResult`（custom dataclass） | 獨立 type → Factor method 無法與其他 metric 用同簽名 |
| `event_around_return` / `multi_horizon_hit_rate` / `mfe_mae_summary` | `MetricOutput \| None` | None 分歧 → caller 要特判、serialize 路徑要特判、`_insufficient_metrics` 要特判 |

**對北極星的傷害**：
- 簡單直覺（軸 1）：使用者呼叫 `multi_split_oos_decay` 得學獨立 type；call
  event 3 metric 要記得 `is not None` 判斷
- 開發最佳實踐（軸 4）：**special-case 擴散**——`from_artifacts` 要讀
  `.survival_ratio` 特例、`None` 要 guard、serialize 要雙路徑；Factor facade
  若不統一型別就被迫加 wrapping 層、每加一個特殊 metric 就要改 Factor
- 彈性（軸 2）：使用者寫 `for m in arts.metric_outputs.values(): print(m.value)`
  會因為 None / OOSResult 不在 dict 裡而「看不到」這些 metric，破壞 uniform
  iteration

**(a) 統一 migrate 到 `MetricOutput`（單一 return contract）**。**建議**。
- `MetricOutput.metadata` 夠彈性容納所有 extra 欄位（`per_split` dict list /
  `sign_flipped` bool / `status` string / `reason` for short-circuit）
- `OOSResult` / `SplitDetail` 保留作 **internal computation helper**
  （calculation 中間步驟），只有最外層 public function 回 `MetricOutput`；
  不破壞內部實作
- `None` → short-circuit `MetricOutput(value=0.0, metadata={"reason": "no_price_data", "p_value": 1.0})`，
  對齊其他 metric 的 `insufficient_` pattern
- Breaking：使用者 / test code 讀 `.survival_ratio` 要改 `.value`、讀
  `.sign_flipped` 要改 `.metadata["sign_flipped"]`；讀 `None` 檢查要改
  `metadata.get("reason") == "no_price_data"`——library 內部跟 tests 範圍限定，
  不洩漏到 `fl.evaluate()` end-user 層

**(b) 維持混合 type，Factor 層 wrapping**
- 破壞軸 4（special-case），每加一個特殊 metric Factor 就要改
- 否決

**(c) 只修 None、保留 `OOSResult`**
- 折衷但不完整；Factor 仍要為 `OOSResult` 單獨 wrap
- 否決

**選 (a)**。排入 §5 **PR 0**（prerequisite），4 個函式一次性改完、
所有 caller + tests 同 PR 遷移。

### 3.11 `f.evaluate()` 的 `return_artifacts` 是否保留？

情境：`fl.evaluate(df, name, return_artifacts=True)` 存在是因為沒
`Factor` 時的唯一 artifacts 取得管道。有了 `f.artifacts` property 後，
這個 flag 有沒有必要？

**(a) `Factor.evaluate()` 不帶 `return_artifacts`，只回 profile；artifacts
走 `f.artifacts`**。**建議**。
```python
f = fl.factor(df, "x")
profile = f.evaluate()      # 只回 profile
arts = f.artifacts          # artifacts 另外取
```
- API 更正交：一個 method 一個職責
- `fl.evaluate` 保留 `return_artifacts` 不變（向後相容 end user code）

**(b) 對稱保留 `Factor.evaluate(return_artifacts=True)`**
- 使用者從 `fl.evaluate` 切過來時不用改思維
- 但有 `f.artifacts` property 就顯得 redundant
- 二選一，視實作時再定；偏 (a)

## 4. 非目標

- **不改 `fl.evaluate` / `fl.evaluate_batch` signature**——`fl.factor` 是
  純加層，已存在的 API 維持 bit-for-bit 不動
- **不改 `factrix.metrics.*` primitive signature**——保留作為 library
  authors / power users / Factor internal 的底層；docs 上標為 "low-level"
- **不改 `Artifacts` dataclass**（§3.7）——保持 data-only bundle 身分
- **不做 `fl.factor_batch()`**——batch 情境應直接用 `fl.evaluate_batch`；
  如果真有「batch 後 drill-down 想用 Factor」的需求，下個 mini-spike 再
  評估 `ProfileSet.to_factors(arts_map)` 或類似 adapter
- **不做 custom metric 的 Factor method 擴展機制**——新增 metric 的路徑
  還是「在 library 內新增 primitive + 在對應 Factor subclass 加 method」，
  有意識 gating（同 `spike_metric_outputs_retention.md:72-96` 結論）
- **不自動 chart / 不 `__repr__` 印摘要**——`Factor` 是 research tool，
  不預期 `print(f)` 當診斷出口
- **不對 CS 外其他 factor_type 做第一版實作**——P0 只做
  `CrossSectionalFactor`（dogfood 飽和度最高），其餘 3 類（Event / MacroPanel
  / MacroCommon）跟進在 PR 2 / 3
- **不做 method chaining（`f.ic().then(...)`）**——metric 是 leaf，不 compose
- **不做 `f.diagnose()` / `f.verdict()` 便利方法**——這些是 Profile 的
  身分，`f.evaluate().diagnose()` 兩步已經夠簡；避免 `Factor` API
  膨脹成「什麼都能做」
- **不改 `build_artifacts` signature**——仍是 `(df, config) -> Artifacts`

## 5. 實作順序（待 §7 sign-off 後）

估時：**1.6-2.1d**，分 ≤6 個 PR（原子 merge，避免中間 state）

0. **PR 0a — 型別統一 migration（§3.13）**：
   - `multi_split_oos_decay` 改回 `MetricOutput`；`OOSResult` / `SplitDetail`
     保留作 internal helper；caller (`cross_sectional.py::from_artifacts`
     L167) 跟 tests 同 PR 遷移（`.survival_ratio` → `.value`、
     `.sign_flipped` → `.metadata["sign_flipped"]`）
   - Tests 綠、Profile 欄位值（`oos_survival_ratio` / `oos_sign_flipped`）
     **bit-for-bit 不變**
   ——**0.2d**

0b. **PR 0b — event None 消除（§3.13）**：
   - `event_around_return` / `multi_horizon_hit_rate` / `mfe_mae_summary`
     的 `return None` → short-circuit `MetricOutput(value=0.0,
     metadata={"reason": "...", "p_value": 1.0})`
   - `event.py::from_artifacts` 相關 `if x is not None` caller guard
     簡化成讀 `metadata["reason"]` 或直接信任 value
   - Tests 綠、event Profile 欄位值不變
   ——**0.2d**

1. **PR 1 — L1 cache pre-populate guard（§3.12）**：`cross_sectional.py`
   （event.py / macro_panel.py / macro_common.py 同 session 一起改）的
   `from_artifacts` 加 `if k in outputs` guard，單獨一個 PR / commit，
   既有 tests 全綠（行為不變，因為第一次呼叫時 `outputs` 都是空）。
   ——**0.2d**
2. **`factrix/factor.py` 新模組**：`Factor` base + `CrossSectionalFactor`
   subclass + `_FACTOR_REGISTRY`。Factor method 逐一 wire up 到
   `factrix.metrics.*` primitive，含 §3.4 per-call override +
   §3.6 cache 讀寫邏輯 + §3.7 `__post_init__` 驗證 +
   §3.10 cross-type AttributeError message。
   ——**0.4d**
3. **`factrix/_api.py::factor` factory function**：`fl.factor(...)`
   寫在 `_api.py`（跟 `evaluate` 放同檔），簽名鏡像 `evaluate`（§3.8），
   dispatch 到 `_FACTOR_REGISTRY`。Top-level export
   (`factrix/__init__.py`) 加 `factor` + `Factor` / `CrossSectionalFactor`
   type hint。——**0.1d**
4. **Tests `tests/test_factor_session.py`**：
   - `fl.factor(df, name)` eager build；重複 call 同 metric 不重算（mock
     `compute_ic` 驗 call count=1）
   - Per-call override (§3.4) 不污染 cache：`f.quantile_spread(n_groups=10)`
     後 `f.artifacts.metric_outputs.get("quantile_spread")` 仍是原 cfg 的結果
     （或不存在）
   - `fl.evaluate(df, name, config=cfg)` 跟 `fl.factor(df, name, config=cfg).evaluate()`
     回 value-equal Profile（§3.5）
   - `f.evaluate()` 跑完後 `f.ic()` 不重算（從 `artifacts.metric_outputs` 讀）
   - `EventFactor` 沒有 `.ic()`——AttributeError with CS-only hint
   - Strict gate：`df` 缺 `forward_return` → ValueError 同 `fl.evaluate`
   - `__post_init__` 驗證：`CrossSectionalFactor(artifacts=event_arts)` →
     TypeError；`CrossSectionalFactor(artifacts=arts_no_name)` → ValueError
   ——**0.4d**
5. **Event / MacroPanel / MacroCommon 跟進**（可拆 PR 2 / 3，或一起 merge）：
   每個 factor_type 一個 Factor subclass，method 集合比照該 Profile 的
   metric 呼叫清單。——**0.4d**
6. **Docs**：README 加 "Research workflow" section 對比
   `fl.evaluate` / `fl.factor`；demo.ipynb 加 section 展示
   `f.ic()` + `f.quantile_spread()` + `f.evaluate()`；
   `factrix.metrics.*` docstring 加 "low-level primitive — for research
   use `fl.factor()`"。——**0.2d**

## 6. 風險

- **Factor subclass 每 metric 新增需同步更新**：新增 CS metric 要同時改
  `CrossSectionalProfile.from_artifacts` 跟 `CrossSectionalFactor`——
  多一個地方要維護。**緩解**：test 驗「CS Profile 用到的每個 metric 在
  `CrossSectionalFactor` 都有對應 method」，CI fail-fast。
- **Cache 讀寫邏輯 subtle**：§3.4 per-call override 情境要繞 cache、§3.5
  cache hit 走 `artifacts.metric_outputs`——邏輯走錯會 silently 回錯結果。
  **緩解**：tests 驗 cache hit / miss / override 三條 path 各自的 call count。
- **使用者 mutate `f.artifacts.prepared`**：Factor cache 會過期但無人知。
  **緩解**：docstring 講 snapshot 語意；避免提供 `f.refresh()` 之類的
  method（強化「immutable 一次性」感）。
- **`fl.factor` 跟 `fl.evaluate` 概念疊加**：使用者可能問「什麼時候用哪個？」
  **緩解**：README 明寫「production 一次性 → `fl.evaluate`；research 多次
  呼叫 → `fl.factor`」；保留兩者，不要試圖統一成一個。
- **per-call override 的回傳**：使用者 call `f.quantile_spread(n_groups=10)`
  拿到結果、以為這個結果進了 `f.evaluate()`——結果沒有（§3.4 決定不污染
  cache）。**緩解**：docstring 明講；若有體感痛點再開 flag 如
  `persist=True` 顯式寫回。
- **Factor 非 thread-safe**：`f.ic()` 寫 `artifacts.metric_outputs["ic"]`
  是非原子；多 thread 共用同一個 `Factor` 會 race。**緩解**：docstring
  明列「單 Factor 一 worker」；batch 場景走 `evaluate_batch` 而非
  Factor loop；未來若有併發使用情境，加鎖或 per-thread Factor instance。

## 7. Sign-off checklist

- [ ] Owner 確認 §3.1 命名 `fl.factor()` 小寫 factory + `Factor` class
      type hint（建議 (a)）
- [ ] Owner 確認 §3.2 eager `build_artifacts` 語意（建議 (a)）
- [ ] Owner 確認 §3.3 per-factor-type subclass 架構（建議 (a)；vs 動態 getattr）
- [ ] Owner 確認 §3.4 per-call kwargs override 機制 + 不污染 cache 的
      policy（建議 (a)）
- [ ] Owner 確認 §3.5 `fl.factor().evaluate()` 跟 `fl.evaluate()` cache
      共用的 value-equal 合約（建議 (a)）
- [ ] Owner 確認 §3.6 Factor method 作為「薄 adapter」呼叫既有 primitive，
      primitive signature 不動（建議 (a)）
- [ ] Owner 確認 §3.7 `Artifacts` 保留 data-only 身分、`Factor` 走
      composition over inheritance（非 Artifacts 子類）
- [ ] Owner 確認 §3.8 `fl.factor()` 簽名鏡像 `fl.evaluate`（建議 (a)）
- [ ] Owner 確認 §3.10 cross-type metric call 走 AttributeError + hint
      message（建議 (a)）
- [ ] Owner 確認 §3.11 `Factor.evaluate()` 不帶 `return_artifacts`（artifacts
      走 property，建議 (a)）；或保留對稱 flag
- [ ] Owner 確認 §3.12 cache 策略：`from_artifacts` 的 L1 metric 加
      `if k in outputs` guard、跟 L2 對齊成單一 source of truth（建議 (a)）
- [ ] Owner 確認 §3.13 型別統一 migration：`multi_split_oos_decay`
      → `MetricOutput`；event 3 個 metric `None` → short-circuit
      `MetricOutput`（建議 (a)）；排入 PR 0a / 0b prerequisite
- [ ] Owner 確認 P0 CS method set = 14 個（`ic` / `ic_ir` / `hit_rate` /
      `ic_trend` / `quantile_spread` / `monotonicity` / `top_concentration` /
      `turnover` / `breakeven_cost` / `net_spread` / `oos_decay` +
      L2 opt-in `regime_ic` / `multi_horizon_ic` / `spanning_alpha`）；
      **排除** `greedy_forward_selection` / `redundancy_matrix`（batch 操作，
      屬 ProfileSet，§3.3）
- [ ] Owner 確認 §3.7 `Factor.__post_init__` 驗證 factor_type / factor_name
- [ ] Owner 確認 P0 scope = CrossSectionalFactor；Event / Macro 跟進在 PR 2 / 3

## 8. Future work（不進本 spike，但記起來）

以下是 review 時可能被提到、但已評估不進本 session 的項目：

- **Batch 整合**：`evaluate_batch(keep_artifacts=True)` 回的
  `dict[name, Artifacts]` 要不要有 `to_factors()` adapter 讓使用者可以
  `factors[name].ic()`？現階段 `arts_map[name]` + 兩行手 wrap 就夠，
  累積實戰 call site 再評估
- **Custom metric 註冊到 Factor**：使用者寫了 own metric、能不能
  `CrossSectionalFactor.register_method("my_metric", my_fn)`？目前沒有
  custom metric 使用者，等真有人寫再評估（同 `spike_metric_outputs_retention.md` 結論）
- **`Factor` 的 `__repr__`**：印 `<CrossSectionalFactor "Mom_20D" n=1200
  periods=250>` 這類摘要——跟 `Profile __repr__` 同屬 mini-issue，可一起做
- **per-call override 顯式寫回 cache**：`f.quantile_spread(n_groups=10, persist=True)`
  把結果寫回 `artifacts.metric_outputs`，讓之後的 `f.evaluate()` 用 override
  的值。目前預設不寫回；若實戰有需求再評估
- **Factor ↔ Artifacts 轉換 helper**：`fl.factor_from_artifacts(arts)` 讓
  已持 `Artifacts` 的使用者直接升級成 Factor。目前 `Factor(artifacts=arts)`
  直接 instantiate 即可，不急
- **`quantile_spread` 敏感度 grid 的 spread_series memoization**：§3.4
  override 路徑每次都重算 spread_series（per-group returns × N periods）。
  單次 ms 級可接受；grid search（n_groups ∈ [3,5,7,10,20] × N factor）時
  累積顯著。若實戰有人抱怨，加 `@lru_cache`-style memoization 於
  `compute_spread_series`，key 含 `(n_groups, forward_periods, prepared id)`
- **`multi_split_oos_decay` 整合進 MetricOutput interface**：目前回
  `OOSDecayResult`（非 `MetricOutput`），Factor P0 排除。若要進 Factor
  method set，需先把 `OOSDecayResult` 的 `survival_ratio` / `sign_flipped`
  等欄位 roundtrip 到 MetricOutput.value / .metadata，獨立 mini-spike
- **from_artifacts guard condition 的 L2 統一**：§3.12 (a) 把 L1 也改成
  `if k in outputs: use cache else compute`；L2 目前也類似但邏輯分散在
  `_augment_level2_intermediates`。未來把兩者統一成一個 helper
  `_memoized_call(outputs, name, fn, *args)` 收斂
- **Factor `compact=True` 支援**：若使用者從 `evaluate_batch(compact=True)`
  的 artifacts_map 構造 Factor，呼叫需要 `prepared` 的 metric（e.g.
  `quantile_spread`）會觸發 `_CompactedPrepared` 的 RuntimeError。目前
  P0 只支援 `fl.factor(df, name)` path（`prepared` 必存），compact path
  等 batch drill-down adapter 出來再考慮
