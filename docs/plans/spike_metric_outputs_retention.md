# Spike — MetricOutput retention on Artifacts for detail views

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> **IMPLEMENTED 2026-04-19** (merge `235b0d1`, core refactor `1f2676e`):
> `Artifacts.metric_outputs` dict added, `from_artifacts` is now a pure
> `(profile, metric_outputs_dict)` tuple, and `describe_profile_values`
> surfaces per-regime / per-horizon / spanning detail. 2026-04-20
> follow-up (commit `dcbe346`): both call sites of the
> `from_artifacts → assign metric_outputs` dance now share the
> `_run_profile_and_attach` helper in `profiles/_base.py`.

**狀態**：IMPLEMENTED（見上方）
**Owner**：jason pan
**日期**：2026-04-19
**解鎖的後續工作**：`fl.describe_profile_values(profile, arts)` + per-regime /
per-horizon / spanning 的 drill-down views，使用者不用重跑 standalone metric
就能看 detail。

---

## 1. 問題陳述

Profile 是 `frozen=True, slots=True` 的 flat scalar dataclass。目前
`from_artifacts()` 呼叫每個 metric 後，從 `MetricOutput` 抽
`.value` / `.stat` / `metadata['p_value']` 填進 Profile 欄位，然後**丟棄**
MetricOutput 物件本身。

造成的 gap：
- **Detail 消失**：`regime_ic.metadata["per_regime"] = {bull: {...}, bear: {...}}`、
  `multi_horizon_ic.metadata["per_horizon"] = {1: {...}, 5: {...}}`、
  `spanning_alpha` 的 per-base-factor betas——這些**全部丟掉**，Profile 只留
  scalar 摘要（`regime_ic_min_tstat` / `multi_horizon_ic_retention` 等）
- **Pretty-print 要靠反推**：想做 `describe_profile_values(profile)` 印每個
  metric 的 value/stat/p/sig 四維度對齊顯示，必須硬編
  `_FIELD_FAMILY_MAP`（哪些 Profile 欄位同屬 `ic` family 等）——加 metric
  就得同步改 mapping
- **Standalone re-run 成本**：研究者要 drill-down 只能重呼叫
  `regime_ic(ic_series, ...)`，等於第二次做同樣計算

想要的 API（目標）：

```python
profile, arts = fl.evaluate(df, "Mom_20D", config=cfg, return_artifacts=True)

# Detail view — scalar 表 + per-regime / per-horizon / spanning sections
fl.describe_profile_values(profile, arts)
fl.describe_profile_values(profile, arts, include_detail=False)  # 只印 scalar

# Targeted drill-down — raw MetricOutput 直接讀
arts.metric_outputs["regime_ic"].metadata["per_regime"]
arts.metric_outputs["multi_horizon_ic"].metadata["per_horizon"]
```

關鍵：**Profile 本身不變**（scalar invariant），多視角都從 Artifacts 抽；
整合成**單一 public function**，`include_detail=True` 時自動 discover 啟用
的 opt-in metric 並印出對應 detail section（regime / multi_horizon /
spanning）。

**Scope boundary**：本 spike **不**處理「基本 pretty-print」—— `print(profile)`
或 Jupyter 的 `_repr_html_`。那是獨立的 Profile `__repr__` 改善，開
另個 mini-issue。`describe_profile_values` 的定位是「**需要 artifacts 才能
給出的完整視圖**」（scalar + detail sections），不是日常 REPL 的第一印象。

## 2. 現況（source-code 驗證）

- `factorlib/evaluation/_protocol.py` — `Artifacts` 只有 `prepared` /
  `intermediates` / `config` / `factor_name`，**沒有** per-metric MetricOutput 存放處
- `factorlib/evaluation/profiles/cross_sectional.py::from_artifacts` L126-254
  呼叫 `ic_metric` / `ic_ir_metric` / `hit_rate` / `ic_trend` / `monotonicity` /
  `multi_split_oos_decay` / `quantile_spread` / `turnover` / `breakeven_cost` /
  `net_spread` / `q1_concentration` 共 ~10 個 metric，各自 unpack 後 MetricOutput 棄置
- `factorlib/metrics/ic.py::regime_ic` 的 MetricOutput 在 `_augment_level2_intermediates`
  裡只讀 `metadata["per_regime"]` 去算 `min_tstat` / `consistent` 存成 1-row
  DataFrame `intermediates["regime_stats"]`——per-regime dict 沒留
- 同理 `multi_horizon_ic` 只留 retention / monotonic，`spanning_alpha` 只留 t / p
- `factorlib/_types.py::MetricOutput` 是 `@dataclass`（非 frozen）；`metadata: dict`
  是 mutable

## 3. 待決策的政策問題

### 3.1 MetricOutput 放哪？

**(a) `Artifacts.metric_outputs: dict[str, MetricOutput]`**

與既有 `intermediates` 平行，統一在 Artifacts 下。`keep_artifacts=True`
opt-in、`compact=True` 不碰（dict 很小）。

- 優點：統一 mental model；Profile 不動；detail view 吃 `(profile, arts)` 乾淨
- 缺點：`from_artifacts` 變 impure（mutate 輸入 Artifacts），要 docstring 講清楚

**(b) Profile 新增私有欄位 `_metric_outputs: dict`**

- 破壞 `frozen=True` 保證
- 破壞 polars-native（`pl.Object` dtype 讓 `filter`/`rank_by` 爛掉）
- 破壞 MLflow serialization（`asdict` 出 nested 不定 schema）
- **否決**

**(c) `evaluate()` 回 tuple `(profile, metric_outputs_dict)`**

- 再多一條 return path（現有 `return_artifacts` 已經複雜化過一次）
- `evaluate_batch` 批次場景要再長出 `metric_outputs_map`
- 使用者體感：兩個 opt-in flag 會混淆

**建議**：**(a)**。加 Artifacts 欄位，走既有 `return_artifacts` / `keep_artifacts`
opt-in 機制。

### 3.2 MetricOutput 的 keys 誰 canonical？

兩個選項：

**(a) `MetricOutput.name`**——metric function 自己返回的 name 欄位（`"ic"` /
`"q1_q5_spread"` / `"regime_ic"` 等）。**建議**。
- 既有、穩定、metric 自己就知道
- 同一 metric 不同 config（e.g. 兩個 regime_labels）會 collision——但單 factor
  下每個 metric 只跑一次，不會撞

**(b) Profile 欄位 prefix**（`"ic"` / `"q1_q5_spread"`）
- 跟 Profile scalar 欄位命名綁一起——但前面 §2.6 已論證過 Profile 欄位不是
  1:1 對應 MetricOutput（`ic_ir` 從 `ic_series` 算，不是 metric function 回傳的）
- 加 metric 要同步記 prefix，抬高維護成本

### 3.3 Contract：哪些 keys 是 canonical、哪些是 opt-in？

Contract 分兩層：

**Always present**（CS 例，每個 factor_type 有自己的 always-set）：
- `ic`, `ic_ir`, `hit_rate`, `ic_trend`, `monotonicity`, `q1_q5_spread`,
  `q1_concentration`, `multi_split_oos_decay`, `turnover`, `breakeven_cost`,
  `net_spread`

**Opt-in present**（config 啟用才存）：
- `regime_ic`（當 `config.regime_labels` 給）
- `multi_horizon_ic`（當 `config.multi_horizon_periods` 給）
- `spanning_alpha`（當 `config.spanning_base_spreads` 給）

`describe_profile_values` 在 `include_detail=True` 時 auto-discover：key
不存在 → 對應 section **省略不印**（不 raise）。使用者直接 index
`arts.metric_outputs["missing_key"]` 才會 `KeyError`——那是 raw MetricOutput
層級，訊息是 Python 預設 dict KeyError。

### 3.4 `from_artifacts` 的 impurity 怎麼框

問題：`from_artifacts` 目前是 **public classmethod**（沒底線前綴），語法上
就是 public API。若 impl 讓它 mutate 輸入 artifacts，使用者直接 call 時會
觸發 silent side effect——「使用者不會直接 call」只是行為假設。三派：

**(a) 改名 `_from_artifacts` 標 internal + docstring 明講 contract**：
- Python 慣例底線前綴 = internal；不阻擋使用者硬 call，但 signal 清楚
- `evaluate` / `evaluate_batch` 改 call `_from_artifacts`
- 成本：每 Profile class 改名 + `@register_profile` 登錄邏輯要讀 `_from_artifacts`
- Impurity 合法化：internal function 允許 side effect 是 Python 慣例
- **否決**：public API 被降級為 internal，寫 custom Profile subclass 的使用者
  必須 implement 帶 underscore 的 method，違反 Python 慣例（subclass hook
  不該是 underscore-prefixed）；且 signal 「不要 call」但其實是 profile
  建構的唯一路徑，矛盾

**(b) 維持 public `from_artifacts` + 改 signature 回 tuple
`(profile, metric_outputs_dict)`**。**建議**（使用者友善方向）。
- Pure function，沒副作用爭議——使用者/subclass 開發者能清楚推論
- API 保持 public（classmethod constructor 本來就是 Profile 建構的對外契約）
- 由 caller (`_evaluate_one` / `evaluate_batch`) 塞回 artifacts：
  ```python
  profile, metric_outputs = profile_cls.from_artifacts(artifacts)
  artifacts.metric_outputs = metric_outputs
  ```
- 契約對 custom Profile subclass 作者明確：「回 (profile, dict)」
- 成本：signature 改動波及 4 個 Profile class + `@register_profile` 登錄
  protocol check + 9 個 test file tuple unpack；但全部是 library 內部 +
  tests，**不外洩到 end-user**（end-user 走 `fl.evaluate()`，不直接 call
  `from_artifacts`）

**(c) 維持 public `from_artifacts` + docstring 警告**：
- 最小改動但語法上不誠實；reviewer 糾過，否決

### 3.5 MetricOutput 的 mutation 如何防

目前 `@dataclass`（非 frozen）+ `metadata: dict`（mutable）。使用者能 mutate。
`metric_outputs` 存在 Artifacts 上後，mutation 風險放大——REPL 使用者很容易
順手 `arts.metric_outputs["ic"].metadata["note"] = "看起來 OK"` 當貼 tag 用，
污染後 `describe_profile_values` / 其他 downstream 讀 `metadata["p_value"]` 就不穩。

**(a) `from_artifacts` 存進 metric_outputs 時把 metadata 包
`types.MappingProxyType`（read-only view）**。**建議**。
```python
from types import MappingProxyType
artifacts.metric_outputs["ic"] = MetricOutput(
    name=ic_m.name, value=ic_m.value, stat=ic_m.stat,
    significance=ic_m.significance,
    metadata=MappingProxyType(ic_m.metadata),  # 包一層 read-only
)
```
- Metric function 內部 `ic_m.metadata` 仍是 dict（建構過程可寫），只有存進
  Artifacts 的**複本**被包 proxy——使用者寫會 `TypeError`
- 實質保護 > 口頭警告；成本一個 import 跟 per-metric 一行
- 不改 MetricOutput dataclass 本身（不 breaking）

**(b) 留原 dict + docstring 警告**：
- Python 友善但無強制；AI agent 會遵守，人類 REPL 不一定
- 污染後 debug 路徑難追（沒 stack trace 指向亂改的那行）

**(c) MetricOutput 整個 `frozen=True`**：過度、breaking——metric function 建構
中多步累積 metadata 會爛。

**殘留風險（proxy 擋不住的）**：即使 metadata 包 proxy，使用者仍可
`metric_output.value = 99.0` 改 top-level 欄位（dataclass 非 frozen）。
且 `MetricOutput.__repr__` 不標示「這份已被 library 視為 read-only」，
視覺上看不出跟一般 MetricOutput 差異。**緩解**：docstring 警告 + 實戰真有人
踩再評估整個 frozen。若日後 frozen 化，同時改 repr 加 `[stashed]` 或類似標記。

### 3.8 Batch drill-down 流程本 session 不做

`evaluate_batch(keep_artifacts=True)` 回 `(ProfileSet, dict[str, Artifacts])`。
使用者要 drill-down top-K 時要手動 `arts_map[profile.factor_name]`——name
對應靠使用者自己管。`ProfileSet` 沒暴露 `get_artifacts(name)` / iterator。

這個 spike **不**解決它——理由：
- 單 profile 的 drill-down API 還沒穩定，先 dogfood 單個 case
- Batch drill-down 可能有更好的 pattern（例如 `ProfileSet` 內部綁 artifacts
  回 profile，或新 method `iter_with_artifacts()`），值得另開 spike
- 目前 manual `arts_map[name]` 不 blocker——3 行 code 使用者自己能寫

**非目標**：`ProfileSet.get_artifacts(name)` / `iter_with_artifacts()` /
batch-scope `describe_profile_values(ps, arts_map)`。下個 mini-spike 處理。

### 3.6 `describe_profile_values` 是否支援 fallback（沒 artifacts 時）？

**(a) 強制 artifacts（簽名：`(profile, artifacts)`，artifacts 必填）**。**建議**。
- 目前沒有 production 使用者，不需要先付 fallback 維護成本
- 省掉 `_FIELD_FAMILY_MAP`（per Profile class 的 field→metric_name mapping）
  及其加 metric 時的同步維護債
- 單一 code path：永遠從 `artifacts.metric_outputs` 讀——實作、測試、debug 都簡單
- 使用者沒傳 `return_artifacts=True` 卻想 pretty-print 時，錯誤訊息可以直接
  指向正解：「`fl.evaluate(df, 'x', return_artifacts=True)` 後傳 artifacts 進來」
- 若日後有 real user 抱怨「我就只想看 scalar 不想背 artifacts」，再補 fallback 不遲

**(b) 支援 fallback**（沒 artifacts → `_FIELD_FAMILY_MAP` 從 Profile scalar 重組）。
- 優點：`fl.evaluate(df, "x")` 沒加 `return_artifacts=True` 也能用
- 缺點：mapping 是 dead weight，加新 metric 忘了改 → pretty-print 缺欄位無錯誤
- 預設否決——YAGNI

### 3.7 Detail view 是一個函式還是多個？

**(a) 單一 `describe_profile_values(profile, artifacts, include_detail=True)`
自動 discover opt-in metric 並印 detail sections**。**建議**。
- 使用者八成想看「全部啟用了的 detail」，一個呼叫搞定
- 新加 Level-2 metric 不需要新 public function（namespace 不膨脹）
- Targeted drill-down 走 raw：`arts.metric_outputs["regime_ic"].metadata`
  已經 public，不用再包一層

**(b) 拆成 `describe_regime_breakdown` / `describe_multi_horizon` /
`describe_spanning` 三個獨立 function**。
- 優點：`from fl.reporting import describe_spanning` 名字一眼懂；各自 fail
  訊息鎖定
- 缺點：namespace 膨脹；format code 很薄其實能共用；targeted scenarios 從
  raw metadata 直接讀也能做到

## 4. 非目標

- **不改 Profile dataclass**（flat scalar 不變、frozen 不變、polars-native 不變）
- **不處理 Profile 基本 pretty-print / `__repr__` / `_repr_html_`**——獨立
  mini-issue，不跟本 spike 綁在一起。`describe_profile_values` 是**含 detail
  的完整視圖**，不是日常 REPL 的第一印象
- **不把 Profile 欄位改成 property 從 metric_outputs 動態算**（破壞 polars）
- **不自動 trigger charts**（charts 是獨立 optional，不偷跑）
- **不加 `Artifacts` method**（`arts.describe_...()`）——detail view 留模組層級
  function
- **不回推 metric_outputs → Profile**（兩者不 1:1，不做反向 API）
- **不對已存在的 `intermediates["regime_stats"]` 等 1-row DataFrame 做重構**
  ——它們現仍是 diagnose rule 的輸入，不動；metric_outputs 是**平行新通道**
- **不改 MetricOutput 的 public shape**（現有 `.name` / `.value` / `.stat` /
  `.significance` / `.metadata` 全部保留；只在存進 Artifacts 時包 metadata
  proxy——見 §3.5）
- **不 default 啟用 keep_artifacts=True**（memory 使用者自決）
- **不做 batch drill-down 流程**（`ProfileSet.get_artifacts(name)` /
  `iter_with_artifacts()` / batch-scope `describe_profile_values`）——使用者
  用 `arts_map[profile.factor_name]` 自己配對，下次 mini-spike 再評估——見 §3.8
- **`describe_profile_values` 只 print 到 stdout，不 return 可程式化物件**
  （`ProfileReport` / rich object 含 `to_dict()`）——AI agent 要程式化讀 data
  已經可以 `dataclasses.asdict(profile)` + `arts.metric_outputs[...]`，新型別
  只是重打包，加了沒 caller
- **不做 custom metric registry**——`describe_profile_values` 只識別 library
  內建 canonical / opt-in keys；使用者自己塞進 `arts.metric_outputs["my_custom"]`
  的東西不會 auto-print detail。等實際有使用者寫 custom metric 再評估 registry
  或 schema hook

## 5. 實作順序（待 §7 sign-off 後）

估時：**1-1.5d**（大部分是 4 factor type × ~10 metrics 的 `from_artifacts` 改動）

1. `Artifacts.metric_outputs: dict[str, MetricOutput] = field(default_factory=dict)`
   加欄位，更新 docstring 說明 vs `intermediates` 的分工——**0.1d**
2. 4 個 `from_artifacts` 改 signature 回 tuple `(Self, dict[str, MetricOutput])`
   （§3.4 option b）+ 改寫 body：累積 local `metric_outputs: dict`，每呼叫一次
   metric 就 `metric_outputs[name] = MetricOutput(..., metadata=MappingProxyType(m.metadata))`
   （§3.5）；最後 `return cls(...), metric_outputs`。`@register_profile` 登錄
   的 protocol signature check 更新。caller (`_evaluate_one` / `evaluate_batch`)
   做 `profile, outputs = profile_cls.from_artifacts(arts); arts.metric_outputs = outputs`。
   9 個 test file 依賴 `Profile.from_artifacts(arts)` 的都要 tuple unpack
   （多半是 `profile, _ = ...`）。**4 個 factor type 同 PR 原子 merge**
   （避免 main 半個 state），CS 先 dogfood drive 決策，Event / MacroPanel /
   MacroCommon 跟進在同一分支。——**0.5d**
3. 新模組 `factorlib/reporting.py`：
   - `describe_profile_values(profile, artifacts, *, include_detail=True)`
     ——artifacts 必填；從 `artifacts.metric_outputs` 讀所有值
   - 內部 helpers（non-public）渲染 regime / multi_horizon / spanning detail
     section——`include_detail=True` 時 auto-discover 啟用的 metric 並逐段印
   - **0.3d**
4. `factorlib/__init__.py` export + README + demo.ipynb 新 section
   (2.3.1 "Drill-down views")——**0.2d**
5. Tests：
   - `tests/test_artifacts_metric_outputs.py`——每個 factor_type 的 canonical
     keys 齊備；opt-in keys 只有 config 啟用時出現；metadata 是 read-only
     view（寫會 TypeError）
   - `tests/test_reporting.py`——`describe_profile_values` smoke test +
     `include_detail=True/False` 切換 + opt-in metric 啟用/未啟用的 detail
     section 行為（啟用 → section 出現；未啟用 → section 省略不印不報錯）
   - **0.3d**

## 6. 風險

- **Memory 佔用**：1000 factor × ~15 MetricOutput × ~5-30 KB metadata ≈ 10-30 MB。
  比 `prepared` 小兩個數量級，可接受。`keep_artifacts=False` 時 Artifacts 跟
  metric_outputs 一起丟，zero persistent cost。
- **Keys schema drift**：使用者依賴 `artifacts.metric_outputs["regime_ic"]`
  存在，library 重構時不小心改 key 會 silently 破壞 user code。**緩解**：
  canonical keys 在 tests 鎖死 + docstring 列齊；改 key 視為 breaking。
- **Mutation 導致 Profile 跟 MetricOutput 數值不一致**：metadata 被包
  `MappingProxyType` 後寫會 TypeError，risk 實質降到 0；但 MetricOutput 的
  `.value` / `.stat` 等 top-level 欄位仍可被 reassign（dataclass 非 frozen）。
  **緩解**：docstring 警告 top-level 欄位不可 mutate；若實戰真有人踩再評估
  整個 frozen。
- **`from_artifacts` signature 改 tuple return 是 breaking**（§3.4 option b）：
  任何下游直接 `profile = Profile.from_artifacts(arts)` 的 code 會 unpack error。
  **緩解**：grep 範圍限於 `factorlib/_api.py` (call site) + 4 個 Profile class
  (定義) + 9 個 test file + `_base.py` protocol check，全部 library 內部；
  end-user 走 `fl.evaluate()` 不直接 call；一併改 + §5 step 2 checklist 列齊。
- **使用者忘 `return_artifacts=True` 卻 call `describe_profile_values`**：
  TypeError（缺 positional arg）或 ValueError（artifacts=None）。**緩解**：
  訊息直接指向正解「在 `fl.evaluate(...)` 加 `return_artifacts=True` 並把
  artifacts 傳進來」。

## 7. Sign-off checklist

- [ ] Owner 確認 §3.1 MetricOutput 放 `Artifacts.metric_outputs`（建議 (a)）
- [ ] Owner 確認 §3.2 keys 用 `MetricOutput.name`（建議 (a)）
- [ ] Owner 確認 §3.3 contract：canonical set + opt-in set，tests 鎖死
- [ ] Owner 確認 §3.4 `from_artifacts` 維持 public + 改 signature 回
      `tuple[Self, dict[str, MetricOutput]]` pure function（建議 (b)；
      使用者友善方向，vs option (a) underscore rename）
- [ ] Owner 確認 §3.5 metadata 用 `MappingProxyType` 包（建議 (a)）
- [ ] Owner 確認 §3.8 batch drill-down 本 session 明列為非目標
- [ ] Owner 確認 §3.6 `describe_profile_values` 強制 artifacts 必填、不做
      fallback（建議 (a)；YAGNI）
- [ ] Owner 確認 §3.7 單一 `describe_profile_values` + auto-discover detail
      （建議 (a)）
- [ ] Owner 確認 detail view 實作放 `factorlib/reporting.py` 新模組
      （非 `_api.py`）

## 8. Future work（不進本 spike，但記起來）

以下是 review 時被提到、但已評估不進本 session 的項目。等實戰出現第一個
blocker 再開 mini-spike 處理：

- **Batch drill-down API**：`ProfileSet.get_artifacts(name)` / `iter_with_artifacts()`
  ——「跑批次 → top-K → 看 Regime detail」是高頻流程。目前使用者手動
  `arts_map[p.factor_name]`，可接受但每人寫一次；累積幾個 call site 就值得
  庫函式化。（見 §3.8）
- **Custom metric 的 detail-view registry**：讓使用者寫 `@register_detail_view`
  之類 hook，自訂 metric 也能被 `describe_profile_values(include_detail=True)`
  自動渲染。目前 library 沒 custom metric 使用者，等真的有第一個才評估
  registry / schema hook。
- **`return_artifacts` vs `keep_artifacts` 命名統一**：兩個 kwarg 語義接近
  但命名不一致（前者 `evaluate`、後者 `evaluate_batch`）。統一成其一是
  cross-session refactor，不在本 spike 範圍。
- **MetricOutput frozen 化**：目前 top-level `.value` / `.stat` 可 reassign。
  若實戰踩到 mutation 造成 debug 惡夢，評估 `@dataclass(frozen=True)` +
  MetricOutput repr 加 `[stashed]` 標記區隔。（見 §3.5）
- **Profile `__repr__` / `_repr_html_` 改善**：讓 `print(profile)` / Jupyter
  顯示變好看；跟本 spike 正交，獨立 mini-issue。（見 §1 scope boundary）
