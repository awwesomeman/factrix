# Plan: Gate → Profile 架構遷移（2-Phase, No Fallback）

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> Status: **Completed — 2026-04-18**
> Driven by: `docs/gate_redesign_v2.md`
> 使用者決策：**不走 fallback / compat shim**。Phase A 建新、Phase B 刪舊，一次 branch merge。
> 最終測試：**345 passed / 0 failed**（branch `refactor/profile-architecture`）。

---

## 1. 目標

把 factorlib 的研究核心抽象從 Gate（pass/fail 決策鏈）換成 `FactorProfile`（per-type typed 剖面）。ADR 已拍板 A–G 七個抉擇、collapse 至 2 個 top-level entry point、無相容層。

---

## 2. 範圍

### In scope
- per-type `FactorProfile` dataclass × 4（cross_sectional / event / macro_panel / macro_common）
- `Profile.from_artifacts()` classmethod 作為唯一建構路徑
- `ProfileSet[P]` polars-native 集合型別
- `factorlib/stats/multiple_testing.py` BHY module
- `profile.verdict()` / `profile.diagnose()` / `profile.canonical_p` property
- `@register_profile(FactorType.X)` decorator + `_PROFILE_REGISTRY`
- 頂層 API：`fl.evaluate()` / `fl.evaluate_batch()` / `fl.list_factor_types()`
- 砍 `EvaluationResult` wrapper、整個 `evaluation/gates/`、`presets.py`、`_caution.py`、舊 `FactorProfile`、`_PROFILE_METRICS` / `_STANDALONE_METRICS` dict、`quick_check` / `compare`
- 更新 `describe_profile()` 內臟（反射 dataclass）

### Out of scope
- Layer 4 deep-dive 工具（保留既有 plots 不動）
- Dashboard / DB 整合（`integrations/` 不動）
- 新增因子類型
- `split_by_group()`（不動）
- `redundancy_matrix()` 由 `plan_factor_screening` 獨立實作，本計畫只確保輸入型別改為 `ProfileSet`

---

## 3. Phase A — 新架構 top-down build

**目標**：feature branch 上落地全部新程式碼；舊程式碼暫留、暫時新舊並 export（僅 `__init__.py` 層，無共用計算路徑）以不破壞既有 test 先通過。

**estimated**：~1 週。

### A.1 核心型別與 protocol
- 新增 `factorlib/_types.py` 加入 `PValue = NewType("PValue", float)`、`Verdict = Literal["PASS", "FAILED"]`、`Diagnostic` dataclass。
- 新增 `factorlib/evaluation/profiles/_base.py`：
  - `FactorProfile` Protocol（`factor_name`、`n_periods`、`canonical_p` property、`verdict()`、`diagnose()`、`CANONICAL_P_FIELD`、`P_VALUE_FIELDS` ClassVar）
  - `register_profile(factor_type)` decorator
  - `_PROFILE_REGISTRY: dict[FactorType, type[FactorProfile]]`

### A.2 per-type Profile dataclass × 4
- `profiles/cross_sectional.py` → `CrossSectionalProfile`
- `profiles/event.py` → `EventProfile`
- `profiles/macro_panel.py` → `MacroPanelProfile`
- `profiles/macro_common.py` → `MacroCommonProfile`

每個 dataclass：
- `@register_profile(FactorType.X) @dataclass(frozen=True, slots=True)`
- 欄位翻譯自現有 `_PROFILE_METRICS`；value metric 用 `float`，p-value 用 `PValue`
- `CANONICAL_P_FIELD: ClassVar[str]` — 釘死（見下方 canonical 決議）
- `P_VALUE_FIELDS: ClassVar[frozenset[str]]`
- `canonical_p: PValue` property → `getattr(self, self.CANONICAL_P_FIELD)`
- `verdict(threshold=2.0)` 二態，只看 canonical p
- `diagnose()` 回 `list[Diagnostic]`
- `from_artifacts(cls, artifacts) -> Self` classmethod：呼叫既有 `factorlib.metrics.*` 函數組裝

**Canonical p 決議**（需 sign-off 後開工）：

| 因子類型 | CANONICAL_P_FIELD | 統計學來源 |
|---------|-------------------|-----------|
| cross_sectional | `ic_p` | IC non-overlapping sampling 的 t-stat → p |
| event_signal | `caar_p` | CAAR non-overlapping sampling（每 `forward_periods` 日一筆）的 t-stat → p。MacKinlay (1997) 標準構造；BMP 作 secondary evidence 於 `diagnose()`。**NOT** `event_window_post`：該欄位僅用於 MFE/MAE 觀察視窗，不對應任何 CAR 顯著性檢驗。 |
| macro_panel | `fm_beta_p` | Fama-MacBeth λ mean 的 Newey-West t-stat → p |
| macro_common | `ts_beta_p` | 跨資產 per-asset TS β 的 cross-sectional t-stat → p（N=1 退化為單一資產 TS t-stat） |

### A.3 Artifacts API 整理
- `Artifacts.from_raw(cls, df, factor_name, config) -> Artifacts` classmethod，封裝既有 `build_artifacts` + preprocess。
- `Artifacts.compact: bool = False` flag（含 guard raise on `.prepared` access when compact）— 對應 `plan_factor_screening` 功能 E。
- 現有 `Artifacts.get(key)` 不動。

### A.4 ProfileSet
- 新增 `factorlib/evaluation/profile_set.py`
- polars-native：`_df: pl.DataFrame`、`_profile_cls: type[P]`
- constructor：
  - `ProfileSet.from_profiles(profiles: list[P]) -> ProfileSet[P]`（檢查 homogeneity，序列化成 DataFrame）
  - `ProfileSet._from_df(df, profile_cls)` internal
- `filter(predicate)`：
  - `pl.Expr` path → 驗證 row-wise boolean（`predicate` 應用後 dtype==Boolean、長度不變），執行 `_df.filter(predicate)`
  - `Callable[[P], bool]` path → `iter_profiles` lambda filter 再重建 df
- `rank_by(field, descending=True)` / `top(n)` / `iter_profiles()` / `to_polars()`

### A.5 Multiple testing
- 新增 `factorlib/stats/multiple_testing.py`:
  ```python
  def bhy_adjust(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
      """BHY step-up procedure. Returns boolean mask of rejections."""

  def bhy_adjusted_p(p_values: np.ndarray) -> np.ndarray:
      """Return per-hypothesis adjusted p-values."""
  ```
- `ProfileSet.multiple_testing_correct(p_source="canonical_p", method="bhy", fdr=0.05) -> ProfileSet[P]`
  - 驗證 `p_source in {"canonical_p"} | profile_cls.P_VALUE_FIELDS` — 否則 `ValueError` 列白名單
  - 計算 `bhy_significant`、`p_adjusted` 兩個新欄位，回傳新 ProfileSet（DataFrame 新增欄位，`_profile_cls` 不變——欄位不屬於 dataclass schema 但屬於 DataFrame view）

### A.6 Diagnose rules
- 新增 `factorlib/evaluation/diagnostics/_rules.py`
  - `Rule` dataclass：`predicate: Callable[[P], bool]`、`severity: Literal["info","warn","veto"]`、`message: str`
  - per-type rule list：`CROSS_SECTIONAL_RULES` / `EVENT_RULES` / ...
  - 從既有 `_caution.py` 翻譯現有規則
- Profile.diagnose() 實作：`return [Diagnostic(r.severity, r.message) for r in RULES if r.predicate(self)]`

### A.7 Top-level facade
- `factorlib/__init__.py` 新增：
  ```python
  def evaluate(df, name, *, factor_type="cross_sectional", **cfg) -> FactorProfile:
      cfg_obj = _config_for_type(factor_type, **cfg)
      artifacts = Artifacts.from_raw(df, name, cfg_obj)
      return _PROFILE_REGISTRY[cfg_obj.factor_type].from_artifacts(artifacts)

  def evaluate_batch(
      factors, *, factor_type="cross_sectional",
      keep_artifacts=True, stop_on_error=False,
      on_error=None, on_result=None,
      **cfg,
  ) -> ProfileSet:
      ...

  def list_factor_types() -> list[str]: ...
  ```
- **`__init__.py` 暫時雙 export**：新 API（`evaluate`, `evaluate_batch`, `ProfileSet`, Profile classes）+ 舊 API（`quick_check`, `batch_evaluate`, `compare`）並存，僅為 Phase A 期間既有測試能繼續 import；**無共用計算路徑**。

### A.8 describe_profile 重寫
- 內臟改為反射 `_PROFILE_REGISTRY[factor_type]` 的 dataclass annotations、ClassVar、docstring，以及 method signatures。
- 舊 `_PROFILE_METRICS` / `_STANDALONE_METRICS` dict 此階段還沒刪，但 `describe_profile` 已不讀。

### A.9 `redundancy_matrix()`（整合自 factor_screening 功能 C）
- 新增 `factorlib/metrics/redundancy.py`：
  ```python
  def redundancy_matrix(
      profiles: ProfileSet,
      method: Literal["factor_rank", "value_series"] = "factor_rank",
      *,
      artifacts: dict[str, Artifacts] | None = None,
  ) -> pl.DataFrame:
      """Pairwise Spearman |ρ| matrix across factors in a ProfileSet.

      method="factor_rank": cross-sectional factor rank correlation (需 artifacts，prepared DataFrame)
      method="value_series": IC / CAAR / beta series correlation（只需 intermediates）
      """
  ```
- 輸入型別：`ProfileSet`（取代原設計的 `dict[str, EvaluationResult]`）
- `method="factor_rank"` 需要 `artifacts` 字典（因為 profile 本身不帶 prepared DataFrame）；`keep_artifacts=False` 時自動降級為 `value_series`（加 warning）
- `method="value_series"` 從 `ProfileSet` 的 profile 上讀對應 metric series（cross_sectional 讀 ic_series、event 讀 event_ic_series 等）
- export 於 `factorlib.__init__`：`fl.redundancy_matrix`

### A.10 測試
- `tests/profiles/test_cross_sectional_profile.py`（及其餘 3 類）：
  - from_artifacts 欄位正確
  - canonical_p property 等於 CANONICAL_P_FIELD 對應欄位
  - verdict() 二態 — 在 threshold=2.0 時與該類 canonical t-stat 比對
  - diagnose() 在退化 fixture 上產出預期 severity / message
- `tests/test_profile_set.py`：
  - from_profiles + homogeneity check
  - filter(pl.Expr) row-wise / 非 boolean 時 raise
  - filter(Callable) 等價
  - rank_by / top / iter_profiles / to_polars
- `tests/stats/test_multiple_testing.py`：
  - bhy_adjust 與 R `p.adjust(method="BY")` 一致（fixture）
  - `multiple_testing_correct` 白名單驗證（ic_p / canonical_p OK；min_p / ic_ir 拒絕）
- `tests/test_api.py`（新）：
  - `fl.evaluate` 單因子
  - `fl.evaluate_batch` 批次 + stop_on_error=False 行為
  - `fl.list_factor_types` 回傳值
- `tests/test_redundancy_matrix.py`（新）：
  - symmetric matrix、對角線為 1
  - factor_rank 與 value_series 在相同 fixture 上的數值關係
  - keep_artifacts=False 時 factor_rank 自動降級 + warning
  - 排除 FAILED 因子（若 ProfileSet 因 stop_on_error=False 漏收）
- **Parity snapshot**：對 5–10 個因子 fixture 同時跑 `fl.evaluate()`（新）與 `fl.quick_check()`（舊）；比較 IC t-stat、spread t-stat 的數值應一致（相同 metric 函數）；verdict 應與 gate pass 在 threshold=2.0 時一致（IC canonical）。差異記錄文件化。

---

## 4. Phase B — 砍舊

**目標**：刪除所有 gate / EvaluationResult / 舊 FactorProfile 相關程式碼；`__init__.py` 清乾淨；notebook / README 更新。

**Status: ✅ Completed (2026-04-18)**。對應 atomic commits：
- `0233c6f` — `refactor!: drop gate-era evaluation layer for FactorProfile API`（-2560 / +203；B.1–B.3 + 部分 B.4 caller 遷移）
- `07c68db` — `feat(metrics): structure short-circuit metadata with reason/n_observed`（14 metric modules；後 §10 補述）
- `655f728` — `feat(profiles): surface insufficient data via profile field and diagnose rule`（§10）
- `ad5bc3e` — `refactor(profile_set): resolve empty-set dtypes via typing.get_type_hints`（§10、B.0.3 結案）

**estimated**：~2 天（不含 B.0 pre-work；含 B.0 約 ~3.5 天）。實際 ~2 天含後續 polish。

---

### B.0 Phase A Post-Review Pre-Work

Phase A 結束時資深 quant / 資深後端檢視留下的遺留項。**必做項為 Phase B 刪除動作的 blocker**：刪除舊路徑前必須完成，否則會喪失 parity / 覆蓋證據。

#### B.0.1 Deletion blockers（必做，~1.5 天）

- [x] **per-type parity tests for event_signal / macro_panel / macro_common** — 2026-04-18
  - 新增 `tests/test_parity_event_profile.py`、`tests/test_parity_macro_panel_profile.py`、`tests/test_parity_macro_common_profile.py`：raw metric t-stat/p 相等、binary verdict 與舊 gate 在 canonical-agreeing 情境一致。各 fixture 標註 BMP-only / hit_rate-only / pooled-only 的 known divergence（新 canonical-binary policy 之設計，非 bug）
  - 新增 `tests/test_verdict_small_n.py`：驗證 `_verdict_from_p` 在 n_dates=20 使用 t-distribution（比 Z 更保守），並鎖定 p-threshold 隨 n 單調遞減
- [x] **`_caution.py` 規則 side-by-side 審計** — 2026-04-18
  - 對照結果：舊 15 條 rules 有 3 條原先因為需要 config 存取而未完全遷移，皆以新增 profile 欄位方式補齊（資深 quant 建議「全部 port」，不 drop）
    - CS：新增 `orthogonalize_applied: bool`，新 rule `cs.orthogonalize_not_applied`（原 `_cs_caution` 讀 `cfg.ortho`）
    - Event：新增 `clustering_adjustment: Literal[...]`，修正 `event.clustering_high` predicate 加 `and p.clustering_adjustment == "none"`（避免已套 kolari_pynnonen 仍噴 warning）
    - Macro panel：新增 `min_cross_section_threshold: int`，`macro_panel.small_cross_section` 改用 profile 欄位而非 hard-code 10（恢復 configurable 能力）
  - `significance_via_hit_rate_only` 已有對應 rule（`event.hit_rate_only`）、格式改為 p-value 組合而非 `via` 欄位但語義等價
  - 剩下 _caution 規則皆有 1:1 對應（見 `diagnostics/_rules.py`）
- [x] **Event canonical p 對齊 ADR** — 採決議 B（2026-04-18 sign-off）
  - 背景：ADR 初稿描述 canonical 為「主事件窗 CAR p」、driven by `EventConfig.event_window_post`（20d）；但 `event_window_post` 自始只用於 MFE/MAE 觀察視窗，從未被任何 CAR 檢驗讀取
  - 決議：維持現況、更新 ADR/plan 將 canonical 描述為「CAAR 於 `forward_periods` 的 non-overlapping sampling t-test」
  - 理由：(1) 與 CS profile 的 `ic_p` 對稱（同樣以 `forward_periods` 採樣），(2) MacKinlay (1997) 標準構造，(3) 新增 CAR-at-`event_window_post` metric 會重複 `caar()` 邏輯、無統計增益、且讓 `event_window_post` 同時承擔 MFE/MAE 與 canonical 兩個職責

#### B.0.2 Breaking bundles（應做，~0.5 天；搭配 Phase B major bump 一併破壞）

- [x] **`oos_decay` 欄位直覺 / 命名** — 2026-04-18（breaking rename）
  - `OOSResult.decay_ratio` / `SplitDetail.decay_ratio` → `survival_ratio`；四類 profile 的 `oos_decay: float` → `oos_survival_ratio: float`；`cs.oos_decay_large` rule → `cs.oos_survival_low`
  - `multi_split_oos_decay(decay_threshold=)` kwarg → `survival_threshold=`；legacy `oos_persistence_gate(decay_threshold=)` 一併更名
  - 語義：`survival_ratio = |mean_OOS| / |mean_IS|`，1.0 = 無衰退（完整存活），0.0 = 完全衰退；與 `< 0.5` 為 concerning 的 rule predicate 方向一致
- [x] **補齊遺漏測試** — 2026-04-18
  - `tests/test_register_profile.py` — 缺 CANONICAL_P_FIELD / 缺 P_VALUE_FIELDS / canonical 不在白名單 / 雙重註冊四個 validation 路徑
  - `tests/test_compacted_prepared.py` — 五個 dunder（`__bool__` / `__len__` / `__iter__` / `__getitem__` / `__contains__`）RuntimeError + `__repr__` safe
  - `test_profile_set.py` — `multiple_testing_correct(p_source=<whitelist member>)` + iter_profiles 與 to_polars 在 filter/rank_by/top/chain 後仍 aligned
  - `test_cross_sectional_profile.py` — `verdict(threshold=0)` = PASS；`verdict(-t) == verdict(+t)` 對稱
  - `test_multiple_testing.py` — `bhy_adjust` 單元素 rejects iff p ≤ fdr（空陣列、全 1 已有）

#### B.0.3 Nice-to-have（可做）

- [x] **`_pairwise_abs_spearman` 性能優化** — 2026-04-18（commit 6f5e719）
  - 單次 `np.corrcoef` 取代 O(M²) 迴圈；null 處理語義由 pairwise-listwise 轉為全 listwise（同 ProfileSet 情境下相同），docstring 記載
- [x] **`register_profile` runtime ClassVar mutation 防護** — 2026-04-18（commit e0786e9）
  - `multiple_testing_correct` 開頭 frozenset 成員檢查；違反時 RuntimeError 指向 mutation 位置
- [x] **`ProfileSet._polars_dtype_for` fragility** — 2026-04-18（commit ad5bc3e）
  - 後續 code review 發現 `tuple[str, ...]` 會不小心 match `"str"` 分支、`Literal` 會掉到 `pl.Object`。採用 `typing.get_origin` / `get_type_hints` + 型別 dispatch 全面改寫
  - 涵蓋 `T | None` 解 Union、`tuple[T, ...]` / `list[T]` → `pl.List(inner)`、`PValue`（NewType）經 `__supertype__` → `Float64`
  - 鎖定於 `tests/test_profile_set.py::TestConstruction::test_empty_schema_uses_typed_dtypes`
- [x] **Doc clarifications** — 2026-04-18（commit 618006c）
  - `P_VALUE_FIELDS` docstring 明確 batch-only 語義
  - `event.caar_sig_bmp_not_confirm` rule 加 threshold rationale（|z|<1.5 涵蓋 BMP 灰區）
  - `_verdict_from_p` 加 Harvey/Liu/Zhu 2016 指引與 t-dist df=n-1 備註
- [ ] **CS profile 的 `ic_tstat` / `ic_p` 樣本數差異**（deferred, nice-to-have）
  - `ic_tstat` 用完整 IC 序列計算、`ic_p` 用 non-overlapping sample 計算
  - 欄位 docstring 補說明或 rename `ic_tstat_sampled` / `ic_tstat_full`

---

### B.1 刪除目錄 / 檔案 ✅
- `factorlib/evaluation/gates/`（整包）
- `factorlib/evaluation/presets.py`
- `factorlib/evaluation/_caution.py`（邏輯已在 A.6 搬進 diagnostics/_rules.py）
- `tests/test_gates.py`、`tests/test_*_gate.py`
- 舊的 `test_p1_additions.py` 中的 bhy_threshold 測試（已搬至 `tests/stats/`）

### B.2 刪除型別與 API ✅
- `factorlib/evaluation/_protocol.py`：
  - 刪除 `GateFn`, `GateResult`, `GateStatus`, `EvaluationStatus`（若不再被 profile 用）
  - 刪除 `EvaluationResult` dataclass
  - 刪除舊 `FactorProfile` dataclass（untyped list）
  - 保留 `Artifacts`
- `factorlib/_api.py`：
  - 刪除 `quick_check()`, `batch_evaluate()`（舊）, `compare()`
  - 刪除 `_PROFILE_METRICS`, `_STANDALONE_METRICS`, `_DESCRIPTIONS`（`_DESCRIPTIONS` 保留或搬進 per-type profile class docstring）
  - 刪除 `batch_evaluate(..., gates=)` 參數痕跡
- `factorlib/evaluation/pipeline.py`：
  - 若 `evaluate()` 已被 `fl.evaluate` 取代 → 整檔刪除
  - 若仍有 helper 被 Artifacts.from_raw 用到 → 保留必要 helper，其餘刪
- `factorlib/_stats.py`：
  - 刪除 `bhy_threshold()`（by `factor_screening`）

### B.3 清理 public exports ✅
- `factorlib/__init__.py`：
  - 刪除 `quick_check`, `batch_evaluate`(舊), `compare`, `GateFn`, `GateResult`, preset lists re-exports
  - 確認 public API：`evaluate`, `evaluate_batch`, `list_factor_types`, `describe_factor_types`, `describe_profile`, `split_by_group`, `ProfileSet`, `Artifacts`, 各 Profile class, `FACTOR_TYPES`, `FactorType`

### B.4 文件 / notebook 更新 ✅
- `README.md`：gate 範例整段改寫為 profile 範例
- `experiments/*.ipynb`：搜尋 `gate_results` / `quick_check` / `compare(` / `EvaluationResult` → 改寫
- `docs/gate_redesign.md`：已標 Superseded，確認 link 正確
- `docs/factor_screening.md`：封存，頂部標 Archived 並指向本計畫
- Release notes 已整合至本計畫 §6（breaking changes table + migration before/after + canonical p 表）

### B.5 回歸測試 ✅
- 刪除後跑全套 `pytest`
- 跑 `experiments/` 下全部 notebook（headless）確認能執行完
- 對 A.10 的 parity fixture 再跑一次，確認結果與 Phase A 快照一致

---

## 5. 檔案改動清單

| 檔案 | Phase | 動作 |
|------|-------|------|
| `factorlib/_types.py` | A.1 | 新增 PValue/Verdict/Diagnostic |
| `factorlib/evaluation/profiles/_base.py` | A.1 | 新增（Protocol + register_profile + registry） |
| `factorlib/evaluation/profiles/cross_sectional.py` | A.2 | 新增 |
| `factorlib/evaluation/profiles/event.py` | A.2 | 新增 |
| `factorlib/evaluation/profiles/macro_panel.py` | A.2 | 新增 |
| `factorlib/evaluation/profiles/macro_common.py` | A.2 | 新增 |
| `factorlib/evaluation/profiles/__init__.py` | A.2 | 新增（import 觸發 decorator） |
| `factorlib/evaluation/_protocol.py` | A.3 → B.2 | 加 Artifacts.from_raw/compact → 刪 Gate/EvaluationResult/舊 FactorProfile |
| `factorlib/evaluation/profile_set.py` | A.4 | 新增 |
| `factorlib/stats/__init__.py` | A.5 | 新增 |
| `factorlib/stats/multiple_testing.py` | A.5 | 新增 |
| `factorlib/evaluation/diagnostics/_rules.py` | A.6 | 新增 |
| `factorlib/__init__.py` | A.7 → B.3 | 加 evaluate/evaluate_batch/list_factor_types/redundancy_matrix（雙 export）→ 清舊 exports |
| `factorlib/_api.py` | A.8 → B.2 | describe_profile 換內臟 → 刪 quick_check/batch_evaluate/compare/metric dicts |
| `factorlib/metrics/redundancy.py` | A.9 | 新增（redundancy_matrix） |
| `factorlib/_stats.py` | B.2 | 刪 bhy_threshold |
| `factorlib/evaluation/pipeline.py` | B.2 | 刪除 gate-era `evaluate()`，保留 `build_artifacts()` + per-type helper |
| `factorlib/evaluation/profile.py` | B.1 | 刪除（`compute_profile` 被 Profile.from_artifacts 取代） |
| `factorlib/evaluation/gates/` | B.1 | 刪除目錄 |
| `factorlib/evaluation/presets.py` | B.1 | 刪除 |
| `factorlib/evaluation/_caution.py` | B.1 | 刪除 |
| `factorlib/charts/__init__.py` | B.2 | `report_charts(result)` → `report_charts(artifacts)` |
| `factorlib/integrations/mlflow.py` | B.2 | `log_evaluation(result)` → `log_profile(profile)` |
| `tests/profiles/*.py` | A.10 | 新增 |
| `tests/test_profile_set.py` | A.10 | 新增 |
| `tests/stats/test_multiple_testing.py` | A.10 | 新增 |
| `tests/test_api.py` | A.10 | 新增（commit 1 從 test_api_new.py rename） |
| `tests/test_redundancy_matrix.py` | A.10 | 新增 |
| `tests/test_insufficient_data.py` | §10 post-polish | 新增（`data.insufficient` 契約 + profile 欄位傳遞 + intentional-skip 排除） |
| `tests/test_gates.py` + `tests/test_parity_*_profile.py`（4） | B.1 | 刪除 |
| `README.md` | B.4 | 改寫 |
| `experiments/*.ipynb` | B.4 | 頂部加 migration banner（保留舊 cells 供參考） |

---

## 6. Breaking Changes（release notes）

版本號：bump **major**（breaking，no compat shim）。

### 6.1 刪除的符號

| 舊符號 | 取代方案 |
|---|---|
| `fl.quick_check(df, name, ...)` | `fl.evaluate(df, name, ...)`（profile-era） |
| `fl.batch_evaluate(..., gates=...)` | `fl.evaluate_batch(...)` 回 `ProfileSet` |
| `fl.compare(results, sort_by=...)` | `profiles.rank_by("ic_ir").to_polars()` |
| `fl.EvaluationResult` | per-type Profile dataclass |
| `fl.GateResult` / `fl.GateFn` | 刪除（`profile.diagnose()` 覆蓋） |
| `fl.CROSS_SECTIONAL_GATES` / `MACRO_PANEL_GATES` / `MACRO_COMMON_GATES` | 刪除 |
| `factorlib.evaluation.gates.*` | 整包刪除 |
| `factorlib.evaluation.presets.default_gates_for` | 刪除 |
| `factorlib.evaluation._caution.check_caution` | 由 `profile.diagnose()` → `list[Diagnostic]` 取代 |
| `factorlib.evaluation.profile.compute_profile` | 刪除（每個 Profile 有自己的 `from_artifacts`） |
| `factorlib.evaluation.pipeline.evaluate(...)`（gate-era） | 移至 `factorlib._api.evaluate`（profile-era dispatch） |
| `factorlib._stats.bhy_threshold(t_stats)` | `factorlib.stats.multiple_testing.bhy_adjust(p_values)` / `ProfileSet.multiple_testing_correct` |

### 6.2 重新命名（語義反轉）

- `OOSResult.decay_ratio` → `survival_ratio`（1.0 = 完整存活）
- `SplitDetail.decay_ratio` → `survival_ratio`
- `multi_split_oos_decay(decay_threshold=)` → `survival_threshold=`
- 四個 Profile 欄位：`oos_decay: float` → `oos_survival_ratio: float`
- CS diagnose rule `cs.oos_decay_large` → `cs.oos_survival_low`

### 6.3 行為變更

- `verdict()` 改為 **二態** (`'PASS' | 'FAILED'`)。舊 `PASS` / `CAUTION` / `VETOED` / `FAILED` 四象限收斂到：verdict（看 canonical p）＋ diagnose（所有脈絡警告）。
- `verdict(threshold)` 以 canonical 統計量的 t 分佈在 `df = n_periods − 1` 轉換門檻；小樣本嚴格比舊 Z 近似保守。
- `verdict` 在 `n_periods < 2` 直接回 `FAILED`（避免 `_p_value_from_t(·, n≤1)` fallback 1.0 導致 `p <= 1.0` 意外 PASS）。
- BHY 吃 **p-values** 而非 t-stats；以 Profile 的 `P_VALUE_FIELDS` 做 same-test-family 白名單。
- Charts: `report_charts(result)` → `report_charts(artifacts)`；profile 刻意不挾帶 DataFrame，需要時用 `build_artifacts(df, config)` 重建。
- MLflow: `FactorTracker.log_evaluation(result, ...)` → `log_profile(profile, ...)`，反射 dataclass 欄位 + verdict/diagnose tags + `tuple[str, ...]` 類欄位（如 `insufficient_metrics`）。

### 6.4 Migration 對照（before → after）

**Single factor**
```python
# before
result = fl.quick_check(df, "Mom_20D")
print(result.status, result.profile.get("ic"))
for gr in result.gate_results: ...
for r in result.caution_reasons: ...

# after
profile = fl.evaluate(df, "Mom_20D", factor_type="cross_sectional")
print(profile.verdict(), profile.canonical_p)
print(profile.ic_mean, profile.ic_tstat, profile.ic_ir)  # typed fields
for d in profile.diagnose():
    print(d.severity, d.code, d.message)
```

**Batch + BHY + rank**
```python
# before
results = fl.batch_evaluate({"A": df1, "B": df2}, factor_type="cross_sectional")
table = fl.compare(results, sort_by="ic")

# after
import polars as pl
profiles = fl.evaluate_batch(
    {"A": df1, "B": df2},
    factor_type="cross_sectional",
)
top = (
    profiles
    .multiple_testing_correct(p_source="canonical_p", fdr=0.05)
    .filter(pl.col("bhy_significant"))
    .rank_by("ic_ir")
    .top(10)
)
print(top.to_polars())
```

**Charts**
```python
# before
figs = report_charts(result)  # result 挾帶 artifacts

# after
from factorlib.evaluation.pipeline import build_artifacts
artifacts = build_artifacts(prepared, config)
figs = report_charts(artifacts)
```

**MLflow**
```python
# before
fl.batch_evaluate(factors, on_result=tracker.log_evaluation)

# after
fl.evaluate_batch(
    factors,
    factor_type="cross_sectional",
    on_result=lambda name, p: tracker.log_profile(
        p, factor_type="cross_sectional",
    ),
)
```

遷移對照亦可參考 `gate_redesign_v2.md` §7 情境 7。

---

## 7. 開工 Checklist

- [ ] ADR `gate_redesign_v2.md` 全部 section sign-off（特別是 §3.G 白名單機制、§4 verdict 二態）
- [ ] Phase A §A.2 table 的 4 個 canonical_p 精確定義（含 t-stat 計算細節、NW lag 參數）sign-off
- [ ] 既有 Q2 review 點逐項消化（event 主事件窗預設值、FM / TS 的 NW/HAC 細節）
- [ ] 建 feature branch `refactor/profile-architecture`
- [ ] 準備 parity fixture：≥20 因子，4 類均勻覆蓋；記錄當前 gate output 作為比對基準

---

## 8. 預估時間

| Phase | 工時 | 主要風險 |
|-------|------|---------|
| A. build new | ~1 週 | canonical_p 細節對齊、parity snapshot 產出與對比 |
| B.0 pre-work (deletion blockers + bundled breaking) | ~2 天 | per-type parity fixture、`_caution.py` 規則審計、event canonical 決議、oos_decay 命名 |
| B. delete old | ~2 天 | notebook / README 漏改 |
| **合計** | **~2 週** | B.0 完成才能進 B.1 |

---

## 9. 與原 `factor_screening.md` 的整合

原篩選計畫（現已封存）的 5 個功能全部吸收進本計畫：

| 原功能 | 本計畫落地於 |
|--------|-------------|
| A. `compare(bhy=True)` | Phase A.7（`evaluate_batch` + `ProfileSet.multiple_testing_correct` + `.to_polars()` 取代 compare） |
| B. `bhy_adjust()` | Phase A.5 |
| C. `redundancy_matrix()` | Phase A.9（輸入型別改 `ProfileSet`） |
| D. Gate primary_p + suggestion | **作廢**（profile canonical_p + diagnose 天然覆蓋） |
| E. `keep_artifacts` | Phase A.3（`Artifacts.compact`）；`evaluate_batch(keep_artifacts=)` 參數在 Phase A post-review 移除為 no-op，Phase B 以 `(profiles, artifacts)` 雙返回取代 |

Phase A 完成後 factorlib 公開 API 即包含篩選計畫的全部能力；不需要另外 follow-up。

---

## 10. Phase B Post-Polish（2026-04-18）

Phase B 完成後資深 quant / 資深後端 fallback 審計發現 12+ 個 metric 早返回路徑用不同格式寫 `metadata["reason"]`、或根本不寫 → profile 端看到 `ic_mean=0.0, ic_tstat=0.0, ic_p=1.0` 無法分辨「真 null signal」vs「樣本不足」。連帶發現 `_verdict_from_p` 有 n≤1 fallback 導致 `1.0 <= 1.0` 意外 PASS 的 latent bug。

**三個 commit 一次收斂**：

| Commit | Scope | Files |
|---|---|---|
| `07c68db feat(metrics): structure short-circuit metadata with reason/n_observed` | 14 個 metric 模組早返回路徑統一寫 `{"reason": "<stable_id>", "n_observed": n, "min_required": N}`；`reason` 詞彙用 prefix 區分 shortfall（`insufficient_*` / `no_*`）與 intentional skip（`not_applicable_*` / `degenerate_*`） | `factorlib/metrics/*.py`（caar, clustering, concentration, corrado, event_quality, fama_macbeth, hit_rate, ic, monotonicity, quantile, spanning, tradability, trend, ts_beta） |
| `655f728 feat(profiles): surface insufficient data via profile field and diagnose rule` | 四類 Profile 新增 `insufficient_metrics: tuple[str, ...]` 欄位；`_insufficient_metrics` helper 掃 metric metadata；新 cross-type `Rule(code="data.insufficient", severity="warn")`，message 動態列出受影響欄位（`Rule.message` 加 `Callable[[P], str]` overload）；**修 `_verdict_from_p(n_periods < 2) → FAILED`** | `factorlib/evaluation/profiles/_base.py`、四類 profile、`diagnostics/_rules.py`、`tests/test_insufficient_data.py`（新）、`tests/test_verdict_small_n.py::TestDegenerateN` |
| `ad5bc3e refactor(profile_set): resolve empty-set dtypes via typing.get_type_hints` | 結案 B.0.3：`_polars_dtype_for` 改用 `typing.get_origin` + 型別 dispatch，正確處理 Union / `tuple[T, ...]` / NewType（`PValue`） | `factorlib/evaluation/profile_set.py`、`tests/test_profile_set.py` |

**設計決策筆記**

- **Prefix vocabulary 優於 StrEnum**：reviewer 建議將 reason 值遷至 `StrEnum`；拒絕，因為會強迫每個 metric 模組 import enum + 指定完整名稱，for a 4-line literal 增加 coupling。prefix (`insufficient_*` / `no_*` / `not_applicable_*` / `degenerate_*`) 的分類成本在 `_insufficient_metrics` 單點集中，call-site 仍是 grep-friendly literal。
- **不做 `MetricOutput.insufficient(...)` factory**：同理；14 個 call-site × 6 行 literal 的重複 vs helper 強迫 import 的權衡下，literal 保留可讀性勝出。
- **`_verdict_from_p(n<2) → FAILED` 的門檻為什麼是 `<2` 而非 `MIN_IC_PERIODS=10`**：兩層語義分離 — `n<2` 是**統計有效性 floor**（t-distribution df ≤ 0 undefined），永遠不能 PASS；`n<MIN_IC_PERIODS` 是**metric 可靠性警告**，以 `data.insufficient` diagnose（warn）呈現但仍允許 verdict PASS（使用者可能做 5-period 探索性 verdict）。
- **`mlflow.log_profile` tuple branch**：`isinstance(value, (tuple, list))` → 以 `","` join 為 tag；否則新加的 `insufficient_metrics` 會被 dataclass field iteration 默默丟棄。

**UX 結果驗證**（smoke test）

```
tiny panel（8 dates × 5 assets）
Verdict: FAILED  canonical_p: 1.0000  n_periods: 0
insufficient_metrics (7): ('ic_mean', 'ic_ir', 'hit_rate', 'ic_trend',
                           'monotonicity', 'q1_q5_spread', 'q1_concentration')
Diagnose[0]: [warn] data.insufficient — One or more metrics short-circuited …
```

最終測試計數：**345 passed / 0 failed**（+15 相對 Phase B 完成時的 330）。

---

## 11. Known Deferred Work

以下項目意識到但**非 Phase B blocker**，留待後續 PR 處理：

- **CS profile 的 `ic_tstat` / `ic_p` 樣本數差異**（B.0.3 deferred）：`ic_tstat` 用完整 IC 序列、`ic_p` 用 non-overlapping sample。Docstring 已補說明，rename 為 `ic_tstat_full` / `ic_tstat_sampled` 的重命名延後。
- **`evaluate_batch` 並行化**：`max_workers` kwarg + `concurrent.futures` 是 research zoo 場景的主要單次成本；API 表面變動屬獨立 PR，不在本次 merge 範圍。
- **`Artifacts.factor_name` 後構造期賦值**：`_api.py` 目前 `artifacts.factor_name = factor_name`，應改為 `build_artifacts(..., factor_name=...)` constructor arg，移除 mutation；pre-existing（Phase A 決策），scope 外。
- **`evaluate_batch` 不回 artifacts**：`redundancy_matrix(method="factor_rank")` 使用者仍需手寫 for-loop 保留 Artifacts；已 docstring 交代，待 `(profiles, artifacts)` 雙返回設計定稿後實作。
- **`clustering_adjustment: ClusteringAdjustment`（Literal）空集 dtype**：`_polars_dtype_for` 目前回 `pl.Object`；Literal → polars dtype 無 1:1 映射，保留明示 fallback 優於猜測。

Phase C 若啟動（新增因子類型 / Layer 4 deep-dive 工具），這些 deferred 再一起盤點。
