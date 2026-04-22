# Plan: factorlib 因子篩選輔助功能

> **Status: Archived — 2026-04-17**
>
> 本計畫原列的 5 個功能（A. compare(bhy=True) / B. bhy_adjust / C. redundancy_matrix / D. Gate primary_p+suggestion / E. keep_artifacts）已**全部吸收進** `docs/plan_gate_redesign.md`（Profile 架構遷移 Phase A）。
>
> - A → Phase A.7（`ProfileSet.multiple_testing_correct` + `.to_polars()`）
> - B → Phase A.5（`factorlib/stats/multiple_testing.py`）
> - C → Phase A.9（輸入型別改 `ProfileSet`）
> - D → **作廢**（profile `canonical_p` + `diagnose()` 天然覆蓋）
> - E → Phase A.3（`Artifacts.compact`）+ Phase A.7（`evaluate_batch(keep_artifacts=False)`）
>
> 新實作請直接看 `plan_gate_redesign.md`；本文件保留作歷史脈絡。
>
> ---
>
> （以下為原 v4 內容，僅供歷史參考。）
>
> Status: v4 — 2026-04-17
> 依 `docs/gate_redesign_v2.md` 重新對齊。Gate 系統全面退場，改以 `FactorProfile` 為核心。
> 原 v3.3 的功能 D（Gate primary_p + suggestion）已作廢，由 profile 的 `canonical_p()` 和 `diagnose()` 天然覆蓋。

---

## 1. 做 4 件事

| 功能 | 解決什麼問題 | 位置 | 狀態 |
|------|------------|------|------|
| **A. `compare(bhy=True)`** | 同時篩 200 因子不做 BHY = 挖 false discovery | `_api.py`；primary_p 來源改為 `profile.canonical_p()` | 實作於 migration Phase 4 |
| **B. `bhy_adjust()` 取代 `bhy_threshold()`** | 用 p-value 做 BHY，支援不同檢定類型混用 | 搬至 `stats/multiple_testing.py`（Phase 3） | 實作於 migration Phase 3 |
| **C. `redundancy_matrix()`** | AI 反覆發明同一類因子 | 新增 `metrics/redundancy.py`；輸入改接 `ProfileSet` | 可獨立實作 |
| **E. `keep_artifacts`** | 1000 因子 × 5MB = 5GB 記憶體 | `_api.py` + `Artifacts.compact` flag | 可獨立實作 |

原功能 D（Gate primary_p + suggestion）**作廢**：
- `primary_p` → `profile.canonical_p()`（per-type hard-coded canonical；cross-sectional 用 IC p、event 用主事件窗 CAR p、FM 用 λ p、TS 用 β p）。
- `suggestion` → `profile.diagnose()`（lazy method，回傳 `list[Diagnostic]`）。

**實作 sequence**：C、E 可先做（與 profile migration 正交）；A、B 在 migration Phase 3–4 實作。

---

## 2. 關鍵設計決策

### 2.1 BHY 是資訊欄位，不是 Gate

`compare()` 輸出的一個 Boolean 欄位，不替使用者做決定：

```
bhy_significant = primary_p 通過 BHY step-up AND status ∈ {PASS, CAUTION}
```

- Jensen et al. (2023)：多重檢定問題真實但不嚴重 → 資訊欄位而非硬性攔截
- Chordia et al. (2020)：真正強的因子都通過 BHY，淘汰的是 marginal 因子
- 有獨立經濟學理由的因子 → 使用者可忽略 BHY，用 OOS 驗證

### 2.2 用 p-value 而非 t-stat

不同 gate 用不同檢定（t, z, binomial），p-value 是共同貨幣。BHY 只需 p-values，不需知道來源（Benjamini & Hochberg 1995）。

### 2.3 Harvey (2016) vs 我們的情境

Harvey 的 t > 3.0 是針對「累積檢定數未知」的啟發式。我們的 N 已知 → 用 BHY 精確校正。

### 2.4 compare() 比較所有指標

`compare()` 輸出 profile 全部指標。BHY 回答「統計上真實嗎？」，profile 回答「好不好用？」，兩者正交。多維度篩選留給使用者 Polars filter。

### 2.5 redundancy_matrix 的 method 選擇

| method | 衡量什麼 | 需要的資料 | keep_artifacts=False |
|--------|---------|-----------|---------------------|
| `factor_rank` | 截面持股重疊度 | `prepared.factor` | 不可用 |
| `value_series` | 預測力時間同步性 | `intermediates` | 可用 |

default 用 `factor_rank`（quant 標準的冗餘度量）。文件說明兩者差異及與 `keep_artifacts` 的互斥。

### 2.6 跨 factor type 的 BHY

混合 type 的 BHY 不必要地保守。加 `bhy_group_by_type=True` (default) 分組做 BHY。

### 2.7 keep_artifacts 的 fail-loud

`Artifacts` 加 `compact: bool = False` flag。compact 時存取 `prepared` 報錯而非返回空 DataFrame。

---

## 3. 功能 A — `compare(bhy=True)`

### 修改：`factorlib/_api.py`

```python
def compare(
    results, *, sort_by="ic", ascending=False,
    bhy: bool = False, bhy_fdr: float = 0.05,
    bhy_group_by_type: bool = True,
) -> pl.DataFrame:
```

新增欄位（`bhy=True` 時）：`primary_p` (Float64), `bhy_significant` (Boolean)

### primary_p 提取

```python
def _extract_primary_p(result: EvaluationResult) -> float | None:
    # 新架構：每類 profile 自訂 canonical test
    if result.status == "FAILED" or result.profile is None:
        return None
    return result.profile.canonical_p()
```

### BHY 判定流程

```
1. 提取所有因子的 primary_p（含 FAILED — 它們是多重檢定負擔的一部分）
2. 按 factor_type 分組（bhy_group_by_type=True 時）
3. 每組呼叫 bhy_adjust(p_values, fdr) → boolean mask
4. bhy_significant = BHY 通過 AND status ∈ {PASS, CAUTION}
```

---

## 4. 功能 B — `bhy_adjust()` 取代 `bhy_threshold()`

### 修改：`factorlib/_stats.py`

```python
def bhy_adjust(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """BHY step-up procedure. 輸入 p-values，回傳 boolean mask."""
```

刪除 `bhy_threshold()`（無核心呼叫者）。同步改 `tests/test_p1_additions.py`。

---

## 5. 功能 C — `redundancy_matrix()`

### 新增：`factorlib/metrics/redundancy.py`

```python
def redundancy_matrix(
    results: dict[str, EvaluationResult],
    method: Literal["factor_rank", "value_series"] = "factor_rank",
) -> pl.DataFrame:
    """因子間 pairwise Spearman |ρ| 矩陣.
    
    method="factor_rank": 截面因子 rank correlation 均值（需要 prepared）
    method="value_series": IC/CAAR/beta series correlation（只需 intermediates）
    """
```

---

## 6. 功能 E — `keep_artifacts`

### 修改：`factorlib/_api.py` + `factorlib/evaluation/_protocol.py`

`batch_evaluate()` 加 `keep_artifacts: bool = True`。

`Artifacts` 加 `compact: bool = False`。compact 時存取 `prepared` raise RuntimeError。

---

## 7. 使用情境

### 分析師

```python
results = fl.batch_evaluate(candidates, keep_artifacts=False)
table = fl.compare(results, sort_by="ic_ir", bhy=True)
corr = fl.redundancy_matrix(results, method="value_series")  # keep_artifacts=False 時用 value_series
finalists = table.filter(pl.col("bhy_significant") & (pl.col("net_spread") > 0))
```

### AI Agent

```python
results = fl.batch_evaluate(candidates, stop_on_error=False)
for name, r in results.items():
    hints = r.profile.diagnose()   # list[Diagnostic] 取代原 gate suggestion
    p = r.profile.canonical_p()    # 單一 canonical p 取代 gate primary_p
table = fl.compare(results, bhy=True)
```

---

## 8. 修改清單

本計畫的 A/B 部分已併入 `plan_gate_redesign.md` Phase 3–4；C/E 獨立實作：

| 功能 | 檔案 | 備註 |
|------|------|------|
| A. compare(bhy=True) | `_api.py` | primary_p 改為 `profile.canonical_p()`，詳見 migration Phase 4 |
| B. bhy_adjust() | `stats/multiple_testing.py` | 新位置，詳見 migration Phase 3 |
| C. redundancy_matrix() | 新增 `metrics/redundancy.py` | 可獨立實作；輸入型別建議為 `ProfileSet` |
| E. keep_artifacts | `_api.py` + `evaluation/_protocol.py` | `Artifacts.compact` flag；獨立實作 |

原功能 D（Gate primary_p + suggestion）**作廢**，無需 gate 改動。

---

## 9. 不做的事

| 不做 | 為什麼 |
|------|--------|
| FactorSpec / Registry / Factor Zoo | Python for loop 已解決 |
| ScreenConfig / ScreenResult / ScreeningVerdict | `batch_evaluate()` + `compare(bhy=True)` 已覆蓋 |
| CorrelationGate / TurnoverGate | Gate protocol 不適合跨因子邏輯 |
| Expression DSL | 另一個專案等級的功能 |
| Dashboard / DB | 已有 integrations/ |
| Gate 重新設計 | 已作廢；gate 系統整體退場，見 `gate_redesign_v2.md` |

---

## 10. Verification Plan

```python
# test_bhy_adjust.py
def test_bhy_adjust_empty():
def test_bhy_adjust_all_significant():
def test_bhy_adjust_none_significant():
def test_bhy_adjust_partial():

# test_compare_bhy.py
def test_bhy_adds_columns():
def test_bhy_scales_with_n():
def test_bhy_failed_not_significant():
def test_bhy_false_backward_compat():
def test_bhy_group_by_type():

# test_redundancy_matrix.py
def test_symmetric():
def test_excludes_failed():
def test_detects_duplicates():
def test_factor_rank_vs_value_series():

# test_keep_artifacts.py
def test_prepared_raises_when_compact():
def test_intermediates_kept():
```

---

## 11. 文獻參考

見 `docs/literature_references.md`：Benjamini & Hochberg (1995), Benjamini & Yekutieli (2001), Harvey et al. (2016), Chordia et al. (2020), Jensen et al. (2023)
