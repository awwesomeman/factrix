# Plan: Gate 系統重新設計

> **Status: Superseded — 2026-04-17**
> **取代者**：`docs/gate_redesign_v2.md` + `docs/plan_gate_redesign.md`
>
> 本文件討論的方案 A–F 之所以難以收斂，根本原因是 Gate 抽象本身同時承擔了「決策」、「推論 p 的生產」、「AI 反饋」三個不同責任，且無法表達不同因子類型的 canonical test 差異。
>
> 結論：**Gate 系統全面退場，改用 per-factor-type `FactorProfile` 作為核心抽象。**
>
> 保留本文件作為決策脈絡。新的實作方向見 ADR 與 migration plan。
>
> ---
>
> （以下為原始內容，僅供歷史參考。）
>
> Status: Draft — 2026-04-17
> 從 `factor_screening.md` v3.2 獨立出來的議題

---

## 1. 問題陳述

因子篩選計畫（v3.2）需要在 gate 上加 `primary_p`（供 BHY）和 `suggestion`（供 AI）。在討論過程中浮現了多個 gate 設計方向，但每個都有 trade-off。先獨立討論 gate 設計，再回頭決定篩選計畫的功能 D 怎麼實作。

---

## 2. 現有 Gate 架構

```python
# _protocol.py
GateFn = Callable[[Artifacts], GateResult]

@dataclass
class GateResult:
    name: str
    status: GateStatus           # "PASS" | "FAILED" | "VETOED"
    detail: dict[str, Any]       # untyped bag
```

```python
# presets.py — gates 是 AND（sequential, short-circuit on FAILED）
CROSS_SECTIONAL_GATES = [significance_gate, oos_persistence_gate]
```

```python
# significance_gate — 內部有 OR 邏輯
def significance_gate(artifacts, *, threshold=2.0):
    # 計算 IC t-stat 和 spread t-stat
    # 任一 >= threshold → PASS
```

### 現有架構的優點
- 極簡：gate 就是一個 function
- 可組合：`functools.partial` 調參數
- 零抽象開銷

### 現有架構缺乏的
- gate 沒有 p-value 欄位 → BHY 不知道用哪個 p
- gate 沒有 suggestion 欄位 → AI 不知道為什麼被拒
- OR 邏輯藏在 gate 內部 → primary_p 的來源不透明

---

## 3. 討論過的方案

### 方案 A：detail dict 放 convention key

```python
detail={"primary_p": p, "suggestion": s}
```

- 優點：零 protocol 改動
- 缺點：convention 可忘/可拼錯，不可發現

### 方案 B：GateResult 加 typed field

```python
@dataclass
class GateResult:
    ...
    primary_p: float | None = None
    suggestion: str = ""
```

- 優點：typed、IDE 補全、自文件化、向後相容
- 缺點：改 `_protocol.py`（但只加 2 個有 default 的 field）

### 方案 C：canonical test（每個 gate 指定一個 canonical 檢定供 BHY 用）

```python
# significance_gate 用 IC 做 canonical，spread 是 backup
primary_p = _p_value_from_t(ic_tstat, n)  # 永遠用 IC
```

- 優點：不需拆 gate
- 缺點：「gate 名字是 significance，但 primary_p 只看 IC」不直覺

### 方案 D：primary_p 跟著 via 走

```python
if "IC" in via:
    primary_p = ic_p
elif "Q1-Q5_spread" in via:
    primary_p = spread_p
else:  # FAIL
    primary_p = min(ic_p, spread_p)
```

- 優點：語義清楚（「gate 用哪個證據做決定，p 就來自那個」）
- 缺點：FAIL 時仍用 min(p)，但影響微乎其微（FAIL 因子的 bhy_significant 永遠 False）

### 方案 E：原子 gate + `any_of` combinator

```python
ic_gate(artifacts) -> GateResult      # 一個 gate = 一個 test = 一個 p
spread_gate(artifacts) -> GateResult

CROSS_SECTIONAL_GATES = [
    any_of(ic_gate, spread_gate),      # OR
    oos_persistence_gate,              # AND
]
```

- 優點：每個 gate 一個 test → primary_p 無歧義；可組合
- 缺點：使用者 customize 時多一層 `any_of` wrapper

### 方案 F：`threshold()` factory — 使用者直接用 metric + threshold

```python
result = fl.evaluate(prepared, "Mom", gates=[
    fl.threshold("ic", stat_gte=2.0),
    fl.threshold("oos_decay", value_gte=0.5),
])
```

- 優點：使用者不需要知道 gate 概念，直接說「哪個指標、什麼門檻」
- 缺點：需要 `compute_metric()` routing 基礎設施（知道怎麼從 artifacts 呼叫每個 metric）；複雜 gate（OOS sign flip）不適用

---

## 4. 評估維度

| 維度 | 說明 |
|------|------|
| **使用者簡單度** | 一般使用者（不改 gate）的體驗 |
| **開發者簡單度** | 寫 custom gate 的體驗 |
| **統計嚴謹性** | primary_p 的正確性 |
| **改動量** | 對現有程式碼的影響 |
| **泛用性** | 未來加新 factor type 或 custom gate 時的擴展性 |

### 方案評估矩陣

| 方案 | 使用者 | 開發者 | 統計 | 改動量 | 泛用性 |
|------|--------|--------|------|--------|--------|
| A. convention key | 不變 | 差（可忘） | 看實作 | 最小 | 差 |
| B. typed field | 不變 | 好（IDE 補全） | 看實作 | 小 | 好 |
| C. canonical | 不變 | 中（需知規則） | 嚴謹 | 小 | 中 |
| D. via-based | 不變 | 好（自然） | 可接受 | 小 | 好 |
| E. atomic + any_of | 稍複雜 | 最好 | 最嚴謹 | 中 | 最好 |
| F. threshold() | 最簡 | 最好 | 嚴謹 | 大 | 好 |

---

## 5. 待決定

1. **primary_p 的來源邏輯**：via-based (D) vs canonical (C) vs atomic (E) vs threshold (F)?
2. **GateResult 要不要加 typed field (B)?** — 不管選哪個 primary_p 方案，這都是獨立的改進
3. **`any_of` / `threshold()` 是否在 scope 內?** — 還是留 follow-up?
4. **是否需要 `compute_metric()` routing?** — 方案 F 需要，其他不需要
5. **OOS persistence 等複雜 gate 怎麼處理?** — sign flip 不是 threshold 能表達的

---

## 6. 與 factor screening plan 的關係

factor screening plan (v3.2) 的功能 D「Gate primary_p + suggestion」依賴本議題的結論。

目前 factor screening plan 的功能 A/B/C/E 不受 gate 設計影響，可以先實作。

待本議題結論後，更新 factor screening plan 的功能 D 部分。
