# Spike T3.S3 — Fast-track two-stage evaluate_batch

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> **REFRAMED 2026-04-18**：本 spike 的 fast_track flag / CheapProfile /
> tuple return / 三種 gate 的原計畫**不執行**。檢討後認定 5 個子問題裡只
> 有 2 個是 library 層真缺口：(C) BHY 分母要能手動指定成 `n_total`、
> (E) 防止重複 BHY 校正。其餘（perf / memory / 一鍵便利性）回到使用者空間
> for-loop 解決。實際 scope 從 3-5d 縮到 <1d，plan 見
> `~/.claude/plans/users-cfh00884862-desktop-dst-code-fact-gentle-lamport.md`。
> 下面原始內容保留作決策背景。

**狀態**：REFRAMED → minimal `n_total` primitive（見上方說明）
**Owner**：jason pan
**日期**：2026-04-18
**解鎖的後續工作**：在 `fl.evaluate_batch` 加 opt-in 的兩階段篩選模式，
讓「1000 候選因子 → 只深入評估 Top 50」的流程不用使用者自己寫 for-loop。

---

## 1. 問題陳述

目前 `evaluate_batch` 對每個因子都跑完整 pipeline（build_artifacts
→ from_artifacts → 全部 metric）。Dogfood 顯示：

- **12 因子 × TW 2023H1-2024H1 panel**：~3s
- **30 因子 × 完整 TW panel**：~92s（retain full）～ 120s（retain compact）

外推到 **1000 因子**：估計 50-70 分鐘，記憶體容易爆。典型量化研究場景是
「丟 1000 候選進來，有 50 個值得看」——等全部跑完才做 BHY 篩選是浪費。

**想要的 API**（目標）：

```python
survivors = fl.evaluate_batch(
    candidates,                 # {name: df} × 1000
    factor_type="cross_sectional",
    fast_track=True,            # 新 opt-in flag
    fast_track_top_n=50,        # 或 fast_track_threshold=0.10
)
```

但加兩階段會動到 BHY 多重檢定的 denominator 語義——這是本 Spike 最難的
一題，不是工程難題，是**統計正確性**難題。

## 2. 現況（source-code 驗證）

- `factrix/_api.py:336-469` — `evaluate_batch` 當前 sequential loop，
  每個因子都跑完整 pipeline
- `factrix/_api.py:_evaluate_one` (Phase 3 polish) — 單因子管線已抽成
  可重用的 helper
- `factrix/evaluation/profile_set.py:232-317` —
  `multiple_testing_correct(p_source, fdr)` 對**整個 ProfileSet** 做 BHY，
  `p_values = np.array([float(p.canonical_p) for p in self._profiles])`
- BHY 前提：**同家族檢定的 p-values**。若 Stage 1 淘汰掉一批因子，它們的
  p-values 不進 BHY，`n`（分母）就變小，FDR 控制的嚴格性降低
- 沒有現成 "cheap profile" 的子集概念——`from_artifacts` 是 all-or-nothing

## 3. 待決策的政策問題

### 3.1 Stage 1 算什麼？

原計畫（Gemini 版）說「只算 `compute_ic() + ic()` 拿 canonical_p 就好」。
我的分析：**這是 p-hacking 的陷阱**。單指標篩選漏掉：
- IC 不顯著但 IC_IR 穩（罕見但存在）
- IC 顯著但 turnover 爆炸
- IC 顯著但 regime-specific（只在 bull 市場）

三個可行方案：

**(a) 單指標 gate：`canonical_p`**

最便宜（O(T)），但統計脆弱。**不建議**。

**(b) Cheap profile subset：IC + hit_rate + monotonicity + ic_trend**

仍然 O(T)（都是 per-date scalar），但訊號品質提升一個量級。
**建議**。

**(c) 完整 Profile 但跳過昂貴指標：不算 multi_split_oos_decay / q1_concentration**

OOS decay 和 q1_concentration 要掃全 panel；前者 O(T × splits)、後者 O(T × N)。
**可行但複雜**，每個 metric 要決定「Stage 1 要不要算」。

**建議**：**(b)**。定義一個 `CheapCrossSectionalProfile`（Profile 的子集），
Stage 1 每個因子回傳這個。Stage 2 存活者再做完整 `from_artifacts`。

### 3.2 Stage 1 gate 形式

三個形式可同時支援（不互斥）：

- `fast_track_top_n: int` — 按 canonical_p 排序取前 N 個
- `fast_track_threshold: float` — canonical_p <= 此值的全部留下
- `fast_track_require: Callable[[CheapProfile], bool]` — 任意 predicate

**建議**：三個都支援。`top_n` 最直覺給 AI-agent 用；`threshold` 給
學術派；`require` 給 advanced user。

### 3.3 BHY 在兩階段下的語義（**最難的一題**）

原始 BHY: reject H0_i iff `p_(i) <= α · i / (n · h(n))`，其中 `n` 是
**檢定總數**。兩階段下 `n` 有兩個候選：

**(a) n = total candidates**（含被 Stage 1 淘汰的）

```python
# Stage 1 用 canonical_p 做 BHY，threshold = α · rank / (1000 · h(1000))
# Stage 2 survivors 不再做新的 BHY
```

優點：**統計正確**——原始完整樣本上的 BHY，沒動。
缺點：Stage 1 的 canonical_p 是**單指標**（IC），用它篩掉因子後，
Stage 2 的存活者已經是 conditioned on 顯著。

**(b) n = Stage 2 survivors**

```python
# 對 survivors 重新做 BHY，n = 50
```

優點：看起來乾淨，「我就是想看這 50 個誰最強」。
缺點：**FDR 失去保證**。Stage 1 的篩選本身是 winner's curse，對
survivors 再 BHY 等於雙重選擇，實際 type I error 會超過宣稱的 α。

**(c) 不跑 BHY**

```python
# Stage 1 = 用 canonical_p 排序 + top_n
# Stage 2 = 完整 Profile，不做 multiple-testing
```

誠實。使用者明知 Top-N 不是 BHY，就是「最看好的 50 個」。FDR 不宣稱。

**建議**：**(a)**。Stage 1 結果同時產出 BHY 結果（on full n），Stage 2
survivors 繼承 `bhy_significant` 欄位，不做第二次 BHY。

實作細節：
- `fl.evaluate_batch(fast_track=True)` 回傳時，`ProfileSet` 的 `_df` 已
  含 `p_adjusted` / `bhy_significant`（跟 `multiple_testing_correct()`
  邏輯一樣，但自動跑過）
- 使用者不需要再呼叫 `.multiple_testing_correct()`——事實上**應該拒絕**
  他這麼做（第二次 BHY 會破壞 (a) 的語義）

### 3.4 Stage 1 淘汰者的回傳形狀

**(a) 只回 survivors**

```python
# return ProfileSet of 50
```

資訊損失：淘汰者的 canonical_p 使用者看不到。

**(b) 回全部，淘汰者的深度欄位 = None**

```python
# return ProfileSet of 1000, 但 950 個的 oos_* / q1_concentration / turnover 都是 None
```

schema 複雜化：Profile 欄位都要變 `| None`。破壞既有 API。

**(c) 回 tuple: (survivors_ProfileSet, cheap_profiles_DataFrame)**

```python
survivors, stage1 = fl.evaluate_batch(..., fast_track=True)
```

完整資訊保留、schema 不動、但 API 變 tuple（類似 `keep_artifacts=True`）。

**建議**：**(c)**。跟 `keep_artifacts=True` 模式對稱。`stage1` 是輕量
DataFrame（1000 × 5 欄位），可以塞進 MLflow 作審計記錄。

### 3.5 與 `keep_artifacts` / `compact` 的互動

`fast_track=True` + `keep_artifacts=True`：只保留 survivors 的 Artifacts
（~50 個），不保留被淘汰者的。這正是 fast_track 的價值所在——記憶體
peak = Stage 1 階段的 1000 × cheap intermediates + Stage 2 的 50 × full。

`fast_track=True` + `compact=True`：自動 imply `keep_artifacts=True`？
還是維持「compact 沒有 keep 會 raise」的嚴格性？**建議維持嚴格**——
使用者明確寫 `keep_artifacts=True, compact=True` 才啟用兩階段記憶體優化。

### 3.6 失敗處理

`stop_on_error` 的語義在兩階段下：
- Stage 1 失敗（IC 算不出）→ 原本 `on_error` 處理，該因子直接跳過（不進
  Stage 2）
- Stage 2 失敗 → 同現有語義

這一層**不需要新 policy**，沿用即可。

## 4. 非目標

- **跨階段 early-stop**：`on_result` 在 Stage 2 仍支援早停，Stage 1 不加
  早停（因為 Stage 1 本來就是 lightweight）
- **並行化 Stage 1**：先不做（留給 T3.S4 Spike）。Stage 1 已經 10x 快於
  Stage 2，串行也可接受
- **Event / Macro 的 fast-track**：CS 先行。Event 的 caar_p 也可以做相同
  架構；Macro 本來就只有 3-10 個 country 規模，不需要 fast-track
- **自定義 Cheap Profile 欄位集**：不讓使用者選，固定為 §3.1 (b) 的 4 個
  metric。減少決策負擔

## 5. 實作順序（待 §7 sign-off 後）

估時：**3-5d**（集中在 Cheap Profile 的設計 + BHY 語義驗證）

1. 設計 `CheapCrossSectionalProfile` dataclass（僅 IC + hit_rate +
   monotonicity + ic_trend）——**0.5d**
2. `_build_cs_cheap_artifacts` / `from_artifacts_cheap`——**0.5d**
3. `evaluate_batch(fast_track=...)` 分支邏輯：Stage 1 loop → BHY on
   full n → Stage 2 只跑 survivors——**1.0d**
4. Tuple return shape + typing overload——**0.3d**
5. Tests：**1.0d**
   - fast_track + top_n = 50 in 1000，檢查 survivors 數量對、原始
     ProfileSet 裡 bhy_significant 正確
   - fast_track + threshold = 0.10，survivors 都小於 threshold
   - fast_track + require = custom predicate
   - BHY semantic sanity：(a) 語義下 survivors 的 p_adjusted 等於單階段
     evaluate_batch + multiple_testing_correct 的結果
6. README / docstring：強調**不要**對 fast_track 回傳的 ProfileSet 再
   呼叫 `.multiple_testing_correct()`——**0.5d**

## 6. 風險

- **統計誤用**：使用者可能仍然對 survivors ProfileSet 跑 BHY。
  **緩解**：`multiple_testing_correct` 檢查 `mt_already_applied` 標記，
  重複呼叫 raise；fast_track 回傳的 ProfileSet 標記為已校正
- **效能不如預期**：如果 Stage 1 cheap profile 實際上跟 full profile
  差不了多少（因為 polars 都是 vectorized，省下的可能是小頭），整個
  feature 價值打折。**緩解**：實作前先 microbenchmark cheap 的單因子
  延遲，確認是 full 的 < 30% 才繼續
- **API complexity**：fast_track 加上三個 gate 參數（top_n / threshold /
  require）+ tuple return + BHY 語義警告，使用者學習曲線陡。**緩解**：
  README 主 Quick Start 不提，開獨立 "Large-batch screening" 段落
- **跟 Phase 2 `with_extra_columns` 的互動**：survivors ProfileSet 可以
  加自訂欄位，cheap profiles DataFrame 是 polars DataFrame 也可以。
  兩者 join 是使用者責任，**不要在 factrix 內 merge**

---

## 7. Sign-off checklist

- [ ] Owner 確認 §3.1 Cheap Profile 的 metric 集合（建議 IC + hit_rate
      + monotonicity + ic_trend）
- [ ] Owner 確認 §3.2 gate 形式（建議三個都支援）
- [ ] Owner 確認 §3.3 BHY 語義（建議 (a)：n = total, Stage 1 做完
      就不再做）
- [ ] Owner 確認 §3.4 回傳形狀（建議 (c) tuple）
- [ ] Owner 確認 §3.5 `compact` 互動規則（建議維持嚴格 keep_artifacts=True
      顯式）
- [ ] 實作前先跑 microbenchmark 確認 Cheap Profile < 30% of Full profile
      延遲（§6 風險）
