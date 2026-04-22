# Spike T3.S2 — Level 2 metrics 接入 Profile

> **IMPLEMENTED 2026-04-18** (commit `3d7ac0a`，§3.4.2 revision `25fc335`)：
> 三個 Level 2 metric 已接入 `CrossSectionalProfile`，opt-in via
> `CrossSectionalConfig.regime_labels` / `multi_horizon_periods` /
> `spanning_base_spreads`。Profile 新增 6 個欄位 + 3 條 diagnose rule。
> 下面原始內容保留作決策背景。

**狀態**：IMPLEMENTED（見上方）
**Owner**：jason pan
**日期**：2026-04-18
**解鎖的後續工作**：把 `regime_ic` / `multi_horizon_ic` / `spanning_alpha`
從「library-only helper」升級為「`evaluate()` 預設回傳的 Profile 欄位」。

---

## 1. 問題陳述

Phase 1 T1.2 把三個 metric 定位為 Level 2：使用者要用就自己 import +
組合。Dogfood 顯示這可行，但有三類使用者會一直撞牆：

1. **AI-agent 批次篩選**：想要 `profile.regime_ic_min_tstat` 直接進
   filter expression，不想再為每個因子另外呼叫一次 metric
2. **Notebook exploratory**：想從 `describe_profile()` 反射時看到這些欄位
3. **MLflow / reporting**：想把整個 Profile 當一列結構化資料 log 出去

Level 2 模式下這三類工作流都要多一層膠水。**問題不是「這些 metric 缺失」**
——它們已實作且測過——**而是 Profile schema 要不要吸收它們，以及怎麼吸收**。

## 2. 現況（source-code 驗證）

- `factorlib/metrics/ic.py:161` — `regime_ic(ic_series, regime_labels)`
  → `MetricOutput`，metadata 帶 `per_regime: dict[str, dict]` 與
  `direction_consistent: bool`
- `factorlib/metrics/ic.py:280` — `multi_horizon_ic(df, periods=[1,5,10,20])`
  → `MetricOutput`，metadata 帶 `per_horizon: dict[int, dict]`；value = 平均 IC、
  stat = min |t|
- `factorlib/metrics/spanning.py:92` — `spanning_alpha(candidate_returns,
  base_returns)` → `MetricOutput`
- **無現有 integration**：`CrossSectionalProfile.from_artifacts`
  (`evaluation/profiles/cross_sectional.py:97-185`) 沒呼叫任何一個
- **無 regime source**：factorlib 沒有內建 regime classifier；使用者必須
  自己傳 `regime_labels: dict[str, str]`（date → label）
- Phase 1 T1.2 已刪 `BaseConfig.multi_horizon_periods`，要加回來

## 3. 待決策的政策問題

### 3.1 Regime 從哪裡來？三個可行選項

**(a) 使用者傳 regime series**

```python
regimes = pl.DataFrame({"date": [...], "regime": ["bull", "bear", ...]})
config = fl.CrossSectionalConfig(regime_labels=regimes)
```

明確、零魔法；使用者自己的責任。

**(b) Config 內建 classifier**

```python
config = fl.CrossSectionalConfig(regime_classifier="vol_tercile")
```

把「判斷 regime」埋進 pipeline。優點是 turnkey，缺點是政策爭議（用哪個
指標切？vol？drawdown？VIX？）每個 classifier 都要養。

**(c) 沒傳 regime 就跳過**

Profile 欄位全部 `None`，diagnose 也 quiet。零 decision cost。

**建議**：**(a) + (c)**。傳 `regime_labels` 才跑，否則 `None`。
不要在 pipeline 內做 classifier。使用者真的需要標準化 regime 再另開
`factorlib.regimes` 模組（out of scope）。

### 3.2 `multi_horizon_ic` 的 `periods` 從哪裡來？

三個選項：

**(a) 重新加回 Config**

```python
config = fl.CrossSectionalConfig(multi_horizon_periods=[1, 5, 10, 20])
```

Phase 1 T1.2 剛因為它是 dead config 而刪掉。加回去的正當性：**現在它會被
真的消費**。

**(b) Profile 建構時固定寫死**

```python
mh_m = multi_horizon_ic(prepared, periods=[1, 5, 10, 20])
```

No API surface。但使用者想看 5/10/20 天以外的 horizon 就沒救。

**(c) Metric 內部沿用現有 default `periods=None` → `[1,5,10,20]`**

實務上跟 (b) 等價。

**建議**：**(a)**。Phase 1 刪它是對的（當時沒消費），現在要消費就加回來。
欄位 docstring 要寫清楚只有 `multi_horizon_ic` 會讀。

> **Post-merge revision (2026-04-18)**：原方案說「always run (cheap)」，
> review 時實測 `multi_horizon_ic` 在 full TW panel 需要 4× `compute_ic`，
> 不 cheap。改為 **opt-in**：`multi_horizon_periods=None` 時 skip，
> 傳 list 才跑。與 regime / spanning 的 opt-in 語義一致。

### 3.3 `spanning_alpha` 的 base factors 從哪裡來？

完全類比 T3.S1 的正交化 basis 問題：**使用者傳 DataFrame**。唯一差別是
spanning 比的是 **returns** 而非 factor exposures：

```python
config = fl.CrossSectionalConfig(
    spanning_base_returns=base_ret_df,  # (date, factor_name, return) long format
)
```

**建議**：跟 T3.S1 對齊——使用者傳 DataFrame，None 就跳過。
**不要**復用 `orthogonalize` 欄位（語義不同：ortho 是 cross-sectional
exposure 回歸、spanning 是 time-series return 回歸）。

### 3.4 Profile 新增欄位（scalar 壓縮）

這是最難決定的一題。三個 metric 各自產生結構化的 metadata，要壓縮成
幾個 scalar 才能放進 frozen dataclass：

#### 3.4.1 regime_ic

| 方案 | 欄位 | 語義 |
|------|------|------|
| A | `regime_ic_min_tstat: float \| None` | 最弱 regime 的 \|t\| —— 保守估計 |
| B | `regime_ic_consistent: bool \| None` | 所有 regime 方向一致？—— 穩定性指標 |
| C（推薦）| **同時放 A + B** | 兩者正交，各自回答不同問題 |

原始 `MetricOutput.metadata["per_regime"]` 是 dict，在 frozen dataclass
存不下。要存結構化明細，必須走 Phase 2 的 `with_extra_columns` 或者
`artifacts.intermediates["regime_ic_detail"]`。**不要把 dict 塞進 Profile**
——破壞 schema homogeneity。

#### 3.4.2 multi_horizon_ic

原始預設 horizons = `[1, 5, 10, 20]` → 4 data points。OLS/Theil-Sen
slope 在 n=4 都脆弱；raw ratio 則遇到 sign flip / 零分母會誤導。折衷：

| 欄位 | 定義 | 回答的問題 |
|------|------|------------|
| `multi_horizon_ic_retention: float \| None` | `ic_at_longest / ic_at_shortest`；`\|ic_short\| < 1e-4` 時為 None | 「長 horizon 還保留多少 day-1 alpha」 |
| `multi_horizon_ic_monotonic: bool \| None` | `\|IC(h)\|` 是否隨 h 單調不升 | 「衰減曲線健康嗎？有反轉嗎？」 |

**為什麼兩個而不是一個**：retention 答 magnitude（"剩 50%"），monotonic
答 shape（"反轉 or 單調衰減"）。一個因子 retention=-0.6 + monotonic=False
代表 **sign 反轉**（momentum 在長 horizon 反轉），交易邏輯與單調衰減完全
不同，單一欄位表達不了。

**明確捨棄**：OLS slope、Theil-Sen slope、exponential half-life fit——
n=4 下都不穩定；需要 slope 的使用者自己從
`artifacts.intermediates["multi_horizon_detail"]` 取 per-horizon dict 算。

#### 3.4.3 spanning_alpha

| 方案 | 欄位 | 語義 |
|------|------|------|
| A | `spanning_alpha_t: float \| None` | alpha t-stat |
| B | `spanning_alpha_p: PValue \| None` | alpha p-value |
| C（推薦）| **A + B** | t 給量級、p 給判決 |

`spanning_alpha_p` 是 p-value，意味著必須**決定要不要加入 `P_VALUE_FIELDS`
whitelist** —— 若加入，使用者可以對它跑 BHY 校正。**建議不加**：
canonical_p 只能是 one per Profile，spanning 是輔助證據不是主判決。

### 3.5 Diagnose rules 新增

建議加三條（都在 R² > 0.7 之類的硬性門檻才 fire）：

- `cs.regime_ic_inconsistent`（severity=warn）
  predicate: `regime_ic_consistent is False`
  訊息："IC direction flips across regimes — factor is regime-dependent."
- `cs.multi_horizon_decay_fast`（severity=warn）
  predicate: `multi_horizon_ic_retention is not None and retention < 0.3`
  訊息："IC retains <30% from shortest to longest horizon — likely
  overfitting to T+1."
- `cs.spanning_alpha_absorbed`（severity=warn）
  predicate: `spanning_alpha_p is not None and spanning_alpha_p > 0.10`
  訊息："Alpha vs base factors not significant — may be a repackaging."

### 3.6 Schema migration 衝擊

新 CrossSectionalProfile 會從 25 個欄位 → 30 個欄位（加 5 個）。
現有 Profile 反射測試（`describe_profile`）會看到更多欄位——預期變動，
不破壞。

**ProfileSet polars schema** 會多出 5 個 float/bool 欄位，全部可為
`None`。empty-set dtype 解析已經能處理 `float | None`（T3.S1 驗證過），
沒有新的 dtype pitfall。

## 4. 非目標

- **內建 regime classifier**：out of scope，另開 `factorlib.regimes` 模組
- **把 per-regime / per-horizon 明細塞進 Profile**：違反 frozen dataclass
  schema；使用者需要明細就走 `artifacts.intermediates` 或
  `with_extra_columns`
- **把 `spanning_alpha_p` 加進 `P_VALUE_FIELDS`**：canonical_p 只能一個，
  spanning 是輔助不是主判決
- **EventProfile / MacroPanelProfile / MacroCommonProfile 的對應欄位**：
  Cross-sectional 先行；三者 regime/decay 語義要各自設計，留給未來 Spike

## 5. 實作順序（待 §7 sign-off 後）

估時：**2-3d**（視 §3.4 採用的欄位數量）

1. 加 `multi_horizon_periods` / `regime_labels` / `spanning_base_returns`
   到 `CrossSectionalConfig`——**0.3d**
2. `_build_cs_artifacts` 在三個 metric 有 input 時呼叫，結果塞
   `intermediates["regime_ic_stats"]` 等 1-row DataFrames——**0.5d**
3. `CrossSectionalProfile` 加欄位 + `from_artifacts` 讀 intermediates——
   **0.5d**
4. 三條 diagnose rule——**0.2d**
5. Unit + integration tests（每條路徑：有 input 跑到、無 input skip、
   極端 R² / decay 觸發 diagnose）——**0.5d**
6. README Level 2 更新：把三個 metric 的「standalone」段落改成
   「pipeline-integrated」說明，保留 standalone 作為 advanced escape——
   **0.2d**

## 6. 風險

- **API surface 膨脹**：加 3 個 config 欄位 + 5 個 Profile 欄位。看起來
  不多，但若之後每個 metric 都這樣加，Profile 會變肥。**緩解**：這三個
  是經過 Level 2 dogfood 驗證確實被需要的；再加新的 metric 前要先跑一遍
  「有沒有人真的用 Level 2 helper 形式」的驗證
- **`multi_horizon_periods` 反覆**：Phase 1 刪它，現在加回來，使用者可能
  困惑。**緩解**：commit message 明確引用 Phase 1 T1.2 + 本 Spike，說明
  為什麼重新納入
- **regime 品質問題**：`regime_ic` 需要至少 2 個 regime 各有 `MIN_IC_PERIODS`
  筆觀測（看 metric 原碼）。小樣本下會大量 skip。**緩解**：Profile 欄位
  設計為 `float | None`，沒跑到就 None；diagnose rule 只在有值時 fire
- **跟 T3.S1 orthogonalize 的互動**：orthogonalize 後 IC 會變小，
  `multi_horizon_ic_decay` 可能被放大效應影響。**緩解**：不是 bug，
  是正確行為；docstring 註記 decay 是**當前** factor 的衰減率（可能已
  residualize）

---

## 7. Sign-off checklist

Signed off 2026-04-18 with one adjustment to §3.4.2 (retention +
monotonic pair instead of single decay ratio).

- [x] §3.1 regime 來源：使用者傳 `regime_labels`，沒傳就跳過
- [x] §3.2 `multi_horizon_periods` 加回 `BaseConfig`
- [x] §3.3 `spanning_base_spreads` 獨立欄位（dict[str, pl.DataFrame]）
- [x] §3.4.1 regime_ic：`regime_ic_min_tstat` + `regime_ic_consistent`
- [x] §3.4.2 multi_horizon：`retention` + `monotonic` 兩欄位（revised
      2026-04-18 — 原計畫單一 decay 欄位無法分辨 sign flip vs monotonic
      decay）
- [x] §3.4.3 spanning：`spanning_alpha_t` + `spanning_alpha_p`；不入
      `P_VALUE_FIELDS`
- [x] §3.5 三條 diagnose rules
