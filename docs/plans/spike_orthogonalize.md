# Spike T3.S1 — 正交化 pipeline 整合

> **SUPERSEDED 2026-04-20**: This document references legacy identifiers (e.g. `q1_q5_spread`, `q1_concentration`) that were renamed through two passes:
>   1. `8f15db8` — `q1_q5_spread` → `long_short_spread`; `q1_concentration` → `top_concentration`
>   2. `24d85eb` (Phase 2a) — `long_short_spread` → `quantile_spread`
>
> See `docs/naming_convention.md` for the current canonical names. This doc is retained as historical record; do **not** use it as the canonical source.

> **IMPLEMENTED 2026-04-18** (commit `931611a`)：pipeline 整合已上 main。
> `CrossSectionalConfig.orthogonalize` 接受 `pl.DataFrame | None`（coverage
> 門檻 + fail-loud），`CrossSectionalProfile` 新增 `orthogonalize_r2_mean` /
> `orthogonalize_n_base` 欄位，並登錄 `cs.orthogonalize_absorbed_most`
> diagnose rule。下面原始內容保留作決策背景。

**狀態**：IMPLEMENTED（見上方）
**Owner**：jason pan
**日期**：2026-04-18
**解鎖的後續工作**：把 `orthogonalize` 重新接回 CS pipeline（Phase 1 T1.1 因為 false-provenance bug 把它拿掉了）。

---

## 1. 問題陳述

Phase 1 T1.1 刪掉了 `CrossSectionalConfig.orthogonalize` 與
`CrossSectionalProfile.orthogonalize_applied`，原因是：**flag 根本沒被
`preprocess_cs_factor` 消費**——Profile 會宣稱 `orthogonalize_applied=True`
但實際上因子數值未經正交化（silent correctness bug）。

`factorlib/preprocess/orthogonalize.py::orthogonalize_factor` 仍是獨立可用的
helper。要把正交化**重新納入 pipeline**，要決定的是「**怎麼接**」，不是
「數學對不對」（數學本身已經 OK）。

## 2. 現況（source code 驗證）

- `orthogonalize_factor`（`preprocess/orthogonalize.py:36`）接受
  `(factor_df, base_factors, factor_col, base_cols)`，回傳
  `OrthogonalizeResult(df, mean_betas, mean_r_squared, n_dates)`。
- `MARKET_DEFAULTS`（`config.py`）原本預留每市場的 basis 命名
  （已於 2026-04-18 按 §3.1 建議刪除 `ortho_factors` 子欄位）
  （`tw: size/value/momentum/industry_tse30`；`us: size/value/momentum/industry_gics`），
  但只是字串標籤，沒有對應的 DataFrame builder。
- `factorlib/factors/size.generate_size` 與
  `factors.industry.encode_industry_dummies` 已存在；
  `generate_momentum_60d` 可作為 TW momentum basis；
  **TW 的 value factor generator 目前缺席**。
- `preprocess_cs_factor`（`preprocess/pipeline.py:62`）目前跑 Step 1-5
  （forward return → winsorize → abnormal return → MAD → z-score）；
  正交化會成為可選的 Step 6。
- `tests/test_orthogonalize.py` 涵蓋 helper 的正確性；目前沒有
  pipeline 層級的整合測試。

## 3. 待決策的政策問題

以下是**真正的**問題——不是「要不要寫」，而是「API 合約長什麼樣」。
每一題都必須在實作前定案。

### 3.1 Basis factors 從哪裡來？

三個方案；第一個是唯一能站得住的 default。

**(a) 使用者傳 DataFrame**——明確、零魔法。

```python
base_df = build_base_factors(panel)    # 使用者自己準備
config = fl.CrossSectionalConfig(
    orthogonalize=base_df,             # truthy 就跑；None 就跳過
    orthogonalize_cols=["size", "mom_60d", *industry_cols],
)
```

**(b) Config 放 callable**——宣告式但把工作藏進 pipeline 裡面；
debug 跟測試重現性很差。

**(c) 從 `MARKET_DEFAULTS` 自動組**——零使用者代碼，但只有當 factorlib
內建每個命名的 factor generator 才有用。TW 的 value generator 不存在，
行不通。

**建議**：走 **(a)**。**把 `MARKET_DEFAULTS.ortho_factors` 整個子欄位刪掉**
——它是 aspirational、不 functional。使用者要便利，就自己用
`fl.orthogonalize_factor` 組（跟今天一樣），把產出的 DataFrame 傳進 config。

### 3.2 Config 欄位形狀

```python
@dataclass(kw_only=True)
class CrossSectionalConfig(BaseConfig):
    ...
    orthogonalize: pl.DataFrame | None = None
    orthogonalize_cols: list[str] | None = None    # None -> 用所有非 key 欄位
    orthogonalize_min_coverage: float = 0.95       # 硬門檻（見 §3.4）
```

- `orthogonalize=None`（預設）→ 跳過，行為與現況相同。
- `orthogonalize=base_df` → Step 5 後呼叫 `orthogonalize_factor`。
- `orthogonalize_cols=None` → 自動取 `date`/`asset_id` 以外的所有欄位。

**拒絕的替代方案**：`orthogonalize: bool` 搭配另一個 `base_factors` kwarg。
兩個欄位必須保持同步會產生一類新的 bug（使用者只設 flag 忘傳 DF，或反之）。
**單一欄位是 DF-or-None 就不可能漂移**。

### 3.3 Profile 欄位

原本被刪的 `orthogonalize_applied: bool` 只是 provenance。換成更有資訊量
的形式便宜又有用：

```python
@dataclass(frozen=True, slots=True)
class CrossSectionalProfile:
    ...
    orthogonalize_r2_mean: float | None        # 沒跑時為 None
    orthogonalize_n_base: int                  # 沒跑時為 0
```

- 消費者立刻知道**有沒有被正交化**、**被削掉多少變異**。
- `orthogonalize_r2_mean > 0.5` 是很好的 diagnose 訊號（「你的因子大部分
  都被 basis 解釋掉了，殘差很小，IC/spread 反映的是殘差而不是原始因子」）。
- **不加 boolean 欄位**——需要時從 `n_base > 0` 推導即可。

### 3.4 Coverage gate

`orthogonalize_factor` 目前遇到 >5% 的 row 沒有 base coverage 時會 log
warning，然後**靜默保留原始值**。把這種靜默 fallback 放進 pipeline，
就是 Phase 1 剛修掉的那類 provenance bug。兩個選項：

**選項 A** —— 低於 `orthogonalize_min_coverage` 直接 **hard fail**
（預設 0.95）。`preprocess_cs_factor` raise，錯誤訊息列出缺席的 dates /
asset_ids。Fail fast；使用者明確同意 gap 才調低門檻。

**選項 B** —— Soft split：`factor` 欄位有 residual 就用 residual、
沒有就用原值，Profile 回報 coverage 比例（例如 `orthogonalize_coverage=0.88`），
太低 fire diagnose。

**建議**：**A**。正交化的目的是產生一條**可解釋的訊號**；半正交化的資料
違背我們重構時想保護的 interpretability。Soft fallback 正是上一次
silent bug 的成因。

### 3.5 Residual 要不要 re-z-score？

OLS residual 均值為 0，但標準差取決於 basis R²。R²=0.3 時 residual
std ≈ 0.84（接近原本 z-score）；R²=0.9 時 residual std ≈ 0.32。

下游指標在 **rank 意義**上 scale-invariant（IC、quantile_spread），但
在**絕對值意義**上不是——`net_spread`（return 單位）、`q1_q5_spread`
（return 單位）都會隨著 scale 跑掉。

**建議**：**正交化後 re-z-score**。保留「`factor` 每日單位變異」這條
不變式，讓下游指標的 reference scale 不會因為切換 `orthogonalize`
就靜默漂移。

### 3.6 要加哪些 diagnose rules

有了 `orthogonalize_r2_mean` 之後，值得出一條：

```
code: cs.orthogonalize_absorbed_most
severity: warn
predicate: p.orthogonalize_r2_mean is not None and p.orthogonalize_r2_mean > 0.7
message: "Basis factors absorbed >70% of variance (R²={r2}). The
         residual signal is small; IC / spread reflect it, not the
         original factor. Check whether the basis set is appropriate."
```

舊的 `cs.orthogonalize_not_applied` 規則**不要**復活——它會對每個沒做
正交化的因子都 fire，也就是所有日常使用場景都會 noise。

## 4. 非目標（刻意不做）

- **自動用 price data 組 basis factors**：需要我們沒有的 TW/US value
  factor generator。讓使用者自己組。
- **多階段正交化**（例如對「batch 裡前面的因子」正交化）：這是 spanning
  alpha 的領域，scope 不同。
- **時變 basis set**（不同日期可用的產業不同）：`orthogonalize_factor`
  已經用 per-date lstsq 處理單日缺席，不需要額外處理。

## 5. 實作順序（待 §7 sign-off 後）

估時：**1.5d**。**§3 的政策決定未定案前不動工**。

1. 新增三個 config 欄位（§3.2）——**0.2d**
2. 在 `preprocess_cs_factor` 加 Step 6（z-score 之後、select 之前）。
   有 basis DF 就呼叫 `orthogonalize_factor`；檢查 coverage 是否過 gate；
   residual 重做 z-score（§3.5）——**0.4d**
3. 把 `orthogonalize_r2_mean` + `orthogonalize_n_base` 從
   `_build_cs_artifacts` 穿透到 `CrossSectionalProfile.from_artifacts`
   （Profile dataclass 加兩個欄位）——**0.3d**
4. 加 diagnose rule（§3.6）——**0.1d**
5. 整合測試：`evaluate(orthogonalize=base_df)` vs 不傳，斷言 IC 有位移、
   `r2_mean` 有值——**0.3d**
6. README Level 2 更新：把 `orthogonalize_factor` 從「自己呼叫」升級為
   「透過 config 傳入」；保留手動 helper 用於 one-off 分析——**0.2d**

## 6. 風險

- **Surface regression**：在同一個 release window 內對同一個 API 做**第二次**
  breaking change。前 Phase-1 寫 `orthogonalize=True` 的使用者剛剛改成不傳，
  現在又要改成傳 DataFrame。**保留名字不變（`orthogonalize`）但改型別**
  （`pl.DataFrame | None` 取代 `bool`），讓 migration 可以 grep 得到。
- **Coverage gate 嚴格度**：0.95 預設對新興市場可能太嚴（產業分類會搬家）。
  `orthogonalize_min_coverage` 要留成可調；上線後觀察真實使用者平均落在哪裡、
  是否需要降預設值。
- **與 `keep_artifacts=True, compact=True` 的互動**：正交化後的 `factor`
  住在 `prepared`，compact 會把 prepared 丟掉。Phase 3 benchmark 確認 compact
  在大批次是必需的。若下游分析要同時看「原始 vs 正交化」，得關掉 compact
  並用 `factor_pre_ortho`（`orthogonalize_factor` 已保留）。這個組合要明確
  文件化。

---

## 7. Sign-off checklist（實作前必須勾）

- [ ] Owner 選 §3.1 (a)／(b)／(c)
- [ ] Owner 確認 §3.2 config 欄位形狀
- [ ] Owner 確認 §3.3 Profile 欄位（scalar R² 還是 per-base β）
- [ ] Owner 選 §3.4 A 還是 B
- [ ] Owner 確認 §3.5 re-z-score 政策
- [ ] Owner 確認 §3.6 diagnose rule 集合
