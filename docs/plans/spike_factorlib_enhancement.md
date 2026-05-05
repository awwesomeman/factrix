# Spike — factrix 改善計畫

> **狀態**：P0 + P1 + P2 + P3 + 所有 COV 項目全部實作完畢（2026-04-20）。
> **來源**：資深 Quant 審查（第一輪 2026-04-20）+ 第二輪深度審查（2026-04-20）+ 第三輪實地驗證（2026-04-20，user 角度）；已修 `8f15db8` / `85b81e3` / `26762b7` / `dcbe346` 後的剩餘項目
> **Owner**：jason pan
> **日期**：2026-04-20
>
> **2026-04-20 實作 commit 對照**：
>   - P0-A/B demo fix：`ef619ef`
>   - P0-C forward_periods gate：`de43bf3`
>   - P0-D 移除 fp= override（5 methods）：`ddc7328`
>   - P1-B `quantile_spread` 統一：`24d85eb`
>   - P1-A `caar_trend` / `beta_trend` 統一：`bd7ae12` + `838dfcc`
>   - P1-C/D `mfe_mae_summary` / `bmp_test` 統一：`01d39f4`
>   - P2-B + COV-H demo full rerun：`fb8eae0`（順修 validate_factor_data / register_rule API drift）
>   - P2-I describe_profile_values 真解：`66bd6db`（移除 artifacts 參數）
>   - Phase 3c P2 + COV demo UX 補完：見下一個 commit（含 P2-A/C/E/F/H/J + COV-A/B/E/F/G/J）
>   - Phase 3d P3-A 8 個 doc SUPERSEDED headers + P3-B/COV-C README：見下一個 commit
> **關聯審查報告**：
>   - 第一輪：`brain/c8b31922.../factrix_review.md`
>   - 第二輪：`brain/1f62e351.../factrix_review_2.md`（含 NEW-A~F / COV-G~J）
>
> **2026-04-20 第三輪驗證修正**（對照 source + demo.ipynb）：
>   - **刪除 P2-D**：`fl.factor()` docstring 實際上只示範 `factor_type=` 單參數（`_api.py:459-463`），查無「同時傳兩者」的錯誤
>   - **刪除 COV-I**：`fl.adapt()` 在 demo cell #4 已執行（`raw_demo = fl.adapt(raw, date='date', asset_id='ticker', price='close_adj')`），查無「完全未示範」
>   - **COV-H 併入 P2-B**：成功輸出的 cell #32 已存在但未執行，本質就是 P2-B 的 16 cells 未執行子集
>   - **P1-A/C/D 改框法**：原 doc 建議「改 cache key 貼齊方法名」會撞破 `factor.py:151-155` 的 invariant `assert result.name == name`（cache key 必須 == primitive MetricOutput.name）。正確修法是統一命名，不是破 invariant
>   - **P1-B 改方向**：user 決策——統一用 `quantile_spread`（field/cache key/primitive 改回對齊方法名）
>   - **移除治標 fallback**：P0-C 方案 B / P0-D warning-only / P1-A/B/D 的 alias 路徑全刪，committed to 正解

---

## 背景

完成 `fl.factor()` session API 和 profile 命名重構後，針對 demo.ipynb 以及
整體 API usability 進行了一輪資深 Quant 審查。本文件記錄審查後仍待處理的改善
項目，依優先級分層，並給出明確的實作指引。

**已修閉項目（本計畫不含）**：
- `q1_q5_spread` → `long_short_spread` / `q1_concentration` → `top_concentration` 命名重構（`8f15db8`）
- `breakeven_cost` / `net_spread` per-call override（`8f15db8`）
- `Factor` base class 從 `fl.__all__` 移除（`85b81e3`）
- Doc prose top/bottom 統一（`26762b7`）
- `_pick_event_return_col` / `_run_profile_and_attach` helper 提取（`dcbe346`）

---

## 優先級 P0：正確性問題（教壞使用者）

### P0-A：Demo regime label 的循環偏誤 ✅ IMPLEMENTED `ef619ef`

**問題**：`demo.ipynb §2.4.2` 用評估期內的收盤報酬來定義 bull/bear regime，
再對**同一段**資料的因子 IC 做 regime 分組——regime 本身就是 forward return
的一部分，屬於 look-ahead 循環偏誤。新用戶最容易複製這個 pattern。

**影響**：Regime IC 結論偏樂觀；若此 pattern 被帶進實際 alpha research，
結論無效。

**修改位置**：`experiments/demo.ipynb §2.4.2`（regime label 建構段）

**建議修法**：
```python
# 改成 look-ahead free 的市場狀態指標
# 做法 A：用前一期市場報酬滾動平均
regime_series = (
    raw_demo
    .filter(pl.col("asset_id") == universe_proxy)   # e.g. ETF 代理市場
    .sort("date")
    .with_columns(
        pl.col("price").pct_change().alias("mkt_ret")
    )
    .with_columns(
        pl.col("mkt_ret").shift(1).alias("lagged_mkt_ret")  # ← 關鍵：shift
    )
    .select(["date", "lagged_mkt_ret"])
)
# 做法 B（更簡潔）：預先算好的 VIX / 景氣指標 join 進來
```

加上 comment 說明為何要 `shift(1)` / look-ahead free 的重要性。

---

### P0-B：Demo orthogonalization 的 size proxy 錯誤 ✅ IMPLEMENTED `ef619ef`

**問題**：`demo.ipynb §2.4.1` 的 `basis_df` 用 `price * volume`（名目成交額）
當作 size factor 做 orthogonalization，但 `raw_demo` 已有 `market_cap` 欄位。
`price × volume` 是流動性指標，不是公司規模代理；這會讓 orthogonalization R²
失準，且帶壞使用者的因子建構認知。

**修改位置**：`experiments/demo.ipynb §2.4.1`

**建議修法**：
```python
# 修改前（錯誤）
basis_df = raw_demo.with_columns(
    (pl.col("price") * pl.col("volume")).log().alias("size")
)

# 修改後（正確）
basis_df = raw_demo.with_columns(
    pl.col("market_cap").log().alias("size")   # raw_demo 已有 market_cap
)
```

---

### P0-C：Strict Gate 未擋住 `forward_periods` 不匹配 ✅ IMPLEMENTED `de43bf3`

**問題**：`evaluate_batch` 的 strict gate 只檢查 `forward_return` 欄位是否
存在，**不檢查 `forward_periods` 是否與傳入的 config 一致**。若使用者把
`preprocess(df, config_A)` 的結果錯傳進 `evaluate_batch(..., config=config_B)`，
而兩者 `forward_periods` 不同，不會報錯，只會靜默用錯誤的展望期算出結果。

**影響**：當使用者在 notebook 混用多個 forward_periods（如同時研究 5 日 / 10
日）版本的 DataFrame 時，此盲點最容易觸發。Demo `§2.3` 展示了兩種不同 config
但沒有明確警示這個風險。

**修改位置**：`factrix/_api.py`（strict gate 邏輯）+ `demo.ipynb §2.3`（加警示 comment）

**修法**：在 preprocessed DataFrame 嵌入 `forward_periods` metadata，strict gate 讀取並比對。

```python
# fl.preprocess 時嵌入（沿用既有 _PREPROCESSED_MARKER 模式）
df = df.with_columns(pl.lit(config.forward_periods).alias("_fl_forward_periods"))

# _api.py strict gate 同 batch 加檢查
embedded_fp = int(df["_fl_forward_periods"][0])
if embedded_fp != config.forward_periods:
    raise ValueError(
        f"forward_periods mismatch: df was preprocessed with {embedded_fp} "
        f"but config has {config.forward_periods}. Re-run fl.preprocess() "
        f"with the correct config."
    )
```

治標方案（只加 demo warning cell）已刪除——gate 在 library 側才符合北極星
「統計正確 U4：end user 走對的路 > 走快的路」。測試 fixtures 同步更新屬必要成本。

---

### P0-D：`f.ic(forward_periods=override)` 只影響 t-test 自由度，不重算 IC series ✅ IMPLEMENTED `ddc7328`

**實作筆記**：scope 擴大到 5 methods — `ic / hit_rate / monotonicity / caar / bmp_test` 的 `forward_periods=` 參數全移除（同一 bug pattern）。repo 內零 call-site，確認為死 API。

**問題**：`CrossSectionalFactor.ic()` 的 `forward_periods` override 把 `fp` 傳入 `ic_fn`，但
底層 `ic_series`（`artifacts.get("ic_series")`）是 `build_artifacts` 時以 `config.forward_periods`
算定的。Override 僅影響非重疊 t-test 的 `n_obs = n_total // fp`，不影響 IC 本身。

**後果**：若 config `forward_periods=5` 但呼叫 `f.ic(forward_periods=10)`，t-stat 以
10 日展望期估計自由度，但 IC series 是 5 日展望期計算的——統計量被系統性低估。
同一 Factor session 裡 `f.ic(forward_periods=10)` 和 `f.ic_ir()`（無 override）共用同
一 IC series，但給出的 p-value 語意不同，邏輯矛盾且無任何警示。

**影響範圍**：`factrix/factor.py::CrossSectionalFactor.ic()` 的 override 路徑；
`hit_rate()` 有相同設計，同樣受影響。

**修改位置**：`factrix/factor.py`（`ic()` / `hit_rate()`）

**修法：移除 `forward_periods=` 參數**——此 override 本質是 API 誤導，假裝能切換
展望期但實際只改 dof。北極星 U4（統計正確）+ U2（易用）都指向「不該讓使用者
用的路，不該開給他」。

```python
# 修改前
def ic(self, forward_periods: int | None = None) -> MetricOutput:
    fp = forward_periods if forward_periods is not None else self.config.forward_periods
    ...

# 修改後
def ic(self) -> MetricOutput:
    # forward_periods 由 config 決定，若要不同展望期請重 preprocess + rebuild session
    return self._cached_or_compute("ic_mean", ic_fn, self.artifacts, ...)
```

`hit_rate()` 同樣處理。這是 API breaking change，但爆炸半徑小（override 只在
Factor session 上存在，Profile / evaluate_batch 路徑不受影響）；docstring 補一行
說明想要不同 fp 的正解：

```
若需不同 forward_periods，請重建 config + 重跑 fl.preprocess + fl.factor()。
同一 session 內切換 fp 會造成 IC series / dof 不一致，已禁用。
```

治標路徑（只加 `warnings.warn`）已刪除——override 存在本身就是陷阱，警告只是
把責任丟給讀說明書的使用者，不符北極星優先序。

---

## 優先級 P1：API 不一致（debug 痛點）

**背景 invariant（`factor.py:151-155`）**：`_cached_or_compute` 內有
`assert result.name == name`，強制 **cache key == primitive `MetricOutput.name`**。
這是刻意的開發正確性契約（memory: project_profile_refactor_status.md L78）。
因此 P1 所有子項不走「改 cache key 貼齊方法名」（會撞破 assert），而是
**三點統一**：`method name == primitive MetricOutput.name == Profile field name`。

### P1-A：`caar_trend` / `beta_trend` 共用 `ic_trend` primitive 導致三地不一致 ✅ IMPLEMENTED `bd7ae12`

**實作筆記**：`ic_trend` primitive 加 `*, name: str = "ic_trend"` 參數；EventFactor/MacroPanel/MacroCommonFactor 各自傳入 `name="caar_trend"` / `"beta_trend"`。附帶修整：`_memoized` / `_cached_or_compute` 的 `name` 參數改名為 `key`（避免與 primitive 的 `name=` kwarg 衝突）。

**問題**：使用者呼叫 `f.caar_trend()`，但 `artifacts.metric_outputs` 的 key
是 `"ic_trend"`（因為 primitive 的 `MetricOutput.name == "ic_trend"`，複用）。
`MacroPanelFactor.beta_trend` 同樣問題（key 也是 `"ic_trend"`，且與 CS 的
`ic_trend` 共 key——跨 factor_type 碰撞風險）。

**影響範圍**：
- `factrix/factor.py::EventFactor.caar_trend`（`:540-544`）
- `factrix/factor.py::MacroPanelFactor.beta_trend`（`:609-616`）
- `factrix/metrics/trend.py::ic_trend`（primitive）

**修法**：讓 primitive 接 `name=` 參數，方法各自傳入自己的名字：

```python
# metrics/trend.py
def ic_trend(series, *, name: str = "ic_trend") -> MetricOutput:
    ...
    return MetricOutput(name=name, value=..., ...)

# factor.py EventFactor
def caar_trend(self) -> MetricOutput:
    return self._cached_or_compute(
        "caar_trend",
        lambda s: ic_trend(s, name="caar_trend"),
        self.artifacts.get("caar_values"),
    )

# factor.py MacroPanelFactor
def beta_trend(self) -> MetricOutput:
    return self._cached_or_compute(
        "beta_trend",
        lambda s: ic_trend(s, name="beta_trend"),
        self.artifacts.get("beta_values"),
    )
```

同步更新 `from_artifacts` 讀 key 的位置（原讀 `"ic_trend"` 改讀各自 key）。
assert `result.name == name` 自動滿足。

---

### P1-B：`quantile_spread` / `long_short_spread` 三地不一致 — **統一用 `quantile_spread`** ✅ IMPLEMENTED `24d85eb` + polish `838dfcc`

**實作筆記**：15 files 改動（CS + MP Profile field / primitive / cache key / describe_profile_values header / 中間 DataFrame / tests / demo prose）。polish commit 修整 `CrossSectionalFactor.quantile_spread` docstring 殘留 "long-short spread" prose。

**問題**：同一 metric 三個地方用兩個名字：
- 方法名：`f.quantile_spread()`
- cache key / primitive `MetricOutput.name` / Profile 欄位：`long_short_spread`
- `describe_profile_values` 印出：`long_short_spread`

使用者試圖呼叫 `f.long_short_spread()` 得 `AttributeError`，或查
`metric_outputs["quantile_spread"]` 得 `KeyError`，兩邊都會撞牆。

**User 決策（2026-04-20）**：**統一用 `quantile_spread`**（不保留
`long_short_spread` alias，不加 method 別名——走三點命名統一的單一正解）。

**修法**（breaking change，需同步所有 call-site）：

| 位置 | 修改前 | 修改後 |
|---|---|---|
| `metrics/quantile.py::quantile_spread` primitive `MetricOutput.name` | `"long_short_spread"` | `"quantile_spread"` |
| CS Profile 欄位 | `long_short_spread` | `quantile_spread` |
| MP Profile 欄位 | `long_short_spread` | `quantile_spread` |
| Factor cache key | `"long_short_spread"` | `"quantile_spread"` |
| `describe_profile_values` header | `long_short_spread` | `quantile_spread` |
| 中間 DataFrame column（若有） | `long_short_spread` | `quantile_spread` |

同步更新：tests / demo.ipynb prose 與 cell / 歷史 doc 的 SUPERSEDED 標頭
（本條新增 `long_short_spread → quantile_spread` 改名事件記錄）。

**注意**：這是 2026-04-20 `8f15db8` 的 `q1_q5_spread → long_short_spread`
改名的部分反轉——user 重新評估後決定 `quantile_spread` 更貼齊方法
語意（「分組計算 spread」）且不需再為 `long_short` 的方向性做額外解釋
（spread 本身已帶方向）。

---

### P1-C：`mfe_mae_summary()` 的 primitive 名與方法名不一致 ✅ IMPLEMENTED `01d39f4`

**問題**：`EventFactor.mfe_mae_summary()` 呼叫 `_cached_or_compute("mfe_mae", ...)`，
cache key 是 `"mfe_mae"`（由 primitive `MetricOutput.name` 決定）而非方法名
`"mfe_mae_summary"`。使用者查 `f.artifacts.metric_outputs["mfe_mae_summary"]` 得
`KeyError`。

**影響範圍**：
- `factrix/factor.py::EventFactor.mfe_mae_summary`（`:507`）
- `factrix/metrics/mfe_mae.py::mfe_mae_summary` primitive（`:160` `name="mfe_mae"`）

**修法**：改 primitive 的 `MetricOutput.name` 為 `"mfe_mae_summary"`，
cache key 同步更新，Profile / demo prose 同步對齊三點統一。

```python
# metrics/mfe_mae.py
def mfe_mae_summary(df) -> MetricOutput:
    ...
    return MetricOutput(name="mfe_mae_summary", ...)   # 原 "mfe_mae"

# factor.py
return self._cached_or_compute("mfe_mae_summary", ...)  # 原 "mfe_mae"
```

---

### P1-D：`bmp_test()` 的 primitive 名與方法名不一致 — **統一用 `bmp_test`** ✅ IMPLEMENTED `01d39f4`

**實作筆記**：EventProfile 欄位 `bmp_sar_mean` → `bmp_test_mean`（`bmp_p` / `bmp_zstat` 保留現有 shorthand 慣例，與 `spread_p` / `spread_tstat` 一致）；diagnostics rule + README + tests 同步。

**問題**：`EventFactor.bmp_test()` 呼叫 `_cached_or_compute("bmp_sar", ...)`，
cache key 是 `"bmp_sar"`，Profile 字段為 `bmp_sar_p`，但方法名是 `bmp_test()`。

**影響範圍**：
- `factrix/factor.py::EventFactor.bmp_test`（`:412`）
- `factrix/metrics/caar.py::bmp_test` primitive（`:227` `name="bmp_sar"`）
- EventProfile 欄位 `bmp_sar_p`

**修法（統一用 `bmp_test`）**：

| 位置 | 修改前 | 修改後 |
|---|---|---|
| `metrics/caar.py::bmp_test` primitive `MetricOutput.name` | `"bmp_sar"` | `"bmp_test"` |
| Factor cache key | `"bmp_sar"` | `"bmp_test"` |
| EventProfile 欄位 | `bmp_sar_p` | `bmp_test_p` |

`bmp_test` 是方法名，也是學術文獻通用名（Boehmer, Musumeci, Poulsen 1991），
比縮寫 `bmp_sar` 更清楚。治標路徑（alias / docstring-only）已刪除。

---

## 優先級 P2：Demo 可用性改善

### P2-A：Charts section 繞過正規 API ✅ IMPLEMENTED

*Done:* Phase 3c

**問題**：`demo.ipynb §2.7` 從內部 import `build_artifacts`，還要手動設
`artifacts.factor_name`，前後 demo 風格不一致。

```python
# 現況（內部路徑）
from factrix.evaluation.pipeline import build_artifacts
artifacts = build_artifacts(prepared, cfg_charts)
artifacts.factor_name = 'Mom_20D'   # 容易忘記

# 應改成正規路徑
_, artifacts = fl.evaluate(mom20_raw, 'Mom_20D', config=cfg_charts,
                            return_artifacts=True)
# 或更直接（用 fl.factor session）
f = fl.factor(mom20_raw, 'Mom_20D', config=cfg_charts)
artifacts = f.artifacts
```

**修改位置**：`experiments/demo.ipynb §2.7`

---

### P2-B：共 16 個 code cell 未執行（遠超 §7，含核心 session demo） ✅ IMPLEMENTED `fb8eae0`

**實作筆記**：`jupyter nbconvert --execute --inplace` 跑完整 kernel，56/56 code cell 全執行、0 錯誤。順修兩個 API drift 問題（非 Phase 2 造成）：
  - Cell 89 `validate_factor_data(raw_demo, ...)` 改成用 `mom20`（preprocessed），因為 CS schema 要求 `factor_raw` 欄位，`raw_demo` 沒有
  - Cell 93 `Rule(fn=...)` → `Rule(predicate=..., message=..., code=..., severity=...)` 對齊現行 Rule API；`register_rule(CrossSectionalProfile, ...)` 改為 `register_rule('cross_sectional', ...)`

COV-H（L2 opt-in 成功輸出）自動隨 rerun 現身於 cell 32 的輸出中。



**問題**：實際統計 `execution_count: null` 的 code cell 共 **16 個**，分佈如下：

| 區段 | 未執行 cells | 影響 |
|------|-------------|------|
| §2.4 Factor session 內部 | cells 17, 19, 21, 23 | 14-metric展示 / cache demo / L2 short-circuit，是核心 showcase |
| §2.4 low-level primitive | cell 25 | − |
| §3.4 EventFactor session | cell 64 | EventFactor 完整 session 從未被驗證 |
| §4.3 MacroPanel session | cell 71 | − |
| §5.3 MacroCommon session | cell 78 | − |
| §7.1–7.8 Library helpers | cells 87~101（8個）| − |

計畫原本描述「§7 多數 cell 未跑過」嚴重性被低估——**§2.4 和 §3.4 的核心 session demo**
也都是 `exec=None`，表示從未驗證能正確執行。

**修改位置**：`experiments/demo.ipynb`（跑完整 kernel，全部 cell 必須有 execution_count）

---

### P2-C：Event Factor session 覆蓋度不足 ✅ IMPLEMENTED

*Done:* Phase 3c (cell 64 already covers all opt-in + corrado in §3.4 list)

**問題**：`demo.ipynb §3` 的 EventFactor session 只呼叫了核心 metric，
price-based metric（`mfe_mae_summary` / `event_around_return` /
`multi_horizon_hit_rate`）及 `corrado_rank_test` 完全未展示，也沒說明
short-circuit 的 opt-in 語意。

**建議**：在 §3 末尾加一個「price-data opt-in」展示 cell：

```python
# Event Factor session — 需要 OHLC 的 opt-in metrics
f_ev = fl.factor(event_df_with_ohlc, 'GoldenCross', config=ev_cfg)

# short-circuit：若 event_df 沒有 high/low/open → 回 MetricOutput(value=0, reason="no_price_data")
mfe = f_ev.mfe_mae_summary()
print(f'mfe_mae_summary: {mfe.metadata.get("reason", "computed")}')

# 有 OHLC 時才有完整數值
har = f_ev.event_around_return()
mhhr = f_ev.multi_horizon_hit_rate()
```

---

### ~~P2-D：`fl.factor()` docstring 範例同時傳 `factor_type` 和 `config`~~ — **已刪除**

2026-04-20 第三輪驗證查實：`_api.py:459-463` 的 docstring 範例只示範
`factor_type="cross_sectional"` 單參數用法，未出現「兩者都傳」的寫法；
docstring L449-451 已明載 `factor_type` 在 `config` 存在時會被忽略。
原項目屬誤報，刪除。

---

### P2-E：`with_extra_columns` 的 positional 對齊未警示排序依賴 ✅ IMPLEMENTED

*Done:* Phase 3c

**問題**：`demo.ipynb §2.5.1` 展示 `with_extra_columns` 時，用
`to_polars()['factor_name'].to_list()` 作為對齊 key，但沒有說明
「在 filter / rank_by 之後做這個對齊就會出錯」的限制。使用者複製後
在有前置 filter 的場景下會靜默地拿到錯誤數值。

**修改位置**：`demo.ipynb §2.5.1`

**建議**：在該 cell 加顯著 comment，說明對齊的使用前提：
```python
# ⚠️ 注意：external_scores 必須與當前 ProfileSet 順序對齊
# 若事先有 filter / rank_by / sort，務必重新以 to_polars()['factor_name']
# 的當下順序來建構 aligned list，否則欄位錯位難以察覺
aligned = [external_scores[name] for name in ps.to_polars()['factor_name'].to_list()]
```

---

### P2-F：`return_artifacts=True` 要重跑 `fl.evaluate` 的 pattern 不佳 ✅ IMPLEMENTED

*Done:* Phase 3c (simplified after P2-I removed artifacts requirement)

**問題**：`demo.ipynb §2.4.3` 先示範了只回 Profile 的 `fl.evaluate(...)`，
之後要取 artifacts 又重執行一次（不同變數名），讓讀者誤以為「要 artifacts
必須重跑是正常的」。

**修改位置**：`demo.ipynb §2.4.3`

**建議**：將對應 cell 改為第一次就帶 `return_artifacts=True`，或直接示範
`fl.factor()` session 的 `f.artifacts` 取法，並加 comment 說明：
```python
# 若需要後續深入分析，從一開始就帶 return_artifacts=True
# 不要重跑 evaluate——用 fl.factor() session 更合適
f = fl.factor(mom20_l2, 'Mom_20D', config=cfg_l2)
profile = f.evaluate()          # 不重算，走 shared cache
arts    = f.artifacts           # 直接取
```

---

### P2-G：Demo section 編號不一致，閱讀導航困難 ✅ IMPLEMENTED

*Done:* already satisfied (§3/§4/§5 have subsections 3.1/3.2/3.3/3.4 etc.)

**問題**：目前編號混用兩種風格：
- §2 下用區各 subsection（`§2.4.1` / `§2.4.2` / `§2.4.3` 等）
- §7 下有 `7.1`~`7.8` 編號
- **§3 / §4 / §5 內部完全沒有 subsection 編號**，讀者在長 notebook 裡導航困難

**修改位置**：`experiments/demo.ipynb`（§3 / §4 / §5 的 markdown cell）

**建議**：將 §3 / §4 / §5 補上 subsection，統一為 `§x.1 問題陳述` /
`§x.2 Factor session` / `§x.3 diagnose` 三層，與 §2 的層次對齊。

---

### P2-H：附錄 B 的變數依賴未說明 ✅ IMPLEMENTED

*Done:* Phase 3c

**問題**：`demo.ipynb 附錄 B`「`describe_profile_values` × 4 factor types」
需要 `event_df`、`industry_prep`、`common_prep` 等變數，它們定義在
很前面的 cell（§3/4/5），如果直接只跑附錄 B 會得到 `NameError`。
無前置條件說明會讓使用者誤以為附錄 B 是獨立的。

**修改位置**：`demo.ipynb 附錄 B` 的 markdown cell

**建議**：在附錄 B 的說明文字頂部加一行身份注明：
> **前置步驟**：本附錄需要先執行 §3（EventFactor）、§4（MacroPanel）、§5（MacroCommon）的全部 cell，
> 切勿單獨執行此段。

---

### P2-I：`describe_profile_values` 的 `artifacts` 在默認路徑下取不到 ✅ IMPLEMENTED `66bd6db`

**實作筆記**：採取更徹底的方案——**移除 `artifacts` 參數**（不保留 optional fallback）。`describe_profile_values(profile)` 直接從 Profile dataclass 渲染 scalar。per-regime / per-horizon / spanning-beta detail 從此函式移除（它們是 `MetricOutput.metadata` 的 dict，不是 Profile scalar），power user 改直接讀 `arts.metric_outputs[key].metadata` drill-down。

理由：原 spike 方案「`artifacts=None` 時走 fallback 路徑」本質就是條件分支 fallback，違反北極星原則。單一參數、單一路徑、無條件分支是真正的「解掉問題」。



**問題**：`fl.describe_profile_values(profile, artifacts)` 需要 `Artifacts` 作為第二參數，
但使用者最自然的路徑是 `profile = fl.evaluate(df, name, config=cfg)`（預設不回 artifacts）。
初次使用者幾乎必然碰壁——第二個參數從何而來？

**修改位置**：`factrix/reporting.py::describe_profile_values`

**實作修法**（2026-04-20，`66bd6db`）：**移除 `artifacts` 參數**，`describe_profile_values(profile)` 單參數從 Profile dataclass 直接渲染。detail sections 從函式移除，成為 power-user 直接讀 `arts.metric_outputs[key].metadata` 的路徑。

```python
# 實際 API
def describe_profile_values(profile: FactorProfile) -> None:
    _print_header(profile)
    _print_value_table(profile)   # iterate dataclass fields; skip None
```

原 optional-artifacts 方案已撤回——`if artifacts is None: ...` 本質仍是 fallback 分支；單一路徑才是真解。

---

### P2-J：`OrthoConfig.min_coverage=0.95` 的嚴格閾值在 demo 無說明 ✅ IMPLEMENTED

*Done:* Phase 3c

**問題**：`CrossSectionalConfig(ortho=basis_df)` 的語法糖預設 `min_coverage=0.95`。
若 `basis_df` 和 factor 資料的 asset/date 覆蓋率不足 95%，pipeline 會 raise 而不是
降級運作，但 demo §2.4.1 沒有任何說明。新使用者首次用 orthogonalization 可能因
資料覆蓋問題得到難以理解的錯誤。

**修改位置**：`demo.ipynb §2.4.1`

**建議**：在 §2.4.1 的 `basis_df` 建構 cell 加一段 comment：

```python
# ⚠️ OrthoConfig(min_coverage=0.95)：若 basis_df 與 factor 資料的交集覆蓋率
# 低於 95%，pipeline 會 raise 而非靜默降級。可放寬：
#   CrossSectionalConfig(ortho=OrthoConfig(base_factors=basis_df, min_coverage=0.8))
# 覆蓋率不足通常代表日期/asset schema 對齊問題，建議先確認再降閾值。
```

---

## 優先級 P3：文件清理

### P3-A：Doc 中的舊命名殘留 ✅ IMPLEMENTED

*Done:* Phase 3d (8 docs stamped with SUPERSEDED banner pointing at both rename passes)

多個歷史設計文件仍有 `q1_q5_spread` / `q1_concentration` 的舊名，
已有 `naming_convention.md` 的 SUPERSEDED 標頭作為全域警示，但若使用者
直接開這些文件可能不知道已過期。

**建議**：在下列文件頭部加 `> SUPERSEDED 2026-04-20: q1_q5_spread → long_short_spread` 標頭。

| 文件 | 殘留名 |
|------|--------|
| `refactor_factrix_routing.md` | `q1_q5_spread` / `q1_concentration` |
| `plan_gate_redesign.md` | `q1_q5_spread` / `q1_concentration` |
| `spike_orthogonalize.md` | `q1_q5_spread` |
| `spike_fast_track_batch.md` | `q1_concentration` |
| `refactor_metric_output.md` | `q1_q5_spread` |
| `plan_event_signal.md` | `q1_concentration` |
| `plan_macro_panel.md` | `q1_concentration` |
| `spike_metric_outputs_retention.md` | `q1_q5_spread` / `q1_concentration` |

### P3-B：`preprocess_cs_factor` vs `preprocess` 差異未說明 ✅ IMPLEMENTED

*Done:* Phase 3d (README Level 0 section)

Demo 只用 `fl.preprocess`，但 `fl.preprocess_cs_factor` 也在 `__init__` re-export。
使用者不知道兩者的差異和適用場景。

**建議**：在 `factrix/README.md` 的 API 一覽表加一行說明，或在
`demo.ipynb §0`（setup cell）加一行 comment 解釋。

### P3-C：`greedy_forward_selection` 未示範 ✅ IMPLEMENTED

*Done:* Phase 3c (demo §2.6 new cell; same as COV-G)

在 `__all__` 暴露但 demo 完全未提。作為「從 N 個 base factor 中 greedy 選
最能 span 的子集」的重要工具，應在 `§2.6` redundancy_matrix 之後加示範。

> ⬆️ **升級建議**：P3-C 應升為 P2 等級。`greedy_forward_selection` 是 screening
> pipeline 最終選因子子集的重要工具，只靠 README 說明不足。

---

## 功能覆蓋缺口（Part 1 Review）

以下項目來自 demo.ipynb 功能覆蓋稽核，屬於「存在於 `__all__` 但 demo 未觸及」
的 API。短期可先補文件 / README 說明，中長期在 demo 加示範 cell。

### COV-A：13 個 factor generator 只展示 2 個 ✅ IMPLEMENTED

*Done:* Phase 3c (cell 10 now lists all generate_* via dir)

`factrix.factors` 有 13 個 generator，demo 只用了 `generate_momentum` /
`generate_volatility`。使用者以為只有這兩個選項，完全不知道有：
`generate_amihud`、`generate_rsi`、`generate_market_beta`、
`generate_overnight_return`、`generate_52w_high_ratio` 等。

**建議**：在 `demo.ipynb §0` 或獨立的 `demo_factor_generators.ipynb`
加一個 cell，列出所有 generator 並用最小範例示範每一族（momentum / vol /
technical / liquidity / event）。

### COV-B：`quantile_spread_vw`（value-weighted spread）未示範 ✅ IMPLEMENTED

*Done:* Phase 3c

`factrix.metrics.quantile_spread_vw` 在 `__all__` 暴露，但 demo 從未呼叫。
使用者不知道有 equal-weight 以外的選項，在有流動性差異的市場（如 TW）
VW spread 更貼近實際可執行的 PnL。

**建議**：在 `demo.ipynb §2.4`（Factor session）加一行：
```python
f.quantile_spread_vw(weight_col='market_cap')   # value-weighted spread
```

### COV-C：`compute_group_returns` 未示範 ✅ IMPLEMENTED

*Done:* Phase 3d (README Level 2 section)

Power user 需要自訂分組統計的入口，但 demo 完全不提，讓使用者以為
只能走 Profile 的黑箱路徑。

**建議**：在 `factrix/README.md` 的 "low-level primitives" 段補一行說明。

### COV-D：`EventFactor.corrado_rank_test` 未示範 ✅ IMPLEMENTED

*Done:* already present in cells 62 and 64

Corrado (1989) 的非參數檢定是 event study 的重要工具，特別是在報酬非常態
的市場。EventProfile 已有欄位但 demo §3 未展示。

**建議**：在 `demo.ipynb §3` 的 Factor session cell 加一行 `f_ev.corrado_rank_test()`。

### COV-E：`ProfileSet.filter(Callable)` 未示範 ✅ IMPLEMENTED

*Done:* Phase 3c (cell 40 now shows both pl.Expr and lambda)

`demo.ipynb §2.5` 只展示了 `filter(pl.Expr)` 版本，Python callable 版本
（`filter(lambda p: p.ic_ir > 0.3)`）完全未示範。兩種介面適合不同場景。

**建議**：在 §2.5 filter 段加一個 callable 版本範例。

### COV-F：MacroPanel / MacroCommon Factor session 覆蓋度不完整 ✅ IMPLEMENTED

*Done:* Phase 3c (cell 71 now covers breakeven_cost/net_spread; §5.3 cell 78 covers full MC method set)

§4 / §5 的 Factor session 方法列表未完整示範：
- `MacroPanelFactor.quantile_spread` 的多組比較（N-aware n_groups）
- `MacroCommonFactor.ts_beta_sign_consistency`

**建議**：在 §4 / §5 各補一個 method 呼叫的 cell。

---

## 功能覆蓋缺口（Part 2 Review）

以下為第二輪深度審查新增的覆蓋缺口。

### COV-G：`greedy_forward_selection` 應升為 P2 示範（見 P3-C） ✅ IMPLEMENTED

*Done:* Phase 3c (§2.6 new cell after redundancy_matrix)

已在 P3-C 標記，此處不重複。

### ~~COV-H：L2 opt-in 的「成功案例」輸出從未展示~~ — **併入 P2-B**

2026-04-20 驗證：demo cell #32 已建構有效的 `regime_df` 並以
`config.regime_labels=regime_df` 跑 `f.regime_ic()` 產出成功輸出——但該 cell
與 cell #23（short-circuit 示範）同屬 P2-B 的 16 個 `execution_count: null`
未執行 cell 子集。P2-B 重跑 notebook 後成功案例自動展現，不需獨立處理。
原 COV-H 併入 P2-B 驗收。

### ~~COV-I：`fl.adapt()` 完全未示範~~ — **已刪除**

2026-04-20 第三輪驗證查實：demo cell #4（已執行）已呼叫
`raw_demo = fl.adapt(raw, date='date', asset_id='ticker', price='close_adj')`
並附註解說明用途。原項目屬誤報，刪除。

### COV-J：`split_by_group` 只展示了結構，沒有銜接 `evaluate_batch` 的 full workflow ✅ IMPLEMENTED

*Done:* Phase 3c (cell 87 extended with per-slice evaluate loop)

**問題**：`demo.ipynb §7.1` 呼叫 `split_by_group` 並 print 了回傳 dict，但沒有
接著展示「如何把切分結果送進 `evaluate_batch` 或 `evaluate`」的完整 workflow。
使用者看到一個字典，不知道下一步怎麼用。

**建議**：在 §7.1 原 cell 後補一個接力 cell：

```python
# split 之後接 evaluate_batch（各 slice 用各自的 N-aware config）
slice_results = {}
for grp_name, (sub_df, grp_cfg) in slices.items():
    p = fl.evaluate(sub_df, f'Mom_20D_{grp_name}', config=grp_cfg)
    slice_results[grp_name] = p
    print(f'[{grp_name}] verdict={p.verdict()}, ic_p={p.ic_p:.4f}')
```

---

## 實作順序建議

```
# ── 一刀修的 demo 錯誤（高 ROI）────────────────────────────────────────────
P0-A (demo regime 偏誤)            ← shift(1)，一行
P0-B (demo size proxy)             ← market_cap，一行

# ── Library 側正確性 gate（統計正確 U4）──────────────────────────────────────
P0-C (forward_periods gate)        ← preprocess embed metadata + gate 檢查，約 0.4d
P0-D (ic()/hit_rate() fp override) ← 移除 forward_periods= 參數，約 0.2d

# ── API 三點命名統一（方法名 == primitive name == Profile field）──────────────
P1-A (caar_trend/beta_trend)       ← primitive 接 name= 參數，約 0.3d
P1-B (quantile_spread)             ← field/key/primitive 改回 quantile_spread，約 0.4d
P1-C (mfe_mae_summary)             ← primitive 改名 mfe_mae_summary，< 0.2d
P1-D (bmp_test)                    ← primitive/field 統一 bmp_test，約 0.2d

# ── Demo UX 改善（demo cell / docstring 修改）──────────────────────────────
P2-A (charts API 路徑)             ← 兩行替換
P2-B (16 cells 未執行 + COV-H)     ← 重跑完整 kernel，含 §2.4/§3.4/§4.3/§5.3/§7
P2-C (Event session opt-in 展示)   ← 加一個 cell
P2-E (with_extra_columns 警示)     ← 加 comment
P2-F (return_artifacts pattern)    ← 加 comment + 改 cell
P2-G (section 編號不一致)          ← §3/4/5 補編號
P2-H (附錄 B 前置資訊)             ← 加身份注明
P2-I (describe_profile_values)     ← artifacts 參數 optional + evaluate() 加 Note
P2-J (ortho min_coverage 說明)     ← demo §2.4.1 加 comment，< 0.05d

# ── 功能覆蓋補齊（demo 加 cell / README 補說明）────────────────────────────
COV-A (factor generators)          ← demo §0 加列舉，< 0.3d
COV-B (quantile_spread_vw)         ← 加一行 demo，< 0.1d
COV-C (compute_group_returns)      ← README 補說明
COV-D (corrado_rank_test)          ← demo §3 加一行（§3.3 已有，§3.4 補進去）
COV-E (filter Callable)            ← demo §2.5 加範例
COV-F (MP/MC session 覆蓋)         ← demo §4/5 各補一 cell
COV-G (greedy_forward_selection)   ← 升 P2；demo §2.6 加示範，約 0.2d
COV-J (split_by_group workflow)    ← §7.1 補 evaluate 接力 cell，< 0.1d

# ── 文件清理（純標頭 / 說明）─────────────────────────────────────────────────
P3-A (舊命名 SUPERSEDED 標頭)      ← 8 個文件加 header，< 0.2d
                                     新增 long_short_spread → quantile_spread 事件
P3-B (preprocess 差異說明)         ← README 補說明
P3-C (greedy_forward_selection)    ← 升 P2（見 COV-G）

# ── 2026-04-20 第三輪驗證後刪除項目 ─────────────────────────────────────────
P2-D (docstring 同傳兩參數)        ← 查無此事，已從 doc 刪除
COV-I (fl.adapt 未示範)            ← 查無此事，已從 doc 刪除（cell #4 已執行）
COV-H (L2 成功案例)                ← 併入 P2-B（cell #32 存在但未執行）
```

---

## 驗收條件

**正確性**
- [ ] regime label 用 `shift(1)` 或 look-ahead free 資料來源（P0-A）
- [ ] `basis_df` 的 size proxy 改為 `market_cap.log()`（P0-B）
- [ ] `fl.preprocess` embed `_fl_forward_periods` + `evaluate*` strict gate 比對 raise（P0-C）
- [ ] `CrossSectionalFactor.ic()` / `hit_rate()` 的 `forward_periods=` 參數移除；docstring 指向 rebuild session 的正解（P0-D）

**API 三點命名統一**（method name == primitive MetricOutput.name == Profile field）
- [ ] `caar_trend` / `beta_trend` 各自的 `MetricOutput.name` 等於方法名（P1-A）
- [ ] `quantile_spread` 三地統一（primitive/Profile field/cache key 全改為 `quantile_spread`，CS + MP）（P1-B）
- [ ] `mfe_mae_summary` 三地統一（primitive `MetricOutput.name="mfe_mae_summary"`，cache key 同步）（P1-C）
- [ ] `bmp_test` 三地統一（primitive `MetricOutput.name="bmp_test"`，EventProfile 欄位 `bmp_test_p`，cache key 同步）（P1-D）

**Demo 可用性**
- [ ] `demo.ipynb §2.7` charts 路徑走 `fl.evaluate(..., return_artifacts=True)` 或 `fl.factor()`（P2-A）
- [ ] `demo.ipynb` **全部 16 個未執行 cell**（§2.4 / §3.4 / §4.3 / §5.3 / §7）的 `execution_count` 非 null（P2-B，含 COV-H 的 L2 成功輸出）
- [ ] Event Factor price-based / corrado opt-in metric 有 cell 展示（P2-C / COV-D）
- [ ] `with_extra_columns` cell 有排序依賴警示 comment（P2-E）
- [ ] §2.4.3 的 artifacts 取法示範用 `fl.factor()` session 路徑（P2-F）
- [ ] §3 / §4 / §5 補上 subsection 編號與 §2 對齊（P2-G）
- [ ] 附錄 B 有前置步驟說明（P2-H）
- [ ] `describe_profile_values(profile, artifacts=None)` 可接受 `None` 只印 scalar table；`evaluate()` docstring 補 Note 指向正路（P2-I）
- [ ] demo §2.4.1 有 `min_coverage` 說明 comment（P2-J）

**功能覆蓋**
- [ ] factor generator 族有列舉示範（COV-A）
- [ ] `quantile_spread_vw` 有 demo 示範（COV-B）
- [ ] `ProfileSet.filter(Callable)` 有 demo 範例（COV-E）
- [ ] MacroPanel / MacroCommon Factor session 方法列表完整（COV-F）
- [ ] `greedy_forward_selection` 有 demo 示範（COV-G / P3-C 升級）
- [ ] §7.1 有 `split_by_group` → `evaluate` 接力的完整 workflow（COV-J）

**文件清理**
- [ ] 8 個歷史文件加 SUPERSEDED 標頭（P3-A），新增 `long_short_spread → quantile_spread` 事件
- [ ] `preprocess` vs `preprocess_cs_factor` 差異在 README 說明（P3-B）
