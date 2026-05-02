"""factrix v0.5 demo — 5 個合法 cell × PANEL/TIMESERIES × BHY × 錯誤路徑.

可執行的功能索引：每個 section 對應 README 一個概念。從 fresh clone
直接跑，無外部資料。生成 ``examples/demo.ipynb`` 走
``jupyter nbconvert`` 或重新執行 ``examples/build_notebook.py``（產生
notebook 時會把每個 ``# %% [markdown]`` / ``# %%`` 區塊轉成對應的
notebook cell）。
"""

# %% [markdown]
# # factrix v0.5 demo
#
# v0.5 user surface ＝ `AnalysisConfig`（4 factories）→ `evaluate` →
# `FactorProfile` → `multi_factor.bhy`。其餘 `describe_analysis_modes`
# / `suggest_config` 是 introspection helpers。
#
# 三條軸：
# - `FactorScope`：因子值是 per-asset (`INDIVIDUAL`) 還是 broadcast (`COMMON`)
# - `Signal`：實數 (`CONTINUOUS`) 還是稀疏觸發 (`SPARSE`)
# - `Metric`：只在 `(INDIVIDUAL, CONTINUOUS)` 細分 IC vs FM
#
# 第四條軸 `Mode` 由資料的 `N = panel["asset_id"].n_unique()` 在
# evaluate-time 自動推導，不是 user-facing。

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import factrix as fl
from factrix.preprocess.returns import compute_forward_return

pl.Config.set_tbl_rows(8)
print("factrix version:", fl.__version__)

# %% [markdown]
# ## 2. 合法 cell 一覽
#
# `describe_analysis_modes()` reverse-query registry SSOT，列出所有合法
# `(scope, signal, metric)` cell × PANEL/TIMESERIES 路由 + 文獻基礎。

# %%
print(fl.describe_analysis_modes())

# %% [markdown]
# JSON 形式給 AI agent / 工具拿：

# %%
modes_json = fl.describe_analysis_modes(format="json")
print(f"{len(modes_json)} legal cells:")
for row in modes_json:
    print(f"  ({row['scope']}, {row['signal']}, {row['metric']})")

# %% [markdown]
# ## 3. INDIVIDUAL × CONTINUOUS × IC — 選股 IC
#
# 個股因子（每股自己的訊號），rank-based predictive ordering。

# %%
raw = fl.datasets.make_cs_panel(
    n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)
print(f"panel shape={panel.shape}  N={panel['asset_id'].n_unique()}")

cfg_ic = fl.AnalysisConfig.individual_continuous(
    metric=fl.Metric.IC, forward_periods=5,
)
prof_ic = fl.evaluate(panel, cfg_ic)

print(f"verdict      = {prof_ic.verdict()}")
print(f"primary_p    = {prof_ic.primary_p:.4g}")
print(f"mode         = {prof_ic.mode}")
print(f"ic_mean      = {prof_ic.stats[fl.StatCode.IC_MEAN]:+.4f}")
print(f"ic_t_nw      = {prof_ic.stats[fl.StatCode.IC_T_NW]:+.2f}")
print(f"nw_lags_used = {prof_ic.stats[fl.StatCode.NW_LAGS_USED]:.0f}")

# %% [markdown]
# `profile.diagnose()` 一次拿全部，給人讀也給 AI agent 拿：

# %%
import json
print(json.dumps(prof_ic.diagnose(), indent=2, default=str))

# %% [markdown]
# ## 4. INDIVIDUAL × CONTINUOUS × FM — Fama-MacBeth λ
#
# `(INDIVIDUAL, CONTINUOUS)` 兩 metric 回答不同研究問題，選擇依研究問題
# 決定，不依資料形狀決定。

# %%
cfg_fm = fl.AnalysisConfig.individual_continuous(
    metric=fl.Metric.FM, forward_periods=5,
)
prof_fm = fl.evaluate(panel, cfg_fm)
print(f"verdict       = {prof_fm.verdict()}")
print(f"primary_p     = {prof_fm.primary_p:.4g}")
print(f"λ_mean        = {prof_fm.stats[fl.StatCode.FM_LAMBDA_MEAN]:+.4f}")
print(f"λ_t_nw        = {prof_fm.stats[fl.StatCode.FM_LAMBDA_T_NW]:+.2f}")

# %% [markdown]
# ## 5. INDIVIDUAL × SPARSE — 事件研究 (CAAR)
#
# 個股事件觸發 `{−1, 0, +1}`，沒 metric 細分。

# %%
ev_raw = fl.datasets.make_event_panel(
    n_assets=80, n_dates=400,
    event_rate=0.02, post_event_drift_bps=15.0,
    signal_horizon=5, seed=42,
)
ev_panel = compute_forward_return(ev_raw, forward_periods=5)

cfg_caar = fl.AnalysisConfig.individual_sparse(forward_periods=5)
prof_caar = fl.evaluate(ev_panel, cfg_caar)
print(f"verdict   = {prof_caar.verdict()}")
print(f"primary_p = {prof_caar.primary_p:.4g}")
print(f"CAAR      = {prof_caar.stats[fl.StatCode.CAAR_MEAN]:+.5f}")
print(f"caar_t_nw = {prof_caar.stats[fl.StatCode.CAAR_T_NW]:+.2f}")
print(f"warnings  = {sorted(w.value for w in prof_caar.warnings)}")

# %% [markdown]
# ## 6. COMMON × CONTINUOUS — broadcast factor (VIX-like)
#
# 每個 date 上所有 asset 共享同一 factor 值。Per-asset TS β → cross-asset
# t-test on `E[β]`。

# %%
def make_broadcast_panel(
    n_assets: int = 30, n_dates: int = 300, seed: int = 11,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    f_t = rng.standard_normal(n_dates)  # broadcast factor
    rows = []
    for t, d in enumerate(dates):
        for i in range(n_assets):
            r = 0.4 * f_t[t] * 0.01 + 0.01 * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"a{i:02d}",
                "factor": float(f_t[t]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


bcast = make_broadcast_panel()
# 驗證 broadcast 性質：每個 date 上 factor 值唯一
is_broadcast = bool(
    bcast.group_by("date").agg(pl.col("factor").n_unique().alias("u"))
    .select((pl.col("u") == 1).all()).item()
)
print(f"broadcast verified: {is_broadcast}  N={bcast['asset_id'].n_unique()}")

cfg_cc = fl.AnalysisConfig.common_continuous(forward_periods=1)
prof_cc = fl.evaluate(bcast, cfg_cc)
print(f"verdict       = {prof_cc.verdict()}")
print(f"primary_p     = {prof_cc.primary_p:.4g}")
print(f"E[β]          = {prof_cc.stats[fl.StatCode.TS_BETA]:+.4f}")

# %% [markdown]
# ## 7. COMMON × SPARSE — broadcast event (FOMC-like)

# %%
def make_broadcast_event_panel(seed: int = 23) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 300, 30
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    d_t = rng.choice([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], size=n_dates)
    rows = []
    for t, d in enumerate(dates):
        for i in range(n_assets):
            r = 0.005 * d_t[t] + 0.01 * rng.standard_normal()
            rows.append({
                "date": d, "asset_id": f"a{i:02d}",
                "factor": float(d_t[t]),
                "forward_return": float(r),
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


cs_event = make_broadcast_event_panel()
cfg_cs = fl.AnalysisConfig.common_sparse(forward_periods=1)
prof_cs = fl.evaluate(cs_event, cfg_cs)
print(f"verdict   = {prof_cs.verdict()}")
print(f"primary_p = {prof_cs.primary_p:.4g}")
print(f"E[β]      = {prof_cs.stats[fl.StatCode.TS_BETA]:+.5f}")

# %% [markdown]
# ## 8. TIMESERIES — N=1 自動降維為時序
#
# `Mode` 不是 user-facing 軸，由 `N = n_unique(asset_id)` 決定。N=1 時
# `(COMMON, *)` 與 `(INDIVIDUAL, SPARSE)` 都走真正的時序 procedure，
# `primary_p` 給真實值不壓 1.0。

# %%
single_asset = bcast.filter(pl.col("asset_id") == "a00")
print(f"N = {single_asset['asset_id'].n_unique()}")

prof_b = fl.evaluate(single_asset, cfg_cc)
print(f"mode      = {prof_b.mode}")
print(f"primary_p = {prof_b.primary_p:.4g}  (TIMESERIES 真實值)")
print(f"warnings  = {sorted(w.value for w in prof_b.warnings)}")

# %% [markdown]
# `(*, SPARSE) × N=1` 時 scope 軸自然 collapse — `individual_sparse` 與
# `common_sparse` 在 N=1 路由到同一個 timeseries dummy procedure，
# `info_notes` 留下 `SCOPE_AXIS_COLLAPSED` audit trail。

# %%
single_event = cs_event.filter(pl.col("asset_id") == "a00")
prof_collapsed = fl.evaluate(
    single_event, fl.AnalysisConfig.common_sparse(forward_periods=1),
)
print(f"info_notes = {sorted(i.value for i in prof_collapsed.info_notes)}")

# 兩個入口在 N=1 等價
prof_via_individual = fl.evaluate(
    single_event, fl.AnalysisConfig.individual_sparse(forward_periods=1),
)
print(
    f"individual_sparse(N=1) primary_p = "
    f"{prof_via_individual.primary_p:.4g}",
)
print(
    f"common_sparse(N=1)     primary_p = "
    f"{prof_collapsed.primary_p:.4g}",
)

# %% [markdown]
# ## 9. 錯誤路徑 — actionable diagnostics
#
# `(INDIVIDUAL, CONTINUOUS) × N=1` 數學上不存在 — `evaluate()` raise
# `ModeAxisError` 並附 `suggested_fix`：

# %%
try:
    fl.evaluate(single_asset, cfg_ic)
except fl.ModeAxisError as e:
    print("ModeAxisError raised")
    print(f"  message       : {e}")
    print(f"  suggested_fix : {e.suggested_fix}")

# %% [markdown]
# `T < MIN_T_HARD = 20` raise `InsufficientSampleError` 並帶 `actual_T`/
# `required_T` 給 caller 程式化 recover：

# %%
short = single_asset.head(15)
try:
    fl.evaluate(short, cfg_cc)
except fl.InsufficientSampleError as e:
    print(f"actual_T   = {e.actual_T}")
    print(f"required_T = {e.required_T}")
    print(f"message    : {e}")

# %% [markdown]
# 違反三軸組合（如 SPARSE 配 IC）在建構 `AnalysisConfig` 階段就 raise，
# 不會跑到 evaluate：

# %%
try:
    fl.AnalysisConfig(
        scope=fl.FactorScope.INDIVIDUAL,
        signal=fl.Signal.SPARSE,
        metric=fl.Metric.IC,
    )
except fl.IncompatibleAxisError as e:
    print(f"IncompatibleAxisError: {e}")

# %% [markdown]
# ## 10. 批次評估 + BHY FDR
#
# BHY 控制**同一 family 內**的 FDR。下面這個批次是「同一個 cell 評估多個
# candidate」的正確用法：

# %%
def make_factor_variant(panel: pl.DataFrame, noise_scale: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return panel.with_columns(
        pl.Series(
            "factor",
            panel["factor"].to_numpy()
            + noise_scale * rng.standard_normal(panel.height),
        ),
    )


candidates = {
    f"variant_{i}": make_factor_variant(panel, noise_scale=0.5 + 0.3 * i, seed=100 + i)
    for i in range(5)
}
profiles = [
    fl.evaluate(p, cfg_ic) for p in candidates.values()
]
for name, prof in zip(candidates, profiles):
    print(f"  {name:12s} primary_p={prof.primary_p:.4g}  verdict={prof.verdict()}")

survivors = fl.multi_factor.bhy(profiles, threshold=0.05)
print(f"\nBHY survivors: {len(survivors)} / {len(profiles)}")

# %% [markdown]
# 跨 family 混批會 emit `RuntimeWarning` — 多個 size=1 family 的 BHY
# 等於 raw threshold，沒有 FDR 校正力：

# %%
mixed = [
    fl.evaluate(panel, cfg_ic),
    fl.evaluate(panel, cfg_fm),
    fl.evaluate(bcast, cfg_cc),
]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fl.multi_factor.bhy(mixed, threshold=0.05)
print(f"warnings raised: {[str(w.message) for w in caught]}")

# %% [markdown]
# ## 11. `suggest_config` — 從資料反向給 factory call

# %%
suggestion = fl.suggest_config(panel)
print(f"suggested  : {suggestion.suggested}")
print(f"reasoning  :")
for axis, why in suggestion.reasoning.items():
    print(f"  {axis:8s}: {why}")
print(f"warnings   : {[w.value for w in suggestion.warnings]}")

# %% [markdown]
# ## 12. 收尾
#
# v0.5 的 user surface 就到此 — 4 個 factory + `evaluate` + `multi_factor.bhy`
# + 兩個 introspection helper。其餘進階用法（`gate=` 自訂 BHY 軸、自訂
# `WarningCode` 過濾邏輯、跨 cell 集合操作）請見 `help(fl.FactorProfile)`、
# `help(fl.multi_factor.bhy)`、`ARCHITECTURE.md`。

# %%
print("demo OK — factrix", fl.__version__)
