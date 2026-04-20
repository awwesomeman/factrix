# factorlib

**Modular factor evaluation toolkit** — 把 cross-sectional / event-signal /
macro-panel / macro-common 四種因子放在同一套 `preprocess → evaluate →
Profile` API 底下，polars-native，每一步都典型有 statistical-discipline 的
默認值。

> **定位：Factor Signal Analyzer**，不是回測引擎。`turnover` /
> `breakeven_cost` / `net_spread` 是理想化的 proxy（等權、無滑價），screening
> 用；不是 tradable P&L。要 realistic execution 請把 screened 因子餵進
> Zipline / Backtrader / 自家引擎。

---

## Install

```bash
pip install factorlib              # core (polars only)
pip install factorlib[charts]      # + plotly 圖表
pip install factorlib[mlflow]      # + mlflow tracking
pip install factorlib[all]         # 全包
```

---

## 30-second smoke test

確認裝好了、環境跑得起來：

```python
import factorlib as fl

raw = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
cfg = fl.CrossSectionalConfig(forward_periods=5)
profile = fl.evaluate(fl.preprocess(raw, config=cfg), "sanity", config=cfg)

print(profile.verdict(), '| ic_mean =', round(profile.ic_mean, 4))
# → PASS | ic_mean = 0.0722
```

三行，不需要任何外部資料（`fl.datasets` 產可重現合成 panel）。通了以後去跑
[`examples/demo.ipynb`](examples/demo.ipynb) 看完整用法。

---

## 為什麼用 factorlib

比起自己 rolling 一套 factor-analysis 腳本，主要差在六件事：

1. **一個 `canonical_p` 驅動 `verdict()`** — 每個 factor_type 有**單一** p-value
   當 PASS/FAILED gate（IC t-test / CAAR / Fama-MacBeth / TS β）。不搞「看 IC
   又看 spread 又看 hit rate 最後人工拍板」的 ad-hoc 聚合。其他 signal 質
   / stability / regime 訊息全部走 `profile.diagnose()` 回 structured
   `Diagnostic`，不偷跑進 verdict。

2. **Typed `Profile` dataclass，不是 dict** — `CrossSectionalProfile`、
   `EventProfile` 等 `frozen + slots`，欄位 IDE discoverable，直接餵 polars
   expression 做 filter / rank / BHY。寫 `profile.ic_mean` 打錯 key IDE 就
   會叫，不會 typo 到半夜 debug。

3. **`fl.factor()` session 統一 metric API** — 單因子研究時所有 standalone
   metric 變成 session method（`f.ic()`、`f.quantile_spread()` 等），
   shared Artifacts cache 讓各 method 和 `f.evaluate()` 共用同一份計算；
   per-call `f.quantile_spread(n_groups=3)` 做 sensitivity sweep 不污染
   cache。

4. **Preprocess 和 evaluate 兩步驟 + strict gate** — `fl.preprocess(raw,
   config=cfg)` 把**所有被烙進 prepared 的 preprocess-time 欄位**（CS 是
   `forward_periods / mad_n / return_clip_pct`；Event / MC 是
   `forward_periods`；MP 另加 `demean_cross_section`）都嵌進 prepared 的
   `_fl_preprocess_sig` marker。`fl.evaluate` / `fl.factor` 逐欄位 diff，
   任何對不上直接 raise 並指名哪個欄位、兩邊各是什麼值。擋掉最惡性的那
   類 bug — 兩邊 config silently 對不上、全下游 metric 無聲無息污染、
   測出來的 IC 看起來合理但其實量錯 horizon。兩步驟保留是為了讓
   `prepared` cache 一次、evaluate-time 欄位（`n_groups, tie_policy, ortho,
   regime_labels, …`）可以 sweep。

5. **批次 + BHY 多重檢定一行搞定** — `ps.multiple_testing_correct(p_source=
   "canonical_p", fdr=0.05)` 用 Benjamini-Yekutieli step-up（比 BH 保守、
   容許 dependence）控制 family-wise FDR；`p_source` 白名單只收
   `Profile.P_VALUE_FIELDS`，複合 p 例如 `min(ic_p, spread_p)` 會被拒，不
   讓 user 餵跨 hypothesis 的 p 進 BHY 壞掉 same-test-family 語意。

6. **Short-circuit = NaN, not 0** — metric 算不出來（sample 太小、缺欄位）
   回傳 `NaN` 加 `metadata["reason"]`，BHY 讀 `metadata["p_value"]=1.0`
   保守拒絕。0 是合法的 factor 結果（IC、β 剛好為 0），NaN 才是「跳過」。
   `describe_profile_values` 把 NaN 顯示成 `—`，skips 看得見。

---

## 四種 factor_type

| Type | 訊號型態 | 典型用法 | Canonical test |
|------|---------|----------|----------------|
| `cross_sectional` | 每期每資產連續值 | momentum / value / size | IC 非重疊 t-test |
| `event_signal` | 離散觸發 `{-1, 0, +1}` | 盈餘公告、黃金交叉 | CAAR 非重疊 t-test |
| `macro_panel` | 連續值、小截面 N<30 | 跨國 CPI、利差配置 | Fama-MacBeth Newey-West |
| `macro_common` | 單一時序、全資產共用 | VIX、黃金、USD index | 截面 t-test on per-asset TS β |

```python
fl.describe_factor_types()           # 所有 factor_type 概觀
fl.describe_profile("event_signal")  # 指定 type 的 Profile dataclass 欄位 + canonical p
```

---

## Scope & non-goals

- **Scope**：daily-to-monthly bar-based factor evaluation。`forward_periods`
  是 **rows，不是 calendar time** — 週頻 panel 寫 `forward_periods=1` 就是
  1-week 前向報酬，同一套 API。
- **非 scope**：intraday / high-frequency（tick-level、sub-second）沒測、
  不支援。per-date CS IC / CAAR / FM λ 的語意在 tick data 上不成立。
- **非 scope**：回測、執行成本建模、滑價、margin call、real portfolio
  optimization。Screening 結果自己帶去回測系統。

---

## 下一步看哪裡

| 想知道 | 看 |
|--------|-----|
| 怎麼用（可執行例子） | [`examples/demo.ipynb`](examples/demo.ipynb) — 四種 factor_type × Level 0–6 的可執行功能索引，`fl.datasets` 產資料、從 fresh clone 可以直接跑 |
| 內部結構（module layout、invariants、Profile contract、Artifacts 策略） | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| 想貢獻（dev workflow、submodule、test 規範、release） | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 版本變動 | [`CHANGELOG.md`](CHANGELOG.md) |
| 個別 metric 的數學、interpretation、corner cases | 對應 module 的 docstring：`help(factorlib.metrics.ic)`、`help(factorlib.metrics.caar)`、... |
