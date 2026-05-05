# factrix

> **Factor Matrix Library** — Polars-native Factor Signal Validator

Factrix 只回答：**「這個因子在統計上真的有效嗎？」**

請帶著 `primary_p` 與 `verdict()` 去下游做回測 / 配置 — `factrix` 不做那些。

📖 **[Documentation](https://awwesomeman.github.io/factrix/)** | [GitHub](https://github.com/awwesomeman/factrix)

---

## 安裝

```bash
# uv（推薦）
uv pip install git+https://github.com/awwesomeman/factrix.git

# conda / pip
pip install git+https://github.com/awwesomeman/factrix.git
```

開發者安裝：

```bash
git clone https://github.com/awwesomeman/factrix.git && cd factrix
uv sync --extra dev
```

---

## 30-second smoke test

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

cfg     = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)
profile = fl.evaluate(panel, cfg)

print(profile.verdict(), '| primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0
```

> `forward_periods` counts rows, not calendar time. factrix is frequency-agnostic.

---

## Research question → factory

| 你想問的問題 | Factory |
|---|---|
| Per-asset 因子能否預測 cross-section 排序？ | `individual_continuous(metric=fl.Metric.IC)` |
| Per-asset 因子的報酬溢酬是多少？ | `individual_continuous(metric=fl.Metric.FM)` |
| 個股事件有沒有 abnormal return？ | `individual_sparse()` |
| Macro 因子對 cross-section 有沒有 systematic exposure？ | `common_continuous()` |
| Macro 事件有沒有市場效應？ | `common_sparse()` |

---

## Batch screening with BHY

```python
candidates = ["mom_5d", "mom_20d", "mom_60d"]
cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)

profiles  = [fl.evaluate(panel.with_columns(pl.col(c).alias("factor")), cfg) for c in candidates]
survivors = fl.multi_factor.bhy(profiles, threshold=0.05)
```

---

## Non-goals

factrix 不做（也不打算做）：portfolio optimization、ML signal、backtest / execution、HFT tick-level。Feed `primary_p` survivors into `skfolio` / `vectorbt` / `zipline` downstream.

---

## 文件

| 想知道 | 看 |
|---|---|
| 安裝、quickstart、三軸概念 | [Get Started](https://awwesomeman.github.io/factrix/getting-started/) |
| PANEL/TIMESERIES、BHY、metric 選擇 | [Guides](https://awwesomeman.github.io/factrix/guides/) |
| 精確公式 / 演算法 / API | [Reference](https://awwesomeman.github.io/factrix/reference/methodology/) |
| 內部架構 / 開發流程 | [Development](https://awwesomeman.github.io/factrix/development/architecture/) |

---

## License

[Apache License 2.0](LICENSE)
