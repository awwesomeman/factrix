"""Generator for examples/demo.ipynb.

Produces a clean notebook showing every factorlib API level across all four
factor_types. The opening section uses ``factorlib.datasets`` so the notebook
runs from a fresh clone — no external parquet required. All sections use
synthetic panels from ``fl.datasets``.

Run from repo root: ``uv run python scripts/build_demo.py``
"""

from __future__ import annotations

import json
from pathlib import Path


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


CELLS: list[dict] = [
    md(
        "# factorlib demo — 四種因子類別 × 完整 API 展示\n"
        "\n"
        "這份 notebook 不是教學文檔，是**可執行的功能索引**：每個 cell 對應 "
        "`factorlib/README.md` 的一段 API，照順序跑完就看過 factorlib 全部功能。\n"
        "\n"
        "涵蓋：\n"
        "- 四種 `factor_type`：`cross_sectional` / `event_signal` / `macro_panel` / `macro_common`\n"
        "- 六個使用層級（Level 0–6）：單因子 → 自訂 config → 個別 metrics → "
        "batch + BHY → redundancy → charts → MLflow\n"
        "\n"
        "資料來源全部是 `fl.datasets` 的合成 panel（seeded），從 fresh clone 可以直接跑，**不需要任何外部 parquet**。\n"
        "如果要改跑自己手上的真實 panel，把 §0.9 那格的 `raw_demo = ...` 換成 `fl.adapt(your_df, ...)` 即可，下游 cells 都建立在 canonical `date / asset_id / price` 之上。"
    ),
    md("## 0. Setup"),
    code(
        "from __future__ import annotations\n"
        "\n"
        "import polars as pl\n"
        "import factorlib as fl\n"
        "\n"
        "pl.Config.set_tbl_rows(8)\n"
        "pl.Config.set_fmt_str_lengths(60)\n"
        "print('factorlib version:', getattr(fl, '__version__', 'dev'))"
    ),
    md(
        "## 0.5 Synthetic quick-start — 不需外部資料\n"
        "\n"
        "`fl.datasets` 提供兩個 seeded 生成器：\n"
        "- `make_cs_panel(n_assets, n_dates, ic_target=...)` — CS 面板，每期 factor 與 N 日 forward return 的 CS 相關性 ≈ `ic_target`。\n"
        "- `make_event_panel(n_assets, n_dates, event_rate, post_event_drift_bps)` — 事件訊號（`factor ∈ {-1, 0, +1}`）加上 post-event drift。\n"
        "\n"
        "輸出 canonical columns `date, asset_id, price, factor`；下一步照一般流程 `fl.preprocess → fl.evaluate`。\n"
        "下面這段 cell 示範完整 end-to-end：合成資料 → preprocess → evaluate → profile。後續 sections 會用更大的合成 panel 展示 factor generators、batch、event、macro 等 API。"
    ),
    code(
        "cfg = fl.CrossSectionalConfig(forward_periods=5)\n"
        "synthetic_raw = fl.datasets.make_cs_panel(\n"
        "    n_assets=100, n_dates=500, ic_target=0.04, forward_periods=5, seed=2024,\n"
        ")\n"
        "print('synthetic raw:', synthetic_raw.shape, synthetic_raw.columns)\n"
        "\n"
        "synthetic_prepared = fl.preprocess(synthetic_raw, config=cfg)\n"
        "synthetic_profile = fl.evaluate(synthetic_prepared, 'synthetic', config=cfg)\n"
        "print(f'realized ic_mean = {synthetic_profile.ic_mean:.4f} '\n"
        "      f'(target was 0.04)')\n"
        "print('verdict:', synthetic_profile.verdict())"
    ),
    md(
        "## 0.9 共用 price panel（給 §1 之後的所有 cells）\n"
        "\n"
        "用 `fl.datasets.make_cs_panel` 產一份 100 資產 × 500 日的合成 price panel，只留 canonical `date / asset_id / price` — factor 欄位在這裡丟掉，後續 sections 會用 `factorlib.factors` 的 generator 從價格重算動量/波動度等真正的候選因子。\n"
        "若要換成真實資料，只要把這格的 `raw_demo = ...` 改成 `fl.adapt(your_df, date=..., asset_id=..., price=...)` 即可。"
    ),
    code(
        "raw_demo = (\n"
        "    fl.datasets.make_cs_panel(n_assets=100, n_dates=500, seed=2024)\n"
        "    .select(['date', 'asset_id', 'price'])\n"
        ")\n"
        "print('rows:', raw_demo.height, '| assets:', raw_demo['asset_id'].n_unique(),\n"
        "      '| span:', raw_demo['date'].min(), '->', raw_demo['date'].max())\n"
        "raw_demo.head(3)"
    ),
    md(
        "## 1. 四種因子類別總覽\n"
        "\n"
        "`fl.describe_factor_types()` 印出所有註冊的 factor_type 與用途；\n"
        "`fl.describe_profile(<type>)` 反射對應 Profile dataclass 的欄位、canonical p 與方法。\n"
        "這兩個是「不用開檔就知道 factorlib 現在長什麼樣」的入口。"
    ),
    code(
        "fl.describe_factor_types()\n"
        "print()\n"
        "print('list_factor_types ->', fl.list_factor_types())"
    ),
    code(
        "for ft in fl.list_factor_types():\n"
        "    fl.describe_profile(ft)"
    ),
    md(
        "## 2. Cross-sectional（選股因子）\n"
        "\n"
        "訊號型態：每期每資產有連續值。典型用法 = 動量 / 價值 / 規模。\n"
        "Canonical test: `ic_p`（IC 非重疊 t-test）。"
    ),
    md("### 2.1 Build factor（用 factorlib 內建 generator）"),
    code(
        "from factorlib.factors import generate_momentum, generate_momentum_60d\n"
        "from factorlib.factors.volatility import generate_volatility\n"
        "\n"
        "mom20 = generate_momentum(raw_demo, lookback=20)\n"
        "print('columns:', mom20.columns)\n"
        "print('shape:', mom20.shape)\n"
        "mom20.select(['asset_id', 'date', 'price', 'factor']).head(3)"
    ),
    md(
        "### 2.2 Level 0 — 最簡單 `fl.evaluate`\n"
        "\n"
        "`fl.preprocess` 與 `fl.evaluate` 是兩步驟：preprocess 產出含 `forward_return` 等中間欄位的 prepared panel，evaluate 再跑 all metrics + diagnostics，回傳 `CrossSectionalProfile`（frozen dataclass）。把 preprocess 留在 user code 是為了避免 config 兩邊不一致偷偷壞掉 canonical_p。"
    ),
    code(
        "cfg_default = fl.CrossSectionalConfig()\n"
        "mom20_prep = fl.preprocess(mom20, config=cfg_default)\n"
        "profile = fl.evaluate(mom20_prep, 'Mom_20D', config=cfg_default)\n"
        "\n"
        "print('type:       ', type(profile).__name__)\n"
        "print('verdict:    ', profile.verdict())\n"
        "print('canonical_p:', f'{profile.canonical_p:.4f}')\n"
        "print('ic_mean:    ', f'{profile.ic_mean:+.4f}')\n"
        "print('ic_tstat:   ', f'{profile.ic_tstat:+.2f}')\n"
        "print('ic_ir:      ', f'{profile.ic_ir:+.3f}')\n"
        "print('long-short spread:', f'{profile.quantile_spread:+.4f}')\n"
        "print('turnover:   ', f'{profile.turnover:.2%}')\n"
        "print('net_spread: ', f'{profile.net_spread:+.4f}')"
    ),
    code(
        "# profile.diagnose() 回傳結構化 Diagnostic — 包含 severity / code / message\n"
        "for d in profile.diagnose():\n"
        "    print(f'[{d.severity:<7s}] {d.code:<35s} {d.message}')"
    ),
    md(
        "### 2.3 Level 1 — 自訂 `CrossSectionalConfig`\n"
        "\n"
        "切換 forward horizon、quantile groups、trading-cost 估計等。"
    ),
    code(
        "cfg = fl.CrossSectionalConfig(\n"
        "    forward_periods=10,            # 10 日 forward return\n"
        "    n_groups=5,                    # 5 組 quantile\n"
        "    mad_n=3.0,\n"
        "    return_clip_pct=(0.01, 0.99),\n"
        "    estimated_cost_bps=30,\n"
        ")\n"
        "mom20_prep_10d = fl.preprocess(mom20, config=cfg)\n"
        "profile_10d = fl.evaluate(mom20_prep_10d, 'Mom_20D_h10', config=cfg)\n"
        "print('verdict @10d:', profile_10d.verdict(), '| ic_ir:', f'{profile_10d.ic_ir:+.3f}')"
    ),
    md(
        "### 2.4 Level 2 — 個別 metrics（繞過 Profile）\n"
        "\n"
        "當你只想算某個統計量、或要餵 metric 進自己的 pipeline，不需要整個 Profile。\n"
        "流程：`fl.preprocess` → 個別 `metrics.*` 函式（全部回 `MetricOutput`）。"
    ),
    code(
        "from factorlib.metrics import (\n"
        "    compute_ic, ic, ic_ir, quantile_spread, monotonicity,\n"
        ")\n"
        "\n"
        "prepared = fl.preprocess(mom20, config=cfg)\n"
        "print('prepared columns:', prepared.columns, '\\n')\n"
        "\n"
        "ic_series = compute_ic(prepared)  # DataFrame[date, ic]\n"
        "ic_m = ic(ic_series, forward_periods=cfg.forward_periods)\n"
        "ir_m = ic_ir(ic_series)\n"
        "spread_m = quantile_spread(prepared, forward_periods=cfg.forward_periods, n_groups=cfg.n_groups)\n"
        "mono_m = monotonicity(prepared, forward_periods=cfg.forward_periods, n_groups=cfg.n_groups)\n"
        "\n"
        "# MetricOutput 的 p-value 放在 .metadata['p_value']，__repr__ 會印摘要\n"
        "for name, out in [('ic', ic_m), ('ic_ir', ir_m), ('quantile_spread', spread_m), ('monotonicity', mono_m)]:\n"
        "    p = out.metadata.get('p_value')\n"
        "    p_str = 'n/a' if p is None else f'{p:.4f}'\n"
        "    stat_str = 'n/a' if out.stat is None else f'{out.stat:+.2f}'\n"
        "    print(f'{name:<15s} value={out.value:+.4f}  stat={stat_str:>6s}  p={p_str}  sig={out.significance or \"\"}')"
    ),
    md(
        "### 2.5 Level 3 — `evaluate_batch` + BHY + rank + top\n"
        "\n"
        "批次評估多因子，回傳 polars-native `ProfileSet`。\n"
        "`multiple_testing_correct` 做 BHY 多重檢定校正；`filter` / `rank_by` / `top` 可鏈式串接。"
    ),
    code(
        "# 準備三個候選 CS 因子\n"
        "mom60 = generate_momentum_60d(raw_demo)\n"
        "vol20 = generate_volatility(raw_demo, lookback=20)\n"
        "\n"
        "factors_map = {\n"
        "    'Mom_20D': mom20,\n"
        "    'Mom_60_5': mom60,\n"
        "    'Vol_20D': vol20,\n"
        "}\n"
        "cs_cfg = fl.CrossSectionalConfig()\n"
        "prepared_map = {name: fl.preprocess(df, config=cs_cfg) for name, df in factors_map.items()}\n"
        "ps = fl.evaluate_batch(prepared_map, config=cs_cfg)\n"
        "print('ProfileSet size:', len(ps), '| profile class:', ps.profile_cls.__name__)\n"
        "ps.to_polars().select(['factor_name', 'ic_mean', 'ic_ir', 'quantile_spread', 'canonical_p'])  # quick glance\n"
    ),
    code(
        "# BHY 多重檢定 + 按 IC_IR 排序 + 取 top 2\n"
        "top = (\n"
        "    ps\n"
        "    .multiple_testing_correct(p_source='canonical_p', fdr=0.10)\n"
        "    .rank_by('ic_ir', descending=True)\n"
        "    .top(2)\n"
        ")\n"
        "top.to_polars().select([\n"
        "    'factor_name', 'ic_p', 'p_adjusted', 'bhy_significant', 'ic_ir',\n"
        "])"
    ),
    code(
        "# 也能傳 pl.Expr 或 Callable[[Profile], bool] 給 filter\n"
        "stable = ps.filter(pl.col('ic_ir').abs() > 0.1)\n"
        "print('factors with |IC_IR| > 0.1:', [p.factor_name for p in stable])"
    ),
    md(
        "### 2.6 Level 4 — Redundancy matrix\n"
        "\n"
        "多因子之間的成對 `|ρ|`，抓出冗餘訊號。"
    ),
    code(
        "# redundancy_matrix 需要 per-factor Artifacts；用 keep_artifacts=True 保留。\n"
        "ps2, arts = fl.evaluate_batch(\n"
        "    prepared_map, config=cs_cfg, keep_artifacts=True,\n"
        ")\n"
        "\n"
        "redund = fl.redundancy_matrix(ps2, method='value_series', artifacts=arts)\n"
        "redund"
    ),
    md(
        "### 2.7 Level 5 — Charts (optional dep: `factorlib[charts]`)\n"
        "\n"
        "`build_artifacts` 保留中間結果（IC series / spread series / quantile group returns…），\n"
        "丟進 `report_charts` 產 plotly 圖。未安裝 `plotly` 會 raise ImportError — 此處 try/except 包起來。"
    ),
    code(
        "from factorlib.evaluation.pipeline import build_artifacts\n"
        "\n"
        "prepared = fl.preprocess(mom20, config=fl.CrossSectionalConfig())\n"
        "artifacts = build_artifacts(prepared, fl.CrossSectionalConfig())\n"
        "artifacts.factor_name = 'Mom_20D'\n"
        "print('artifacts.intermediates keys:', sorted(artifacts.intermediates.keys()))\n"
        "\n"
        "try:\n"
        "    from factorlib.charts import report_charts\n"
        "    figs = report_charts(artifacts)\n"
        "    print(f'produced {len(figs)} figures:', list(figs)[:5])\n"
        "    # 在 notebook 直接顯示第一張\n"
        "    next(iter(figs.values())).show()\n"
        "except ImportError as e:\n"
        "    print('charts skipped — install with `pip install factorlib[charts]`:', e)"
    ),
    md(
        "### 2.8 Level 6 — MLflow tracking (optional dep: `factorlib[mlflow]`)\n"
        "\n"
        "`on_result` callback 在 `evaluate_batch` 的每個因子算完後觸發，\n"
        "可用來 log profile 到 MLflow experiment。這裡只示範寫法，不實跑（避免副作用）。"
    ),
    code(
        "# 不實跑 — 只展示 API 形狀\n"
        "demo = '''\n"
        "from factorlib.integrations.mlflow import FactorTracker\n"
        "\n"
        "tracker = FactorTracker('Factor_Zoo')\n"
        "ps = fl.evaluate_batch(\n"
        "    factors_map,\n"
        "    factor_type='cross_sectional',\n"
        "    on_result=lambda name, p: tracker.log_profile(p, factor_type='cross_sectional'),\n"
        ")\n"
        "'''\n"
        "print(demo)"
    ),
    md(
        "## 3. Event-signal（事件交易因子）\n"
        "\n"
        "訊號型態：離散觸發 `{-1, 0, +1}`。Canonical test: `caar_p`（CAAR 非重疊 t-test）。\n"
        "範例：黃金交叉 / 死亡交叉 → 事件後 5 日有異常報酬嗎？"
    ),
    md("### 3.1 Build golden/death-cross event signal"),
    code(
        "event_df = (\n"
        "    raw_demo.sort(['asset_id', 'date'])\n"
        "    .with_columns(\n"
        "        pl.col('price').rolling_mean(5).over('asset_id').alias('ma5'),\n"
        "        pl.col('price').rolling_mean(20).over('asset_id').alias('ma20'),\n"
        "    )\n"
        "    .with_columns(\n"
        "        pl.when(pl.col('ma5') > pl.col('ma20')).then(1)\n"
        "          .otherwise(-1).alias('cross_state')\n"
        "    )\n"
        "    .with_columns(\n"
        "        (pl.col('cross_state') - pl.col('cross_state').shift(1).over('asset_id')).alias('delta')\n"
        "    )\n"
        "    .with_columns(\n"
        "        pl.when(pl.col('delta') == 2).then(1.0)   # 黃金交叉\n"
        "        .when(pl.col('delta') == -2).then(-1.0)   # 死亡交叉\n"
        "        .otherwise(0.0).alias('factor')\n"
        "    )\n"
        "    .filter(pl.col('factor').is_not_null() & pl.col('ma20').is_not_null())\n"
        "    .select(['asset_id', 'date', 'price', 'factor'])\n"
        ")\n"
        "print('factor value counts:')\n"
        "print(event_df['factor'].value_counts().sort('factor'))"
    ),
    md("### 3.2 evaluate（自動切到 `EventProfile`）"),
    code(
        "ev_cfg = fl.EventConfig(\n"
        "    forward_periods=5,\n"
        "    event_window_pre=5,\n"
        "    event_window_post=20,\n"
        "    cluster_window=3,\n"
        "    adjust_clustering='none',\n"
        ")\n"
        "event_prepared = fl.preprocess(event_df, config=ev_cfg)\n"
        "ev_profile = fl.evaluate(event_prepared, 'GoldenCross', config=ev_cfg)\n"
        "\n"
        "print('type:           ', type(ev_profile).__name__)\n"
        "print('n_events:       ', ev_profile.n_events)\n"
        "print('n_periods (dates):', ev_profile.n_periods)\n"
        "print('verdict:        ', ev_profile.verdict())\n"
        "print('CAAR mean:      ', f'{ev_profile.caar_mean:+.4f}')\n"
        "print('CAAR p (canonical):', f'{ev_profile.caar_p:.4f}')\n"
        "print('BMP z:          ', f'{ev_profile.bmp_zstat:+.2f}  p={ev_profile.bmp_p:.4f}')\n"
        "print('event_hit_rate: ', f'{ev_profile.event_hit_rate:.2%}  p={ev_profile.event_hit_rate_p:.4f}')\n"
        "print('profit_factor:  ', f'{ev_profile.profit_factor:.3f}')\n"
        "print('clustering_hhi (normalized):', ev_profile.clustering_hhi_normalized)"
    ),
    code(
        "for d in ev_profile.diagnose():\n"
        "    print(f'[{d.severity:<7s}] {d.code:<35s} {d.message}')"
    ),
    md("### 3.3 Level 2 — individual event metrics"),
    code(
        "from factorlib.metrics import (\n"
        "    compute_caar, caar, bmp_test, event_hit_rate,\n"
        "    corrado_rank_test, clustering_diagnostic,\n"
        ")\n"
        "\n"
        "ev_prepared = fl.preprocess(event_df, config=ev_cfg)\n"
        "\n"
        "caar_series = compute_caar(ev_prepared)   # per-event-date signed AR\n"
        "print('caar:         ', caar(caar_series, forward_periods=5))\n"
        "print('BMP:          ', bmp_test(ev_prepared, forward_periods=5))\n"
        "print('event_hit_rate:', event_hit_rate(ev_prepared))\n"
        "print('corrado rank: ', corrado_rank_test(ev_prepared))\n"
        "print('clustering:   ', clustering_diagnostic(ev_prepared, cluster_window=3))"
    ),
    md(
        "## 4. Macro panel（跨國/小截面配置因子）\n"
        "\n"
        "訊號型態：連續值 + 小截面 `N < 30`。典型用法 = 跨國 CPI / 利差配置。\n"
        "Canonical test: `fm_beta_p`（Fama-MacBeth Newey-West t-test）。\n"
        "\n"
        "這裡另外生一份 25 資產的合成 panel，當「25 個 pseudo-country」，示範小截面情境下的 `MacroPanelConfig`。"
    ),
    md("### 4.1 Build small-N panel"),
    code(
        "small_raw = (\n"
        "    fl.datasets.make_cs_panel(n_assets=25, n_dates=500, seed=7)\n"
        "    .select(['date', 'asset_id', 'price'])\n"
        ")\n"
        "# factor = 相對 60 日均價偏離（mean-reversion proxy）\n"
        "small_panel = small_raw.with_columns(\n"
        "    (pl.col('price') / pl.col('price').rolling_mean(60).over('asset_id') - 1).alias('factor')\n"
        ").drop_nulls('factor')\n"
        "print('small-N panel:', small_panel['asset_id'].n_unique(), 'assets |', small_panel.height, 'rows')\n"
        "small_panel.head(3)"
    ),
    md("### 4.2 evaluate + `MacroPanelConfig`"),
    code(
        "mp_cfg = fl.MacroPanelConfig(\n"
        "    forward_periods=5,\n"
        "    n_groups=3,               # 小截面 → 少分組\n"
        "    demean_cross_section=False,\n"
        "    min_cross_section=10,\n"
        ")\n"
        "small_prepared = fl.preprocess(small_panel, config=mp_cfg)\n"
        "mp_profile = fl.evaluate(small_prepared, 'SmallRelValue', config=mp_cfg)\n"
        "\n"
        "print('verdict:              ', mp_profile.verdict())\n"
        "print('fm_beta_mean (λ):     ', f'{mp_profile.fm_beta_mean:+.5f}')\n"
        "print('fm_beta_tstat:        ', f'{mp_profile.fm_beta_tstat:+.2f}')\n"
        "print('fm_beta_p (canonical):', f'{mp_profile.fm_beta_p:.4f}')\n"
        "print('pooled_beta:          ', f'{mp_profile.pooled_beta:+.5f}  p={mp_profile.pooled_beta_p:.4f}')\n"
        "print('beta_sign_consistency:', f'{mp_profile.beta_sign_consistency:.2%}')\n"
        "print('quantile_spread:         ', f'{mp_profile.quantile_spread:+.4f}')\n"
        "print('median cross-section N:', mp_profile.median_cross_section_n)"
    ),
    md(
        "## 5. Macro common（全市場共用因子）\n"
        "\n"
        "訊號型態：單一時序，每個資產共用同一個 factor 值。典型用法 = VIX / 黃金 / USD index。\n"
        "Canonical test: `ts_beta_p`（per-asset 時序 OLS β 的截面 t-test）。\n"
        "\n"
        "範例：市場截面波動率（所有資產日報酬的 cross-sectional std）→ 對每檔股票的 exposure 穩定嗎？"
    ),
    md("### 5.1 Build market vol + broadcast 到每個資產"),
    code(
        "# 市場波動率（每日：跨資產日報酬 std）\n"
        "mkt_vol = (\n"
        "    raw_demo.sort(['asset_id', 'date'])\n"
        "    .with_columns(pl.col('price').pct_change().over('asset_id').alias('ret'))\n"
        "    .group_by('date').agg(pl.col('ret').std().alias('factor'))\n"
        "    .sort('date').drop_nulls('factor')\n"
        ")\n"
        "print('market vol head:')\n"
        "print(mkt_vol.head(3))\n"
        "\n"
        "# 為了控 runtime，sample 50 檔標的\n"
        "sample_assets = raw_demo.select('asset_id').unique().head(50)['asset_id'].to_list()\n"
        "common_df = (\n"
        "    raw_demo.filter(pl.col('asset_id').is_in(sample_assets))\n"
        "    .select(['date', 'asset_id', 'price'])\n"
        "    .join(mkt_vol, on='date', how='inner')\n"
        ")\n"
        "print('common panel shape:', common_df.shape)"
    ),
    md("### 5.2 evaluate + `MacroCommonConfig`"),
    code(
        "mc_cfg = fl.MacroCommonConfig(\n"
        "    forward_periods=5,\n"
        "    ts_window=60,\n"
        "    tradable=False,\n"
        ")\n"
        "common_prepared = fl.preprocess(common_df, config=mc_cfg)\n"
        "mc_profile = fl.evaluate(common_prepared, 'MktVol', config=mc_cfg)\n"
        "\n"
        "print('verdict:              ', mc_profile.verdict())\n"
        "print('n_assets:             ', mc_profile.n_assets)\n"
        "print('ts_beta_mean:         ', f'{mc_profile.ts_beta_mean:+.5f}')\n"
        "print('ts_beta_tstat:        ', f'{mc_profile.ts_beta_tstat:+.2f}')\n"
        "print('ts_beta_p (canonical):', f'{mc_profile.ts_beta_p:.4f}')\n"
        "print('mean R²:              ', f'{mc_profile.mean_r_squared:.4f}')\n"
        "print('beta_sign_consistency:', f'{mc_profile.ts_beta_sign_consistency:.2%}')"
    ),
    md(
        "## 6. 切換因子類別的 cheat sheet\n"
        "\n"
        "只要換 `factor_type=` 字串（或對應的 `XxxConfig`），同一組 `fl.evaluate` / "
        "`fl.evaluate_batch` / `ProfileSet` API 都能用——差別在回傳的 Profile dataclass "
        "欄位不同、canonical p 對應的統計量不同。"
    ),
    code(
        "cheat = pl.DataFrame({\n"
        "    'factor_type': ['cross_sectional', 'event_signal', 'macro_panel', 'macro_common'],\n"
        "    'signal_shape': ['每期每資產連續值', '離散 {-1,0,+1}', '連續值，小截面 N<30', '單一時序、全資產共用'],\n"
        "    'Config':       ['CrossSectionalConfig', 'EventConfig', 'MacroPanelConfig', 'MacroCommonConfig'],\n"
        "    'Profile':      ['CrossSectionalProfile', 'EventProfile', 'MacroPanelProfile', 'MacroCommonProfile'],\n"
        "    'canonical_p':  ['ic_p', 'caar_p', 'fm_beta_p', 'ts_beta_p'],\n"
        "    'core_question': ['排序能預測截面報酬差異嗎？', '事件後報酬有異常嗎？',\n"
        "                       '宏觀指標能預測跨資產配置嗎？', '資產對共同因子的 exposure 穩定嗎？'],\n"
        "})\n"
        "cheat"
    ),
    md(
        "---\n"
        "\n"
        "## 附錄 A：全 factor_type 收到的 Profile 一覽\n"
        "\n"
        "把本 notebook 跑出來的四個 profile 並排，展示不同 factor_type 的 verdict + canonical_p 是同一套介面。"
    ),
    code(
        "summary = pl.DataFrame([\n"
        "    {\n"
        "        'factor_type': 'cross_sectional',\n"
        "        'factor_name': profile.factor_name,\n"
        "        'verdict': profile.verdict(),\n"
        "        'canonical_p': profile.canonical_p,\n"
        "        'n_periods': profile.n_periods,\n"
        "    },\n"
        "    {\n"
        "        'factor_type': 'event_signal',\n"
        "        'factor_name': ev_profile.factor_name,\n"
        "        'verdict': ev_profile.verdict(),\n"
        "        'canonical_p': ev_profile.canonical_p,\n"
        "        'n_periods': ev_profile.n_periods,\n"
        "    },\n"
        "    {\n"
        "        'factor_type': 'macro_panel',\n"
        "        'factor_name': mp_profile.factor_name,\n"
        "        'verdict': mp_profile.verdict(),\n"
        "        'canonical_p': mp_profile.canonical_p,\n"
        "        'n_periods': mp_profile.n_periods,\n"
        "    },\n"
        "    {\n"
        "        'factor_type': 'macro_common',\n"
        "        'factor_name': mc_profile.factor_name,\n"
        "        'verdict': mc_profile.verdict(),\n"
        "        'canonical_p': mc_profile.canonical_p,\n"
        "        'n_periods': mc_profile.n_periods,\n"
        "    },\n"
        "])\n"
        "summary"
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "examples" / "demo.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
