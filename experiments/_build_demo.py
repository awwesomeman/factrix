"""Generator for experiments/demo.ipynb — run once, then discard.

Produces a clean notebook showing every factorlib API level across all four
factor_types against the bundled TW daily panel.
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
        "- 資料：`tw_stock_daily_2017_2025.parquet`（TW 全市場日線），切出 2023-01–2024-06 當 demo 視窗。"
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
    code(
        "# 載入原始 TW 日線，用 fl.adapt 把 user 欄位名對應到 factorlib canonical names\n"
        "# (canonical = date / asset_id / price)\n"
        "raw = pl.read_parquet('../tw_stock_daily_2017_2025.parquet')\n"
        "raw_demo = fl.adapt(raw, date='date', asset_id='ticker', price='close_adj')\n"
        "\n"
        "# validation schema 要求 date 為 Datetime(ms)，原始是 pl.Date → 轉型\n"
        "raw_demo = raw_demo.with_columns(pl.col('date').cast(pl.Datetime('ms')))\n"
        "\n"
        "# 切出 2023-01 ~ 2024-06 demo 視窗（控制 runtime）\n"
        "raw_demo = raw_demo.filter(\n"
        "    (pl.col('date') >= pl.datetime(2023, 1, 1))\n"
        "    & (pl.col('date') < pl.datetime(2024, 7, 1))\n"
        ")\n"
        "print('rows:', raw_demo.height, '| assets:', raw_demo['asset_id'].n_unique())\n"
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
        "一行跑完 preprocess + all metrics + diagnostics，回傳 `CrossSectionalProfile`（frozen dataclass）。"
    ),
    code(
        "profile = fl.evaluate(mom20, 'Mom_20D', factor_type='cross_sectional')\n"
        "\n"
        "print('type:       ', type(profile).__name__)\n"
        "print('verdict:    ', profile.verdict())\n"
        "print('canonical_p:', f'{profile.canonical_p:.4f}')\n"
        "print('ic_mean:    ', f'{profile.ic_mean:+.4f}')\n"
        "print('ic_tstat:   ', f'{profile.ic_tstat:+.2f}')\n"
        "print('ic_ir:      ', f'{profile.ic_ir:+.3f}')\n"
        "print('q1_q5 spread:', f'{profile.q1_q5_spread:+.4f}')\n"
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
        "    q_top=0.2,                     # Q1 取前 20%\n"
        "    orthogonalize=False,\n"
        "    mad_n=3.0,\n"
        "    return_clip_pct=(0.01, 0.99),\n"
        "    estimated_cost_bps=30,\n"
        ")\n"
        "profile_10d = fl.evaluate(mom20, 'Mom_20D_h10', config=cfg)\n"
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
        "for name, out in [('ic', ic_m), ('ic_ir', ir_m), ('q1_q5_spread', spread_m), ('monotonicity', mono_m)]:\n"
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
        "ps = fl.evaluate_batch(factors_map, factor_type='cross_sectional')\n"
        "print('ProfileSet size:', len(ps), '| profile class:', ps.profile_cls.__name__)\n"
        "ps.to_polars().select(['factor_name', 'ic_mean', 'ic_ir', 'q1_q5_spread', 'canonical_p'])  # quick glance\n"
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
        "# redundancy_matrix 需要 per-factor Artifacts；evaluate_batch 不保留，\n"
        "# 要自己 loop fl.evaluate 再 build_artifacts 並收集。\n"
        "from factorlib.evaluation.pipeline import build_artifacts\n"
        "\n"
        "arts: dict[str, object] = {}\n"
        "for name, fdf in factors_map.items():\n"
        "    prep = fl.preprocess(fdf, config=fl.CrossSectionalConfig())\n"
        "    a = build_artifacts(prep, fl.CrossSectionalConfig())\n"
        "    a.factor_name = name\n"
        "    arts[name] = a\n"
        "\n"
        "redund = fl.redundancy_matrix(ps, method='value_series', artifacts=arts)\n"
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
        "ev_profile = fl.evaluate(event_df, 'GoldenCross', config=ev_cfg)\n"
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
        "由於 TW 資料只有單一市場，這裡用 **TWSE 產業分類** 當 pseudo-country（N≈30）。"
    ),
    md("### 4.1 Build industry-aggregated panel"),
    code(
        "industry = (\n"
        "    raw_demo.group_by(['date', 'twse_ind'])\n"
        "    .agg(pl.col('price').mean())\n"
        "    .rename({'twse_ind': 'asset_id'})\n"
        "    .drop_nulls(['asset_id'])\n"
        "    .sort(['asset_id', 'date'])\n"
        ")\n"
        "# factor = 產業相對 60 日均價偏離（簡單的 mean-reversion proxy）\n"
        "industry = industry.with_columns(\n"
        "    (pl.col('price') / pl.col('price').rolling_mean(60).over('asset_id') - 1).alias('factor')\n"
        ").drop_nulls('factor')\n"
        "print('industry N:', industry['asset_id'].n_unique())\n"
        "industry.head(3)"
    ),
    md("### 4.2 evaluate + `MacroPanelConfig`"),
    code(
        "mp_cfg = fl.MacroPanelConfig(\n"
        "    forward_periods=5,\n"
        "    n_groups=3,               # 小截面 → 少分組\n"
        "    demean_cross_section=False,\n"
        "    min_cross_section=10,\n"
        ")\n"
        "mp_profile = fl.evaluate(industry, 'IndRelValue', config=mp_cfg)\n"
        "\n"
        "print('verdict:              ', mp_profile.verdict())\n"
        "print('fm_beta_mean (λ):     ', f'{mp_profile.fm_beta_mean:+.5f}')\n"
        "print('fm_beta_tstat:        ', f'{mp_profile.fm_beta_tstat:+.2f}')\n"
        "print('fm_beta_p (canonical):', f'{mp_profile.fm_beta_p:.4f}')\n"
        "print('pooled_beta:          ', f'{mp_profile.pooled_beta:+.5f}  p={mp_profile.pooled_beta_p:.4f}')\n"
        "print('beta_sign_consistency:', f'{mp_profile.beta_sign_consistency:.2%}')\n"
        "print('q1_q5_spread:         ', f'{mp_profile.q1_q5_spread:+.4f}')\n"
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
        "mc_profile = fl.evaluate(common_df, 'MktVol', config=mc_cfg)\n"
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
    out = Path(__file__).parent / "demo.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
