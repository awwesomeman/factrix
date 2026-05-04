# factrix 指標適用範圍 & Fallback 對照

逐 Profile 欄位列 **N（資產數）** / **T（時序長度）** 下限，以及不達門檻時的實際行為。方便使用者在決定 data shape 時先查表、debug 看到 `NaN` 欄位時先對照。

> 本文職責：逐 Profile 欄位的 **N/T 門檻**與**不達門檻時的 fallback 語意**。**精確公式**不在這裡 — 請讀對應 `.py` module 的 docstring（`help(factrix.metrics.<name>)`）；論文依據請看 [statistical-methods.md](./statistical-methods.md)。文件整體分工見 [README.md](https://github.com/awwesomeman/factrix/blob/main/README.md)。
>
> **Authoritative source 仍是 code**（`metadata["reason"]`、metric 模組 docstring）。本表為使用者便捷參考，可能漂移；若有疑義以 `metadata["reason"]` 為準，或 `git blame` 對應 `MIN_*` 常數定義。

---

## 術語定義

本文和 README 用以下術語描述 factrix 對非典型輸入的反應；語意有嚴格區分：

| 術語 | 定義 | 典型場景 | 通知管道 |
|---|---|---|---|
| **raise** | 立即拋出 exception，停止執行 | 結構錯誤（wrong factor_type、N=1 on CS/MP）、config 缺漏 | `ValueError` / `TypeError` |
| **fallback** | 切到一條**語意不同但仍有意義**的替代計算路徑，Profile 欄位有值但解讀須調整 | `macro_common` N=1 切到單資產 OLS；`event_signal` N=1 讓 CAAR 退化成時間平均 | `diagnose()` info rule |
| **degraded** | 同 fallback，但**診斷欄位本身被停用**（Profile 欄位為 `None`）；計算仍跑，只是失去 cross-section 相關診斷 | `event_signal` N=1 → `clustering_hhi=None` | `diagnose()` info rule + Profile 欄位 `None` |
| **short-circuit** | 樣本不足、input 缺失等情況下，metric 不嘗試計算，直接回傳 `NaN` + `metadata["reason"]` | T < MIN_ASSETS_PER_DATE_IC、N < MIN_TS_OBS 的 asset 全部被 skip 等 | `MetricOutput.metadata["reason"]` + `insufficient_metrics` tuple + cross-type `data.insufficient` rule |
| **drop / skip** | 計算跑完，但某些 row / asset / date 被**靜默過濾**（例：per-date N<10 的 date 不進 IC 序列） | `compute_ic` 每日 N<10 drop；`compute_ts_betas` T<20 skip | `Artifacts.intermediates["coverage"]` 1-row summary |
| **short-circuit → canonical p=1.0** | short-circuit 時 canonical p 保守壓到 1.0 確保 `verdict()` FAILED | 所有 metric short-circuit | 隱含 canonical p；`verdict()` 輸出 `FAILED` |

**何時用哪個詞**：
- Profile 欄位**有值但語意變**了 → 「fallback」
- Profile 欄位**是 `None`** → 「degraded」
- Profile 欄位**是 `NaN`** → 「short-circuit」
- **資料某些 row 被丟** → 「drop / skip」
- **使用者的 API 呼叫被擋** → 「raise」

---

## 樣本量的三個維度（先看懂這個再查表）

談「樣本夠不夠」要先分清楚三種數量，不同 factor_type 的 canonical test 看到的「有效樣本」是不同組合：

| 維度 | 符號 | 意義 | factrix 欄位 |
|---|---|---|---|
| 資產數 | **N** | 每個 date 截面上有多少 asset | `asset_id` 的 unique 數 |
| 時序長度 | **T** | 總共有多少個 date | `date` 的 unique 數 |
| 事件數 | **K** | 稀疏事件類 factor 的非零觸發數 | `filter(factor != 0).height` |
| 非重疊時序 | **T/h** | T 除以 `forward_periods=h`，避免 overlap 重複計入 | `_sample_non_overlapping` 內部 |

**時間顆粒**（daily / weekly / monthly bar）不直接影響 factrix 的統計閾值 — 在 factrix 裡，`forward_periods` 是**行數**，不是日曆時間。週頻 panel 寫 `forward_periods=1` 就是 1 週前瞻報酬。但時間顆粒會透過 **T 的大小**和 **重疊結構**間接影響：日頻 1 年 ≈ T=250；月頻 10 年 ≈ T=120；同一個 MIN_ASSETS_PER_DATE_IC=10 在兩者下意義不同（日頻容易滿、月頻需要多年歷史）。

### 各 factor_type 的 canonical test 實際看什麼樣本

| factor_type | canonical test | 有效樣本 | 另一個維度的門檻 |
|---|---|---|---|
| `cross_sectional` | IC 非重疊 t-test | **T/h**（期數）| 每 date N ≥ 2 才有 rank；per-date N 太小會 drop 掉該期 |
| `macro_panel` | Fama-MacBeth λ 的 t-test | **T**（λ 序列長度，`MIN_FM_PERIODS=20`）| 每 date N ≥ 3 做 stage-1 OLS |
| `macro_common` (N≥2) | 跨資產 t-test on per-asset β | **N**（資產數）| 每 asset T ≥ `MIN_TS_OBS=20` 估 β |
| `macro_common` (N=1) | 單資產 OLS t-test（fallback，無 HAC 修正）| **T**（時序長度）| canonical p 保守壓到 1.0 |
| `event_signal` | CAAR 非重疊 t-test | **K/h**（非重疊事件數）| N 無下限；K ≥ `MIN_EVENTS=10` |

**常見誤會**：「我有 1000 筆資料」不等於「樣本夠」 — 要看是 1000 個 (date, asset) pair、還是 1000 個事件、還是 1000 個非重疊期。同樣 1000 筆，在不同 factor_type 下可能樣本很充足或嚴重不足。

### 總樣本量（N×T）什麼時候關鍵

- **`macro_panel` stage-1 OLS**：每期用 N 個 asset 做 regression，dof = N − 2；N = 3 是最低底線但結果不穩。N × T 規模決定整體推論的平均精度。
- **`cross_sectional` per-date IC**：per-date rank correlation 用 N 個 asset；N 小則單期 IC 噪音大，但多期平均（T）可以平掉。N × T 總量 < 1000 時 IC 時序 CI 變寬。
- **`macro_common` per-asset regression**：每個 asset 用 T 觀測，N × T 總量決定 ts_beta 分佈的穩定度。

### 最小可用組合（rule of thumb）

| 目標 | N | T | 總觀測 (N×T) | 備註 |
|---|---|---|---|---|
| `cross_sectional` 勉強可信 | ≥ 30 | ≥ 250 日 = 1 年（或 ≥ 24 月） | ≥ 7,500 | 日頻 1 年 / 月頻 2 年 |
| `cross_sectional` 建議門檻 | ≥ 100 | ≥ 500 日 = 2 年 | ≥ 50,000 | 業界常見標準 |
| `macro_panel` 勉強可信 | ≥ 5 | ≥ 24（滿足 MIN_FM_PERIODS 加 overlap 緩衝）| ≥ 120 | 跨國配置，月頻 2 年 |
| `macro_panel` 建議 | ≥ 10 | ≥ 60 | ≥ 600 | |
| `macro_common` (N≥2) 勉強 | ≥ 5 | ≥ 24 per asset | ≥ 120 | |
| `macro_common` (N=1) | 1 | ≥ 24 | ≥ 24 | 保守 verdict，改看 tstat |
| `event_signal` 勉強 | 任意 | — | K ≥ 10（K 才是關鍵）| 事件可跨 N 或跨 T 累積 |

**這些 rule of thumb 不是硬閾值** — factrix 不會在 N=29 時 raise，但你自己要知道 CI 會很寬。

---

## 全域 fallback 行為

- **Metric short-circuit**：算不出來時統一回傳
  - `value = NaN`
  - `metadata["reason"] = "insufficient_*"` 或其他具體原因
  - `metadata["p_value"] = 1.0`（保守預設，BHY 一定拒絕）
- **`describe_profile_values()`**：NaN 顯示為 `—`，使用者一眼看見哪些欄位 skip；底部附 diagnostic 計數提示（`veto`/`warn`/`info` 數），引導 `profile.diagnose()`
- **Verdict**：canonical p = 1.0 → verdict = `FAILED`（但這時 "FAILED" 的意義是「資料不足以判斷」而非「因子無效」）

### Fallback 通知管道對照

factrix 預設採**拉式**通知（user 主動呼叫 `.diagnose()` / 讀 `metadata` / 讀 Artifacts），不主動 `warnings.warn`。這和 `verdict()` + `diagnose()` 的整體哲學一致 — framework 偵測、user 決定。

| Fallback 類型 | 通知管道 | 如何取用 |
|---|---|---|
| **結構錯誤**：wrong factor_type、N=1 panel 餵 CS/MP | `raise ValueError` / `TypeError` | 不可避免；必須改 code |
| **語意退化**：N=1 event CAAR 退化成單資產時間平均；N=1 macro_common canonical p → 1.0 | `profile.diagnose()` 回傳 info-severity `Diagnostic` | `for d in profile.diagnose(): print(d)` |
| **Metric 層短路**：樣本不足（MIN_* 未達）→ 欄位 NaN | `MetricOutput.metadata["reason"]` + Profile 欄位 `insufficient_metrics`（tuple）+ cross-type rule `data.insufficient` 觸發 warn-severity diagnose | `profile.insufficient_metrics`；或 `artifacts.metric_outputs["ic"].metadata["reason"]` |
| **逐行 / 逐資產 drop**：`compute_ic` 丟 per-date N<10 的 date；`compute_ts_betas` skip T<20 的 asset | `artifacts.intermediates["coverage"]` 1-row DataFrame（欄位 `axis / n_total / n_kept / n_dropped / drop_reason`，CS 的 `axis='dates'`、MC 的 `axis='assets'`） | `art.get("coverage").row(0, named=True)` 讀對應數字 |
| **Factor-type 選錯提示**：`describe_profile_values` 底部顯示 diagnostic 計數 | stdout print | 視覺線索，非 programmatic |
| **既有 UserWarning**：`quantile_spread` 小 N per group、`redundancy_matrix` 退化、tie_ratio 過高等 | `warnings.warn(UserWarning)` | 標準 Python warnings 機制；`pytest.warns` 可捕捉 |

**設計原則**：
- **結構錯誤 raise**：不讓使用者 silently 得到錯誤結果
- **語意退化走 diagnose**：使用者需要時主動問；不噪
- **統計短路同時走 metadata 和 diagnose**：提供機器可讀欄位（metadata）+ 人類可讀彙總（diagnose 的 `data.insufficient` rule）
- **Silent drop 走 Artifacts intermediates**：合法使用情境（unbalanced panel）很常見；不適合 push；但使用者可以**主動查**
- **不做 `warnings.warn` 上 per-element 事件**：批次 evaluate 500 個 factor × 10 個 warning 會變成 5000 行 log 噪音

---

## 閾值常數

常數定義散落兩處：通用門檻在 `factrix/_types.py`；metric-specific 門檻在各 metric 模組。下表右欄標明每個常數的實際定義位置 — 以 code 為 authoritative。

| 常數 | 值 | 意義 | 定義位置 |
|---|---|---|---|
| `MIN_ASSETS_PER_DATE_IC` | 10 | IC 時序 t-test 的最小非重疊期數 | `factrix/_types.py` |
| `MIN_EVENTS` | 10 | CAAR / BMP / Corrado 最小事件數 | `factrix/_types.py` |
| `MIN_OOS_PERIODS` | 5 | OOS decay 每 split 最小期數 | `factrix/_types.py` |
| `MIN_PORTFOLIO_PERIODS` | 5 | quantile_spread 時序最小期數 | `factrix/_types.py` |
| `MIN_MONOTONICITY_PERIODS` | 5 | monotonicity 最小期數 | `factrix/_types.py` |
| `MIN_FM_PERIODS` | 20 | Fama-MacBeth λ 序列最小期數 | `factrix/metrics/fama_macbeth.py` |
| `MIN_TS_OBS` | 20 | per-asset TS regression 最小觀測數 | `factrix/metrics/ts_beta.py` |

---

## `cross_sectional`（`CrossSectionalProfile` 欄位）

**整體 N=1 panel**：`_validate_n_assets` 於 `build_artifacts` raise `ValueError`，訊息指向 `MacroCommonConfig`。

| 欄位 | N 下限 (per date) | T 下限 | 不達門檻時 |
|---|---|---|---|
| `ic_mean` / `ic_p` / `ic_nw_p` / `ic_ir` | ≥ 2 per date（rank 需要兩個以上 asset） | `MIN_ASSETS_PER_DATE_IC=10` 非重疊期 | NaN + p=1.0，`reason=insufficient_ic_periods` |
| `quantile_spread` / `spread_p` | ≥ `n_groups`（預設 10） | `MIN_PORTFOLIO_PERIODS=5` | NaN；per-date N < n_groups 時該期被 drop |
| `monotonicity` | ≥ `n_groups` | `MIN_MONOTONICITY_PERIODS=5` | NaN |
| `top_concentration` | ≥ `n_groups` | — | NaN 若 top bucket 空 |
| `turnover` (rank-stability: `1 − mean(Spearman ρ)`；**診斷用**，**不**入 cost 公式) | ≥ 2 per date（rank 需要 ≥ 2 asset） | ≥ `2·forward_periods + 1` raw dates | NaN, `reason=insufficient_dates` |
| `notional_turnover` (notional Q1/Q_n churn；**cost 公式唯一合法 driver**) | ≥ `n_groups` per date（否則 top / bot 可能為空 → 該期被 drop） | ≥ `forward_periods + 1` raw dates（等價於取樣後 ≥ 2 個 rebalance 配對）| NaN, `reason=insufficient_dates` \| `no_valid_pairs` |
| `breakeven_cost` / `net_spread` | 繼承 `notional_turnover` 門檻 | 同 | `inf` 若 notional_turnover → 0；NaN 若 notional_turnover 短路 |
| `ic_trend` | 同 `ic_*` | 同 IC | NaN |
| `oos_survival_ratio` / `oos_sign_flipped` | 同 IC | T 可分 2 splits × `MIN_OOS_PERIODS=5` | NaN |
| `regime_ic` *(opt-in)* | 同 IC | 每 regime ≥ `MIN_ASSETS_PER_DATE_IC` | 失守 regime 略過；metadata 保留成功 regime |
| `multi_horizon_ic` *(opt-in)* | 需 `price` 欄位 | 每 horizon ≥ `MIN_ASSETS_PER_DATE_IC` × 步長 | 長 horizon 單獨失守時該欄 NaN |
| `spanning_alpha` *(opt-in)* | — | 基底 spreads T 需對齊 factor spread | NaN |

### 注意事項
- Low-cardinality factor（binary / bucketed）：`tie_policy='ordinal'` 可能讓結果受 sort order 干擾；超過門檻會 `UserWarning` 提示切 `tie_policy='average'`
- `ortho` 不在 `_fl_preprocess_sig`，允許 "preprocess 一次 / evaluate 時 sweep basis" 的使用模式；代價是使用者需自保 preprocess cfg 和 evaluate cfg 的 ortho 是同一個（否則 silently 應用 evaluate 時的 ortho）
- **`turnover` vs `notional_turnover` 選用**：把 `breakeven_cost` / `net_spread` 當**實盤 bps 閾值**在 dashboard / filter 上比時，一律以 `notional_turnover` 為單位（middle-rank shuffle 不計成本、尾部 churn 才算）。`turnover` 只適合用來做「哪個因子的排序比較穩」這類 factor-ranking，不宜直接喂給成本公式——詳見 `statistical_methods.md` 的 Turnover & Trading-Cost Proxy 段落與兩個指標的 bias 方向說明
- **Diagnostic rule**：`notional_turnover > 0.5` 時，`profile.diagnose()` 會吐出 `cs.high_notional_turnover` (CrossSectional) 或 `macro_panel.high_notional_turnover` (MacroPanel)，severity = `warn`。要在 ProfileSet 大批 filter 時用 `diagnose_all()` 篩 code 即可，不必自己寫 threshold 判斷。

---

## `event_signal`（`EventProfile` 欄位）

| 欄位 | N | 事件數 | T 下限 | 不達門檻時 |
|---|---|---|---|---|
| `caar_mean` / `caar_p` (canonical) | 無 (N=1 可用) | `MIN_EVENTS=10` | — | NaN + p=1.0 |
| `bmp_p` | N 無硬下限；SE 為 event-wise SAR 的 std (跨事件)，故 N=1 仍可算 | `MIN_EVENTS` | — | NaN 若事件數不足。**N=1 時 BMP 喪失原本對「cross-sectional event-induced variance inflation」的防禦（沒有 cross-section），數值上退化成類似單資產 mean-adjusted CAAR 的 t-test** |
| `corrado_rank_p` | 無 | `MIN_EVENTS` | — | NaN |
| `event_hit_rate` / `hit_rate_p` | 無 | `MIN_EVENTS` | — | NaN |
| `profit_factor` / `event_skewness` | 無 | 非零事件數 > 1 | — | NaN |
| **`clustering_hhi`** / `clustering_hhi_normalized` | **N = 1 → 自動停用**，Profile 欄位 = `None` | — | — | `None`（不是 NaN，型別是 `float \| None`）|
| `event_ic` / `event_ic_p` | — | `MIN_EVENTS` 且 `\|factor\|` 須有變化 | — | `None` 若 magnitude 全同 |
| `signal_density` | 無 | 無 | — | 純資訊指標，總是可算 |
| `caar_trend` / `oos_survival_ratio` | 同 caar | T 可分 2 splits × `MIN_OOS_PERIODS` | — | NaN |

### N=1 語意變化（要注意）
- **CAAR 從「跨資產平均異常報酬」退化成「單資產時間平均異常報酬」** — 數值還是可算，但你失去了「事件在不同資產上表現一致」這層證據
- 單資產下，同一 asset 不同時點的事件常有自相關（例：earnings pre-announcement leak）— 這層風險目前 factrix 不檢測

---

## `macro_panel`（`MacroPanelProfile` 欄位）

**整體 N=1 panel**：`_validate_n_assets` 於 `build_artifacts` raise `ValueError`，訊息指向 `MacroCommonConfig`。

**小 N 典型範圍**：2 ≤ N < 30（跨國 / 跨資產類別）

| 欄位 | N (per date) | T 下限 | 不達門檻時 |
|---|---|---|---|
| `fm_beta_mean` / `fm_beta_p` / `fm_beta_tstat` (canonical) | ≥ 3 per date（stage-1 OLS 需 2 係數 + ≥1 dof） | λ 序列長度 ≥ `MIN_FM_PERIODS=20` | NaN + p=1.0，`reason=insufficient_fm_periods` |
| `pooled_beta` / `pooled_p` | — | N × T dof ≥ 基本門檻 | NaN |
| `fm_pooled_sign_consistency` | 兩者皆需可算 | 同上 | NaN |
| `beta_trend` | 同 FM | 同 FM | NaN |
| `oos_survival_ratio` / `oos_sign_flipped` | 同 FM | T 可分 2 splits × `MIN_OOS_PERIODS` | NaN |

### 小 N 的 HAC lag 警示
`fama_macbeth` 使用 Newey-West `lags = max(⌊T^(1/3)⌋, forward_periods − 1)`。當 T (λ 序列長度) 接近 `3 × lags` 時，HAC 估計邊際可靠；實務上 T < 24 + `forward_periods − 1` 建議雙讀 `fm_beta_p` 與 `pooled_p`，正負號不一致時觸發 `macro_panel.fm_pooled_sign_mismatch` veto rule。

---

## `macro_common`（`MacroCommonProfile` 欄位）

**支援 N=1**：語意是「單一時序因子在一支資產上的 predictive power」。詳見下方「N=1 行為」。

| 欄位 | N | T (per asset) | N=1 行為 |
|---|---|---|---|
| `ts_beta_mean` / `ts_beta_p` / `ts_beta_tstat` (canonical) | — | `MIN_TS_OBS=20` | **Fallback → 單資產 OLS t-test（plain SE，無 HAC）；`ts_beta_p` 保守壓到 1.0**（見下方）|
| `mean_r_squared` | — | 同上 | N=1 時為該單一資產的 TS 回歸 R² |
| `ts_beta_sign_consistency` | **≥ 2** | 同上 | N=1 → NaN（沒有 sign 分佈可比）|
| `oos_survival_ratio` / `oos_sign_flipped` | — | T 可分 2 splits × `MIN_OOS_PERIODS` | N=1 仍可算（純時序 split）|
| `beta_trend` | — | 同 T 門檻 | N=1 仍可算 |
| `factor_adf_p` | — | 單時序長度 ≥ ~10 meaningful | 高值（> 0.10）→ Stambaugh bias flag |

### N=1 的保守 fallback 設計與解讀
底層切到 `ts_beta_single_asset_fallback`：
1. 對該單一資產做**plain OLS** TS regression 算 β 與 t-stat（SE 為 `sqrt(σ² · (X'X)⁻¹)`，**未做 HAC 修正**）
2. **canonical `ts_beta_p` 保守地壓到 1.0** — 為避免 N=1 時使用者誤把單資產顯著誤讀成跨資產 premium
3. `ts_beta_tstat` 欄位仍保留實際 t-stat 值 — 供使用者自己讀
4. `diagnose()` 發出 `macro_common.single_asset` info 提示退化路徑

**SE 警示**：`ts_beta_tstat` 的分母是 plain OLS SE，**對報酬的 autocorrelation 和 heteroskedasticity 沒有防禦**。嚴格推論（尤其單序列 vol clustering 明顯時）建議外接 `arch` package 跑 GARCH-adjusted SE 或 wild bootstrap。factrix 內部不內建 HAC 修正 — Stambaugh bias 的根源是 predictor persistence，HAC 只修 SE 不修 bias，內建會給使用者 false confidence。

**使用者心態建議**：
- 別用 `verdict() == "PASS"` 判斷 — 永遠會 FAILED
- 讀 `ts_beta_tstat` 和 `factor_adf_p` 的實際值
- 若 `factor_adf_p > 0.10` 同時 `|ts_beta_tstat| > 2`：Stambaugh bias 高度懷疑，建議外接 IVX (`arch` package) 或 Campbell-Yogo Q-test

---

## 跨類通用

| 情境 | 統一行為 |
|---|---|
| Metric 短路 | NaN + `metadata["reason"]` + `metadata["p_value"]=1.0` |
| BHY FDR 控制 | `P_VALUE_FIELDS` 白名單限制；短路欄位 p=1.0 自然拒絕 |
| `verdict()` 門檻 | 預設 `t=2.0`；`2.0 ≤ \|t\| < 3.0` 時若 warn-level diagnostic 命名替代 p-source，升級為 `PASS_WITH_WARNINGS` |
| `describe_profile_values()` | NaN → `—`；`None` → `—`；skips 顯而易見 |

---

## `redundancy_matrix` 的適用範圍

`redundancy_matrix` 是**診斷工具**（量化 factor 之間的重疊程度），**不是 combiner**（不產生合成信號）。輸入是 `ProfileSet`（保證同一 factor_type），輸出是 |Spearman ρ| 矩陣。

兩個方法在不同 factor_type 下的適用情況：

| method | `cross_sectional` | `macro_panel` | `event_signal` | `macro_common` |
|---|---|---|---|---|
| **`factor_rank`**（per-date 資產間 rank 相關）| 主要使用情境；語意最清楚 | 可用（小 panel 較噪）| 可用（但大多為 0 事件稀疏）| **拒絕（raise ValueError）** — factor 跨資產恆定，ranking 全部 tied 屬退化 |
| **`value_series`**（Profile 值序列 Spearman 相關）| IC 序列相關 | λ 序列相關 | CAAR 序列相關 | per-asset β 序列相關；支援且語意清楚 |

**怎麼選 method**：
- 問「這些 factor 會選到一樣的股票嗎」→ `factor_rank`
- 問「這些 factor 績效在時序上一起動嗎」→ `value_series`
- `macro_common` 只能 `value_series`（hard-coded guard）
- Compact mode artifacts 只能 `value_series`（`factor_rank` 需 prepared panel；code 會自動降級並發 `UserWarning`）

---

## `ts_quantile_spread` / `ts_asymmetry` 的適用範圍

兩個 **standalone diagnostic**（不進 registry、不影響 `verdict()`）— 補位 `(COMMON, CONTINUOUS, *)` cell 的 OLS β 假設線性 + 對稱所漏掉的 shape：

- `ts_quantile_spread`：factor 自己歷史分桶 → 桶間條件期望檢定（top-bottom Wald）+ Spearman 單調性 diagnostic。抓 U-shape / inverted-U / extreme-only。
- `ts_asymmetry`：factor 正負兩側 response 是否對稱（method A 條件期望、method B 分段斜率）。抓 long-side ≠ short-side。

兩者皆 OLS + NW HAC + Wald，與 `ts_beta_t_nw` 對齊；**不**用 Welch t（iid 假設在 overlapping forward return 下破裂）。

### 適用性（gates）

| Gate | 條件 | 影響 |
|------|------|------|
| **A**. distinct 數量 | `n_unique(factor) ≥ n_groups × 2` | `ts_quantile_spread` 是否可跑 |
| **B**. 雙側存在 | `(factor>0).any() and (factor<0).any()` | `ts_asymmetry` 兩個方法的最低門檻 |
| **C**. 雙側內變異 | `n_unique(factor[factor>0]) ≥ 2 and n_unique(factor[factor<0]) ≥ 2` | `ts_asymmetry` method B 的額外條件 |

### 適用性矩陣

| 你手上的資料 | `ts_quantile_spread` | `ts_asymmetry` |
|---|---|---|
| **COMMON × CONTINUOUS, PANEL** (N≥2) | ✓ 收成 `equal_weight` per-date（`metadata["aggregation"]` 記錄）| ✓ |
| **COMMON × CONTINUOUS, TIMESERIES** (N=1) | ✓ | ✓ |
| **INDIVIDUAL × CONTINUOUS** (任何 mode) | ✗ guard → 改用 cross-sectional `quantile_spread` | ✗ → `quantile_spread.long_alpha / short_alpha` |
| **(\*, SPARSE / binary / ternary)** | ✗ Gate A → `event_quality.*` | ✗ Gate B → `event_hit_rate` |
| **factor 全 ≥ 0 或全 ≤ 0** | ✓ | ✗ Gate B（無雙側）→ 整個 metric 短路，reason 寫入 `metadata["reason"]`（method 不分 A/B，兩個都不出）；Gate C 失敗才是「method A 出、method B 不出」，由 `metadata["method_b_skipped"]` 記錄 |

任一 gate 失敗 → `MetricOutput` 帶 `metadata["reason"]` + redirect hint，**不 silently 回 NaN**（與 §跨類通用 §Metric 短路 慣例一致）。

### API policy

- `n_groups: int = 5` default、user override、`n_periods // n_groups < 5` soft warning、`n_periods < MIN_PORTFOLIO_PERIODS` hard short-circuit。**不** auto-adjust。
- v1 PANEL 收法固定 `equal_weight`；`value_weight` / `factor_weight` 後續 issue。

設計細節 / Wald regression 公式 / 為何不用 Welch → 見 [issue #5](https://github.com/awwesomeman/factrix/issues/5) 與兩個 module docstring。

---

## 常見 decision tree（給 debug 用）

```
我的 Profile 顯示 canonical p = 1.0 → FAILED
  │
  ├─ describe_profile_values 看到 canonical 欄位顯示 "—"？
  │    └─ YES → 資料不足，查 metadata["reason"]
  │
  └─ 有數字但 p=1.0？
       └─ 可能是 N=1 macro_common 保守 fallback
           └─ 讀 ts_beta_tstat / factor_adf_p 實際值
```

```
我懷疑 factor_type 選錯了
  │
  ├─ N=1？
  │    ├─ 共用時序（VIX、DXY）      → macro_common（保守 fallback，canonical p=1.0）
  │    ├─ 稀疏事件（單標的 earnings）→ event_signal（clustering_hhi 自停用，CAAR 語意改）
  │    └─ 連續因子（單標的 P/E-like）→ README coverage gap（目前無 first-class 支援）
  ├─ 2 ≤ N < 30？ → macro_panel（連續）或 event_signal（稀疏）
  ├─ N ≥ 30？ → cross_sectional（連續）或 event_signal（稀疏）
  └─ 看不出訊號 geometry？ → README「怎麼選 factor_type」2D 表
```

---

## 版本同步提示

本表若與實際 metric 行為不符，**請以 code 行為為準**。更新路徑：
1. 改 `_types.py` 的通用 `MIN_*` 常數、或 `metrics/fama_macbeth.py` / `metrics/ts_beta.py` 內的 metric-specific 常數 → 更新本文上方「閾值常數」表（含定義位置欄）
2. 改 `from_artifacts` 的 fallback 邏輯 → 更新對應 factor_type 區塊
3. 新增 metric → 在對應 factor_type 區塊加一行（N 下限 / T 下限 / fallback）

長期計劃：把本表資訊 programmatic 化進 `fl.describe_profile()`，避免手動同步漂移。
