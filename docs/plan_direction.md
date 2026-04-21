# factorlib 策略方向與下一步

> 日期：2026-04-21
> 用途：避免後續再重新走過同樣的討論；作為未來 refactor 決策的 anchor。
>
> **範圍註記**：本檔是 `factor-analysis` workspace 的內部工作文件，不構成 factorlib 套件的對外文件。下方若干 `../../docs/...` 連結指向 parent repo 的研究素材；獨立 clone factorlib submodule 的使用者不會看到那些檔案，但核心 factorlib 決策（scope、Path B'、未定項目）獨立於 parent repo。

---

## 1. 脈絡

### 1.1 原目標宣言（專案願景）
> 讓 quant 輕鬆分析不同因子類別 — 使用者側要一致 / 易用 / 可擴充 / 統計正確，開發者側要一致 / 可擴充 / 正確。預期可以找出用於單資產交易的因子、選股策略因子等等。

### 1.2 實際工作優先級（真正的 driver）
| 優先級 | 場景 | 性質 |
|---|---|---|
| **P1（急）** | 資產配置研究（廣義，不止區域股市配置） | 工作主軸 |
| **P2（長期）** | 單資產交易策略開發 | 個人主軸；factorlib 原初動機 |
| P3（潛在） | 選股策略 | 不急；factorlib 目前最成熟 |
| P4（潛在） | 套利策略 | 工具類別不同（cointegration-driven），factorlib 不適合 |
| P5（期待） | HFT | 明確 out-of-scope |

### 1.3 關鍵錯位
factorlib 目前投資最深的 `cross_sectional`（P3）是**最不急**的場景；P1/P2 支援**不到 30%**。

---

## 2. 核心診斷（多輪 review 收斂）

### 2.1 統計面（資深 quant 視角）
- **N=1 不是 first-class**：僅 `macro_common` 有顯式 fallback，`cross_sectional` / `macro_panel` 單資產時 silently 短路成 FAILED（訊息是「樣本不足」但真因是「選錯 type」）。
- **單資產 canonical 設計錯誤**：`ts_beta_single_asset_fallback` 把 p 壓到 1.0，uninformative 而非 conservative。正確應走 HAC-corrected predictive regression β t-test。
- **`MIN_IC_PERIODS=10` 語意重載**：同一常數同時當「時間軸樣本數 gate」和「per-date 資產數」使用，該拆成 `MIN_IC_PERIODS_T` / `MIN_CS_UNIVERSE`。
- **`verdict()` t=2.0 門檻**：單因子 discovery 在 Harvey-Liu-Zhu (2016) 下應 t≥3；現有 `PASS_WITH_WARNINGS` 機制可用來在 `2.0 ≤ |t| < 3.0` 發出警告。
- **Effect size 不是一等公民**：Profile 輸出以 `canonical_p` 為中心，effect size 散落各欄位。**這是 BL integration 的直接 blocker** — BL 的 `(P, Q, Ω)` 需要 effect magnitude + SE，不是 p-value。

### 2.2 架構面（資深後端視角）
- **Dispatch 寫在 7 個地方**：`_api.FACTOR_TYPES`、`preprocess/pipeline.py`、`preprocess_sig`、`build_artifacts`、`_FACTOR_REGISTRY`、`diagnose_profile` 等。extensibility 4/10。
- **`_fl_preprocess_sig` 三個 silent-wrongness 洞**：
  - (a) `config=None` 默認成 `CrossSectionalConfig()` — event 資料 silently CS-preprocess
  - (b) sig 不 key on `factor_type`
  - (c) `ortho` basis 在 `build_artifacts` 才算 — preprocess / evaluate 間切換 ortho 會 silently 產出不同 Profile
- **`P_VALUE_FIELDS` 無 invariant check**：欄位重命名會造成 BHY runtime `KeyError`，而非 class-def 期報錯。
- **Config / Profile 對稱性已裂**：`CrossSectionalConfig` 7+ knobs vs `MacroCommonConfig` 2 knobs；`ic_trend` / `beta_trend` / `caar_trend` 不同 stem；OOS 欄位在 4 個 Profile 各複製一份。

---

## 3. 最終決策

### 3.1 路徑決策：Path B'
**factorlib 維持為 signal validator，不吸收 allocation / strategy 方法論。**

```
Input: 價量 + 宏觀 + 估值
  ↓
[1] factorlib — signal 有效性驗證 (canonical p + effect size + diagnose)
  ↓
[2] ML signal layer — xgboost + shap（下游，不是 factorlib 職責）
  ↓
[3] Black-Litterman / view integration — PyPortfolioOpt 或未來 factorlib-allocation
  ↓
[4] Optimizer — skfolio / cvxpy / riskfolio-lib
  ↓
Output: 資產權重
```

**明確：factorlib 只做 [1]。** [2]–[4] 生態已有成熟工具，重造輪子的 ROI 為負。

### 3.2 實施哲學：research-first
- P1 研究 deadline 真實存在；2 個月 infrastructure refactor 是奢侈品。
- 現有 factorlib 核心邏輯對 P1 正常路徑統計上夠正確（`macro_panel` / `macro_common`）。
- 具體 refactor 需求要靠實際研究的 friction 驗證，不靠 speculation。
- v0.x 階段 API 破壞自由仍在，晚點重構不會更貴。

### 3.3 對使用者研究素材的 framing 修正
`../../docs/study/regional_equity_allocation.md` section 六「開發建議路徑」目前的 framing 是 Path A（把 MVO / HRP / BL 納入 factorlib）— **應改寫為 Path B' framing**：factorlib 只負責 signal validation，allocation 交下游。此修改建議留到第一個 P1 研究跑完後再做（那時會更清楚介面該長什麼樣）。

---

## 4. 立即行動：Safety Minimum（約 1 週 / 15–20 小時）

這三項是 **silent-wrongness fix**，不做會污染研究結論。**不是 refactor，是 bug fix。**

### 4.1 修 `_fl_preprocess_sig` 三個洞
- ✅ **(a) 完成**：`preprocess()` 在 `config=None` 時改成 raise TypeError，列出四個 config 選項。
- ✅ **(b) 完成**：`factor_type` 納入 `preprocess_sig` 第一個 key；既有的 diff 邏輯自動捕捉跨類 mismatch。
- 🟡 **(c) 延後 / 不做**：ortho fingerprint 進 sig 會破壞 legitimate sweep pattern（`prepared = preprocess(df, cfg_base); for basis in bases: evaluate(prepared, ..., config=replace(cfg_base, ortho=basis))`）。silent-wrongness 風險是 scenario 1（cfg 複製漏同步）；sweep 是 scenario 2（跨 basis 比較）。為了保住 scenario 2，暫不改。若後續 friction log 出現 scenario 1 被踩 ≥2 次，再評估改用「build_artifacts 時 stamp 第二 marker」的方案。README 會補充「使用同一個 cfg 實例」的文化守則。

### 4.2 `factor_type` mismatch 明確 raise
- ✅ **完成**：`factorlib/_validators.py` 新 module 提供 `validate_n_assets`；從 `preprocess_cs_factor` / `preprocess_macro_panel`（fail fast）和 `build_artifacts`（backstop）雙層呼叫。
- 涵蓋兩種退化：
  - (a) 全域 N=1：單資產 panel 餵 CS/MP
  - (b) staggered schedule：全域 N≥2 但 max per-date n_unique < 2 (CS) / < 3 (MP) — 每個 date 都會被 metric 層短路，原本悄悄產生空 IC / λ
- 錯誤訊息指向 `MacroCommonConfig` 與 `docs/plan_direction.md §7 待定決策`。

### 4.3 `P_VALUE_FIELDS` invariant test
- ✅ **完成**：`tests/test_profile_invariants.py` 13 個 parametrized 測試。
  - `CANONICAL_P_FIELD ⊂ P_VALUE_FIELDS`
  - 每個 whitelist 欄位在 Profile dataclass 中存在且 annotation 為 `PValue`（或 `PValue | None`）
  - per-type rules 和 `CROSS_TYPE_RULES` 的 `recommended_p_source` 都在對應 whitelist
  - 包括 `_CUSTOM_RULES`（執行時註冊的）

### 4.4 後續 review 補丁（subagent review 後）
- ✅ `redundancy_matrix(method='factor_rank')` 在 `macro_common` 加 runtime guard（原本 silent 返回 0 矩陣；現 raise ValueError）
- ✅ metric_applicability.md 修正：
  - `macro_common` N=1 fallback 從誤稱 "Newey-West" 改為「plain OLS」+ HAC 警語
  - 「閾值常數」加「定義位置」欄，修正原本「單一 source of truth: `_types.py`」誤導（`MIN_FM_PERIODS` / `MIN_TS_OBS` 實際在 metric 模組）
- ✅ 此 plan doc 從 parent repo 搬進 factorlib submodule（原本內部連結斷）
- ✅ N=1 guard 強化：新 `factorlib/_validators.py`；除全域 N=1 外補 staggered schedule 偵測（max per-date n_unique < 2 for CS / < 3 for MP）；同步從 `preprocess_cs_factor` / `preprocess_macro_panel` 呼叫（fail fast），`build_artifacts` 保留 backstop

### 4.5 Fallback 一致性最小修補（agent review Class D/E）
採用最小修補而非全框架重構（α/β/γ 都不是；走「現有 `diagnose()` + `metadata` + `Artifacts` 三管道分工」）：
- ✅ 新增 `event.single_asset` info-severity rule — 補齊 event_signal N=1 在 diagnose 的 asymmetry（原本只有 `macro_common.single_asset`）；用 `clustering_hhi is None` 作 N=1 signal
- ✅ `Artifacts.intermediates["coverage"]` — `_build_cs_artifacts` 記錄 `n_dates_total / n_dates_kept / n_dates_dropped_low_n`；`_build_macro_common_artifacts` 記錄 `n_assets_total / n_assets_kept / n_assets_skipped_short_t`。合法使用情境（unbalanced panel）常見，不適合 push，但使用者可以主動查
- ✅ `describe_profile_values()` 底部加 diagnostic 計數提示（`N veto, N warn, N info — call .diagnose() for details`），有 fire 才出；不污染 clean fast-path
- ✅ metric_applicability.md 加「Fallback 通知管道對照」表 — raise / diagnose / metadata / Artifacts summary / UserWarning 五種管道各自職責

**明確不做**（保留給 friction log 證偽）：
- Per-element silent drop 不加 `warnings.warn`（批次 evaluate 會噪）
- 不做全面 "hybrid γ" 框架（複雜度無實證需求）
- `n_assets` 欄位不新加入 `EventProfile`（用 `clustering_hhi is None` 當 proxy 夠精確）

---

## 5. 明確不做（至少 Phase 1 研究完成前）

| 項目 | 不做的理由 |
|---|---|
| Registry dispatch (`FactorTypeSpec` + `SPEC_REGISTRY`) | 短期只修一次 silent-wrongness，不新增 factor_type；7 處 dispatch 不會痛 |
| `SingleAsset` 第 5 個 FactorType | 現有 `macro_common` N=1 fallback 在 Safety minimum 後對 P2 初期夠用；真的痛再加 |
| `effect_size` 升格 first-class 欄位 | 第一個 BL adapter 寫時再評估；現在從 Profile 手挖欄位成本低 |
| `factorlib-allocation` 新 repo | speculation-driven；等研究 friction log 累積再抽 |
| Regime conditioning 方法論 | HMM / threshold 的實作屬下游 allocation layer，不是 validator |
| `scipy` 加入 core dep | 無具體痛點；現有自刻夠用 |
| MVO / HRP / BL 進 factorlib | 生態工具成熟（`skfolio` / `PyPortfolioOpt`）；定位漂移 |
| Structural break / GARCH SE / IVX | 外接 `ruptures` / `arch`；lean-dep 原則保留 |
| ML signal layer（xgboost + shap） | 不是 factorlib 職責；allocation layer 或 notebook |
| Backtest engine | README 已明確 non-goal |

---

## 6. Friction Log 協議

### 6.1 檔案位置
`../../docs/friction_log.md`（位於 parent repo factor-analysis，不隨 factorlib 發行）

### 6.2 Entry 格式
```markdown
## YYYY-MM-DD
- 任務脈絡：當下在做什麼研究
- 痛點：具體遇到什麼不順
- workaround：這次怎麼繞過去
- [tag: short_tag_for_grouping]
```

### 6.3 Review 節奏
- 每 4–8 週 review 一次
- 或累計 10+ entries 時 review

### 6.4 決策規則
| Tag 出現次數 | 行動 |
|---|---|
| ≥ 3 次 | 值得做成 library feature；進 roadmap |
| 1–2 次 | notebook util function 或放 `factorlib-allocation` 候選 |
| 0 次（本計劃預測的痛點沒發生） | 從 roadmap 刪除，Speculation 被證偽 |

---

## 7. 待定決策（Safety minimum + 1 個 P1 研究跑完後重審）

這些是目前 subagent 建議但**尚未 commit** 的項目。research phase 結束後回來決定：

1. **`SingleAsset` 是否成為第 5 個 FactorType**？或繼續用 `macro_common` N=1 fallback？
   - 決策信號：friction log 中 `n1_dispatch` / `single_asset_*` tag 出現頻率。
2. **`effect_size` 是否升格為 Profile 一等欄位**？
   - 決策信號：寫第一個 BL adapter 時的摩擦程度。
3. **`factorlib-allocation` 是否開 repo**？
   - 決策信號：notebook 中 allocation 相關 util 是否重複 3 次以上。
4. **`FactorTypeSpec` registry 是否做**？
   - 決策信號：有沒有第二次需要新增 factor_type（例如為了 P1 才發現要一個 `cross_asset_common`）。
5. **`scipy` 是否加入 core dep**？
   - 決策信號：下一個要加的統計方法是否涉及重造 scipy 既有功能（如 bootstrap CI、特殊分佈）。
6. **`../../docs/study/regional_equity_allocation.md` section 六是否改寫**？
   - 決策信號：第一個 P1 研究跑完後，實際用到的 pipeline 形狀。

---

## 8. 生態位置定位（給未來參考）

| 層級 | 工具 | 職責 | 備註 |
|---|---|---|---|
| Signal validation | **factorlib** | canonical p + effect size + diagnose | 本 repo 專責此層 |
| Signal combination | `xgboost` + `shap` | 多因子合成 expected return + 可解釋性 | 見 `study/regional_equity_allocation_ml.md` |
| View integration | `PyPortfolioOpt.BlackLittermanModel` | ML view + 市場先驗 → posterior μ | 或未來 `factorlib-allocation` 薄 wrapper |
| Optimizer | `skfolio` / `cvxpy` / `riskfolio-lib` | 解 w* | 不自己重造 |
| Regime detection | `hmmlearn` / 自寫 threshold | state classifier | 放 allocation layer |
| Structural break | `ruptures` | 斷點偵測 | 不進 factorlib core |
| ARCH / GARCH SE | `arch` | vol clustering 推論 | 不進 factorlib core |
| Backtest glue | `vectorbt` / `bt` / 自寫 | 執行模擬 | 未來 `factorlib-strategy` 考慮薄 adapter |

---

## 9. 參考

### 9.1 關鍵對話輸入
- Senior-quant subagent review A（第一輪）：findings on N=1 silent degeneracy、MIN constant overload、verdict threshold。
- Senior-backend subagent review A（第一輪）：7-site dispatch、`P_VALUE_FIELDS` invariant、preprocess_sig holes。
- Senior-quant subagent review B（第二輪，更新目標後）：`Hypothesis` abstraction 建議、methodology fit audit、rename `macro_common` → `time_series_predictor` 建議、scipy 放寬建議。
- Senior-backend subagent review B（第二輪）：monorepo + 多 package 統一版本建議、`FactorTypeSpec` 設計、5 個 FactorType（vs 雙軸）、測試 minimum viable set。

### 9.2 研究素材
- `../../docs/study/regional_equity_allocation.md` — 區域股市配置業界實務與學術方法論。
- `../../docs/study/regional_equity_allocation_ml.md` — ML 可解釋性在區域配置的應用。
- （注意：section 六「開發建議路徑」目前 framing 偏 Path A，待改寫）

### 9.3 factorlib 現況關鍵檔案
- `factorlib/_types.py` — `FactorType` enum、`MIN_*` 常數、`Verdict` / `Diagnostic` 型別
- `factorlib/_api.py` — 公開 API 表面（dispatch 集中處之一）
- `factorlib/evaluation/profiles/` — 四個 Profile class
- `factorlib/evaluation/diagnostics/_rules.py` — 診斷規則定義
- `factorlib/preprocess/pipeline.py` — `_fl_preprocess_sig` 三個洞的所在
- `factorlib/metrics/` — canonical test 實作
- `pyproject.toml` — 依賴管理

---

## 10. 當前狀態 / 下一步

- [ ] Safety minimum 4.1：修 `_fl_preprocess_sig` 三個洞
- [ ] Safety minimum 4.2：`factor_type` mismatch 明確 raise
- [ ] Safety minimum 4.3：`P_VALUE_FIELDS` invariant test
- [ ] 建立 `../../docs/friction_log.md` 空檔（parent repo）
- [ ] 開跑第一個 P1 研究（候選：跨國 CAPE + Momentum 驗證，對應 parent repo [`docs/study/regional_equity_allocation.md`](../../docs/study/regional_equity_allocation.md) 第五節）
- [ ] 4–8 週後回來 review friction log，重審第 7 節待定決策

---

## Meta-notes（給未來的自己）

1. **Scope 克制原則**：每次考慮加新 feature 前問三題 — (a) 讓 factor validation 做得更好嗎？(b) 需要新 dependency 嗎？(c) 在補既有承諾還是開新承諾？三題都過才做。
2. **Research-first 不等於 infrastructure-never**：friction log 的意義是讓 speculation 被 empirical signal 取代，而非拒絕所有抽象。
3. **不要再被 subagent 的 ambitious refactor plan 拉偏**：solo maintainer + 工作 deadline 下，subagent 的「2 個月 refactor」常常是 greenfield 思維，要有意識過濾。
4. **factorlib 的護城河是 canonical p + diagnose 的 statistical discipline**，不是 feature 數量。保住這個護城河比擴大範圍重要。
