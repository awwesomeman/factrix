# 文獻出處與採用論點

> 本文件整理 FQS v2/v3 設計過程中參考的所有學術論文與業界文獻，按主題分類，
> 並標注每篇文獻在本框架中被採用的具體論點。

---

## 目錄

1. [評分架構設計](#1-評分架構設計)
2. [反對 Composite Score / 反對統一不同因子類型](#2-反對-composite-score--反對統一不同因子類型)
3. [統計門檻與多重比較](#3-統計門檻與多重比較)
4. [Factor Zoo 與因子複製](#4-factor-zoo-與因子複製)
5. [因子篩選、收縮與增量 Alpha](#5-因子篩選收縮與增量-alpha)
6. [因子集中度與容量](#6-因子集中度與容量)
7. [因子評估指標](#7-因子評估指標)
8. [事件研究方法論](#8-事件研究方法論)
9. [小截面因子評估](#9-小截面因子評估) — pseudo-replication、panel random effects、Fama-MacBeth
10. [權重理論與 MCDM](#10-權重理論與-mcdm)
11. [業界實務](#11-業界實務)
12. [跨市場因子評估](#12-跨市場因子評估) — 含宏觀因子訊號評估方法論
13. [統計穩健性與交易成本](#13-統計穩健性與交易成本)

---

## 1. 評分架構設計

### Cochrane (2011)

- **出處：** Cochrane, J. H. (2011). "Presidential Address: Discount Rates." *Journal of Finance* 66(4), 1047-1108.
- **採用論點：** 因子評估的根本結構可拆為 time-series predictability（時序可預測性）與 cross-sectional variation（截面差異）兩個正交維度。所有資產類別共享相同的評估框架。
- **在本框架的應用：** 支持以「根本問題」（訊號是否真實、能否獲利）而非「計算方法」（IC、portfolio return）定義維度，使框架跨因子類型通用。

### Berkin & Swedroe (2016)

- **出處：** Berkin, A. L. & Swedroe, L. E. (2016). *Your Complete Guide to Factor-Based Investing.* BAM Alliance Press.
- **採用論點：** 因子五標準——Persistent（持續）、Pervasive（普遍）、Robust（穩健）、Investable（可投資）、Intuitive（可解釋）。前三者對應 Reliability，Investable 對應 Profitability，Intuitive 為定性判斷不適合量化打分。
- **在本框架的應用：** 確認 2 維度（Reliability / Profitability）的分類能涵蓋主流因子評估標準中可量化的部分。

### Arnott, Harvey & Markowitz (2019)

- **出處：** Arnott, R. D., Harvey, C. R. & Markowitz, H. M. (2019). "A Backtesting Protocol in the Era of Machine Learning." *Journal of Financial Data Science* 1(1), 64-74. ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654))
- **採用論點：** 提出七類回測規範（Research Motivation、Multiple Testing、Data Issues、Cross-Validation、Model Dynamics、Model Complexity、Research Culture），其中 **Model Complexity** 明確建議優先選擇簡單結構與正則化，過度複雜的模型在 OOS 表現通常更差。
- **在本框架的應用：** 支持「以最小複雜度達成統計可靠評分」的核心原則；反對在框架中同時引入過多新機制（熵權+分位數+MC+雙層正交 = 複雜度超支）。

### Macrosynergy (2024)

- **出處：** Macrosynergy. "How to Measure the Quality of a Trading Signal." ([連結](https://macrosynergy.com/research/how-to-measure-the-quality-of-a-trading-signal/))
- **採用論點：** 跨資產訊號評估三層框架——(1) Accuracy（準確度：balanced accuracy、Kendall correlation）、(2) Risk-adjusted performance（風險調整績效：naive Sharpe、Sortino）、(3) Panel structure（面板結構：跨截面 vs 跨時間的均值）。明確處理 panel/macro 訊號與截面股票訊號的差異。
- **在本框架的應用：** 支持以通用問題（而非特定計算方法）定義維度的設計；確認框架需要能處理不同 panel 結構（cross-section vs time-series）的訊號。

---

## 2. 反對 Composite Score / 反對統一不同因子類型

### Goodhart (1984)

- **出處：** Goodhart, C. A. E. (1984). "Problems of Monetary Management: The U.K. Experience." In *Monetary Theory and Practice: The U.K. Experience*. Macmillan, pp. 91-121.
- **採用論點：** Goodhart's Law："When a measure becomes a target, it ceases to be a good measure." 原為貨幣政策語境，廣泛引用於量化金融的指標設計討論。當 composite score 成為因子生成器的最佳化目標時，會產出刷分因子——形式上各指標達標但無真正經濟意義。
- **在本框架的應用：** factrix 放棄 composite score 的核心理由之一。採用 `verdict()` pass/fail 設計，評估程序是篩選工具而非最佳化目標。

### MacKinlay (1997)

- **出處：** MacKinlay, A. C. (1997). "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1), 13-39.
- **採用論點：** 事件研究（event study）是一套完整獨立的方法論體系，包含 CAR（累積異常報酬）、BHAR（買入持有異常報酬）、calendar-time portfolio 等方法。其統計框架（事件窗口、估計窗口、異常報酬定義、截面聚合）與截面因子評估（IC、quantile sort、factor portfolio）是完全不同的統計問題。
- **在本框架的應用：** factrix 評估框架聚焦於截面個股因子的核心依據。事件訊號應使用 event study 方法論獨立評估，不應強行套入截面因子的評估程序。兩者共享基礎設施（MLflow、Dashboard UI）但不共享評估邏輯。

---

## 3. 統計門檻與多重比較

### Benjamini & Hochberg (1995)

- **出處：** Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B* 57(1), 289-300.
- **採用論點：** 提出 False Discovery Rate (FDR) 概念及 BH step-up procedure。FDR 控制的是「被宣稱顯著的假設中，false discovery 的期望比例」，而非 Bonferroni 的「至少出現一個 false positive 的機率」(FWER)。BH 比 Bonferroni 寬鬆但仍提供嚴格的統計保證，適用於大規模同時檢定（如數百因子篩選）。**BH procedure 的輸入是 p-values，與產生 p-value 的檢定類型無關**（t 檢定、z 檢定、bootstrap 等均可混用）。
- **在本框架的應用：** `bhy()` / `multiple_testing_correct()` 的理論基礎。BHY（下條）是 BH 的推廣，但核心演算法相同。factrix 的實作使用 p-values 作為輸入（而非 t-stats），確保不同 factor type 的不同檢定類型可以統一處理。

### Benjamini & Yekutieli (2001)

- **出處：** Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *Annals of Statistics* 29(4), 1165-1188.
- **採用論點：** 將 BH procedure 推廣到相依檢定（dependent tests）。因子之間通常高度相關（如不同 lookback 的動能因子），違反 BH 的獨立性假設。BHY 加入校正因子 `c(m) = Σ(1/i)`，在任意依賴結構下控制 FDR。代價是門檻更保守（約為 BH 的 1/ln(m) 倍）。
- **在本框架的應用：** `_stats.py` 中 `bhy_threshold()` 和新增的 `bhy_adjust()` 均使用 BHY correction（含 `c(m)` 校正因子），因為因子間相關結構未知且通常為正相關。

### Harvey, Liu & Zhu (2016)

- **出處：** Harvey, C. R., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies* 29(1), 5-68.
- **採用論點：** 整理 300+ 已發表因子，指出傳統 t > 2.0 門檻在大量多重檢定下不足。建議新因子的 t-stat 應 > 3.0（Bonferroni、Holm、BHY 校正）。隨因子數量增長，門檻應進一步提高。
- **重要情境限定：** Harvey 的 t > 3.0 建議是針對「整個學術圈歷史上累積做過的所有因子檢定」，其中檢定總數未知且受 publication bias 影響（只有顯著結果會被發表）。**這與單次篩選 session 中明確知道做了 N 次檢定的情境不同。** 在已知 N 的情境下，BHY 可精確計算門檻，不需要 Harvey 的 t > 3.0 啟發式近似。
- **在本框架的應用：** `bhy()` 使用 BHY 精確校正（而非固定 t > 3.0），因為篩選 session 中 N 已知。Harvey 的 t > 3.0 保留作為文獻參考，不作為 factrix 的門檻。

### Harvey (2017)

- **出處：** Harvey, C. R. (2017). "Presidential Address: The Scientific Outlook in Financial Economics." *Journal of Finance* 72(4), 1399-1440. ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2893930))
- **採用論點：** 提出回測檢查清單：需有經濟動機、OOS 測試、跨市場穩健性檢驗、交易成本考量。建議使用最小貝式因子（minimum Bayes factor）作為 p-value 的替代。
- **在本框架的應用：** 支持多層過濾設計（`oos_survival_ratio` veto rule、`verdict()` t-stat 門檻）——不依賴單一指標，而是多層過濾。

### Chordia, Goyal & Saretto (2020)

- **出處：** Chordia, T., Goyal, A. & Saretto, A. (2020). "Anomalies and False Rejections." *Review of Financial Studies* 33(5), 2134-2179.
- **採用論點：** 確認多重檢定問題的嚴重性；實證顯示 BHY 校正比 Bonferroni 更適當（後者假設檢定間獨立，但因子高度相關）。**關鍵實證發現：將 BHY 套用在 2000+ anomalies 上，真正強的因子（t > 3-4）幾乎都通過 BHY 校正。** BHY 主要淘汰的是 marginally significant (t ≈ 2.0-2.5) 的因子——而這些恰好是最可能是 false discovery 的族群。
- **在本框架的應用：** `compare(bhy=True)` 的實證支持——BHY 不會「殺死好因子」，它淘汰的是「剛好及格」的因子。使用者若對某個被 BHY 淘汰的因子有獨立的經濟學理由，可以忽略 BHY 結果單獨用 OOS 資料驗證。

### Harvey & Liu (2020)

- **出處：** Harvey, C. R. & Liu, Y. (2020). "Lucky Factors." *Journal of Financial Economics* 137(1), 116-142.
- **採用論點：** Bayesian multiple testing framework，自然整合因子間相關結構。考慮 multiple testing 後，大量「t > 2.0」的因子不再顯著。比 Bonferroni/BHY 更適合因子間有相關性的場景。
- **在本框架的應用：** `bhy()` / `multiple_testing_correct()` 連續校正的理論依據——從固定門檻升級為連續校正時，本文論證了為什麼離散階梯門檻不夠原則性。詳見 [§13](#13-統計穩健性與交易成本)。

### de Prado (2018)

- **出處：** de Prado, M. L. (2018). *Advances in Financial Machine Learning.* Wiley. Chapter 12: "Backtesting through Cross-Validation."
- **採用論點：** 提出 Combinatorial Purged Cross-Validation (CPCV) 處理金融資料的 train/test split 問題。單一 train/test 切分對切分點位置脆弱——多組 purged splits 降低對切分點落在何處（例如 COVID、GFC）的敏感度。適用於 OOS decay 評估。
- **在本框架的應用：** 支持 `oos_survival_ratio` 多切分設計——使用 3+ 個切分點（而非單一 80/20）取中位數，降低 regime-change 敏感度。

---

## 4. Factor Zoo 與因子複製

### Hou, Xue & Zhang (2020)

- **出處：** Hou, K., Xue, C. & Zhang, L. (2020). "Replicating Anomalies." *Review of Financial Studies* 33(5), 2019-2133.
- **採用論點：** 嘗試複製 452 個已發表異常因子，約 65% 在統一方法論下複製失敗。主因：(1) 小型股主導——市值加權後效果消失、(2) 退市報酬處理不當、(3) 前視偏誤。
- **在本框架的應用：** 支持前處理正交化（Step 6）去除 Size/Industry 曝露的設計——因子集中於小型股或特定產業時，正交化後 alpha 若消失代表因子無獨立價值。

### McLean & Pontiff (2016)

- **出處：** McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71(1), 5-32.
- **採用論點：** 研究 97 個異常因子的樣本內 vs 樣本外表現。平均 OOS 衰減約 32%（發表後溢價降低）。約 58% 的 IS 報酬在 OOS 存活。發表在頂級期刊的因子衰減更多（更多套利資金流入）。
- **在本框架的應用：** OOS_Decay 閘門門檻設為 0.5（比平均衰減 32% 更寬鬆），僅過濾嚴重 overfit 的因子。OOS_Decay 同時作為 Reliability 維度的評分指標和 VETO 閘門。

### Jensen, Kelly & Pedersen (2023)

- **出處：** Jensen, T. I., Kelly, B. T. & Pedersen, L. H. (2023). "Is There a Replication Crisis in Finance?" *Journal of Finance* 78(5), 2465-2518.
- **採用論點：** 重新檢驗約 150 個因子，發現約 85% 在使用原始方法論時可複製，但大多數被其他因子涵蓋（subsumed）。**多重檢定問題是真實的，但沒有 Harvey (2016) 宣稱的那麼嚴重**——大多數因子在 OOS 和跨市場資料中仍然 replicate。具有強經濟學動機（economic motivation）的因子複製率更高。
- **在本框架的應用：** 支持 `redundancy_matrix()` 的設計——因子個別有效但彼此冗餘是比 false discovery 更常見的問題。同時支持 `bhy()` 作為可選工具（資訊性）而非硬截斷（攔截性）的設計決策：多重檢定問題是真的，但不應過度保守地攔截所有 marginally significant 因子。

### Green, Hand & Zhang (2017)

- **出處：** Green, J., Hand, J. R. M. & Zhang, X. F. (2017). "The Characteristics that Provide Independent Information about Average US Monthly Stock Returns." *Review of Financial Studies* 30(4), 1389-1436.
- **採用論點：** 檢驗哪些公司特徵提供獨立的截面預測力。聯合檢定下僅 12-15 個特徵存活（大量特徵是冗餘的）。
- **在本框架的應用：** 強化 Incremental Alpha 的必要性——即使 Stage 1 選出高分因子，聯合檢定後可能只有少數有獨立價值。

### Chen & Zimmermann (2022)

- **出處：** Chen, A. Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional Asset Pricing." *Critical Finance Review* 11(2), 207-264. ([網站](https://www.openassetpricing.com/); [GitHub](https://github.com/OpenSourceAP/CrossSection))
- **採用論點：** 建立 319 個因子的標準化可複製資料庫。評估標準極簡但有效：(1) long-short portfolio t-stat、(2) monotonicity check、(3) IS vs OOS 比較。複製品質高：reproduced t-stats 對原始的回歸斜率 0.88-0.90，R² = 82-83%。
- **在本框架的應用：** 確認本框架的核心指標選擇（t-stat + monotonicity + OOS decay）與學術標準一致。Chen & Zimmermann 刻意不評估 turnover/capacity/decay——這些是實務面的補充，對應本框架的 Profitability 維度。

---

## 5. 因子篩選、收縮與增量 Alpha

### Kozak, Nagel & Santosh (2020)

- **出處：** Kozak, S., Nagel, S. & Santosh, S. (2020). "Shrinking the Cross-Section." *Journal of Financial Economics* 135(2), 271-292.
- **採用論點：** 使用貝式收縮估計隨機折現因子（SDF）。核心洞見：個別因子溢價難以精確估計（寬信賴區間），但由多因子構建的 SDF 估計相對穩定。因子的價值在於對 SDF 的**邊際貢獻**，非獨立表現。
- **在本框架的應用：** 支持 Stage 2 的設計邏輯——單因子品質（Stage 1）和邊際貢獻（Stage 2）是兩個獨立問題，不應混合成一個加權分數。

### Feng, Giglio & Xiu (2020)

- **出處：** Feng, G., Giglio, S. & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance* 75(3), 1327-1370.
- **採用論點：** 使用 double-selection LASSO 判定因子是否有增量 alpha。發現大量因子在控制已知因子後是冗餘的。提出 model-selection 框架取代逐一因子檢定。
- **在本框架的應用：** 支持 Stage 2 Incremental Alpha 的 spanning test 設計；確認「因子間比較」不應混入「因子個體評分」。

### Barillas & Shanken (2017)

- **出處：** Barillas, F. & Shanken, J. (2017). "Which Alpha?" *Review of Financial Studies* 30(4), 1316-1338.
- **採用論點：** 新因子有用 ⟺ 對現有因子集的 spanning regression alpha ≠ 0。增量 Sharpe ratio：`ΔSR² = α² / σ²(ε)`。
- **在本框架的應用：** Stage 2 Incremental Alpha 的理論基礎。FQS 作為品質門檻篩入合格池，池內按 spanning regression α 排序。

### Barillas & Shanken (2018)

- **出處：** Barillas, F. & Shanken, J. (2018). "Comparing Asset Pricing Models." *Journal of Finance* 73(2), 715-754.
- **採用論點：** 提出 squared Sharpe ratio 比較框架——擴展模型的 SR² 減去基準模型的 SR² 應為正值，等價於新因子 α ≠ 0。
- **在本框架的應用：** 提供 Incremental Alpha 的替代衡量方式（ΔSR²），可作為未來 Stage 2 的進階指標。

### Gibbons, Ross & Shanken (1989)

- **出處：** Gibbons, M. R., Ross, S. A. & Shanken, J. (1989). "A Test of the Efficiency of a Given Portfolio." *Econometrica* 57(5), 1121-1152.
- **採用論點：** GRS 檢定——測試一組 test asset 的 alpha 是否聯合為零。用於評估因子模型是否充分定價截面報酬。
- **在本框架的應用：** 可作為 Stage 2 的多因子聯合檢定工具，驗證選入因子集是否存在遺漏的定價資訊。

---

## 6. 因子集中度與容量

### Hou, Xue & Zhang (2020)（同 §4，集中度面向）

- **額外採用論點：** 約 65% 的因子在市值加權後消失，主因是因子集中於小型股或特定產業。這是「類型曝露集中」的典型案例——因子的 alpha 本質上來自 Size 或 Industry 曝露。
- **在本框架的應用：** 支持 Step 6 前處理正交化去除已知因子曝露的設計。正交化後 alpha 若消失，代表因子是已知因子的偽裝。

### Israel & Moskowitz (2013)（同 §11，容量面向）

- **額外採用論點：** 系統性分析因子的**容量（capacity）**限制。許多因子的 alpha 集中在小型股和做空端——前者流動性有限（衝擊成本高），後者有借券限制。因子容量決定了實務可部署的資金規模。
- **在本框架的應用：** 支持 Effective_Breadth 代理指標的設計——衡量 Q1 內訊號的分散度（HHI 倒數），作為容量的代理。因子品質（quality）與因子容量（capacity）是不同問題：Reliability + Profitability 評估品質，Effective_Breadth 補充容量資訊。

### 集中度問題的三層防線設計邏輯

本框架將集中度問題分為三種類型，各由不同機制處理：

1. **類型曝露集中**（Q1 集中在某產業/市值區間）→ Step 6 正交化治病因
2. **過擬合集中**（AI 因子意外挑到 IS 期間特定標的）→ OOS_Decay VETO
3. **個股集中 / 容量限制**（alpha 確實只存在少數標的，OOS 也有效）→ Effective_Breadth 代理指標

前兩種是因子品質問題（false positive），第三種是因子容量問題（true positive but low capacity）。正交化和 OOS_Decay 無法處理第三種，因為因子是「真的」——它只是覆蓋面窄。Effective_Breadth 填補此盲點。

---

## 7. 因子評估指標

### Grinold & Kahn (2000)

- **出處：** Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- **採用論點：** 主動管理基本定律：`IR = IC × √Breadth`。IC_IR（Information Coefficient 的 Information Ratio）為核心訊號品質指標。ICIR > 0.5 為「有趣」的經驗門檻。
- **在本框架的應用：** IC_IR 作為 Reliability 維度的核心指標之一；IC_IR 的 score map `[0.1, 0.6]` 基於此文獻。同時暗示小截面（低 Breadth）下即使 IC 高，IR 仍可能低。

### Patton & Timmermann (2010)

- **出處：** Patton, A. J. & Timmermann, A. (2010). "Monotonicity in Asset Returns: New Tests with Applications to the Term Structure, the CAPM, and Portfolio Sorts." *Journal of Financial Economics* 98(3), 605-625. ([PDF](https://public.econ.duke.edu/~ap172/Patton_Timmermann_sorts_JFE_Dec2010.pdf); [R Package](https://cran.r-project.org/web/packages/monotonicity/))
- **採用論點：** 提出正式的單調性（MR）檢定框架。以嚴格單調性為備擇假設（非虛無假設）。支持最多 15 組；小 N 建議使用 all-pairs 版本（相較 adjacent-pairs 有更高 power）。Bootstrap 處理序列相關。
- **在本框架的應用：** Monotonicity 指標的理論基礎。台股 ~2000 檔使用 5 組（quintile），符合文獻建議。

### Novy-Marx & Velikov (2016)

- **出處：** Novy-Marx, R. & Velikov, M. (2016). "A Taxonomy of Anomalies and Their Trading Costs." *Review of Financial Studies* 29(1), 104-147.
- **採用論點：** 提出損益兩平交易成本公式：`Breakeven Cost = Gross Alpha / (2 × Turnover)`。實證顯示許多高頻因子在扣除交易成本後不賺錢。Turnover 應與 alpha 結合評估，而非單獨衡量。
- **在本框架的應用：** Breakeven Cost 取代 v1 單獨的 Turnover（rank autocorrelation）指標。Breakeven Cost 的 score map `[10 bps, 100 bps]` 參考此文獻。

---

## 8. 事件研究方法論

### MacKinlay (1997)（同 §2）

- **額外說明：** 事件研究方法論的經典參考。CAR（累積異常報酬）計算流程：(1) 定義事件窗口與估計窗口、(2) 以市場模型估計正常報酬、(3) 計算異常報酬 AR = R - E(R)、(4) 跨事件聚合為 CAAR。此方法論與截面因子評估（IC、quantile sort）是平行的方法體系，不應混用。
- **在本框架的應用：** 未來 `scoring/event_study.py` 獨立模組的方法論依據。

---

## 9. 小截面因子評估

### Pseudo-Replication 與 Panel Random Effects

#### Macrosynergy (2024) — Testing Macro Trading Factors

- **出處：** Macrosynergy. "Testing Macro Trading Factors." ([連結](https://research.macrosynergy.com/testing-macro-trading-factors/))
- **採用論點：** 跨國 panel（如 10-20 國指數 × T 期）的因子顯著性檢定不能用簡單 pooled regression——同一期的觀測共享全球衝擊，假設獨立會造成 **pseudo-replication**，t-stat 膨脹 2-3 倍。解法：**period-specific random effects** panel regression，自動在 pooled 和 fixed-effect 之間取最適權重。實證顯示 CPI 與報酬的負向關係在 random effects 下 p-value 僅 87-88%，遠低於 pooled regression 的「高度顯著」。
- **在本框架的應用：** 小 N 場景（國家指數、區域因子）下的 IC t-test 等效於 pooled regression，存在 pseudo-replication 風險。未來如支援 macro 因子評估，應以 panel random effects 取代 IC t-test 作為顯著性檢定。

#### Macrosynergy (2024) — Panel Regression with JPMaQS

- **出處：** Macrosynergy. "Panel Regression with JPMaQS." ([連結](https://macrosynergy.com/academy/notebooks/panel-regression-with-jpmaqs/))
- **採用論點：** 提供 Python 實作範例，示範如何用 `LinearMixedModel`（`statsmodels` 或 `macrosynergy` 套件）估計 period random effects。關鍵步驟：(1) 構建 panel DataFrame（entity × time）、(2) 指定 random effect group = time period、(3) 對比 pooled OLS vs RE 的 coefficient significance。
- **在本框架的應用：** `factrix/metrics/fama_macbeth.py` 的 macro 因子評估路徑實作，此文件提供直接可參考的實作模式。

### Fama-MacBeth 在小 N 下的適用性

#### Fama & MacBeth (1973)

- **出處：** Fama, E. F. & MacBeth, J. D. (1973). "Risk, Return, and Equilibrium: Empirical Tests." *Journal of Political Economy* 81(3), 607-636.
- **採用論點：** 每期跑截面回歸得到 β_t，再對 {β_1, ..., β_T} 做時序 t 檢定。**顯著性取決於 T（時序期數），不是 N（截面資產數）。** 即使每期只有 10 個國家，只要 T > 60，β 的時序均值仍可做有效推論。
- **在本框架的應用：** Fama-MacBeth β 是 IC 的參數版本。小 N 時 β 比 rank IC 更有效率（利用量級資訊，不只排序）。未來 macro 因子路徑可以 Fama-MacBeth β 取代 Spearman IC。

#### Petersen (2009)

- **出處：** Petersen, M. A. (2009). "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches." *Review of Financial Studies* 22(1), 435-480.
- **採用論點：** Fama-MacBeth 標準誤在存在截面相關時是正確的（time clustering），但在存在時序自相關時低估（serial correlation in β_t）。解法：Newey-West 調整後的 t-stat。實證比較 OLS、clustered SE、Fama-MacBeth、fixed effects 在不同 panel 結構下的表現。
- **在本框架的應用：** 任何 Fama-MacBeth 實作都應使用 Newey-West SE，否則時序自相關會膨脹 t-stat——和 IC 的 non-overlapping sampling 動機一致。

### 因子強度與截面大小

### Giglio, Xiu & Zhang (2025)

- **出處：** Giglio, S., Xiu, D. & Zhang, D. (2025). "Test Assets and Weak Factors." *Journal of Finance*. ([NBER WP](https://www.nber.org/papers/w29002); [BFI WP](https://bfi.uchicago.edu/wp-content/uploads/2021/07/BFI_WP_2021-79.pdf))
- **採用論點：** **因子強度不是因子本身的固有屬性，而是截面大小的函數。** 同一個流動性因子在 size/value-sorted portfolios 中可能是「弱因子」，在 liquidity-sorted portfolios 中是「強因子」。小截面（N=10-30）下標準 IC 和 quantile-spread 方法 noise 極大。建議使用 targeted portfolio sorts 和 Supervised PCA。
- **在本框架的應用：** 支持 global_macro / group_region 因子類型在 IC_IR 和 Monotonicity 上自動降權的設計——adaptive sigmoid 會因 t-stat 低而降權，而非強制排除這些指標。同時說明為何不同 factor_type 可以共享維度但需要不同的指標路由。

### IC 在小截面的限制

- **出處：** Kolm, P. N. & Ritter, G. (2020). "On the Bayesian interpretation of Black–Litterman." *European Journal of Operational Research* 280(2), 564-572. 及相關 IC 文獻。([arXiv 參考](https://arxiv.org/pdf/2010.08601))
- **採用論點：** IC 在少量資產時不可靠——「sample size is important」。IC 值本身「small in magnitude and volatile across time」，在小 universe 中作為績效衡量指標效果差。Fundamental Law（IR = IC × √Breadth）說明低 Breadth 下高 IC 仍產生低 IR。
- **在本框架的應用：** global_macro（N~10-30）和 group_region（N~10-30）以 Hit_Rate 和 OOS_Decay 為主要 Reliability 指標，IC_IR 作為輔助參考。

---

## 10. 權重理論與 MCDM

### DeMiguel, Garlappi & Uppal (2009)

- **出處：** DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies* 22(5), 1915-1953.
- **採用論點：** 在 14 個資料集的實證中，1/N 等權策略在 OOS Sharpe ratio、CEQ return、turnover 等指標上常勝過均值-變異數最佳化等 14 種策略。原因：估計誤差（estimation error）> 最佳化增益（optimization gain），尤其在參數多、樣本少的情境。
- **在本框架的應用：** 維度權重固定 50/50 等權的理論基礎。不使用最佳化（predictive regression）或自適應（熵權法）來決定維度權重，因為 2 維度 + 有限因子樣本下，estimation error 會 dominate。

### Mukhametzyanov (2021)

- **出處：** Mukhametzyanov, I. Z. (2021). "Specific Character of Objective Methods for Determining Weights of Criteria in MCDM Problems: Entropy, CRITIC, SD." *Decision Making: Applications in Management and Engineering* 4(2), 76-105. ([連結](https://dmame-journal.org/index.php/dmame/article/view/194))
- **採用論點：** 純客觀權重法（Entropy、CRITIC、SD）在 MCDM 中的限制：(1) 熵權法不考慮指標間相關性——相關指標會膨脹權重、(2) 小樣本下熵估計不穩定、(3) 低變異指標權重趨零（by design，但可能排除重要但穩定的指標）。建議混合主觀+客觀方法。
- **在本框架的應用：** 確認不在 Stage 1 使用熵權法的決策——2 維度下只有 2 個值算熵，統計意義不足，且存在上述限制。等權更穩健。

---

## 11. 業界實務

### Tulchinsky (2019) — WorldQuant

- **出處：** Tulchinsky, I. (2019). *Finding Alphas: A Quantitative Approach to Building Trading Strategies.* Wiley. 另參考 WorldQuant BRAIN Alpha Documentation ([Scribd](https://www.scribd.com/document/728780335/World-Quant-Brain-Alpha-Documentation))。
- **採用論點：** WorldQuant 工業化 alpha 生產流程。核心指標：Sharpe ratio、turnover、drawdown、與現有 book 的相關性。新 alpha 僅在與現有 portfolio 相關性 < 0.3-0.5 時才有邊際價值。公開的 fitness 公式：`fitness = sqrt(abs(returns) / max(turnover, 0.125)) × Sharpe`，需 > 1.0 才通過。
- **在本框架的應用：** 支持 Breakeven Cost（結合 alpha 和 turnover）的設計思路；Stage 2 Incremental Alpha 對應 WorldQuant 的「與現有 book 低相關」要求。

### Israel & Moskowitz (2013)

- **出處：** Israel, R. & Moskowitz, T. J. (2013). "The Role of Shorting, Firm Size, and Time on Market Anomalies." *Journal of Financial Economics* 108(2), 275-301.
- **採用論點：** 系統性分析因子容量（capacity）與執行成本。許多因子的 alpha 集中在小型股和做空端——實務中難以大規模執行。
- **在本框架的應用：** 支持前處理正交化去除 Size 曝露；Breakeven Cost 指標間接反映容量限制。

### Asness (2015)

- **出處：** Asness, C. S. (2015). "The Great Divide." *Institutional Investor*.
- **採用論點：** 因子需同時具備統計證據（statistical evidence）與經濟邏輯（economic rationale）。單有統計顯著性不足以證明因子有效。
- **在本框架的應用：** 本框架聚焦可量化的統計評估，經濟邏輯由使用者自行判斷（記錄在 MLflow 的 `logic_description` tag 中）。

### Almgren & Chriss (2001)

- **出處：** Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk* 3(2), 5-39.
- **採用論點：** 市場衝擊的 square-root law：交易成本 ∝ √(order size / daily volume)。線性成本假設（如 Breakeven Cost 公式）在大資金量下高估因子的經濟可行性。
- **在本框架的應用：** Breakeven Cost profile 指標的使用限制——線性近似在小規模策略下可接受，但大 AUM 下低估成本。

### Asness, Moskowitz & Pedersen (2013)

- **出處：** Asness, C. S., Moskowitz, T. J. & Pedersen, L. H. (2013). "Value and Momentum Everywhere." *Journal of Finance* 68(3), 929-985.
- **採用論點：** 在 5 大資產類別（4 國個股、國家指數、政府債券、外匯、商品，共 8 個市場）測試 Value 和 Momentum。因子報酬在跨資產類別間的相關性高於被動曝露間的相關性。
- **在本框架的應用：** 支持框架的通用性設計——相同的評估維度（Reliability / Profitability）應能跨資產類別運行。跨市場方法論詳見 [§12 跨市場因子評估](#12-跨市場因子評估)。

---

## 12. 跨市場因子評估

> 本節整理「因子是全球現象還是本地現象」這一核心辯論的相關文獻，
> 涵蓋全球 vs 本地因子模型、跨市場聚合方法、以及業界標準化實務。

### 核心辯論：Global vs Local Factor Models

#### Griffin (2002)

- **出處：** Griffin, J. M. (2002). "Are the Fama and French Factors Global or Country Specific?" *Review of Financial Studies* 15(3), 783-803.
- **採用論點：** **本地因子模型在 R² 和定價誤差上全面優於全球因子模型。** 加入外國因子反而讓 in-sample 和 out-of-sample 定價更不準確。研究涵蓋美國、英國、日本、加拿大四國，將全球 HML/SMB 分解為 domestic 和 foreign 成分後比較。結論：實務應用（資本成本估計、績效歸因）應以 country-specific 為基準。
- **在本框架的應用：** 支持 factrix 逐市場獨立評估的設計——跨市場有效性應作為附加品質信號，而非取代本地 `verdict()` 評估。

#### Fama & French (2012)

- **出處：** Fama, E. F. & French, K. R. (2012). "Size, Value, and Momentum in International Stock Returns." *Journal of Financial Economics* 105(3), 457-472.
- **採用論點：** 23 國分 4 區域（北美、歐洲、日本、亞太），value premium 在所有區域存在，momentum 除日本外均存在。**關鍵發現：拒絕跨區域整合定價假說**——local model 使用 local explanatory returns 對 local average returns 的解釋力顯著優於 global model。因子構建方式：區域內 2×3 sorts。
- **在本框架的應用：** 因子是普遍的（pervasive：在多個區域出現）但定價機制是本地的（local：各區域的因子溢價大小和結構不同）。支持「本地 `verdict()` 為主、跨市場一致性為輔」的分層設計。

#### Fama & French (2017)

- **出處：** Fama, E. F. & French, K. R. (2017). "International Tests of a Five-Factor Asset Pricing Model." *Journal of Financial Economics* 123(3), 441-463.
- **採用論點：** 五因子模型（Mkt, SMB, HML, RMW, CMA）的國際測試。歐美亞太大致有效，**但日本異常**——有強 BTM 效應卻幾乎無 profitability/investment 效應。Spanning test 確認 global factors 無法解釋 regional returns。全球版模型在所有區域的定價誤差均大於本地版。
- **在本框架的應用：** 日本的例外說明「Pervasive ≠ Identical Everywhere」——同一因子在不同市場可能有不同的表現模式。跨市場評估應容許部分市場的 `verdict()` 未通過。

### 跨市場因子複製與衰減

#### Jacobs & Müller (2020)

- **出處：** Jacobs, H. & Müller, S. (2020). "Anomalies Across the Globe: Once Public, No Longer Existent?" *Journal of Financial Economics* 135(1), 213-230.
- **採用論點：** 241 個異常因子在 39 個股票市場的 pre/post-publication 表現。**美國是唯一有顯著 post-publication decay 的市場**，國際市場的因子獲利性在發表後大致維持。原因：跨國投資障礙（資本管制、資訊不對稱、本地偏好）使套利資金無法自由流動，市場保持分割。
- **在本框架的應用：** `oos_survival_ratio` 的跨市場含義：台股或其他新興市場的 OOS decay 可能天然低於美股，因為套利效率較低。0.5 門檻在不同市場的嚴格程度不同——但這是市場結構差異，不是門檻設計問題。

#### Baltussen, Swinkels & van Vliet (2021)

- **出處：** Baltussen, G., Swinkels, L. & van Vliet, P. (2021). "Global Factor Premiums." *Journal of Financial Economics* 142(3), 1128-1154.
- **採用論點：** 24 個因子溢價跨股票、債券、商品、外匯四大資產類別，**217 年歷史**（1800-2016）。OOS 測試顯示因子溢價存在且衰減有限。方法論：(1) Dual replication/OOS 設計——先在已知期間複製，再在更早歷史資料中 OOS 測試；(2) 使用 minimum Bayes factor（Harvey 建議）校正 p-hacking。
- **在本框架的應用：** 支持以超長歷史 + 跨資產類別證據作為因子「真實性」的判據。Dual replication 設計可與 factrix `oos_survival_ratio` 的 IS/OOS 概念對應。

#### Jensen, Kelly & Pedersen (2023)（同 §4，跨市場面向）

- **額外採用論點：** 153 因子在 **93 國**測試。在美國 in-sample 有效的因子，在 93 國 out-of-sample 也普遍有效——跨國證據強化（而非削弱）因子有效性。提供 [JKP Global Factor Data](https://jkpfactors.com/) 開源資料庫（Python），標準化 93 國的因子構建流程。([GitHub](https://github.com/bkelly-lab/ReplicationCrisis))
- **在本框架的應用：** JKP 是目前最全面的跨國因子複製框架。其 country-by-country 構建 + 聚合測試的方法論，可作為未來跨市場 Layer 的標準流程參考。

### 全球因子模型構建

#### Hou, Karolyi & Kho (2011)

- **出處：** Hou, K., Karolyi, G. A. & Kho, B.-C. (2011). "What Factors Drive Global Stock Returns?" *Review of Financial Studies* 24(8), 2527-2574.
- **採用論點：** 49 國 27,000+ 股票，30 年。Momentum + Cash-flow-to-price 模型優於 global CAPM 和 Fama-French 模型，但 local models 在多數異常因子上仍有更低定價誤差。**核心方法：Global/Local 分解**——對每個因子構建 global 版本和 country-specific 版本，再將 global 分解為 domestic + foreign 成分：`F_global = w_d × F_domestic + w_f × F_foreign`，測試 foreign component 是否有增量解釋力。
- **在本框架的應用：** Global/Local 分解是衡量「因子的解釋力有多少是真正全球性的 vs 本地性的」最嚴謹的方法。可作為未來跨市場因子分析的進階工具。

#### Chen, Han, Tang & Zhu (2024)

- **出處：** Chen, J., Han, Y., Tang, G. & Zhu, Y. (2024). "Taming the Global Factor Zoo." *Journal of International Money and Finance*.
- **採用論點：** 36 國 48,120+ 股票（1990-2024），使用 iterative two-step LASSO 從 152 個異常因子中篩選出 **7 因子全球模型**（market, profit growth, quality, momentum, investment, size, debt issuance）。Selected factors 月均溢價 32-81 bps（|t| ≥ 3）。全球模型在多數國家優於 local 和 foreign models。本質上是 Feng, Giglio & Xiu (2020) 的全球延伸。
- **在本框架的應用：** LASSO 選出的 7 因子可作為跨市場 Step 6 正交化的候選 base factors，取代目前僅針對台股設計的 Size/Value/Momentum/Industry。

#### Karolyi & Wu (2022)

- **出處：** Karolyi, G. A. & Wu, Y. (2022). "A New Partial-Segmentation Approach to Modeling International Stock Returns." *Journal of Financial and Quantitative Analysis* 57(2), 459-496.
- **採用論點：** 提出「部分分割」（partial segmentation）方法：同時構建 local 和 foreign 因子，測試哪種組合最佳。46 國 37,000+ 股票，20 年。發現 **local + foreign 的混合模型**優於純 global 或純 local。市場既非完全整合也非完全分割。
- **在本框架的應用：** 理論上支持「本地 `verdict()` 為主 + 跨市場信號為輔」的分層架構——兩者提供互補資訊，不應只取其一。

### 新興市場與市場微結構差異

#### Cakici, Fabozzi & Tan (2013)

- **出處：** Cakici, N., Fabozzi, F. J. & Tan, S. (2013). "Size, Value, and Momentum in Emerging Market Stock Returns." *Emerging Markets Review* 16, 46-65.
- **採用論點：** 18 個新興市場測試。Value 在所有市場均有效。Momentum 除東歐外均有效。溢價差異大：亞洲 1.01%/月、東歐 1.92%/月、拉美 0.87%/月。**Local factors 的解釋力遠優於 US 或 global factors**，顯示新興市場的分割程度高於已開發市場。
- **在本框架的應用：** 台股作為新興/邊緣已開發市場，因子表現可能與美股有顯著差異。跨市場比較時應區分市場發展程度（developed vs emerging），避免直接套用美股經驗。

#### Chaieb, Langlois & Scaillet (2021)

- **出處：** Chaieb, I., Langlois, H. & Scaillet, O. (2021). "Factors and Risk Premia in Individual International Stock Returns." *Journal of Financial Economics* 141(2), 669-692.
- **採用論點：** 跨國個股（非組合）層級的 panel 估計。**Local market factor 在所有市場（已開發+新興）都不可被 global/regional factors 替代。** 所有因子在大多數國家有顯著風險溢價，但定價誤差大且隨時間變化。
- **在本框架的應用：** 確認 Step 6 正交化應使用 local market factor（而非 global market）。定價誤差的時變性支持 Regime IC 的設計——因子有效性可能在不同時期/體制下變化。

#### Tobek & Hronec (2021)

- **出處：** Tobek, O. & Hronec, M. (2021). "Does It Pay to Follow Anomalies Research? Machine Learning Approach with International Evidence." *Journal of Financial Markets* 56, 100592.
- **採用論點：** 153 個異常因子的國際測試。**重要不對稱性：美股過去表現可預測非美市場的 OOS 贏家，但非美市場過去表現無法預測美股 OOS 贏家。** Neural networks 聚合異常因子後在全球 Fama-French 五因子模型上產生 value-weighted alpha 0.843%/月（t = 5.668）。
- **在本框架的應用：** 如果未來建立跨市場因子篩選，美股證據可作為其他市場的先驗（prior），但反向不成立。暗示 Bayesian hierarchical 模型中，美股應貢獻較大的先驗權重。

### 跨市場聚合的統計方法

#### Bryzgalova, Huang & Julliard (2023)

- **出處：** Bryzgalova, S., Huang, J. & Julliard, C. (2023). "Bayesian Solutions for the Factor Zoo: We Just Ran Two Quadrillion Models." *Journal of Finance* 78(1), 487-557.
- **採用論點：** Bayesian Model Averaging (BMA) 跨所有可能的因子組合。可靠估計風險溢價、偵測弱因子、自動選擇最佳模型。BMA-SDF 在 in-sample 和 out-of-sample 均優於現有模型。方法可推廣至跨市場：以市場作為 BMA 的另一個維度。
- **在本框架的應用：** 提供 Stage 2 Incremental Alpha 的跨市場延伸思路——不僅在市場內篩選冗餘因子，也可跨市場評估因子的全球邊際貢獻。

### 業界跨市場標準化實務

#### MSCI Barra GEM / FaCS

- **出處：** MSCI. (2018). "MSCI FaCS Methodology." 另參考 Barra Global Equity Model (GEM3) 及 Global Total Market Equity Model (GEMLT) 文件。
- **採用論點：** 業界標準的跨市場因子標準化方法——**country-specific mean + global standard deviation**。即：每個市場內部去均值（消除國家效應），但用全球標準差做尺度（確保跨國可比）。模型結構將報酬分解為 World + Country + Industry + Style + Currency 五層。個股對本國 country factor 有 unit exposure，對他國為 zero。Volatility 等本質跨市場可比的因子使用 global mean。
- **在本框架的應用：** 如果未來需要跨市場比較因子 signal 的強度，MSCI 的 country-specific mean + global std 是業界公認的標準化方式。但 factrix `verdict()` pass/fail 設計天然迴避了這個問題——比較的是通過/未通過，不是 signal 的絕對值。

---

### 宏觀因子訊號評估方法論

#### Macrosynergy (2024) — How to Measure the Quality of a Trading Signal

- **出處：** Macrosynergy. "How to Measure the Quality of a Trading Signal." ([連結](https://macrosynergy.com/research/how-to-measure-the-quality-of-a-trading-signal/))
- **採用論點：** 跨資產訊號評估三層框架：(1) **Accuracy**（balanced accuracy、Pearson/Kendall/Spearman correlation——三種都報，不只用 Spearman）、(2) **Risk-adjusted performance**（signal-weighted portfolio 的 Sharpe ratio、Sortino ratio）、(3) **Panel structure**（區分 cross-sectional mean vs time-series mean 的訊號品質）。明確處理 panel/macro 訊號與截面股票訊號的差異。
- **在本框架的應用：** 小 N 宏觀因子不適合 quantile sort，應改用 (1) 多種 correlation 指標 + (2) long-short portfolio SR 作為核心評估工具。三層框架可作為未來 macro pipeline 的設計藍圖。

#### Macrosynergy (2024) — Systematic Stock Selection with Macro Factors

- **出處：** Macrosynergy. "Systematic Stock Selection with Macro Factors." ([連結](https://macrosynergy.com/research/systematic-stock-selection-with-macro-factors/))
- **採用論點：** 以 22 檔美股（跨 GICS sector，1992-2020）為小 universe，用宏觀因子做選股。即使在小 universe 中，macro-aware portfolios 累積超額報酬 30%-100%。方法論：**sequential statistical learning**——用歷史資料選最佳 macro factor model，再用該 model 估計 forward-looking factor loadings。最終收斂於 5 個 dominant factors（relative credit growth、real appreciation、real yields、consumption growth、trade balance）。
- **在本框架的應用：** 證明宏觀因子在小 universe 下仍有效，但需要不同的評估框架（panel regression + portfolio SR，而非 IC + quantile sort）。

#### Asness, Moskowitz & Pedersen (2013)（跨資產因子評估方法論）

- **出處：** 同 §11。
- **額外採用論點（方法論面向）：** 在國家指數（N ≈ 13-18）等小截面中的做法：(1) **不做 quintile sort**，改做 top/bottom 1/3 long-short；(2) 以 long-short portfolio 的 **time-series Sharpe ratio** 和 **alpha**（對 global market 回歸）評估因子有效性；(3) 跨 8 個市場（4 國個股、國家指數、債券、外匯、商品）統一使用相同方法。
- **在本框架的應用：** 小 N 場景的 gold standard。證明 long-short SR + time-series alpha 是跨資產類別通用的因子評估方法，不依賴截面大 N。

---

## 13. 統計穩健性與交易成本

> 本節整理 v3 統計穩健性檢視（2026-04）中新增的文獻引用，
> 涵蓋 Bayesian 多重檢定、stepdown 校正、交易成本調整因子選擇、穩健回歸、因子擁擠。

### Harvey & Liu (2020)

- **出處：** Harvey, C. R. & Liu, Y. (2020). "Lucky Factors." *Journal of Financial Economics* 137(1), 116-142.
- **採用論點：** 提出 Bayesian multiple testing framework，處理大量因子同時檢定的 false discovery 問題。相較 Bonferroni（假設檢定間獨立）和 BHY（允許相關但為 frequentist），Bayesian 方法自然整合因子間的相關結構，提供 posterior probability of being a true factor。核心結論：考慮 multiple testing 後，大量「t > 2.0」的因子不再顯著。
- **在本框架的應用：** `bhy()` / `verdict()` 改進的理論依據——從固定門檻 2.0 升級為 BHY 連續校正時，本文提供了為什麼 BHY 比離散階梯門檻（2.0/2.5/3.0）更原則性的論證。未來若需更精確的校正，可進一步採用本文的 Bayesian framework。

### Romano & Wolf (2005)

- **出處：** Romano, J. P. & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized Data Snooping." *Econometrica* 73(4), 1237-1282.
- **採用論點：** 提出 stepdown 多重檢定校正方法，控制 family-wise error rate (FWER)。相較 Bonferroni，stepdown 方法通過 bootstrap 估計聯合分佈，考慮檢定統計量間的相關性，因此在保持相同 FWER 控制的前提下有更高的統計力（拒絕更多虛假因子）。
- **在本框架的應用：** `verdict()` 的 OR 條件（IC_IR OR Q1-Q5 spread）等效於兩次檢定。可使用 Romano-Wolf stepdown correction 處理此 OR 帶來的多重檢定膨脹，比 Bonferroni 近似（門檻從 2.0 提高至 ~2.24）更精確。

### DeMiguel, Martin-Utrera & Nogales (2020)

- **出處：** DeMiguel, V., Martin-Utrera, A. & Nogales, F. J. (2020). "Transaction Costs and Trading Volume in the Cross-Section of Stock Returns." *Journal of Financial Economics* 135(1), 271-292.
- **採用論點：** 實證顯示考慮交易成本後，最優因子集合可能完全改變——高 alpha 但高 turnover 的因子在扣除成本後可能不如低 alpha 但低 turnover 的因子。提出 cost-adjusted factor selection 框架，將交易成本內嵌於因子篩選（而非事後作為 profile 指標）。
- **在本框架的應用：** 支持 Net_Spread（`Q1-Q5 Spread - 2 × cost × Turnover`）作為 Profile 指標——雖然不進入 `verdict()` 判斷（成本是 implementation 問題），但讓使用者快速判斷扣除成本後的殘餘 alpha。本文的核心洞見是 Breakeven Cost 和 Gross Spread 分開報告不足以捕捉交互效應。

### Sen (1968)

- **出處：** Sen, P. K. (1968). "Estimates of the Regression Coefficient Based on Kendall's Tau." *Journal of the American Statistical Association* 63(324), 1379-1389.
- **採用論點：** 提出 Theil-Sen estimator（中位數斜率估計）——使用所有相鄰觀測點對的斜率中位數作為回歸係數估計。Breakdown point = 29.3%（即最多 29.3% 的資料為離群值時仍給出正確估計），遠優於 OLS（breakdown point = 0%）。且不需要殘差常態分佈假設。
- **在本框架的應用：** IC_Trend 指標使用 Theil-Sen estimator 替代 OLS 斜率或比值法。金融時間序列中常有 outlier（如 COVID 期間的 IC 暴跌），Theil-Sen 不會被單一極端觀測扭曲。`scipy.stats.theilslopes` 提供斜率與 confidence interval，可直接判斷 IC 衰減是否統計顯著。

### Lou & Polk (2022)

- **出處：** Lou, D. & Polk, C. (2022). "Comomentum: Inferring Arbitrage Activity from Return Correlations." *Review of Financial Studies* 35(7), 3272-3302.
- **採用論點：** 提出 comomentum 指標——momentum 贏家之間（及輸家之間）的超額相關性作為因子擁擠（crowding）的代理。當 comomentum 高時，momentum 因子預測力下降（crowding 壓縮 alpha）。建立了因子 crowding → alpha decay 的因果機制：套利資本集中 → 同向交易 → 相關性上升 → 超額報酬消失。
- **在本框架的應用：** 為 IC_Trend 的經濟學解釋提供理論框架——IC 隨時間衰減不僅是 data-mining 的結果（McLean & Pontiff, 2016），也可能是因子 crowding 的結果。AI 生成的因子尤其可能面臨快速 crowding（多個 AI 系統挖掘相似訊號）。IC_Trend 的 Theil-Sen 斜率顯著為負時，可能反映 crowding decay 而非 overfitting decay——兩者的 implication 不同（前者未來可能反轉，後者不會）。
