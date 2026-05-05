# FactorLib 架構設計藍圖：全自動因子探索與基礎設施

本文檔總結了從「資深量化研究員」與「後端工程師」的雙重視角，針對 `factrix` 套件在**「大規模因子探索/篩選」**以及**「開源套件架構」**上的最佳設計實踐。

---

## 1. 架構核心原則：計算與基礎設施分離

當 `factrix` 作為一個讓外部使用者或跨部門分析師安裝的套件時，「邊界（Scope）控制」是確保套件容易被採用且能無限擴展的關鍵。

> [!IMPORTANT]
> **混合模式（Hybrid Approach）**
> 核心庫 (`factrix`) 必須保持純粹的無狀態 (Stateless)，針對 DB 與儀表板採用 **Batteries Included but Swappable (內建輕量預設，但可抽換)** 的策略。這能讓新使用者零門檻上手，同時滿足企業級的擴展需求。

### 1.1 儲存層 (Database) 設計
- **提供標準介面 (Protocol / ABC)**：定義統一的 `BaseFactorStore`，讓所有對因子的讀寫都透過此介面進行。套件核心不強迫綁定 PostgreSQL 或 MongoDB。
- **開箱即用的預設**：內建基於 `SQLite` 或 `Parquet` 的儲存實作（如 `SqliteStore`），讓使用者 `pip install` 後一行代碼就能啟動狀態紀錄。
- **企業級擴展**：使用者可以自行繼承 `BaseFactorStore` 來串接自己公司的 AWS S3 或私有雲資料庫。

### 1.2 儀表板 (Dashboard) 設計
- **靜態分析 (層次一)**：提供類似 `alphalens` 的 API，使用者在 Jupyter Notebook 中呼叫 `result.plot_summary()` 即可用 `matplotlib/plotly` 渲染出 IC 衰減或分層回測圖。
- **可互動 Web UI (層次二 - 可選安裝)**：將儀表板作為 Optional Dependency (`pip install factrix[dashboard]`)。提供基於 Streamlit 或 Dash 的 CLI 命令（如：`factrix server --db=sqlite:///factors.db`），使用者可以立刻獲得類似 MLflow 的精美視覺化。

---

## 2. 面向 AI 與分析師的大規模探索輔助模組

在「AI 自主生成、測試、反饋」的現代量化架構中，必須具備強大的防禦機制，避免 AI 產出過擬合（Overfitting）或高換手（高交易成本）的無效因子。

### 2.1 因子除重與正交化檢驗 (De-duplication)
AI 很容易反覆發明「動能」或「市值」因子的變形。
- **設計**：在 Pipeline 中新增 `CorrelationGate` 或 `ResidualizationGate`。
- **機制**：系統自動將新因子與現有「因子庫 (Factor Zoo)」中的基準因子進行正交化。若殘差 IC 趨弱，即使原生 IC 很高也會被 Veto，迫使 AI 尋找具有增量資訊 (Marginal Alpha) 的邏輯。

### 2.2 微觀結構與交易成本防護 (Turnover Guard)
AI 經常發掘短期反轉，但往往會被滑價吃光利潤。
- **設計**：在 `compute_profile` 加入自相關性 (Autocorrelation) 與因子衰減 (Signal Decay) 計算。
- **機制**：建立 `TurnoverGate`，檢核隱含換手率與預測 Spread 的比值。

### 2.3 經濟學邏輯與表達式沙盒 (Symbolic DSL Sandbox)
- **設計**：禁止 AI 直接撰寫 Python 代碼以防前視偏差 (Look-ahead bias) 與安全風險。設計一套特有的 DSL（例如 `Rank(Ts_Mean(Close, 20))`）。
- **機制**：`factrix` 負責將 DSL 安全地解析為底層的 `Polars` 運算式。

### 2.4 Agent-Friendly 診斷反饋 (Structured JSON Feedback)
> [!TIP]
> AI 無法透過看圖表來改進，因此發生 FAILED 或 VETO 時，需要回傳幫助 AI「反思 (Reflection)」的診斷結構。

- **範例反饋**：
  ```json
  {
    "status": "FAILED",
    "failed_gate": "IC稳定性校验 (ICIR)",
    "metrics": {"IC": 0.04, "ICIR": 0.2}, 
    "suggestion_for_ai": "IC均值达标，但波动极大。建议尝试加入截面中性化或移动平均。"
  }
  ```

---

## 3. 模組與目錄發展藍圖 (Architecture Blueprint)

為實現上述功能，`factrix` 專案目錄將可按以下結構進行模組化切割：

```text
factrix/
├── core/                  # 純粹的運算引擎 (Polars)
├── tracking/              # 狀態與儲存層
│   ├── base.py            # BaseFactorStore (協議/介面)
│   └── local_sqlite.py    # 開箱即用的預設實作
├── evaluation/
│   ├── pipeline.py        # 驗證總控
│   ├── gates.py           # 加入 CorrelationGate, TurnoverGate
│   └── agent_api.py       # 面向 LLM Agent 的結構化反饋 API
├── factors/
│   ├── dsl.py             # 公式解析沙盒 (將字串安全轉為 Polars 表達式)
│   └── zoo.py             # 記錄/檢索優秀因子的 Factor Registry
├── integrations/
│   └── strategy_builder.py# 自動將因子打包並傳遞給後續 LiveTrader 引擎
└── ui/                    # 獨立的儀表板模組 (可選依賴)
    ├── cli.py             # CLI 啟動點 (factrix server)
    └── dashboard_app.py   # Streamlit/Dash 畫面邏輯
```
