# Spike T3.S4 — 跨因子平行化（n_jobs）

**狀態**：**CLOSED — deferred indefinitely**（2026-04-18）
**Owner**：—
**解鎖的後續工作**：**無**。Benchmark 過了 §2.1 門檻但 ROI 過低；實作
未啟動。Benchmark 腳本（`experiments/benchmark_evaluate_batch_cpu.py`）
保留供未來重評。

## 關閉理由（2026-04-18）

Benchmark 顯示 CPU 37%（§2.3 填好），技術上有並行空間，但：

1. **沒有場景等不及**：30 因子 67s 是 notebook 互動；1000 因子 ~37min
   是 offline screening。沒有「blocker 等不及」的情境。
2. **實測加速比可能 < 2x**：37% mean 但 p95 77.6%，代表 Polars 已吃
   近一半；剩餘的 undersaturation 多半是 Python for-loop / MetricOutput
   構造等，process parallelism 未必能抓。Spawn 啟動成本 + pickle 成本
   在 30 因子 × 2s 的規模下容易吃掉節省。
3. **Maintenance 成本高**：pickle 失敗、worker OOM、Ctrl-C 清理、闭包
   picklable 限制——每個都是長期 bug 磁鐵（§6 已列）。
4. **使用者有替代路徑**：orchestration 層切成 N 個 Python process 各
   跑 subset，**零 factorlib 改動**，達到同樣加速。
5. **Polars 會自己改善**：每個 release 都在擠 thread utilization；半年
   後可能自然變 60%+，現在加 `n_jobs` 將來反而是 over-parallelism。

**重新開啟的觸發條件**：使用者實際回報「批次跑超過 1 小時」且
orchestration-層切分無法解決，才重啟本 Spike。

---

## 1. 問題陳述

`evaluate_batch` 當前是 sequential for-loop。使用者期望直覺：「我有 16 核
CPU，為什麼不併發跑 16 個因子？」

Gemini 版原計畫說「加 `n_jobs=4`，10K 因子可 4x 加速」。**我的分析**：
這是 premature optimization，因為：

1. **Polars 本身已經多執行緒**：`compute_ic`、`compute_spread_series`、
   `orthogonalize_factor` 內部都是 vectorized + Rust threads。單因子
   pipeline 可能已經把你的 CPU 吃滿。
2. **Polars Rust threads + Python fork() = 死鎖**。多進程要用 `spawn`，
   但 spawn 有顯著啟動成本（每個子進程重跑 `import polars`）。
3. **Thread contention**：如果 Polars 預設開 16 執行緒，4 個 worker × 16
   threads = 64 threads 搶 16 核，context switch 成本超過並行收益。

**正確的問法不是「怎麼加 n_jobs」，而是「現在 CPU 有多飽？若已飽和，
加並行只會更慢」。**

## 2. 前提驗證（benchmark 必做）

**不准在 benchmark 之前寫任何 `n_jobs=` 代碼**。

### 2.1 Benchmark 設計

在 TW 完整 panel + 100 因子下，測量：

```python
# 單因子單獨跑時的 CPU 利用率
time fl.evaluate(one_factor_df, "x", factor_type="cross_sectional")
# → monitor CPU usage via psutil.cpu_percent(interval=0.5)

# 批次 sequential 跑時的平均 CPU 利用率
time fl.evaluate_batch(factor_dict, factor_type="cross_sectional")
# → 同上
```

**決策門檻**：

| 平均 CPU 利用率 | 結論 | 行動 |
|---------------|------|------|
| > 70%         | Polars 已飽和 | **不做 n_jobs**，close spike |
| 40-70%        | 部分飽和 | 寫 microbenchmark 比較 n_jobs=2/4/8 實測 speedup |
| < 40%         | IO 或 Python overhead bound | **可能值得**做 n_jobs，但仍需確認瓶頸類型 |

### 2.2 Benchmark 尚未跑

本 Spike 卡在這一步。`experiments/benchmark_compact.py` 沒測 CPU 利用率。
需要新增 `experiments/benchmark_evaluate_batch_cpu.py`。

**Owner 動工前的第零件事**：跑 benchmark，填上面的表格，把結果 append
到本文件的 §2.3。若 CPU > 70%，Spike 直接 close，不進入 §3。

### 2.3 Benchmark 結果

```
日期: 2026-04-18
Panel: TW 2017-2025, 3.8M rows × 2029 assets
因子數: 30（lookback sweep 於 5 個 generator × 6 個 lookback）
硬體: 8 physical cores / 8 logical
POLARS_MAX_THREADS: 未設（Polars 預設）

單因子 evaluate 平均 CPU: 34.9% (p95 68.3%)  elapsed=1.85s
批次 sequential 平均 CPU: 37.0% (p95 77.6%)  elapsed=67.3s / per_factor=2.24s
結論: UNDERSATURATED（< 40% mean），技術上可並行

但 §2.1 門檻並非唯一判斷點。結合 ROI（見首段「關閉理由」）後
仍決定 close。p95 77.6% 顯示 Polars 已抓接近一半，可收割的空間
比 37% mean 看起來的小。
```

腳本：`experiments/benchmark_evaluate_batch_cpu.py`。重跑方式：
`python experiments/benchmark_evaluate_batch_cpu.py`。

## 3. 若 benchmark 顯示值得做的政策問題

**本節只有在 §2.1 benchmark 落在「< 40%」才適用**。

### 3.1 進程 vs 執行緒？

- **Threads**：`concurrent.futures.ThreadPoolExecutor`。Polars 會釋放 GIL
  做計算，理論上 threads 可擴展。但若 Polars 自己已開滿 threads，我們
  再加 threads 是 oversubscription。
- **Processes**：`multiprocessing.get_context("spawn").Pool`。繞開 GIL，
  但 IPC 成本高（每個 worker 要重新 `import polars`；輸入 DataFrame 要
  pickle）。

**建議**：**Processes with spawn**。Threads 上 polars 的行為不可靠。
需要明文禁用 fork（macOS 預設 spawn；Linux 預設 fork，要顯式切）。

### 3.2 子進程內的 `POLARS_MAX_THREADS`

若主進程看到 16 核，每個 worker 預設也開 16 threads，4 workers = 64
threads 搶 16 核。必須在子進程 `os.environ["POLARS_MAX_THREADS"] = "4"`
（= cpu_count // n_jobs）。

**建議**：在 worker 啟動時自動設定，使用者不用管。

### 3.3 子進程啟動成本分攤

Spawn 成本 ≈ 500ms-2s（import polars + factorlib）。單因子 sequential
算需要 3s。**worker 啟動成本 < 單因子時間才划算**。

**建議**：`n_jobs=2/4` 默認用 persistent worker pool（不是 per-call
spawn）。使用者 `evaluate_batch` 呼叫完 pool 保留到程式結束。

### 3.4 API 形狀

```python
fl.evaluate_batch(
    candidates,
    factor_type="cross_sectional",
    n_jobs=4,                    # None or 1 = sequential (default)
    # 跟 scikit-learn 慣例對齊：-1 = cpu_count()
)
```

**建議**：`n_jobs=None` 當 sequential（保留預設行為），`n_jobs=-1` =
所有核，正整數 = 明確核數。

### 3.5 與 fast_track / keep_artifacts / compact 的互動

- `n_jobs` + `fast_track=True`：只並行 Stage 2（50 個 survivors），Stage 1
  串行（本來就快）
- `n_jobs` + `keep_artifacts=True`：worker 必須回傳 artifacts，pickle 成本
  高。**警告使用者可能變慢**
- `n_jobs` + `compact=True`：OK，pickle 的是 compact artifacts
- `n_jobs` + `on_result` callback：**callback 在哪個 process 跑？**
  若在主進程，要把每個 worker 的結果 queue 回來序列化；若在 worker，
  callback 閉包要 picklable（很多 lambda 不行）。**建議 callback 在
  主進程**（保留 side-effect 可控性）

### 3.6 失敗處理

Worker 死了怎麼辦？
- pickle 失敗 → 主進程 raise，pool 整鍋扔
- OOM → SIGKILL，pool 要能 detect 並重啟 worker
- 使用者 Ctrl-C → pool 要乾淨關閉

**建議**：用 `concurrent.futures.ProcessPoolExecutor` 而非 raw
`multiprocessing.Pool`，exception 傳播更乾淨。

## 4. 非目標

- **分散式運算**（Dask / Ray）：out of scope。如果使用者到這個規模，
  應該自己在 airflow/ray 層呼叫 factorlib，而不是讓 factorlib 變成分散
  式框架
- **GPU**：OLS / IC 都是 Polars / NumPy，沒有 GPU 收益
- **自動調 `n_jobs`**：根據 panel 大小、因子數量自動決定。**過度工程**，
  使用者自己會估
- **並行化 Stage 1 in fast_track**：Stage 1 本來就快，先不動

## 5. 實作順序（**只有 §2 benchmark 過關才啟動**）

估時：**2-3d**（純並行化邏輯 + 測試）

1. `experiments/benchmark_evaluate_batch_cpu.py`——**0.5d**
2. 填 §2.3 benchmark 結果，決定 go / no-go——**0.2d**

**若 go：**

3. Worker setup 函式（設定 `POLARS_MAX_THREADS`、初始化 pool）——**0.5d**
4. `evaluate_batch` 加 `n_jobs` 分支，主進程負責：任務分發、callback
   dispatch、錯誤蒐集——**0.8d**
5. Tests：
   - `n_jobs=2` 跑 10 因子，結果與 sequential 相同
   - `n_jobs=4` + `keep_artifacts=True` 回傳正確 artifacts dict
   - Worker 中途 raise，`stop_on_error=False` 有被 `on_error` 捕捉到
   - Ctrl-C 不留 zombie process（integration test）
   - **0.8d**
6. README：明確標「僅在 Polars CPU 未飽和時有收益」——**0.2d**

## 6. 風險

- **結果不一致**：不同 run 的因子順序可能不同（pool 回傳順序無保證）。
  **緩解**：主進程收齊結果後按原始 factor name 順序重排，保證輸出
  deterministic
- **pickle 失敗**：使用者 DataFrame 有 custom 欄位（pl.Object dtype
  裡塞 Python objects）就 pickle 不過。**緩解**：在 worker
  disaptch 前試 pickle，失敗直接 raise 帶清楚訊息
- **實際沒快**：使用者升 `n_jobs=8` 發現跑得比 `n_jobs=1` 慢 2x。
  **緩解**：docstring 要寫「先 benchmark，再開」。**可考慮** 加
  `verbose=True` mode 印出 worker startup 時間、每個因子耗時，讓
  使用者自己判斷
- **維護負擔**：並行化 code path 永遠是 bug 溫床（pickle 邊界、signal
  handling、OOM）。**緩解**：若 benchmark 結果顯示 speedup < 2x，
  **不要做**

---

## 7. Sign-off checklist

- [x] Benchmark 已跑（2026-04-18，§2.3）
- [x] 結論：**close**（非因 CPU > 70%，而因 ROI 過低見首段）
- [ ] ~~§3 實作項目~~（不適用，Spike 已關閉）
