"""Benchmark scenario modules — one per analysis-axis cell (#380 §2).

- ``continuous`` — Continuous × Individual scenarios (S1 / S2 / S3 /
  P1 plus per-metric micros).
- ``algo`` — algo scenarios that bypass ``run_metrics`` (S4 greedy
  forward selection).
- ``sparse`` — Sparse × Individual scenarios (S5 event-study bundle
  + M-corrado).
- ``dummy`` — minimal end-to-end smoke proving wrapper → JSONL →
  validator.
"""
