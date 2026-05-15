"""Benchmark scenario modules, grouped by analysis-axis cell.

- ``continuous`` — Continuous × Individual scenarios.
- ``algo`` — scenarios that bypass ``run_metrics`` (greedy forward
  selection).
- ``sparse`` — Sparse × Individual scenarios (event-study).
- ``dummy`` — minimal end-to-end smoke proving wrapper → JSONL →
  validator.
"""
