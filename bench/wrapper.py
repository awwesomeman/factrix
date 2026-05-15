"""Measurement wrapper — produces one ``BenchRecord`` per invocation.

Separates ``setup`` (data prep + I/O) from ``compute`` (the metric work
under measurement) so the JSONL records both phases. Ratio analysis
defaults to ``compute_s`` (the measured phase) over ``wall_s`` so
machine-to-machine I/O differences do not pollute comparisons.
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from bench.metric_sets import METRIC_SET_VERSION
from bench.preflight import collect_env, quiesce
from bench.schema import SCHEMA_VERSION, BenchRecord, CacheState, Env, Status


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb(proc: psutil.Process, baseline_mb: float) -> float:
    """Best-effort peak RSS — psutil exposes current RSS, not peak."""
    try:
        mi = proc.memory_info()
        return max(mi.rss / (1024**2), baseline_mb)
    except Exception:
        return baseline_mb


def measure[T](
    setup: Callable[[], T],
    compute: Callable[[T], Any],
    *,
    scenario_id: str,
    axis_cell: str,
    scale: dict[str, Any],
    metric_set: str,
    metric_set_version: str = METRIC_SET_VERSION,
    run_idx: int = 0,
    is_warmup: bool = False,
    cache_state: CacheState = "warm",
    env: Env | None = None,
) -> BenchRecord:
    """Run ``setup`` then ``compute`` under measurement.

    ``setup`` is the data-prep phase (synthetic generation, deserialize,
    cache prime); ``compute`` is the work we care about. Exceptions in
    either phase are caught and recorded as ``status="error"`` so the
    JSONL row is preserved for retrospective analysis.

    The ``scale`` dict is wrapped with ``axis_cell`` and validated by
    the pydantic discriminator — passing a shape that doesn't match
    the declared ``axis_cell`` is a fail-loud error.
    """
    if env is None:
        env = collect_env()

    quiesce()
    proc = psutil.Process()
    baseline_rss_mb = proc.memory_info().rss / (1024**2)
    tracemalloc.start()
    started_at = _utc_now_iso()
    setup_s: float | None = None
    compute_s: float | None = None
    wall_s: float | None = None
    cpu_s: float | None = None
    peak_rss_mb: float | None = None
    peak_alloc_mb: float | None = None
    status: Status = "ok"
    error_message: str | None = None

    wall0 = time.perf_counter()
    cpu0 = time.process_time()

    try:
        t0 = time.perf_counter()
        artifact = setup()
        t1 = time.perf_counter()
        rss_after_setup = _peak_rss_mb(proc, baseline_rss_mb)
        compute(artifact)
        t2 = time.perf_counter()
        rss_after_compute = _peak_rss_mb(proc, rss_after_setup)

        setup_s = t1 - t0
        compute_s = t2 - t1
        wall_s = t2 - wall0
        cpu_s = time.process_time() - cpu0
        peak_rss_mb = rss_after_compute
        _, traced_peak = tracemalloc.get_traced_memory()
        peak_alloc_mb = traced_peak / (1024**2)
    except MemoryError as e:
        status = "oom"
        error_message = repr(e)
        wall_s = time.perf_counter() - wall0
        cpu_s = time.process_time() - cpu0
    except BaseException as e:
        status = "error"
        error_message = repr(e)
        wall_s = time.perf_counter() - wall0
        cpu_s = time.process_time() - cpu0
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            tracemalloc.stop()
            raise
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        gc.collect()

    record = BenchRecord.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "metric_set": metric_set,
            "metric_set_version": metric_set_version,
            "axis_cell": axis_cell,
            "scale": {"axis_cell": axis_cell, **scale},
            "run_idx": run_idx,
            "is_warmup": is_warmup,
            "cache_state": cache_state,
            "status": status,
            "error_message": error_message,
            "started_at": started_at,
            "wall_s": wall_s,
            "setup_s": setup_s,
            "compute_s": compute_s,
            "cpu_s": cpu_s,
            "peak_rss_mb": peak_rss_mb,
            "peak_alloc_mb": peak_alloc_mb,
            "env": env.model_dump(),
        }
    )
    return record


@contextmanager
def jsonl_writer(path: str | Path):
    """Append-mode JSONL sink yielding a single ``write(record)`` fn."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = p.open("a", encoding="utf-8")
    try:

        def write(record: BenchRecord) -> None:
            fh.write(record.model_dump_json() + "\n")
            fh.flush()

        yield write
    finally:
        fh.close()


def write_records(path: str | Path, records: list[BenchRecord]) -> None:
    """Bulk-write ``records`` to ``path`` as JSONL (overwrites)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read raw (unvalidated) JSON rows from ``path``."""
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
