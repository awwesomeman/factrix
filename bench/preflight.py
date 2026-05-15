"""Pre-measurement setup: lock thread counts, seed RNGs, collect env.

The thread-count environment variables must be set **before** numpy /
BLAS are imported into the running process; once a BLAS backend has
read them, later mutation is ignored. ``lock_threads`` is therefore
safe to call from harness entry points but cannot rescue a process
that already imported numpy with unbounded threads — callers that
care should set the env vars in their shell as well.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

import numpy as np
import psutil
from factrix.datasets import DATASET_SPEC_VERSION

from bench.schema import Env

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def lock_threads(n: int = 1) -> None:
    """Pin BLAS / OMP thread counts.

    Sets every common env knob to ``n``. Effective only when called
    before numpy / scipy / polars have been imported by the running
    process; idempotent thereafter but a no-op for already-initialized
    BLAS state.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for key in _THREAD_ENV_VARS:
        os.environ[key] = str(n)


def seed_numpy(seed: int = 0) -> None:
    """Seed numpy's legacy global RNG.

    Per-call ``np.random.default_rng(seed)`` is preferred elsewhere;
    this exists so legacy code paths that touch ``np.random.*`` are
    not a hidden source of run-to-run drift.
    """
    np.random.seed(seed)


def quiesce() -> None:
    """Run before each measurement: force a full GC pass."""
    gc.collect()
    gc.collect()
    gc.collect()


def _detect_blas() -> str:
    try:
        info = np.show_config(mode="dicts")  # type: ignore[call-arg]
    except TypeError:
        return "unknown"
    except Exception:
        return "unknown"
    if not isinstance(info, dict):
        return "unknown"
    blas = info.get("Build Dependencies", {}).get("blas", {})
    name = blas.get("name") if isinstance(blas, dict) else None
    return str(name) if name else "unknown"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _factrix_version() -> str:
    try:
        from importlib.metadata import version

        return version("factrix")
    except Exception:
        return "unknown"


def _omp_threads() -> int:
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw and raw.isdigit():
        return int(raw)
    return psutil.cpu_count(logical=True) or 1


def collect_env() -> Env:
    """Snapshot the runtime environment into the JSONL ``env`` field."""
    vm = psutil.virtual_memory()
    return Env(
        git_sha=_git_sha(),
        factrix_version=_factrix_version(),
        dataset_spec_version=DATASET_SPEC_VERSION,
        python=platform.python_version(),
        numpy=np.__version__,
        blas=_detect_blas(),
        omp_threads=_omp_threads(),
        cpu_model=platform.processor() or platform.machine() or "unknown",
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
        ram_gb=round(vm.total / (1024**3), 2),
        os=f"{platform.system().lower()}-{platform.machine().lower()}",
    )


@dataclass(frozen=True)
class Preflight:
    """Result of preflight setup, useful for echoing into logs."""

    threads: int
    seed: int
    env: Env

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["env"] = self.env.model_dump()
        return d


def preflight(*, threads: int = 1, seed: int = 0) -> Preflight:
    """One-shot: lock threads, seed numpy, GC, snapshot env."""
    lock_threads(threads)
    seed_numpy(seed)
    quiesce()
    return Preflight(threads=threads, seed=seed, env=collect_env())
