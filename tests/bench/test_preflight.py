"""bench.preflight — thread lock, seed determinism, env snapshot."""

from __future__ import annotations

import os

import numpy as np
from bench.preflight import collect_env, lock_threads, preflight, seed_numpy


def test_lock_threads_sets_env():
    lock_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    lock_threads(1)


def test_seed_numpy_is_deterministic():
    seed_numpy(42)
    a = np.random.random(5)
    seed_numpy(42)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_collect_env_fields_present():
    env = collect_env()
    assert env.dataset_spec_version == "1"
    assert env.omp_threads >= 1
    assert env.cpu_cores >= 1
    assert env.ram_gb > 0
    assert env.os


def test_preflight_returns_locked_state():
    pre = preflight(threads=1, seed=7)
    assert pre.threads == 1
    assert pre.seed == 7
    assert pre.env.dataset_spec_version == "1"
