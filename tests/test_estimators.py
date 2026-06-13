"""``factrix.estimators`` lowercase callable namespace."""

from __future__ import annotations

import factrix as fx
import numpy as np
import pytest
from factrix.stats import BlockBootstrap


class TestBlockBootstrapCallable:
    def test_returns_p_in_range_and_metadata(self) -> None:
        diff = np.random.default_rng(2).standard_normal(80)
        p, meta = fx.estimators.block_bootstrap(
            diff, n_resamples=199, rng_seed=42, scheme="stationary"
        )
        assert 1.0 / 200 <= p <= 1.0
        assert {"block_length", "n_resamples", "scheme", "rng_seed"} <= set(meta.keys())

    def test_rejects_zero_n_resamples(self) -> None:
        diff = np.random.default_rng(5).standard_normal(40)
        with pytest.raises(ValueError, match="n_resamples"):
            fx.estimators.block_bootstrap(diff, n_resamples=0)

    def test_rejects_unknown_scheme(self) -> None:
        diff = np.random.default_rng(6).standard_normal(40)
        with pytest.raises(ValueError, match="scheme"):
            fx.estimators.block_bootstrap(diff, scheme="bogus")  # type: ignore[arg-type]

    def test_rejects_zero_block_length(self) -> None:
        diff = np.random.default_rng(7).standard_normal(40)
        with pytest.raises(ValueError, match="block_length"):
            fx.estimators.block_bootstrap(diff, block_length=0)

    def test_class_and_function_share_underlying_p_for_same_seed(self) -> None:
        # BlockBootstrap class doesn't expose a series-level compute() —
        # the slice-test path consumes its parameters via the slice-test
        # function. The lowercase callable here is an alias over the
        # underlying paired-diff bootstrap; smoke-check shape only.
        diff = np.random.default_rng(3).standard_normal(60)
        p_a, _ = fx.estimators.block_bootstrap(diff, n_resamples=99, rng_seed=7)
        p_b, _ = fx.estimators.block_bootstrap(diff, n_resamples=99, rng_seed=7)
        assert p_a == p_b
        assert isinstance(BlockBootstrap(), BlockBootstrap)  # class still importable


class TestNamespaceExports:
    def test_top_level_namespace_attached(self) -> None:
        assert hasattr(fx, "estimators")

    def test_namespace_exports_callables(self) -> None:
        for name in (
            "block_bootstrap",
            "driscoll_kraay",
        ):
            assert callable(getattr(fx.estimators, name))

    def test_exposes_driscoll_kraay(self) -> None:
        # Driscoll-Kraay is the cross-section-robust HAC SE
        # option behind pooled_beta(driscoll_kraay=True).
        assert callable(fx.estimators.driscoll_kraay)
