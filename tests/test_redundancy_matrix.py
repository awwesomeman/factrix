"""redundancy_matrix tests: symmetry, diagonal, auto-downgrade."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorlib.evaluation._protocol import _COMPACTED_PREPARED
from factorlib.evaluation.profile_set import ProfileSet
from factorlib.metrics.redundancy import redundancy_matrix


class TestShape:
    def test_symmetric_and_unit_diagonal(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        m = redundancy_matrix(ps, method="factor_rank", artifacts=artifacts)
        n = len(profiles)
        names = [p.factor_name for p in profiles]
        assert m.height == n
        # Diagonal
        for i, n_i in enumerate(names):
            val = m.filter(pl.col("factor") == n_i)[n_i].item()
            assert abs(val - 1.0) < 1e-12
        # Symmetric
        for i in range(n):
            for j in range(i + 1, n):
                ni, nj = names[i], names[j]
                a = m.filter(pl.col("factor") == ni)[nj].item()
                b = m.filter(pl.col("factor") == nj)[ni].item()
                assert abs(a - b) < 1e-12


class TestMethods:
    def test_factor_rank_and_value_series_both_work(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        fr = redundancy_matrix(ps, method="factor_rank", artifacts=artifacts)
        vs = redundancy_matrix(ps, method="value_series", artifacts=artifacts)
        assert fr.shape == vs.shape

    def test_value_series_values_in_unit_interval(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        m = redundancy_matrix(ps, method="value_series", artifacts=artifacts)
        names = [p.factor_name for p in profiles]
        for n_i in names:
            vals = m.drop("factor")[n_i].to_numpy()
            assert ((vals >= 0.0) & (vals <= 1.0 + 1e-9)).all()


class TestAutoDowngrade:
    def test_compact_artifact_triggers_warning(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        # Mark one artifact compact
        compact_name = profiles[0].factor_name
        object.__setattr__(artifacts[compact_name], "prepared", _COMPACTED_PREPARED)
        object.__setattr__(artifacts[compact_name], "compact", True)

        ps = ProfileSet(profiles)
        with pytest.warns(UserWarning, match="auto-downgrading"):
            m = redundancy_matrix(ps, method="factor_rank", artifacts=artifacts)
        # Still returns a shape-correct matrix
        assert m.height == len(profiles)


class TestErrors:
    def test_no_artifacts(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(ValueError, match="requires artifacts="):
            redundancy_matrix(ps)

    def test_missing_factor(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        # Drop one from artifacts
        first = profiles[0].factor_name
        slim = {k: v for k, v in artifacts.items() if k != first}
        with pytest.raises(KeyError, match="missing factor"):
            redundancy_matrix(ps, method="value_series", artifacts=slim)

    def test_empty_profile_set(self):
        empty = ProfileSet(
            [],
            profile_cls=__import__(
                "factorlib.evaluation.profiles", fromlist=["CrossSectionalProfile"]
            ).CrossSectionalProfile,
        )
        with pytest.raises(ValueError, match="empty"):
            redundancy_matrix(empty, artifacts={})

    def test_unknown_method(self, cs_profiles_and_artifacts):
        profiles, artifacts = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(ValueError, match="Unknown method"):
            redundancy_matrix(ps, method="mahalanobis", artifacts=artifacts)  # type: ignore[arg-type]
