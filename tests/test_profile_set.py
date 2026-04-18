"""ProfileSet: construction, filter, rank, multiple-testing correct."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorlib.evaluation.profiles import (
    CrossSectionalProfile,
    EventProfile,
)
from factorlib.evaluation.profile_set import ProfileSet


class TestConstruction:
    def test_infers_type_from_non_empty(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        assert ps.profile_cls is CrossSectionalProfile
        assert len(ps) == len(profiles)

    def test_empty_requires_profile_cls(self):
        with pytest.raises(ValueError, match="profile_cls"):
            ProfileSet([])

    def test_empty_with_profile_cls(self):
        ps = ProfileSet([], profile_cls=CrossSectionalProfile)
        assert ps.profile_cls is CrossSectionalProfile
        assert len(ps) == 0
        # to_polars should still yield a DataFrame with the schema
        df = ps.to_polars()
        assert df.height == 0
        assert "factor_name" in df.columns
        assert "ic_p" in df.columns

    def test_empty_schema_uses_typed_dtypes(self):
        # Empty-set dtype resolution goes through get_type_hints +
        # _polars_dtype_for; lock the three non-trivial cases:
        # PValue (NewType), float | None, and tuple[str, ...].
        from factorlib.evaluation.profiles import EventProfile
        ps = ProfileSet([], profile_cls=EventProfile)
        df = ps.to_polars()
        assert df.schema["caar_p"] == pl.Float64                # PValue → float
        assert df.schema["clustering_hhi"] == pl.Float64        # float | None
        assert df.schema["insufficient_metrics"] == pl.List(pl.Utf8)
        assert df.schema["oos_sign_flipped"] == pl.Boolean

    def test_rejects_mixed_types(
        self, cs_profile_strong, event_profile_strong=None
    ):
        # Build an EventProfile on the spot to mix
        from datetime import datetime, timedelta
        rng = np.random.default_rng(0)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        rows = []
        for d in dates:
            for i in range(10):
                f = int(rng.choice([-1, 0, 0, 1]))
                r = 0.5 * f + 0.5 * rng.standard_normal()
                rows.append({
                    "date": d, "asset_id": f"ev{i}",
                    "factor": float(f), "forward_return": float(r),
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        from factorlib.config import EventConfig
        from factorlib.evaluation.pipeline import build_artifacts
        art = build_artifacts(df, EventConfig())
        art.factor_name = "ev_aux"
        ev_prof = EventProfile.from_artifacts(art)

        with pytest.raises(TypeError, match="single-type"):
            ProfileSet([cs_profile_strong, ev_prof])


class TestPolarsView:
    def test_canonical_p_column_added(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        df = ps.to_polars()
        assert "canonical_p" in df.columns
        # For CS, canonical_p == ic_p
        np.testing.assert_allclose(
            df["canonical_p"].to_numpy(), df["ic_p"].to_numpy(),
        )

    def test_row_count_matches(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        assert ps.to_polars().height == len(profiles)


class TestFilter:
    def test_pl_expr_boolean(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        ps2 = ps.filter(pl.col("ic_ir") > 0.3)
        # iter_profiles order matches df order
        assert len(ps2) == ps2.to_polars().height
        for p in ps2.iter_profiles():
            assert p.ic_ir > 0.3

    def test_lambda(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        ps2 = ps.filter(lambda p: p.ic_tstat > 2.0)
        for p in ps2.iter_profiles():
            assert p.ic_tstat > 2.0

    def test_rejects_non_boolean_expr(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(TypeError, match="Boolean"):
            ps.filter(pl.col("ic_p") + 0.01)

    def test_rejects_aggregation(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(RuntimeError, match="changed row count"):
            ps.filter(pl.col("ic_p").mean())

    def test_rejects_wrong_arg_type(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(TypeError, match="pl.Expr or Callable"):
            ps.filter("ic_p > 0")


class TestRankTop:
    def test_rank_by_descending(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).rank_by("ic_ir")
        ir_values = [p.ic_ir for p in ps.iter_profiles()]
        assert ir_values == sorted(ir_values, reverse=True)

    def test_rank_by_ascending(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).rank_by("ic_p", descending=False)
        p_values = [p.ic_p for p in ps.iter_profiles()]
        assert p_values == sorted(p_values)

    def test_rank_by_unknown_field(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(KeyError, match="not in CrossSectionalProfile"):
            ps.rank_by("nonexistent")

    def test_top(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).rank_by("ic_ir").top(2)
        assert len(ps) == 2

    def test_top_rejects_negative(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        with pytest.raises(ValueError):
            ProfileSet(profiles).top(-1)


class TestProfileDfOrderingInvariant:
    """iter_profiles() and to_polars() must stay in lockstep.

    The class exposes two views of the same sequence (typed profiles
    and a polars DataFrame). A bug where one view reorders but not the
    other would silently misalign BHY inputs / filter masks. These tests
    lock the invariant after each reshaping operation.
    """

    def _assert_aligned(self, ps):
        names_from_iter = [p.factor_name for p in ps.iter_profiles()]
        names_from_df = ps.to_polars()["factor_name"].to_list()
        assert names_from_iter == names_from_df

    def test_filter_expr_preserves_alignment(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).filter(pl.col("ic_tstat") > 0)
        self._assert_aligned(ps)

    def test_filter_callable_preserves_alignment(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).filter(lambda p: p.ic_mean > 0)
        self._assert_aligned(ps)

    def test_rank_by_preserves_alignment(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).rank_by("ic_ir")
        self._assert_aligned(ps)

    def test_top_preserves_alignment(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).rank_by("ic_p", descending=False).top(2)
        self._assert_aligned(ps)

    def test_chain_preserves_alignment(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = (
            ProfileSet(profiles)
            .filter(pl.col("ic_p") <= 1.0)
            .rank_by("ic_ir")
            .top(3)
        )
        self._assert_aligned(ps)


class TestWithExtraColumns:
    """Attach user-computed columns to the polars view.

    Invariants checked:
      - Dataclass tuples unchanged
      - Extra columns visible via to_polars / filter / rank_by
      - Row-count mismatch and name collision raise ValueError
      - Survives filter / rank_by / top / multiple_testing_correct
    """

    def test_dict_input_attaches_columns(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).with_extra_columns(
            {"my_score": [1.0, 2.0, 3.0, 4.0]}
        )
        df = ps.to_polars()
        assert "my_score" in df.columns
        assert df["my_score"].to_list() == [1.0, 2.0, 3.0, 4.0]
        # underlying profiles untouched
        for original, new in zip(profiles, ps.iter_profiles()):
            assert original is new

    def test_dataframe_input_attaches_columns(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        extra = pl.DataFrame({
            "earnings_ic": [0.01, 0.05, -0.02, 0.0],
            "tag": ["a", "b", "c", "d"],
        })
        ps = ProfileSet(profiles).with_extra_columns(extra)
        df = ps.to_polars()
        assert df["earnings_ic"].to_list() == [0.01, 0.05, -0.02, 0.0]
        assert df["tag"].to_list() == ["a", "b", "c", "d"]

    def test_row_count_mismatch_raises(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        with pytest.raises(ValueError, match="row-count mismatch"):
            ProfileSet(profiles).with_extra_columns({"bad": [1.0, 2.0]})

    def test_rejects_dataclass_field_collision(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        with pytest.raises(ValueError, match="already exist"):
            ProfileSet(profiles).with_extra_columns(
                {"ic_mean": [0.0] * len(profiles)}
            )

    def test_rejects_canonical_p_collision(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        with pytest.raises(ValueError, match="already exist"):
            ProfileSet(profiles).with_extra_columns(
                {"canonical_p": [0.5] * len(profiles)}
            )

    def test_extras_survive_filter(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).with_extra_columns(
            {"my_score": [4.0, 3.0, 2.0, 1.0]}
        )
        ps2 = ps.filter(pl.col("my_score") >= 3.0)
        assert len(ps2) == 2
        assert ps2.to_polars()["my_score"].to_list() == [4.0, 3.0]

    def test_extras_survive_rank_by_and_top(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = (
            ProfileSet(profiles)
            .with_extra_columns({"my_score": [1.0, 4.0, 2.0, 3.0]})
            .rank_by("my_score", descending=True)
            .top(2)
        )
        # df and profiles are reordered together
        assert ps.to_polars()["my_score"].to_list() == [4.0, 3.0]
        names_df = ps.to_polars()["factor_name"].to_list()
        names_iter = [p.factor_name for p in ps.iter_profiles()]
        assert names_df == names_iter

    def test_extras_survive_multiple_testing_correct(
        self, cs_profiles_and_artifacts,
    ):
        profiles, _ = cs_profiles_and_artifacts
        ps = (
            ProfileSet(profiles)
            .with_extra_columns({"my_score": [0.1, 0.2, 0.3, 0.4]})
            .multiple_testing_correct(fdr=0.10)
        )
        df = ps.to_polars()
        assert "my_score" in df.columns
        assert "p_adjusted" in df.columns

    def test_empty_set_accepts_empty_extras(self):
        ps = ProfileSet([], profile_cls=CrossSectionalProfile)
        ps2 = ps.with_extra_columns(pl.DataFrame({"my_score": [], "tag": []}))
        assert "my_score" in ps2.to_polars().columns
        assert ps2.to_polars().height == 0

    def test_chained_calls_append_independently(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = (
            ProfileSet(profiles)
            .with_extra_columns({"a": [1.0] * len(profiles)})
            .with_extra_columns({"b": [2.0] * len(profiles)})
        )
        df = ps.to_polars()
        assert {"a", "b"} <= set(df.columns)


class TestMultipleTestingCorrect:
    def test_default_p_source_is_canonical(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).multiple_testing_correct(fdr=0.10)
        df = ps.to_polars()
        assert "bhy_significant" in df.columns
        assert "p_adjusted" in df.columns
        assert "mt_p_source" in df.columns

    def test_whitelist_rejects_non_p_field(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(ValueError, match="not a valid p-value source"):
            ps.multiple_testing_correct(p_source="ic_ir")

    def test_whitelist_rejects_composed_name(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(ValueError, match="not a valid p-value source"):
            ps.multiple_testing_correct(p_source="min_p")

    def test_unknown_method(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        with pytest.raises(ValueError, match="Unknown multiple-testing method"):
            ps.multiple_testing_correct(method="bonferroni")  # type: ignore[arg-type]

    def test_empty_set_adds_columns(self):
        ps = ProfileSet([], profile_cls=CrossSectionalProfile)
        adj = ps.multiple_testing_correct(fdr=0.05)
        assert "bhy_significant" in adj.to_polars().columns

    def test_preserves_typed_profiles(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).multiple_testing_correct(fdr=0.50)
        # The profile dataclasses themselves are unchanged
        for p in ps.iter_profiles():
            assert isinstance(p, CrossSectionalProfile)

    def test_filter_on_bhy_significant(self, cs_profiles_and_artifacts):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).multiple_testing_correct(fdr=0.50)
        strong = ps.filter(pl.col("bhy_significant"))
        # strong_profile (signal_coef=0.5) should definitely survive
        names = [p.factor_name for p in strong.iter_profiles()]
        assert "strong" in names

    def test_whitelist_member_accepted(self, cs_profiles_and_artifacts):
        """p_source can name any whitelisted field, not just canonical_p."""
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles).multiple_testing_correct(
            p_source="spread_p", fdr=0.10,
        )
        df = ps.to_polars()
        assert df["mt_p_source"].unique().to_list() == ["spread_p"]
        # Adjusted p must line up with the spread_p input (same row order).
        adj = df["p_adjusted"].to_list()
        assert len(adj) == len(profiles)

    def test_rejects_post_registration_invariant_break(
        self, cs_profiles_and_artifacts,
    ):
        profiles, _ = cs_profiles_and_artifacts
        ps = ProfileSet(profiles)
        original = CrossSectionalProfile.CANONICAL_P_FIELD
        try:
            # Simulate a downstream monkey-patch that silently puts the
            # class back into an invalid shape. BHY must refuse rather
            # than feed the caller a meaningless correction.
            CrossSectionalProfile.CANONICAL_P_FIELD = "not_a_real_field"
            with pytest.raises(RuntimeError, match="@register_profile"):
                ps.multiple_testing_correct(fdr=0.05)
        finally:
            CrossSectionalProfile.CANONICAL_P_FIELD = original
