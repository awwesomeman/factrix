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

    def test_factor_rank_uses_mean_abs_not_abs_mean(self):
        """Two factors whose per-date rank correlation flips sign across
        dates are still redundant for stock selection (same mechanism,
        sign flip). mean(|rho|) reports high redundancy; |mean(rho)|
        would wrongly report near zero."""
        import numpy as np
        import polars as pl
        from datetime import datetime, timedelta
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation.pipeline import build_artifacts
        from factorlib.evaluation.profiles import CrossSectionalProfile

        # Build two factors: factor B equals factor A on odd dates and
        # -factor A on even dates. Per-date Spearman alternates +1/-1,
        # averaging to zero; but |rho| is always 1 → mean |rho| = 1.
        rng = np.random.default_rng(777)
        n_dates, n_assets = 40, 20
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]

        rows_a, rows_b = [], []
        for d_i, d in enumerate(dates):
            f = rng.standard_normal(n_assets)
            r = 0.4 * f + 0.6 * rng.standard_normal(n_assets)
            sign = 1.0 if d_i % 2 == 0 else -1.0
            for i in range(n_assets):
                rows_a.append({"date": d, "asset_id": f"a{i}",
                               "factor": float(f[i]),
                               "forward_return": float(r[i])})
                rows_b.append({"date": d, "asset_id": f"a{i}",
                               "factor": float(sign * f[i]),
                               "forward_return": float(r[i])})
        df_a = pl.DataFrame(rows_a).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        df_b = pl.DataFrame(rows_b).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        cfg = CrossSectionalConfig()
        art_a = build_artifacts(df_a, cfg); art_a.factor_name = "A"
        art_b = build_artifacts(df_b, cfg); art_b.factor_name = "B"
        profiles = [
            CrossSectionalProfile.from_artifacts(art_a),
            CrossSectionalProfile.from_artifacts(art_b),
        ]
        ps = ProfileSet(profiles)

        m = redundancy_matrix(
            ps, method="factor_rank", artifacts={"A": art_a, "B": art_b},
        )
        rho_ab = m.filter(pl.col("factor") == "A")["B"].item()
        # mean(|rho|) should be near 1 (per-date rhos are +/-1).
        # |mean(rho)| would give ~0 which would be the bug.
        assert rho_ab > 0.9, (
            f"Expected mean|rho| ~ 1.0 for sign-flipping factors; "
            f"got {rho_ab:.4f} -- suggests |mean(rho)| aggregation bug."
        )


class TestDegenerateInputs:
    def test_zero_variance_factor_warns_and_maxes_redundancy(self):
        """A constant factor (every rank tied) is upstream-broken. Report
        it as maximally redundant with others so downstream filters drop
        it, rather than silently reporting 0 redundancy (orthogonal)."""
        import numpy as np
        import polars as pl
        from datetime import datetime, timedelta
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation.pipeline import build_artifacts
        from factorlib.evaluation.profiles import CrossSectionalProfile

        rng = np.random.default_rng(4242)
        n_dates, n_assets = 40, 20
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]

        rows_good, rows_const = [], []
        for d in dates:
            f_good = rng.standard_normal(n_assets)
            r = 0.4 * f_good + 0.6 * rng.standard_normal(n_assets)
            for i in range(n_assets):
                rows_good.append({"date": d, "asset_id": f"a{i}",
                                  "factor": float(f_good[i]),
                                  "forward_return": float(r[i])})
                rows_const.append({"date": d, "asset_id": f"a{i}",
                                   "factor": 1.0,
                                   "forward_return": float(r[i])})

        df_good = pl.DataFrame(rows_good).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        df_const = pl.DataFrame(rows_const).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )

        cfg = CrossSectionalConfig()
        art_good = build_artifacts(df_good, cfg); art_good.factor_name = "good"
        art_const = build_artifacts(df_const, cfg); art_const.factor_name = "const"
        profiles = [
            CrossSectionalProfile.from_artifacts(art_good),
            CrossSectionalProfile.from_artifacts(art_const),
        ]
        ps = ProfileSet(profiles)
        with pytest.warns(UserWarning, match="zero-variance"):
            m = redundancy_matrix(
                ps,
                method="value_series",
                artifacts={"good": art_good, "const": art_const},
            )
        # Off-diagonal is 1.0 (maximally redundant → surfaces the bug).
        rho = m.filter(pl.col("factor") == "good")["const"].item()
        assert rho == pytest.approx(1.0), (
            f"Zero-variance factor should be marked |ρ|=1 to force "
            f"downstream deduplication; got {rho}"
        )

    def test_staggered_history_warns(self):
        """Factors with differently-sized value series trigger the
        intersection warning so users aren't blindsided by the tighter
        effective window."""
        import numpy as np
        import polars as pl
        from datetime import datetime, timedelta
        from factorlib.config import CrossSectionalConfig
        from factorlib.evaluation.pipeline import build_artifacts
        from factorlib.evaluation.profiles import CrossSectionalProfile

        rng = np.random.default_rng(5151)
        n_assets = 20

        def _panel(n_dates: int, start_offset: int, seed_salt: int) -> pl.DataFrame:
            dates = [
                datetime(2024, 1, 1) + timedelta(days=i + start_offset)
                for i in range(n_dates)
            ]
            rng_local = np.random.default_rng(5151 + seed_salt)
            rows = []
            for d in dates:
                f = rng_local.standard_normal(n_assets)
                r = 0.4 * f + 0.6 * rng_local.standard_normal(n_assets)
                for i in range(n_assets):
                    rows.append({"date": d, "asset_id": f"a{i}",
                                 "factor": float(f[i]),
                                 "forward_return": float(r[i])})
            return pl.DataFrame(rows).with_columns(
                pl.col("date").cast(pl.Datetime("ms"))
            )

        df_full = _panel(n_dates=60, start_offset=0, seed_salt=1)
        df_short = _panel(n_dates=30, start_offset=30, seed_salt=2)

        cfg = CrossSectionalConfig()
        art_full = build_artifacts(df_full, cfg); art_full.factor_name = "full"
        art_short = build_artifacts(df_short, cfg); art_short.factor_name = "short"
        profiles = [
            CrossSectionalProfile.from_artifacts(art_full),
            CrossSectionalProfile.from_artifacts(art_short),
        ]
        ps = ProfileSet(profiles)
        with pytest.warns(UserWarning, match="missing dates"):
            redundancy_matrix(
                ps,
                method="value_series",
                artifacts={"full": art_full, "short": art_short},
            )


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
