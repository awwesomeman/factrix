"""Deflated Sharpe Ratio and Probability of Backtest Overfitting."""

from __future__ import annotations

import math
from itertools import combinations, pairwise

import factrix as fx
import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix.stats import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from scipy.stats import rankdata

# Bailey & López de Prado (2014) numerical example: T = 1250, annualized
# SR = 2.5 and V = 0.5 over 250 periods a year, N = 100, skew -3, kurtosis 10.
PAPER = {
    "n_obs": 1250,
    "n_trials": 100,
    "sharpe_variance": 0.5 / 250,
    "skew": -3.0,
    "kurtosis": 10.0,
}
PAPER_SHARPE = 2.5 / math.sqrt(250)


def test_exported_from_stats_namespace() -> None:
    assert fx.stats.deflated_sharpe_ratio is deflated_sharpe_ratio
    assert (
        fx.stats.probability_of_backtest_overfitting
        is probability_of_backtest_overfitting
    )


class TestDeflatedSharpeRatio:
    def test_paper_numerical_example(self) -> None:
        # The paper reports DSR ≈ 0.9004 (SR_0 ≈ 0.1132 per period).
        assert deflated_sharpe_ratio(PAPER_SHARPE, **PAPER) == pytest.approx(
            0.9004, abs=5e-5
        )

    def test_hand_computed_zero_variance_reduces_to_psr_at_zero(self) -> None:
        # V = 0 puts SR_0 at 0: DSR = Φ(SR * sqrt(T - 1) / sqrt(1 - γ3 SR + (γ4-1)/4 SR²)).
        # SR = 0.1, T = 101, γ3 = 0, γ4 = 3: z = 0.1 * 10 / sqrt(1.005) = 0.99751.
        got = deflated_sharpe_ratio(
            0.1, n_obs=101, n_trials=10, sharpe_variance=0.0, skew=0.0, kurtosis=3.0
        )
        z = 1.0 / math.sqrt(1.005)
        assert got == pytest.approx(0.5 * (1 + math.erf(z / math.sqrt(2))), abs=1e-12)

    def test_falls_as_trial_count_rises(self) -> None:
        params = {k: v for k, v in PAPER.items() if k != "n_trials"}
        values = [
            deflated_sharpe_ratio(PAPER_SHARPE, n_trials=n, **params)
            for n in (2, 5, 10, 100, 1_000, 10_000)
        ]
        assert all(a > b for a, b in pairwise(values))

    def test_negative_skew_and_fat_tails_lower_dsr(self) -> None:
        normal = {**PAPER, "skew": 0.0, "kurtosis": 3.0}
        assert deflated_sharpe_ratio(PAPER_SHARPE, **PAPER) < deflated_sharpe_ratio(
            PAPER_SHARPE, **normal
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_trials", 1),
            ("n_trials", 2.5),
            ("n_trials", True),
            ("n_obs", 1),
            ("n_obs", 100.0),
            ("sharpe_variance", -1e-6),
            ("sharpe_variance", math.nan),
            ("skew", math.inf),
            ("kurtosis", math.nan),
            # Excess kurtosis passed by mistake: 0 < 1 + skew² for any skew.
            ("kurtosis", 0.0),
            ("kurtosis", 9.0),  # < 1 + (-3)² = 10
        ],
    )
    def test_rejects_invalid_input(self, field: str, value: object) -> None:
        with pytest.raises(UserInputError, match=field):
            deflated_sharpe_ratio(PAPER_SHARPE, **{**PAPER, field: value})

    def test_rejects_non_finite_sharpe(self) -> None:
        with pytest.raises(UserInputError, match="sharpe"):
            deflated_sharpe_ratio(math.nan, **PAPER)

    def test_rejects_zero_variance_sharpe_estimator(self) -> None:
        # Two-point distribution (kurtosis = 1 + skew²) at SR = 2 / skew makes
        # 1 - γ3 SR + (γ4 - 1)/4 SR² = 1 - 2 + 1 = 0.
        with pytest.raises(UserInputError, match="kurtosis"):
            deflated_sharpe_ratio(
                1.0,
                n_obs=100,
                n_trials=10,
                sharpe_variance=0.01,
                skew=2.0,
                kurtosis=5.0,
            )


def _blocks(*columns: list[list[float]]) -> np.ndarray:
    """Stack per-trial lists of blocks into a (T, N) matrix."""
    return np.column_stack([np.concatenate(col) for col in columns])


class TestProbabilityOfBacktestOverfitting:
    def test_hand_computed_always_overfit(self) -> None:
        # S = 2: split {0} trains on block 0, {1} on block 1. A wins block 0
        # (SR 2/√2 vs 1/√2) and loses block 1, B the reverse: both splits pick
        # an out-of-sample loser, PBO = 2/2.
        m = _blocks([[1, 3], [0, 2]], [[0, 2], [1, 3]])
        assert probability_of_backtest_overfitting(m, n_splits=2) == 1.0

    def test_hand_computed_never_overfit(self) -> None:
        m = _blocks([[1, 3], [1, 3]], [[0, 2], [0, 2]])
        assert probability_of_backtest_overfitting(m, n_splits=2) == 0.0

    def test_hand_computed_fraction(self) -> None:
        # S = 4, blocks [μ-1, μ+1]. B has μ = 0 everywhere, so its Sharpe is 0
        # on any half; A's Sharpe on a half has the sign of its two block
        # means' sum. A's means are [4, 2, -1, -3] (total 2):
        #   {0,1}: IS +6 A wins, OOS -4 A last     -> overfit
        #   {0,2}: IS +3 A wins, OOS -1 A last     -> overfit
        #   {0,3}: IS +1 A wins, OOS +1 A first    -> not
        #   {1,2}: IS +1 A wins, OOS +1 A first    -> not
        #   {1,3}: IS -1 B wins, OOS +3 B last     -> overfit
        #   {2,3}: IS -4 B wins, OOS +6 B last     -> overfit
        a = [[mu - 1, mu + 1] for mu in (4, 2, -1, -3)]
        b = [[-1, 1]] * 4
        m = _blocks(a, b)
        assert probability_of_backtest_overfitting(m, n_splits=4) == pytest.approx(
            4 / 6
        )

    def test_median_rank_counts_as_overfit(self) -> None:
        # N = 3: the in-sample winner landing at out-of-sample rank 2 gives
        # ω = 2/4, λ = 0, which the paper's indicator on (-∞, 0] counts.
        # Split {0}: A wins (SR 3/√2), then ranks 2nd on block 1 behind B.
        # Split {1}: B wins (3/√2), then ranks 2nd on block 0 behind A.
        m = _blocks(
            [[2, 4], [1, 3]],
            [[0, 2], [2, 4]],
            [[-1, 1], [-1, 1]],
        )
        assert probability_of_backtest_overfitting(m, n_splits=2) == 1.0

    def test_pure_noise_is_near_one_half(self) -> None:
        # Sweep over 2000 seeds of this design: mean 0.506, per-seed sd 0.205
        # (a single noise matrix ranges from 0.04 to 0.99), so a 200-seed mean
        # has sd ≈ 0.0145 and 0.5 ± 0.05 is about 3.5 sd.
        values = [
            probability_of_backtest_overfitting(
                np.random.default_rng(seed).standard_normal((320, 20)), n_splits=10
            )
            for seed in range(200)
        ]
        assert np.mean(values) == pytest.approx(0.5, abs=0.05)

    def test_real_signal_is_near_zero(self) -> None:
        # Same noise plus a 0.5-sd mean on one trial: over 200 seeds the
        # largest PBO measured was 0.008.
        values = []
        for seed in range(50):
            m = np.random.default_rng(seed).standard_normal((320, 20))
            m[:, 7] += 0.5
            values.append(probability_of_backtest_overfitting(m, n_splits=10))
        assert max(values) <= 0.02

    def test_matches_naive_cscv_across_batches(self) -> None:
        # S = 14 gives C(14, 7) = 3432 splits, more than one 1024-split batch.
        # The reference concatenates each half and ranks directly.
        s = 14
        m = np.random.default_rng(3).normal(0.01, 1.0, size=(140, 6))
        blocks = np.split(m, s)
        overfit = []
        for picked in combinations(range(s), s // 2):
            ins = np.concatenate([blocks[i] for i in picked])
            oos = np.concatenate([blocks[i] for i in range(s) if i not in picked])
            sr_is = ins.mean(axis=0) / ins.std(axis=0, ddof=1)
            sr_oos = oos.mean(axis=0) / oos.std(axis=0, ddof=1)
            omega = rankdata(sr_oos)[np.argmax(sr_is)] / (m.shape[1] + 1)
            overfit.append(np.log(omega / (1 - omega)) <= 0)
        assert probability_of_backtest_overfitting(m, n_splits=s) == np.mean(overfit)

    @pytest.mark.parametrize(
        ("shape", "n_splits", "match"),
        [
            ((40, 1), 4, "trial_returns"),  # one trial
            ((40,), 4, "trial_returns"),  # not 2-D
            ((40, 3), 3, "n_splits"),  # odd
            ((40, 3), 0, "n_splits"),
            ((40, 3), 4.0, "n_splits"),
            ((42, 3), 4, "n_splits"),  # does not divide T
            ((40, 3), 80, "n_splits"),  # more blocks than periods
            ((2, 3), 2, "at least 4 periods"),  # halves of one period
        ],
    )
    def test_rejects_invalid_shape_or_splits(
        self, shape: tuple[int, ...], n_splits: object, match: str
    ) -> None:
        m = np.random.default_rng(0).standard_normal(shape)
        with pytest.raises(UserInputError, match=match):
            probability_of_backtest_overfitting(m, n_splits=n_splits)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite_returns(self, bad: float) -> None:
        m = np.random.default_rng(0).standard_normal((40, 3))
        m[5, 1] = bad
        with pytest.raises(UserInputError, match="finite"):
            probability_of_backtest_overfitting(m, n_splits=4)

    def test_rejects_constant_trial(self) -> None:
        m = np.random.default_rng(0).standard_normal((40, 3))
        m[:, 2] = 0.1
        with pytest.raises(UserInputError, match="constant on 4 of 4 blocks"):
            probability_of_backtest_overfitting(m, n_splits=4)

    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_rejects_trial_constant_on_one_half_only(self, level: float) -> None:
        # Noisy in block 0, flat afterwards: the half of blocks {1, 2} has no
        # Sharpe ratio. Its variance from sums cancels to ~1e-16 rather than
        # 0, and a finiteness check alone returned PBO 1.0 (level 0) and 0.0
        # (level 1) for exactly these matrices.
        m = np.random.default_rng(1).standard_normal((40, 3))
        m[10:, 2] = level
        with pytest.raises(UserInputError, match="constant on 3 of 4 blocks"):
            probability_of_backtest_overfitting(m, n_splits=4)
