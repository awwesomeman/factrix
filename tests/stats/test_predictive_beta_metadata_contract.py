"""``predictive_beta`` metadata must be reproducible from the keys it names.

Two contracts, both split out of the review of #1034:

- the ill-conditioned-bandwidth warning has to name the key a reader can look
  up (#1035). ``predictive_beta`` is the only metric whose ``n_periods`` is the
  truncated augmented-design row count rather than the count the screen reads,
  so the bare token sends a reader to the wrong number;
- the ``no_amihud_hurvich_fit`` short circuit runs no kernel, so its
  ``har_lags`` is the bandwidth resolved for the *attempt*. #1034 tightened the
  documented contract to "the bandwidth the kernel ran at", which that branch
  cannot satisfy; the value is kept and the exception is documented instead of
  overloading the ``None``-means-``h = 1`` sentinel (#1036).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factrix.metrics.predictive_beta import predictive_beta

STAT_KEYS_DOCS = Path("docs/reference/stat-keys-by-metric.md")


def _ts_panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    n = len(x)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "asset_id": ["A"] * n,
            "factor": x,
            "forward_return": y,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _run(n: int, h: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 0.1 * x + rng.standard_normal(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=h)
    return result, [str(w.message) for w in caught]


class TestIllConditionedMessageNamesALookupableKey:
    """#1035: the count printed must be the value under the key named."""

    @pytest.mark.parametrize(("n", "h"), [(90, 63), (60, 42), (120, 21)])
    def test_message_count_matches_the_metadata_key_it_names(
        self, n: int, h: int
    ) -> None:
        result, messages = _run(n, h, seed=n * 100 + h)
        assert not math.isnan(result.value)
        bandwidth = [m for m in messages if "hac_bandwidth_ill_conditioned" in m]
        assert bandwidth, "the bandwidth screen should fire in this regime"
        message = bandwidth[0]

        assert "n_periods_finite /" in message
        assert f"on the {result.metadata['n_periods_finite']} finite pairs" in message

    def test_message_does_not_send_a_reader_to_the_truncated_row_count(self) -> None:
        """``n_periods`` is the augmented design's rows; the screen reads neither."""
        result, messages = _run(90, 63, seed=9063)
        message = next(m for m in messages if "hac_bandwidth_ill_conditioned" in m)

        assert result.metadata["n_periods"] != result.metadata["n_periods_finite"]
        assert "n_periods /" not in message


class TestShortCircuitBandwidthIsTheAttemptedOne:
    """#1036: no kernel runs there, so the contract states the exception."""

    @pytest.mark.parametrize(("n", "h"), [(20, 19), (20, 16), (22, 18)])
    def test_short_circuit_keeps_the_resolved_bandwidth(self, n: int, h: int) -> None:
        """Pinned deliberately: ``None`` there would overload the ``h = 1`` sentinel."""
        result, _ = _run(n, h, seed=n * 10 + h)

        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_amihud_hurvich_fit"
        assert isinstance(result.metadata["har_lags"], int)

    def test_h1_still_reports_no_bandwidth(self) -> None:
        result, _ = _run(240, 1, seed=11)

        assert not math.isnan(result.value)
        assert result.metadata["har_lags"] is None

    def test_docs_state_the_short_circuit_exception(self) -> None:
        """The tightened ``har_lags`` contract must carry its one exception."""
        docs = STAT_KEYS_DOCS.read_text(encoding="utf-8")
        assert "no_amihud_hurvich_fit" in docs
        assert "the bandwidth resolved for the attempt" in docs

    def test_docs_do_not_repeat_at_at(self) -> None:
        """Typo introduced by #1034's wording; folded in here per review."""
        docs = STAT_KEYS_DOCS.read_text(encoding="utf-8")
        assert "ran at at" not in docs
