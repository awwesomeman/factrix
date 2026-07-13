"""``fx.multi_factor.bhy`` on the EvaluationResult contract."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import BhyResult, bhy
from factrix._results import MetricResult

from .conftest import make_result, make_spec


def test_returns_dict_keyed_by_metric_name_even_for_single_metric():
    make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.001, metric="ic") for i in range(5)]
    out = bhy(results, metrics=["ic"], q=0.05)
    assert isinstance(out, dict)
    assert set(out) == {"ic"}
    assert isinstance(out["ic"], BhyResult)


def test_multi_primary_runs_independent_screens():
    make_spec("ic")
    make_spec("ic_ir")
    results = [
        make_result(
            factor=f"f{i}",
            p=0.0001,
            metric="ic",
            extra_outputs={
                "ic_ir": MetricResult(
                    value=0.4,
                    p_value=0.5,
                    alternative="two-sided",
                    n_obs=100,
                    name="ic_ir",
                    metadata={"p_value": 0.5},
                )
            },
        )
        for i in range(4)
    ]
    out = bhy(results, metrics=["ic", "ic_ir"], q=0.05)
    assert set(out) == {"ic", "ic_ir"}
    assert len(out["ic"]) == 4
    assert len(out["ic_ir"]) == 0


def test_empty_input_raises():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        bhy([], metrics=["ic"], q=0.05)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1, float("nan"), True])
def test_q_must_be_open_unit_interval(q):
    make_spec("ic")
    results = [make_result(factor="f", p=0.01, metric="ic")]
    with pytest.raises(UserInputError, match="open interval"):
        bhy(results, metrics=["ic"], q=q)  # type: ignore[arg-type]


def test_dict_input_suggests_values():
    make_spec("ic")
    results = {
        f"f{i}": make_result(factor=f"f{i}", p=0.01, metric="ic") for i in range(3)
    }
    with pytest.raises(UserInputError, match=r"list\(results\.values\(\)\)"):
        bhy(results, metrics=["ic"], q=0.05)  # type: ignore[arg-type]


def test_list_of_dict_keys_suggests_values():
    make_spec("ic")
    results = {
        f"f{i}": make_result(factor=f"f{i}", p=0.01, metric="ic") for i in range(3)
    }
    mistaken: list[str] = []
    mistaken.extend(results)

    with pytest.raises(UserInputError, match=r"list\(results\.values\(\)\)"):
        bhy(mistaken, metrics=["ic"], q=0.05)  # type: ignore[arg-type]


def test_no_surviving_results_returns_empty_record():
    make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.9, metric="ic") for i in range(5)]
    out = bhy(results, metrics=["ic"], q=0.05)
    assert len(out["ic"]) == 0


def test_to_frame_keeps_eliminated_factors_adj_p():
    make_spec("ic")
    results = [make_result(factor=f"hit{i}", p=0.0001, metric="ic") for i in range(2)]
    results += [make_result(factor=f"miss{i}", p=0.6, metric="ic") for i in range(3)]
    out = bhy(results, metrics=["ic"], q=0.05)["ic"]

    frame = out.to_frame()
    # Every tested factor is present, not just survivors.
    assert frame.height == 5
    assert set(frame["factor"]) == {"hit0", "hit1", "miss0", "miss1", "miss2"}
    # Eliminated factors keep a finite adjusted p-value (the whole point) and
    # are flagged not-survived — they are no longer discarded.
    missed = frame.filter(~pl.col("survived"))
    assert missed.height == 3
    assert missed["adj_p"].is_finite().all()
    # The surviving-subset views remain the complement.
    assert {r.factor for r in out.survivors} == {"hit0", "hit1"}
    assert len(out.adj_p) == 2


def test_insufficient_short_circuits_are_dropped_from_family():
    make_spec("ic")
    valid = make_result(factor="valid", p=0.01, metric="ic")
    insufficient = make_result(
        factor="thin",
        p=1.0,
        metric="ic",
        value=float("nan"),
        metadata={"reason": "insufficient_ic_periods"},
    )
    out = bhy([valid, insufficient], metrics=["ic"], q=0.05)["ic"]

    assert out.n_tests == {(): 1}
    assert [r.factor for r in out.survivors] == ["valid"]


def test_expand_over_forward_periods_partitions_by_horizon():
    make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.001, metric="ic", forward_periods=1)
        for i in range(3)
    ] + [
        make_result(factor=f"f{i}", p=0.9, metric="ic", forward_periods=5)
        for i in range(3)
    ]
    out = bhy(results, metrics=["ic"], expand_over=("forward_periods",), q=0.05)
    assert out["ic"].expand_over == ("forward_periods",)
    assert set(out["ic"].n_tests) == {(1,), (5,)}
    survivor_factors = {r.factor for r in out["ic"].survivors}
    assert survivor_factors == {"f0", "f1", "f2"}


def test_expand_over_param_key():
    make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.001, metric="ic", params={"region": "US"})
        for i in range(3)
    ] + [
        make_result(factor=f"f{i}", p=0.9, metric="ic", params={"region": "EU"})
        for i in range(3)
    ]
    out = bhy(results, metrics=["ic"], expand_over=("region",), q=0.05)
    assert set(out["ic"].n_tests) == {("US",), ("EU",)}


def test_mixed_horizons_without_expand_over_warns():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, metric="ic", forward_periods=1),
        make_result(factor="f2", p=0.01, metric="ic", forward_periods=5),
    ]
    with pytest.warns(RuntimeWarning, match="mixes forward_periods"):
        bhy(results, metrics=["ic"], q=0.5)


def test_same_factor_at_different_horizons_is_two_hypotheses():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, metric="ic", forward_periods=1),
        make_result(factor="f1", p=0.02, metric="ic", forward_periods=5),
    ]
    with pytest.warns(RuntimeWarning, match="pooled"):
        out = bhy(results, metrics=["ic"], q=0.5)["ic"]
    assert out.n_tests == {(): 2}
    assert [(r.factor, r.forward_periods) for r in out.survivors] == [
        ("f1", 1),
        ("f1", 5),
    ]


def test_singleton_buckets_warn():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.001, metric="ic", params={"region": "US"}),
        make_result(factor="f2", p=0.001, metric="ic", params={"region": "EU"}),
    ]
    with pytest.warns(RuntimeWarning, match="single result"):
        bhy(results, metrics=["ic"], expand_over=("region",), q=0.5)


def test_primary_must_be_list():
    make_spec("ic")
    with pytest.raises(UserInputError, match="always a list"):
        bhy([make_result(factor="f", p=0.01, metric="ic")], metrics="ic")  # type: ignore[arg-type]


def test_primary_must_be_non_empty():
    with pytest.raises(UserInputError, match="non-empty"):
        bhy([], metrics=[])


def test_primary_element_must_be_str():
    with pytest.raises(UserInputError, match="str metric label"):
        bhy([], metrics=[123])  # type: ignore[list-item]


def test_duplicate_factor_and_horizon_without_expand_over_raises():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, metric="ic"),
        make_result(factor="f1", p=0.02, metric="ic"),
    ]
    with pytest.raises(UserInputError, match="unique"):
        bhy(results, metrics=["ic"])


def test_missing_primary_metric_raises():
    make_spec("ic")
    make_spec("alpha")
    results = [make_result(factor="f1", p=0.01, metric="ic")]
    with pytest.raises(UserInputError, match="other"):
        bhy(results, metrics=["other"])


def test_descriptive_metric_all_none_p_raises():
    """A descriptive metric (every result's p-value is ``None``) cannot be
    used for FDR control — ``bhy`` rejects it upfront with a clear message."""
    make_spec("mfe_mae")
    results = [make_result(factor=f"f{i}", p=None, metric="mfe_mae") for i in range(4)]
    with pytest.raises(UserInputError, match="FDR control"):
        bhy(results, metrics=["mfe_mae"])


def test_partial_none_p_raises_per_factor():
    """When only some results lack a p-value the pre-flight does not fire;
    resolution still raises on the offending factor."""
    make_spec("ic")
    results = [
        make_result(factor="f0", p=0.01, metric="ic"),
        make_result(factor="f1", p=None, metric="ic"),
    ]
    with pytest.raises(UserInputError, match="f1"):
        bhy(results, metrics=["ic"])


def test_factor_as_expand_over_key_raises():
    make_spec("ic")
    results = [make_result(factor="f1", p=0.01, metric="ic")]
    with pytest.raises(UserInputError, match="hypothesis identifier"):
        bhy(results, metrics=["ic"], expand_over=("factor",))


def test_missing_param_key_raises():
    make_spec("ic")
    results = [make_result(factor="f1", p=0.01, metric="ic", params={"region": "US"})]
    with pytest.raises(UserInputError, match="universe"):
        bhy(results, metrics=["ic"], expand_over=("universe",))


def test_missing_param_key_aggregates_all_offenders():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, metric="ic", params={"region": "US"}),
        make_result(factor="f2", p=0.01, metric="ic", params={}),
        make_result(factor="f3", p=0.01, metric="ic", params={"region": "EU"}),
    ]
    with pytest.raises(UserInputError) as excinfo:
        bhy(results, metrics=["ic"], expand_over=("region",))
    msg = str(excinfo.value)
    # Only the two gaps are reported; the result that carries the key is absent.
    assert "factor='f2' missing 'region'" in msg
    assert "factor='f1'" not in msg
    assert "factor='f3'" not in msg


def test_missing_param_key_aggregates_multiple_keys():
    make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, metric="ic", params={"region": "US"}),
        make_result(factor="f2", p=0.01, metric="ic", params={"sector": "tech"}),
    ]
    with pytest.raises(UserInputError) as excinfo:
        bhy(results, metrics=["ic"], expand_over=("region", "sector"))
    msg = str(excinfo.value)
    assert "factor='f1' missing 'sector'" in msg
    assert "factor='f2' missing 'region'" in msg


def test_adj_p_monotonic_within_bucket():
    make_spec("ic")
    p_values = [0.001, 0.01, 0.02, 0.5, 0.9]
    results = [
        make_result(factor=f"f{i}", p=p, metric="ic") for i, p in enumerate(p_values)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = bhy(results, metrics=["ic"], q=0.99)
    frame = out["ic"].to_frame()
    assert frame.height == len(p_values)
    adj_p = frame["adj_p"].to_numpy()
    assert adj_p[0] <= adj_p[1] <= adj_p[2]
    assert np.all(adj_p <= 1.0)


def test_bhy_result_repr_and_html():
    make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.001, metric="ic") for i in range(3)]
    out = bhy(results, metrics=["ic"], q=0.5)["ic"]
    text = repr(out)
    assert "BhyResult" in text
    assert "f0" in text
    html = out._repr_html_()
    assert "<table" in html and "adj_p" in html


def test_params_join_identity_so_swept_knob_pools_in_one_family():
    """A knob swept in `params` disambiguates without splitting the family.

    Before params joined the identity, the same factor evaluated across
    several `base_tf` values collided on `(factor, forward_periods)` and the
    only pooling-safe escape was encoding the knob into the factor name.
    """
    make_spec("ic")
    results = [
        make_result(factor="mom", p=0.001, metric="ic", params={"base_tf": tf})
        for tf in ("1h", "4h", "1d")
    ]
    out = bhy(results, metrics=["ic"], q=0.05)

    # One family of three — not three families, and not a duplicate-id raise.
    assert out["ic"].expand_over == ()
    assert out["ic"].n_tests == {(): 3}


def test_metadata_does_not_disambiguate_a_duplicate_hypothesis():
    """Bookkeeping labels must not rescue a genuinely duplicated hypothesis."""
    make_spec("ic")
    results = [
        make_result(factor="mom", p=0.001, metric="ic", result_metadata={"run_id": run})
        for run in ("a", "b")
    ]
    with pytest.raises(UserInputError, match="metadata is bookkeeping"):
        bhy(results, metrics=["ic"], q=0.05)


def test_expand_over_on_a_metadata_key_is_rejected():
    """Partitioning a family on bookkeeping is a category error, not a lookup miss."""
    make_spec("ic")
    results = [
        make_result(
            factor=f"f{i}",
            p=0.001,
            metric="ic",
            result_metadata={"run_id": "a"},
        )
        for i in range(3)
    ]
    with pytest.raises(UserInputError, match="never partitions a family"):
        bhy(results, metrics=["ic"], expand_over=("run_id",), q=0.05)


def test_params_keys_ride_along_so_distinct_axes_do_not_collide():
    """Same value on different param keys must stay two hypotheses."""
    make_spec("ic")
    results = [
        make_result(factor="mom", p=0.001, metric="ic", params={"base_tf": "1h"}),
        make_result(factor="mom", p=0.002, metric="ic", params={"universe": "1h"}),
    ]
    out = bhy(results, metrics=["ic"], q=0.05)
    assert out["ic"].n_tests[()] == 2
