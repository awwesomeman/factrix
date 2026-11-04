"""factrix's own lazy queries collect on the in-memory engine.

From polars 2 a bare ``LazyFrame.collect()`` runs on the streaming engine,
whose ``group_by`` ``mean`` / ``sum`` / ``corr`` can differ in the last bits
between two identical calls. Every statistic factrix returns would then be irreproducible at the
ulp level, and the batch == list-of-one contract of the batchable primitives
breaks. Repeated in-memory collects are bit-identical, and in-memory is what
``"auto"`` resolved to under polars 1, so factrix pins it on every query it
builds.

The user's own ``LazyFrame`` (collected once at the API boundary) is left on
the user's engine: that query and its engine choice belong to the caller.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics._helpers import _aggregate_to_per_date
from factrix.metrics._primitives._common_betas import compute_common_betas
from factrix.metrics._primitives._ic import compute_ic

FACTRIX_DIR = pathlib.Path("factrix")

# (module, enclosing function) of the one collect that runs the caller's query.
_USER_QUERY_COLLECTS = {("factrix/_data_input.py", "_coerce_data")}


def _collect_calls() -> list[tuple[str, str, int, ast.Call]]:
    """Every ``<expr>.collect(...)`` call under ``factrix/``."""
    found: list[tuple[str, str, int, ast.Call]] = []
    for path in sorted(FACTRIX_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "collect"
                ):
                    found.append((path.as_posix(), func.name, node.lineno, node))
    return found


def _engine(call: ast.Call) -> object:
    for kw in call.keywords:
        if kw.arg == "engine" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def test_every_factrix_query_pins_the_in_memory_engine() -> None:
    calls = _collect_calls()
    exempt = {(path, fn) for path, fn, _, _ in calls} & _USER_QUERY_COLLECTS
    # The scan must see the user-query collect, or the exemption names a site
    # that moved and the walk below proves nothing about it.
    assert exempt == _USER_QUERY_COLLECTS
    offenders = [
        f"{path}:{line} ({fn})"
        for path, fn, line, call in calls
        if (path, fn) not in _USER_QUERY_COLLECTS and _engine(call) != "in-memory"
    ]
    assert len(calls) > len(_USER_QUERY_COLLECTS)
    assert not offenders, offenders


def _panel(n_assets: int, n_dates: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.array(
        [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)],
        dtype="datetime64[us]",
    )
    return pl.DataFrame(
        {
            "date": np.tile(dates, n_assets),
            "asset_id": np.repeat([f"A{i:04d}" for i in range(n_assets)], n_dates),
            "factor": rng.standard_normal(n_assets * n_dates),
            "common": np.tile(rng.standard_normal(n_dates), n_assets),
            "forward_return": rng.standard_normal(n_assets * n_dates) * 0.02 + 0.001,
        }
    )


# The three producers whose output varied between identical calls on polars
# 2.1 without the pin. The panel is sized so the streaming engine splits each
# group across threads; on a single-threaded runner this cannot fail.
@pytest.mark.parametrize(
    "produce",
    [
        pytest.param(lambda d: compute_ic(d)["factor"], id="compute_ic"),
        pytest.param(
            lambda d: compute_common_betas(d, factor_cols=["common"])["common"],
            id="compute_common_betas",
        ),
        pytest.param(_aggregate_to_per_date, id="_aggregate_to_per_date"),
    ],
)
def test_repeated_calls_are_bitwise_identical(produce) -> None:
    data = _panel(300, 500, seed=1)
    first = produce(data)
    for _ in range(5):
        again = produce(data)
        for col in first.columns:
            if first[col].dtype.is_float():
                assert np.array_equal(
                    first[col].to_numpy(), again[col].to_numpy(), equal_nan=True
                ), col
