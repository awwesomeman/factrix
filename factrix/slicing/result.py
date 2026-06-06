"""Container returned by :func:`factrix.slicing.by_slice`.

``SliceResult`` is a ``Mapping[str, MetricResult]`` — every existing
``dict``-shaped consumer (``for k, v in result.items()``,
``result["bull"]``, ``len(result)``) keeps working. The added value is
:meth:`SliceResult.to_frame`: a fixed-schema long-form ``pl.DataFrame``
that flattens per-slice ``MetricResult`` rows for plotting,
leaderboards, and Notebook rendering.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import polars as pl

from factrix._results import MetricResult


class SliceResult(Mapping[str, MetricResult]):
    """Mapping of slice label → :class:`MetricResult` with a frame renderer.

    Returned by :func:`factrix.slicing.by_slice`. Iteration order matches
    the upstream ``polars.DataFrame.partition_by`` order (insertion-order
    of distinct values, not lexicographic) — call ``.sort("slice")`` on
    the rendered frame if a stable order is needed downstream.

    The container deliberately exposes only the universal projection
    (``value``, ``stat``, ``p``). For metric-specific
    metadata (``tie_ratio``, ``shanken_correction``, ...), build the
    DataFrame directly from ``[(k, m.metadata) for k, m in result.items()]``.

    Warning:
        ``p`` in the rendered frame is the **per-slice** marginal
        p-value computed by the metric on that slice alone — *not*
        adjusted for the K parallel tests across slices. Filtering
        ``df.filter(pl.col("p_value") < 0.05)`` across K=10 sectors
        inflates the family-wise error rate (FWER); under H0 you expect
        ≈ 0.4 "significant" slices by chance. For cross-slice
        inference with FWER / false discovery rate (FDR) control, use
        :func:`factrix.slice_pairwise_test` (Holm / Romano-Wolf /
        Bonferroni) or :func:`factrix.slice_joint_test` (omnibus χ²)
        instead. The container is for exploration; the inference
        functions are for claims.

    Examples:
        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic, compute_ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=240)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> ic_df = compute_ic(panel)["factor"].with_columns(
        ...     pl.col("date").dt.year().alias("year")
        ... )
        >>> per_year = fx.by_slice(ic, ic_df, label="year")
        >>> isinstance(per_year, fx.SliceResult)
        True
        >>> len(per_year) >= 1
        True
    """

    def __init__(self, data: Mapping[str, MetricResult]) -> None:
        self._data: dict[str, MetricResult] = dict(data)

    def __getitem__(self, key: str) -> MetricResult:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"SliceResult({self._data!r})"

    def to_frame(self, *, slice_col: str = "slice") -> pl.DataFrame:
        """Render slices as a long-form ``pl.DataFrame``.

        Schema (always, in order): ``slice_col``, ``value``, ``stat``,
        ``p_value``. ``stat`` and ``p_value`` are ``None`` when the
        underlying ``MetricResult`` does not carry them (descriptive
        metric, short-circuit failure path, etc.).

        Args:
            slice_col: Output column name for the slice label. Default
                ``"slice"``; override when the caller needs to avoid a
                clash with an existing identity column they intend to
                join in upstream.

        Returns:
            ``pl.DataFrame`` with one row per slice. Row order follows
            iteration order of ``self``.

        Examples:
            >>> import polars as pl
            >>> import factrix as fx
            >>> from factrix.preprocess import compute_forward_return
            >>> from factrix.metrics import ic, compute_ic
            >>> raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=240)
            >>> panel = compute_forward_return(raw, forward_periods=5)
            >>> ic_df = compute_ic(panel)["factor"].with_columns(
            ...     pl.col("date").dt.year().alias("year")
            ... )
            >>> per_year = fx.by_slice(ic, ic_df, label="year")
            >>> df = per_year.to_frame()
            >>> set(df.columns) >= {"slice", "value"}
            True

            Override the slice column name to avoid collision:

            >>> df2 = per_year.to_frame(slice_col="year")
            >>> "year" in df2.columns
            True
        """
        cols = (slice_col, "value", "stat", "p_value")
        rows: dict[str, list[object]] = {c: [] for c in cols}
        for key, m in self._data.items():
            rows[slice_col].append(key)
            rows["value"].append(m.value)
            rows["stat"].append(m.stat)
            rows["p_value"].append(float(m.p_value) if isinstance(m.p_value, int | float) else None)
        schema: dict[str, type[pl.DataType]] = {
            slice_col: pl.Utf8,
            "value": pl.Float64,
            "stat": pl.Float64,
            "p_value": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()
