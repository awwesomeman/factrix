"""JSONL record schema for the benchmark harness.

Pinned here so downstream consumers do not need to roll their own
parser. ``scale`` is an open schema keyed on ``axis_cell`` — new
axis cells extend the union without breaking existing parsers.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"

AxisCell = Literal[
    "continuous_individual_panel",
    "sparse_individual_panel",
]

Status = Literal["ok", "oom", "error", "timeout"]
CacheState = Literal["cold", "warm", "unknown"]


class ContinuousIndividualPanelScale(BaseModel):
    """Scale shape for the `continuous_individual_panel` axis cell."""

    model_config = ConfigDict(extra="forbid")
    axis_cell: Literal["continuous_individual_panel"]
    n_factors: int = Field(ge=1)
    n_dates: int = Field(ge=1)
    n_assets: int = Field(ge=1)


class SparseIndividualPanelScale(BaseModel):
    """Scale shape for the `sparse_individual_panel` axis cell."""

    model_config = ConfigDict(extra="forbid")
    axis_cell: Literal["sparse_individual_panel"]
    n_events: int = Field(ge=0)
    n_assets: int = Field(ge=1)
    n_dates: int = Field(ge=1)
    window_pre: int = Field(ge=0)
    window_post: int = Field(ge=0)


Scale = Annotated[
    ContinuousIndividualPanelScale | SparseIndividualPanelScale,
    Field(discriminator="axis_cell"),
]


class Env(BaseModel):
    """Runtime environment fingerprint embedded in every record."""

    model_config = ConfigDict(extra="forbid")
    git_sha: str
    factrix_version: str
    dataset_spec_version: str
    python: str
    numpy: str
    blas: str
    omp_threads: int = Field(ge=1)
    cpu_model: str
    cpu_cores: int = Field(ge=1)
    ram_gb: float = Field(ge=0)
    os: str


class BenchRecord(BaseModel):
    """One row of the benchmark JSONL."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    scenario_id: str
    metric_set: str
    metric_set_version: str
    axis_cell: AxisCell
    scale: Scale
    run_idx: int = Field(ge=0)
    is_warmup: bool
    cache_state: CacheState
    status: Status
    error_message: str | None
    started_at: str  # ISO8601 UTC, e.g. "2026-05-15T08:30:00Z"
    wall_s: float | None = Field(ge=0)
    setup_s: float | None = Field(ge=0)
    compute_s: float | None = Field(ge=0)
    cpu_s: float | None = Field(ge=0)
    peak_rss_mb: float | None = Field(ge=0)
    peak_alloc_mb: float | None = Field(ge=0)
    env: Env

    @model_validator(mode="after")
    def _scale_axis_cell_matches(self) -> BenchRecord:
        inner = getattr(self.scale, "axis_cell", None)
        if inner != self.axis_cell:
            raise ValueError(
                f"axis_cell mismatch: record={self.axis_cell!r} but scale={inner!r}"
            )
        return self


def record_json_schema() -> dict[str, Any]:
    """Export the JSON Schema for ``BenchRecord``."""
    return BenchRecord.model_json_schema()
