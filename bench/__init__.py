"""Benchmark harness foundation — wrapper, schema, validator.

Sub-issue #381 of #380. Design SSOT is the v1 planning comment on #380.
This package is dev tooling; not exported from the public ``factrix``
namespace.
"""

from bench.schema import (
    SCHEMA_VERSION,
    BenchRecord,
    Env,
    record_json_schema,
)

__all__ = [
    "SCHEMA_VERSION",
    "BenchRecord",
    "Env",
    "record_json_schema",
]
