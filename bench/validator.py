"""Self-validation of harness output (#380 §9.1).

The harness reads back every JSONL it writes through ``validate_file``;
the same code path is exposed as the ``python -m bench.validate`` CLI
for ad-hoc auditing of historical baselines.

Fail-loud is intentional: a silent shape drift now is an unparseable
baseline six months from now.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from bench.schema import SCHEMA_VERSION, BenchRecord


@dataclass(frozen=True)
class ValidationFailure:
    """One bad row in a JSONL file."""

    line_no: int
    error: str


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating one JSONL file."""

    path: Path
    n_rows: int
    failures: list[ValidationFailure]

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_file(path: str | Path) -> ValidationReport:
    """Parse every line of ``path`` as ``BenchRecord``; collect failures."""
    p = Path(path)
    failures: list[ValidationFailure] = []
    n_rows = 0
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            n_rows += 1
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as e:
                failures.append(ValidationFailure(line_no, f"invalid JSON: {e}"))
                continue
            sv = data.get("schema_version")
            if sv != SCHEMA_VERSION:
                failures.append(
                    ValidationFailure(
                        line_no,
                        f"schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}",
                    )
                )
                continue
            try:
                BenchRecord.model_validate(data)
            except ValidationError as e:
                failures.append(ValidationFailure(line_no, str(e)))
    return ValidationReport(path=p, n_rows=n_rows, failures=failures)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate one or more JSONL files."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: python -m bench.validate <path.jsonl> [<path.jsonl> ...]")
        return 0 if args else 2

    overall_ok = True
    for path in args:
        if not Path(path).is_file():
            overall_ok = False
            print(f"FAIL  {path}  (file not found)", file=sys.stderr)
            continue
        report = validate_file(path)
        if report.ok:
            print(f"OK  {path}  ({report.n_rows} rows)")
        else:
            overall_ok = False
            print(
                f"FAIL  {path}  ({len(report.failures)} / {report.n_rows} rows)",
                file=sys.stderr,
            )
            for f in report.failures:
                print(f"  line {f.line_no}: {f.error}", file=sys.stderr)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
