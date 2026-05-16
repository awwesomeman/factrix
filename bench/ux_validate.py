"""``python -m bench.ux_validate <jsonl-dir>`` — assert UX latency targets.

Reads every ``*.jsonl`` under the given directory, picks the measured
rows (``is_warmup=false ∧ status="ok"``), and asserts each row's
``wall_s`` against ``bench.ux_targets.UX_TARGETS``.

Output is a markdown table to stdout. Exit code is non-zero when any
row fails its target (a *red flag*); stretch rows, scenarios not in
the target table, and ``status="oom" / "error"`` rows are reported
but do not block. The latter matters at ``xlarge`` — an OOM on a
1000-factor screen is real data, not a CI failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bench.ux_targets import UX_TARGETS_VERSION, UxTarget, lookup_target

RowVerdict = Literal["pass", "fail", "stretch", "skip"]


@dataclass(frozen=True)
class RowReport:
    """One asserted row in the UX validation report."""

    scenario_id: str
    wall_s: float | None
    wall_s_max: float | None
    verdict: RowVerdict
    peak_rss_mb: float | None
    n_threads: int
    n_factors: int | None
    status: str
    source_file: str


@dataclass(frozen=True)
class IncidentReport:
    """Non-blocking row (oom / error / unknown scenario)."""

    scenario_id: str
    status: str
    error_message: str | None
    source_file: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_n_factors(scale: dict[str, Any]) -> int | None:
    value = scale.get("n_factors")
    return int(value) if isinstance(value, int) else None


def _classify(row: dict[str, Any], target: UxTarget | None) -> RowVerdict:
    if target is None:
        return "skip"
    if target.wall_s_max is None:
        return "stretch"
    wall_s = row.get("wall_s")
    if wall_s is None:
        return "fail"
    return "pass" if wall_s <= target.wall_s_max else "fail"


def evaluate_dir(jsonl_dir: Path) -> tuple[list[RowReport], list[IncidentReport]]:
    """Read every JSONL in ``jsonl_dir`` and classify each measured row.

    Returns ``(row_reports, incidents)`` where ``incidents`` collects
    OOM / error / unknown-scenario rows that should appear in the
    report but not gate the exit code.
    """
    rows: list[RowReport] = []
    incidents: list[IncidentReport] = []
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for raw in _read_jsonl(path):
            scenario_id = raw["scenario_id"]
            status = raw["status"]
            if status != "ok":
                incidents.append(
                    IncidentReport(
                        scenario_id=scenario_id,
                        status=status,
                        error_message=raw.get("error_message"),
                        source_file=path.name,
                    )
                )
                continue
            if raw.get("is_warmup", False):
                continue
            scale = raw.get("scale", {})
            n_factors = _extract_n_factors(scale)
            target = lookup_target(scenario_id, n_factors=n_factors)
            verdict = _classify(raw, target)
            rows.append(
                RowReport(
                    scenario_id=scenario_id,
                    wall_s=raw.get("wall_s"),
                    wall_s_max=target.wall_s_max if target else None,
                    verdict=verdict,
                    peak_rss_mb=raw.get("peak_rss_mb"),
                    n_threads=int(raw.get("env", {}).get("omp_threads", 1)),
                    n_factors=n_factors,
                    status=status,
                    source_file=path.name,
                )
            )
    return rows, incidents


_VERDICT_GLYPH: dict[RowVerdict, str] = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "stretch": "➖ stretch",
    "skip": "➖ skip",
}


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def render_markdown(rows: list[RowReport], incidents: list[IncidentReport]) -> str:
    """Render the asserted rows + non-blocking incidents as markdown."""
    lines: list[str] = []
    lines.append(f"## UX validation (UX_TARGETS_VERSION={UX_TARGETS_VERSION})")
    lines.append("")
    lines.append(
        "| scenario | n_factors | wall_s | target | verdict | "
        "peak_rss_mb | n_threads | source |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.scenario_id} | {r.n_factors if r.n_factors is not None else '—'} "
            f"| {_fmt(r.wall_s)} | {_fmt(r.wall_s_max)} | "
            f"{_VERDICT_GLYPH[r.verdict]} | {_fmt(r.peak_rss_mb, '.0f')} | "
            f"{r.n_threads} | {r.source_file} |"
        )
    if incidents:
        lines.append("")
        lines.append("### Incidents (non-blocking)")
        lines.append("")
        lines.append("| scenario | status | message | source |")
        lines.append("|---|---|---|---|")
        for inc in incidents:
            msg = (inc.error_message or "").replace("|", "\\|")
            lines.append(
                f"| {inc.scenario_id} | {inc.status} | {msg} | {inc.source_file} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — assert JSONL dir vs UX targets."""
    parser = argparse.ArgumentParser(
        prog="python -m bench.ux_validate",
        description="Assert benchmark JSONL records against UX latency targets.",
    )
    parser.add_argument(
        "jsonl_dir",
        type=Path,
        help="Directory containing <scenario_id>.jsonl files to validate.",
    )
    args = parser.parse_args(argv)

    if not args.jsonl_dir.is_dir():
        parser.error(f"not a directory: {args.jsonl_dir}")

    rows, incidents = evaluate_dir(args.jsonl_dir)
    sys.stdout.write(render_markdown(rows, incidents))
    red_flag = any(r.verdict == "fail" for r in rows)
    return 1 if red_flag else 0


if __name__ == "__main__":
    raise SystemExit(main())
