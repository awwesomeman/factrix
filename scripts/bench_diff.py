"""Ad-hoc before / after compute_s + peak_rss_mb ratio for two bench output dirs.

Usage::

    python scripts/bench_diff.py <before-dir> <after-dir>

Prints a markdown table to stdout for direct paste into PR descriptions.
Compatibility-gates schema_version / metric_set_version /
env.dataset_spec_version / axis_cell / cache_state — refuses to compare
mismatched runs (exit 1) rather than silently coercing.

Scope: PR-description helper for the #378 optimisation loop. Not a
long-term CLI; if it grows past ad-hoc shape, open a follow-up issue to
promote into a proper ``bench.compare`` module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_COMPAT_FIELDS = (
    "schema_version",
    "metric_set_version",
    "axis_cell",
    "cache_state",
)


def _compat_fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        *(record[k] for k in _COMPAT_FIELDS),
        record["env"]["dataset_spec_version"],
    )


def _scale_key(record: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(record["scale"].items()))


def _load(
    dirpath: Path,
) -> dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]]:
    if not dirpath.is_dir():
        raise SystemExit(f"ERROR: not a directory: {dirpath}")
    records: dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]] = {}
    for f in sorted(dirpath.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("is_warmup") or r.get("status") != "ok":
                continue
            key = (r["scenario_id"], _scale_key(r))
            if key in records:
                raise SystemExit(f"ERROR: duplicate scenario key {key!r} in {dirpath}")
            records[key] = r
    return records


def _scale_label(scale: dict[str, Any]) -> str:
    for k in ("n_factors", "n_events"):
        if k in scale:
            return f"{k}={scale[k]}"
    return "-"


def _format_table(
    before: dict[tuple[str, Any], dict[str, Any]],
    after: dict[tuple[str, Any], dict[str, Any]],
) -> str:
    lines = [
        "| scenario_id | scale | compute_s_before | compute_s_after | ratio | peak_rss_mb_ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(before):
        scenario_id, scale_key = key
        b = before[key]
        a = after[key]
        b_c = b["compute_s"]
        a_c = a["compute_s"]
        ratio = a_c / b_c if b_c else float("nan")
        b_rss = b["peak_rss_mb"]
        a_rss = a["peak_rss_mb"]
        rss_ratio = a_rss / b_rss if b_rss else float("nan")
        lines.append(
            f"| {scenario_id} | {_scale_label(dict(scale_key))} | "
            f"{b_c:.3f} | {a_c:.3f} | {ratio:.3f} | {rss_ratio:.3f} |"
        )
    return "\n".join(lines)


def diff(before_dir: Path, after_dir: Path) -> str:
    """Return the markdown ratio table. Raises SystemExit on incompatibility."""
    before = _load(before_dir)
    after = _load(after_dir)

    missing_in_after = sorted(set(before) - set(after))
    missing_in_before = sorted(set(after) - set(before))
    if missing_in_after or missing_in_before:
        lines = ["ERROR: scenario set mismatch between before and after"]
        if missing_in_after:
            lines.append(f"  missing in after: {missing_in_after}")
        if missing_in_before:
            lines.append(f"  missing in before: {missing_in_before}")
        raise SystemExit("\n".join(lines))

    mismatches: list[str] = []
    for key in sorted(before):
        b_fp = _compat_fingerprint(before[key])
        a_fp = _compat_fingerprint(after[key])
        if b_fp != a_fp:
            diffs = {
                field: (b_fp[i], a_fp[i])
                for i, field in enumerate((*_COMPAT_FIELDS, "env.dataset_spec_version"))
                if b_fp[i] != a_fp[i]
            }
            mismatches.append(f"  {key[0]} ({_scale_label(dict(key[1]))}): {diffs}")
    if mismatches:
        raise SystemExit(
            "ERROR: compatibility mismatch — refusing to compare\n"
            + "\n".join(mismatches)
        )

    return _format_table(before, after)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ad-hoc before/after ratio for two bench output dirs."
    )
    p.add_argument("before_dir", type=Path)
    p.add_argument("after_dir", type=Path)
    args = p.parse_args(argv)
    print(diff(args.before_dir, args.after_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
