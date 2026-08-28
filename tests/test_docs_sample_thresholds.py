"""Drift guards for the sample-size numbers in ``metric-applicability.md``.

The page is hand-written and number-dense: a constants table that
restates every ``MIN_*`` literal, and a per-family ``Min sample`` column
that declares which tier gates each metric. Nothing generates either, so
both are parsed here and asserted against the live objects:

1. Constants table — every row's value, tier and source module are
   checked against the constant imported from that module.
2. Inline claims — the ``` `NAME` (value) ``` / ``` `NAME = value` ```
   forms scattered through the reference and development pages are
   checked against the same constants.
3. Per-metric floors — the HARD / WARN tiers each row claims are checked
   against the resolved :attr:`MetricSpec.sample_threshold`.

The floors in the spec are resolved at the metric's default overlap
horizon, so they are scaled multiples of the raw constants; only the
*presence* of a tier per axis is comparable, and that is what (3)
asserts.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest
from factrix._metric_index import MetricSpec, public_specs

APPLICABILITY = pathlib.Path("docs/reference/metric-applicability.md")
DOCS_ROOT = pathlib.Path("docs")

# Superseded design write-ups; they quote constants that no longer exist.
_EXCLUDED_DOC_DIRS = ("plans",)

_CONSTANT_TOKEN = r"(MIN_[A-Z0-9_]+|N_GROUPS_FLOOR|PERSISTENT_SERIES_AUTOCORR)"
_VALUE = r"([0-9]+(?:\.[0-9]+)?)"

_TABLE_ROW = re.compile(
    rf"^\| `{_CONSTANT_TOKEN}` \| {_VALUE} \| [^|]* \| (hard|warn) \| "
    r"`([^`]+)` \|",
    re.M,
)

_INLINE_CLAIMS = (
    # `MIN_X = 4`
    re.compile(rf"`{_CONSTANT_TOKEN}\s*=\s*{_VALUE}`"),
    # `MIN_X` = 4
    re.compile(rf"`{_CONSTANT_TOKEN}`\s*=\s*{_VALUE}\b"),
    # `MIN_X` (4)
    re.compile(rf"`{_CONSTANT_TOKEN}`\s*\(\s*{_VALUE}\s*\)"),
)

# Rows of the per-family ``Metric | Sample axis | Min sample`` tables.
_METRIC_ROW = re.compile(
    r"^\| \[`([a-z0-9_]+)`\]\[factrix\.metrics\.[a-z0-9_.]+\] \| [^|]* \| ([^|]*) \|",
    re.M,
)

# A metric whose gate cannot be declared as a static panel-shape floor, so
# ``sample_threshold`` is deliberately empty while the page documents the
# in-body floor. Each entry mirrors the reason stated at the metric.
_NO_STATIC_FLOOR: dict[str, str] = {
    # Gates within-period asset couples; DataProperties.n_pairs is a row
    # count, so the floor is enforced in-body only.
    "directional_pair_accuracy": "asset_pairs axis is not a DataProperties axis",
    # Per-offset event counts are factor-context-dependent.
    "event_around_return": "per-offset floor, no static panel-shape floor",
    # >= 2 events is a math-degeneracy guard, not the statistical floor.
    "signal_density": "in-body degeneracy guard, not a declared floor",
}


def _doc_pages() -> list[pathlib.Path]:
    return sorted(
        p
        for p in DOCS_ROOT.rglob("*.md")
        if not any(part in _EXCLUDED_DOC_DIRS for part in p.parts)
    )


def _constants_table() -> list[tuple[str, str, str, str]]:
    """Return ``(name, value, tier, source module path)`` per table row."""
    rows = _TABLE_ROW.findall(APPLICABILITY.read_text(encoding="utf-8"))
    assert rows, f"{APPLICABILITY}: no sample-size constants table rows parsed"
    return rows


def _module_of(source_path: str) -> str:
    return source_path.removesuffix(".py").replace("/", ".")


def _live_value(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


@pytest.fixture(scope="module")
def constant_namespace() -> dict[str, object]:
    """Every sample-size constant an inline claim may name, keyed by name.

    The page's own ``Source module`` column supplies the metric-side
    modules, so a new constant module needs no second declaration here;
    the two shared constant modules are added for the tokens that live
    outside the table (``N_GROUPS_FLOOR``, ``PERSISTENT_SERIES_AUTOCORR``).
    """
    namespace = {
        name: _live_value(_module_of(source), name)
        for name, _value, _tier, source in _constants_table()
    }
    for module_name in ("factrix._types", "factrix._stats.constants"):
        module = importlib.import_module(module_name)
        namespace |= {
            name: getattr(module, name)
            for name in vars(module)
            if re.fullmatch(_CONSTANT_TOKEN, name)
        }
    return namespace


def test_constants_table_matches_code() -> None:
    """Each table row's value, tier and source module must be live."""
    mismatches: list[str] = []
    for name, value, tier, source in _constants_table():
        module = _module_of(source)
        try:
            live = _live_value(module, name)
        except (AttributeError, ModuleNotFoundError):
            mismatches.append(f"{name}: not importable from {module}")
            continue
        if float(value) != float(live):  # type: ignore[arg-type]
            mismatches.append(f"{name}: doc says {value}, {module} says {live}")
        expected_tier = "hard" if name.endswith("_HARD") else "warn"
        if tier != expected_tier:
            mismatches.append(
                f"{name}: doc tier column says {tier!r}, name implies {expected_tier!r}"
            )
    assert not mismatches, (
        f"{APPLICABILITY} sample-size constants table is stale:\n  "
        + "\n  ".join(mismatches)
    )


def test_inline_constant_claims_match_code(
    constant_namespace: dict[str, object],
) -> None:
    """Inline ``NAME (value)`` / ``NAME = value`` claims must match the code."""
    mismatches: list[str] = []
    for page in _doc_pages():
        text = page.read_text(encoding="utf-8")
        for pattern in _INLINE_CLAIMS:
            for name, value in pattern.findall(text):
                if name not in constant_namespace:
                    mismatches.append(
                        f"{page}: `{name}` is not in the sample-size constants "
                        f"table of {APPLICABILITY}"
                    )
                    continue
                live = constant_namespace[name]
                if float(value) != float(live):  # type: ignore[arg-type]
                    mismatches.append(f"{page}: `{name}` = {value}, code says {live}")
    assert not mismatches, "Inline constant claims are stale:\n  " + "\n  ".join(
        mismatches
    )


def _declared_tiers(min_sample_cell: str) -> tuple[bool, bool]:
    """Return ``(claims a hard floor, claims a warn tier)`` for one doc cell."""
    hard = bool(re.search(r"_HARD\b|>=|≥", min_sample_cell))
    warn = bool(re.search(r"_WARN\b|warn", min_sample_cell))
    return hard, warn


def _spec_tiers(spec: MetricSpec) -> tuple[bool, bool]:
    """Return ``(has a min_* floor, has a warn_* floor)`` for one spec."""
    threshold = spec.sample_threshold
    axes = ("periods", "assets", "pairs", "events")
    return (
        any(getattr(threshold, f"min_{axis}") is not None for axis in axes),
        any(getattr(threshold, f"warn_{axis}") is not None for axis in axes),
    )


def _per_metric_rows() -> dict[str, str]:
    """Return ``metric name -> Min sample cell`` for the per-family tables.

    Only the ``Other metrics by family`` region carries floors; the
    event-study tables further down share the row shape but describe
    abnormal-return primitives.
    """
    text = APPLICABILITY.read_text(encoding="utf-8")
    region = text.split("## Other metrics by family", 1)[1].split(
        "## Sample-size constants", 1
    )[0]
    rows = dict(_METRIC_ROW.findall(region))
    assert rows, f"{APPLICABILITY}: no per-metric rows parsed"
    # ``as `other``` rows inherit the referenced row's floor verbatim.
    for name, cell in list(rows.items()):
        alias = re.fullmatch(r"\s*as `([a-z0-9_]+)`\s*", cell)
        if alias is not None:
            rows[name] = rows[alias.group(1)]
    return rows


def test_per_metric_floors_match_spec() -> None:
    """Per-family ``Min sample`` cells must declare the tiers the spec carries."""
    specs = {spec.name: spec for _stem, spec in public_specs()}

    mismatches: list[str] = []
    for name, cell in _per_metric_rows().items():
        if name in _NO_STATIC_FLOOR:
            continue
        spec = specs.get(name)
        if spec is None:
            mismatches.append(f"{name}: documented but not a registered metric")
            continue
        doc_hard, doc_warn = _declared_tiers(cell)
        spec_hard, spec_warn = _spec_tiers(spec)
        if doc_hard != spec_hard:
            mismatches.append(
                f"{name}: doc {'declares' if doc_hard else 'declares no'} hard "
                f"floor, spec {'has' if spec_hard else 'has no'} min_* floor"
            )
        if doc_warn != spec_warn:
            mismatches.append(
                f"{name}: doc {'declares' if doc_warn else 'declares no'} warn "
                f"tier, spec {'has' if spec_warn else 'has no'} warn_* floor"
            )
    assert not mismatches, (
        f"{APPLICABILITY} per-metric floors disagree with MetricSpec:\n  "
        + "\n  ".join(mismatches)
    )


def test_no_static_floor_exemptions_still_empty() -> None:
    """The exempted metrics must still declare no static floor.

    Keeps :data:`_NO_STATIC_FLOOR` from silently masking a metric that
    later gained a real ``sample_threshold``.
    """
    specs = {spec.name: spec for _stem, spec in public_specs()}
    stale = [
        f"{name} ({reason})"
        for name, reason in _NO_STATIC_FLOOR.items()
        if any(_spec_tiers(specs[name]))
    ]
    assert not stale, (
        "These metrics now declare a sample_threshold — drop them from "
        "_NO_STATIC_FLOOR and document the floor:\n  " + "\n  ".join(stale)
    )
