"""Drift guard: ``MetricResult.metadata`` keys ↔ docs reference page.

Code → doc: AST-scans every public ``factrix/metrics/*.py`` for literal
string keys that flow into ``MetricResult.metadata`` and asserts each
appears as a backtick token in ``docs/reference/stat-keys-by-metric.md``.

Doc → code: every key the page's per-metric bullets name must still
appear in the metric's own module or in the shared internal modules that
build its metadata. Keys are assembled from f-strings and ``**kwargs``
splats as well as dict literals, so the reverse direction reads source
text rather than the AST — it catches a key deleted from the code and
left behind in the page, not a key that moved between metrics.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
from factrix._metric_index import public_specs

METRICS_DIR = pathlib.Path("factrix/metrics")
DOCS_PAGE = pathlib.Path("docs/reference/stat-keys-by-metric.md")

# Per-metric subsection heading, e.g. ``#### `ic_ir` ``.
_SECTION_HEADING = re.compile(r"^`([a-z0-9_]+)`")
# The role bullets that enumerate metadata keys (``*short-circuit*`` names
# ``reason`` values, not keys, and is excluded).
_KEY_BULLET = re.compile(
    r"^- \*(?:primary|secondary-test|descriptive)\*[^\n]*(?:\n(?!- |#|$)[^\n]*)*",
    re.M,
)
_BACKTICK_TOKEN = re.compile(r"`([a-z][a-z0-9_]*)`")

_COMMON_KEYS: frozenset[str] = frozenset(
    {"p_value", "stat_type", "h0", "method", "reason"}
)

# Explicit-keyword params of ``_short_circuit_output`` — control flags that
# do not surface as ``MetricResult.metadata`` keys at runtime.
_HELPER_CONTROL_KWARGS: frozenset[str] = frozenset(
    {"n_obs", "n_obs_axis", "descriptive", "alternative"}
)

# Inner keys of nested dict / list-of-dict metadata payloads. Documented
# at the outer-key level (e.g. ``per_regime`` covers its inner shape),
# not as standalone bullets.
_NESTED_KEYS: frozenset[str] = frozenset(
    {
        "mean_ic",
        "std_ic",
        "stat",
        "significance",
        "p_adjusted_bhy",
        "z_stat",
        "hit_rate",
        "is_ratio",
        "mean_is",
        "mean_oos",
        "survival_ratio",
        "sign_flipped",
    }
)


def _public_metric_modules() -> list[pathlib.Path]:
    return sorted(p for p in METRICS_DIR.glob("*.py") if not p.stem.startswith("_"))


def _dict_string_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {
        k.value
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


def _is_metadata_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "metadata"


def _emitted_metadata_keys(path: pathlib.Path) -> set[str]:
    """Collect literal string keys flowing into ``MetricResult.metadata``.

    Covers:
    - ``MetricResult(..., metadata={...})`` kwarg dict literal.
    - ``metadata = {...}`` / annotated assign — later passed by name.
    - ``metadata["key"] = ...`` subscript writes.
    - ``metadata.update({...})`` calls.
    - ``_short_circuit_output(name, reason, **extras)`` call-site kwargs
      — each kwarg name is splatted into ``metadata`` by the helper.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # MetricResult(..., metadata={...})
            for kw in node.keywords:
                if kw.arg == "metadata":
                    keys |= _dict_string_keys(kw.value)
            # metadata.update({...})
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and _is_metadata_name(func.value)
                and node.args
            ):
                keys |= _dict_string_keys(node.args[0])
            # _short_circuit_output(name, reason, k1=v1, k2=v2, ...)
            if isinstance(func, ast.Name) and func.id == "_short_circuit_output":
                keys.update(
                    kw.arg
                    for kw in node.keywords
                    if kw.arg is not None and kw.arg not in _HELPER_CONTROL_KWARGS
                )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_metadata_name(target):
                    keys |= _dict_string_keys(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and _is_metadata_name(node.target)
            and node.value is not None
        ):
            keys |= _dict_string_keys(node.value)
        if (
            isinstance(node, ast.Subscript)
            and _is_metadata_name(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)

    return keys


@pytest.fixture(scope="module")
def docs_text() -> str:
    return DOCS_PAGE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    _public_metric_modules(),
    ids=lambda p: p.stem,
)
def test_metadata_keys_documented(path: pathlib.Path, docs_text: str) -> None:
    emitted = _emitted_metadata_keys(path)
    candidates = emitted - _COMMON_KEYS - _NESTED_KEYS
    missing = sorted(k for k in candidates if f"`{k}`" not in docs_text)
    assert not missing, (
        f"{path.name}: metadata keys emitted but not referenced in "
        f"{DOCS_PAGE}: {missing}"
    )


def _shared_source() -> str:
    """Source text of the internal modules that assemble metric metadata."""
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(pathlib.Path("factrix").rglob("*.py"))
        if p.stem.startswith("_")
        or p.parent.name in {"_primitives", "inference", "_stats"}
    )


def _documented_keys(section: str) -> set[str]:
    """Backtick tokens named on the role bullets of one metric subsection."""
    keys: set[str] = set()
    for bullet in _KEY_BULLET.finditer(section):
        keys |= set(_BACKTICK_TOKEN.findall(bullet.group(0)))
    return keys


def test_documented_keys_still_exist_in_code(docs_text: str) -> None:
    """Every key the page names must still appear in the metric's sources."""
    name_to_stem = {spec.name: stem for stem, spec in public_specs()}
    shared = _shared_source()

    sections = re.split(r"^#### ", docs_text, flags=re.M)[1:]
    stale: list[str] = []
    for section in sections:
        heading = _SECTION_HEADING.match(section.split("\n", 1)[0])
        if heading is None:
            continue
        name = heading.group(1)
        stem = name_to_stem.get(name)
        if stem is None:
            stale.append(f"{name}: documented but not a registered metric")
            continue
        sources = (METRICS_DIR / f"{stem}.py").read_text(encoding="utf-8") + shared
        stale += [
            f"{name}: `{key}` is documented but no longer in factrix sources"
            for key in sorted(_documented_keys(section))
            if not re.search(rf"\b{key}\b", sources)
        ]
    assert not stale, (
        f"{DOCS_PAGE} names metadata keys the code no longer carries:\n  "
        + "\n  ".join(stale)
    )
