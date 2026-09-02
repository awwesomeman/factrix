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
from factrix.metrics._helpers import _drop_stat_keys

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
_KEY_TOKEN = re.compile(r"[a-z][a-z0-9_]*")

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


def _dict_string_keys(
    node: ast.AST,
    bindings: dict[str, set[str]] | None = None,
) -> set[str]:
    if isinstance(node, ast.IfExp):
        return _dict_string_keys(node.body, bindings) | _dict_string_keys(
            node.orelse, bindings
        )
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
        elif key is None and isinstance(value, ast.Name) and bindings is not None:
            keys |= bindings.get(value.id, set())
    return keys


def _dict_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Resolve local dict literals, unpacking, and keyword ``update`` calls."""
    bindings: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: list[ast.Name] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = [
                    target for target in node.targets if isinstance(target, ast.Name)
                ]
                value = node.value
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        bound = bindings.setdefault(target.value.id, set())
                        before = len(bound)
                        bound.add(target.slice.value)
                        changed |= len(bound) != before
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
                value = node.value
            if value is not None:
                resolved = _dict_string_keys(value, bindings)
                for target in targets:
                    before = len(bindings.setdefault(target.id, set()))
                    bindings[target.id] |= resolved
                    changed |= len(bindings[target.id]) != before

            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr != "update" or not isinstance(node.func.value, ast.Name):
                continue
            target = node.func.value.id
            resolved: set[str] = set()
            for arg in node.args:
                resolved |= _dict_string_keys(arg, bindings)
                if isinstance(arg, ast.Name):
                    resolved |= bindings.get(arg.id, set())
            resolved |= {kw.arg for kw in node.keywords if kw.arg is not None}
            before = len(bindings.setdefault(target, set()))
            bindings[target] |= resolved
            changed |= len(bindings[target]) != before
    return bindings


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
    bindings = _dict_bindings(tree)
    keys: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # MetricResult(..., metadata={...})
            for kw in node.keywords:
                if kw.arg == "metadata":
                    keys |= _dict_string_keys(kw.value, bindings)
                    if isinstance(kw.value, ast.Name):
                        keys |= bindings.get(kw.value.id, set())
            # metadata.update({...})
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and _is_metadata_name(func.value)
            ):
                for arg in node.args:
                    keys |= _dict_string_keys(arg, bindings)
                    if isinstance(arg, ast.Name):
                        keys |= bindings.get(arg.id, set())
                keys |= {kw.arg for kw in node.keywords if kw.arg is not None}
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
                    keys |= _dict_string_keys(node.value, bindings)
        if (
            isinstance(node, ast.AnnAssign)
            and _is_metadata_name(node.target)
            and node.value is not None
        ):
            keys |= _dict_string_keys(node.value, bindings)
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


def _shared_literal_keys() -> set[str]:
    """Quoted key tokens in internal metadata builders plus dynamic drop keys."""
    keys = set(_drop_stat_keys("periods")) | set(_drop_stat_keys("assets"))
    paths = (
        p
        for p in sorted(pathlib.Path("factrix").rglob("*.py"))
        if p.stem.startswith("_")
        or p.parent.name in {"_primitives", "inference", "_stats"}
    )
    for path in paths:
        keys |= _literal_keys(path)
    return keys


def _literal_keys(path: pathlib.Path) -> set[str]:
    """Return identifier-shaped string literals, never bare variable names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _KEY_TOKEN.fullmatch(node.value)
    }


def test_shared_drop_stat_keys_documented(docs_text: str) -> None:
    missing = sorted(
        key
        for axis in ("periods", "assets")
        for key in _drop_stat_keys(axis)
        if f"`{key}`" not in docs_text
    )
    assert not missing, f"shared drop-stat keys missing from {DOCS_PAGE}: {missing}"


def test_metadata_scan_follows_local_dict_updates(tmp_path: pathlib.Path) -> None:
    module = tmp_path / "metric.py"
    module.write_text(
        """
def metric():
    method_b = {}
    method_b.update(h0_method_b="beta_pos = beta_neg")
    metadata = {**method_b}
    return MetricResult(metadata=metadata)
""",
        encoding="utf-8",
    )

    assert "h0_method_b" in _emitted_metadata_keys(module)


def test_metadata_scan_does_not_treat_local_names_as_keys(
    tmp_path: pathlib.Path,
) -> None:
    module = tmp_path / "metric.py"
    module.write_text(
        """
def metric():
    n_dropped = 3
    return MetricResult(metadata={"dropped_periods": n_dropped})
""",
        encoding="utf-8",
    )

    assert _emitted_metadata_keys(module) == {"dropped_periods"}


def _documented_keys(section: str) -> set[str]:
    """Backtick tokens named on the role bullets of one metric subsection."""
    keys: set[str] = set()
    for bullet in _KEY_BULLET.finditer(section):
        keys |= set(_BACKTICK_TOKEN.findall(bullet.group(0)))
    return keys


def test_documented_keys_still_exist_in_code(docs_text: str) -> None:
    """Every key the page names must still appear in the metric's sources."""
    name_to_stem = {spec.name: stem for stem, spec in public_specs()}
    shared = _shared_literal_keys()

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
        metric_path = METRICS_DIR / f"{stem}.py"
        emitted = (
            _emitted_metadata_keys(metric_path) | _literal_keys(metric_path) | shared
        )
        stale += [
            f"{name}: `{key}` is documented but no longer in factrix sources"
            for key in sorted(_documented_keys(section))
            if key not in emitted
        ]
    assert not stale, (
        f"{DOCS_PAGE} names metadata keys the code no longer carries:\n  "
        + "\n  ".join(stale)
    )
