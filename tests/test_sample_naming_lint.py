"""Naming and single-source lints for sample-size constants and metadata keys.

Companion to the ``compute_* <-> PIPELINE`` naming lint (FX001/FX002) in
``test_metric_index.py``; the grammars themselves are documented in
``docs/development/architecture.md`` (Sample guards, metadata key grammar).

- **FX003** AST-scans every ``factrix/**/*.py`` for module-level ``MIN_*`` /
  ``DEFAULT_MIN_*`` constant assignments and asserts each name matches
  ``MIN_[<DOMAIN>_]<AXIS>[_<TIER>]`` (the axis token is mandatory). Scope is
  deliberately the module-level ``MIN_*`` constant — an open set anyone can
  extend in any module, with no other guard. The closed ``SampleThreshold``
  field set (backed by ``_AXES``) is covered elsewhere.
- **FX004** public ``@metric`` decorators declare ``sample_threshold``.
- **FX005** metadata keys in ``factrix/metrics/**`` name their axis instead of
  the neutral ``obs`` / ``sample`` vocabulary, which the grammar reserves for
  ``MetricResult.n_obs`` and ``fx.inference``. Complements
  ``test_docs_stat_keys_by_metric``, which checks that the keys a metric emits
  are documented — not how they are spelled.
- **FX006** the two single-sourced signature defaults (``overlap_periods`` /
  ``n_groups``) reference the ``DEFAULT_*`` constant rather than repeating its
  value, so the shared bucketing and stride cannot drift apart per module.
"""

from __future__ import annotations

import ast
import pathlib
import re

PACKAGE_DIR = pathlib.Path("factrix")

# A name that looks like a sample-size floor constant — what the lint inspects.
_CANDIDATE = re.compile(r"^(DEFAULT_)?MIN_[A-Z0-9_]+$")

# The axis-token grammar it must follow: optional ``DEFAULT_`` prefix, optional
# ``<DOMAIN>_`` prefix(es), a mandatory axis token, optional ``_HARD`` / ``_WARN``.
_VALID = re.compile(
    r"^(DEFAULT_)?MIN_([A-Z0-9]+_)*(PERIODS|ASSETS|EVENTS|PAIRS)(_HARD|_WARN)?$"
)


def _module_level_names(path: pathlib.Path) -> set[str]:
    """Names bound by a top-level ``Assign`` / ``AnnAssign`` in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_sample_constant_naming_lint() -> None:
    """Lint FX003: every ``MIN_*`` constant carries an axis token."""
    violations: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        for name in _module_level_names(path):
            if _CANDIDATE.match(name) and not _VALID.match(name):
                violations.append(f"{path}: {name}")
    assert not violations, (
        "FX003: sample-size constants must match "
        "MIN_[<DOMAIN>_]<AXIS>[_<TIER>] with AXIS in "
        "{PERIODS, ASSETS, EVENTS, PAIRS}:\n  " + "\n  ".join(violations)
    )


def test_user_facing_metrics_declare_sample_threshold() -> None:
    """Lint FX004: public metric decorators make pre-flight policy explicit."""
    violations: list[str] = []
    for path in sorted((PACKAGE_DIR / "metrics").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and getattr(decorator.func, "id", "") == "metric"
                ):
                    continue
                if not any(kw.arg == "sample_threshold" for kw in decorator.keywords):
                    violations.append(f"{path}:{node.lineno}: {node.name}")
    assert not violations, (
        "FX004: user-facing @metric decorators must explicitly declare "
        "sample_threshold=...; use SampleThreshold() for deliberate no-op "
        "pre-flight floors:\n  " + "\n  ".join(violations)
    )


# --- FX005: metadata keys carry an axis token, not the neutral vocabulary ----

METRICS_DIR = PACKAGE_DIR / "metrics"

# A metadata key spelled with the NEUTRAL sample vocabulary (``obs`` /
# ``sample``) rather than an axis token — what FX005 inspects. Counters of
# things that are not a sample axis (``n_groups``, ``n_top``, ``n_clusters``,
# ``n_wins``) are deliberately out of scope: they are not sample sizes, so the
# axis grammar does not apply to them.
_NEUTRAL_KEY = re.compile(r"^n_[a-z0-9_]*(obs|sample)[a-z0-9_]*$")

# Same axis tokens as FX003, matched anywhere in the key.
_KEY_AXIS = re.compile(r"(^|_)(periods|assets|pairs|events)(_|$)")

# Documented exemptions — the two names ``docs/development/architecture.md``
# (metadata grammar) allows to stay neutral:
#   ``n_obs`` / ``n_obs_axis``  the first-class ``MetricResult`` fields; the
#       axis is carried in ``n_obs_axis``, so the field itself is axis-free by
#       design and appears as a string in schema and serialisation code.
#   ``n_obs_sampled``  owned by ``fx.inference`` (``inference/series_mean.py``),
#       which the grammar exempts because one inference path serves every axis.
#       ``ic`` only READS it off ``result.metadata``; it does not define it.
#   ``n_resamples``  a count of bootstrap replications, not of observations on
#       any axis; it only matches the "sample" substring by accident.
_FX005_ALLOWED = frozenset({"n_obs", "n_obs_axis", "n_obs_sampled", "n_resamples"})


def _string_constants(path: pathlib.Path) -> set[str]:
    """Every ``str`` literal in one source file."""
    return {
        node.value
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_metadata_key_naming_lint() -> None:
    """Lint FX005: metric metadata keys name their axis instead of "obs"."""
    violations: list[str] = []
    for path in sorted(METRICS_DIR.rglob("*.py")):
        for text in _string_constants(path):
            if (
                _NEUTRAL_KEY.match(text)
                and text not in _FX005_ALLOWED
                and not _KEY_AXIS.search(text)
            ):
                violations.append(f"{path}: {text!r}")
    assert not violations, (
        "FX005: metric metadata keys must carry an axis token "
        "(periods / assets / pairs / events) rather than the neutral "
        "'obs' / 'sample' vocabulary, which is reserved for "
        "MetricResult.n_obs and fx.inference:\n  " + "\n  ".join(violations)
    )


# --- FX006: shared signature defaults come from the DEFAULT_* constants ------

# The two knobs ``factrix._types`` declares a single source of truth for. Their
# value (5) may not be repeated as a bare literal in a signature: the cost
# algebra pairing quantile_spread with notional_turnover is only valid when
# every consumer defaults to the SAME bucketing and stride, which is exactly
# what drifting per-module literals broke before.
_SSOT_DEFAULTS = {
    "overlap_periods": "DEFAULT_FORWARD_PERIODS",
    "n_groups": "DEFAULT_N_GROUPS",
}


def test_shared_defaults_reference_the_constant() -> None:
    """Lint FX006: no bare numeric default for the single-sourced knobs."""
    violations: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(
                zip(
                    positional[len(positional) - len(args.defaults) :],
                    args.defaults,
                    strict=True,
                )
            )
            pairs += [
                (a, d)
                for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
                if d is not None
            ]
            for arg, default in pairs:
                want = _SSOT_DEFAULTS.get(arg.arg)
                if want is None or not isinstance(default, ast.Constant):
                    continue
                # ``None`` / ``1`` are the deliberate non-default semantics
                # documented in architecture.md ("Three defaults are in use").
                if default.value == 5:
                    violations.append(
                        f"{path}:{node.lineno}: {node.name}({arg.arg}={default.value!r}) "
                        f"— import {want} from factrix._types"
                    )
    assert not violations, (
        "FX006: the single-sourced defaults must reference the constant, not "
        "repeat its value:\n  " + "\n  ".join(violations)
    )
