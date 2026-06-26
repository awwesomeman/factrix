"""Naming guard: bare ``df`` / ``_df`` identifiers are banned in ``factrix/``.

``df`` is ambiguous — *degrees of freedom* in a statistics context,
*DataFrame* in the polars idiom. The architecture doc kills the
collision by **position** (see
``docs/development/architecture.md`` → "Naming: ``data`` vs ``df_*``"):

- ``df_…`` **prefix** → degrees of freedom (``df_num``, ``df_denom``,
  ``df_resid``, ``df_t``). KEPT.
- ``…_df`` **suffix** with a content prefix → DataFrame (``ic_df``,
  ``caar_df``, ``factor_df``, ``sorted_df``). KEPT.
- bare ``df`` / ``_df`` → BANNED — the unqualified token is exactly the
  ambiguous case. A standalone frame uses ``data`` (or a semantic noun).

This walks every ``factrix/`` module with :mod:`ast` and fails if any
function/lambda parameter, assignment target (a bare ``Name``), or
annotated/dataclass field is named exactly ``df`` or ``_df``.

The scipy distribution kwarg (``sp_stats.chi2.sf(q, df=h)``) is *not*
flagged: it is a keyword argument inside a :class:`ast.Call`, never a
parameter/assignment/field *definition*, so the visitor below — which
only inspects definitions and assignment targets — never sees it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

FACTRIX_DIR = pathlib.Path("factrix")

# The exact banned identifiers. ``df_num``/``ic_df``/etc. are NOT here:
# this is an equality check, not a prefix/suffix match.
_BANNED = {"df", "_df"}


def _python_modules() -> list[pathlib.Path]:
    """Every ``.py`` module under ``factrix/``."""
    return sorted(FACTRIX_DIR.rglob("*.py"))


def _banned_definitions(tree: ast.AST) -> list[str]:
    """Return ``"<lineno>: <name>"`` for each banned-name definition.

    Inspected definition sites:

    * function / lambda parameters (positional, ``*args``-style,
      keyword-only, ``**kwargs``),
    * assignment targets that are a bare ``Name`` (``df = ...``,
      including tuple/list unpacking and augmented/walrus assignment),
    * annotated assignments / dataclass fields (``df: pl.DataFrame``).
    """
    hits: list[str] = []

    def _flag(name: str | None, lineno: int) -> None:
        if name in _BANNED:
            hits.append(f"line {lineno}: {name}")

    def _flag_target(node: ast.expr) -> None:
        # Recurse into tuple/list unpacking targets; flag bare Names.
        if isinstance(node, ast.Name):
            _flag(node.id, node.lineno)
        elif isinstance(node, ast.Tuple | ast.List):
            for elt in node.elts:
                _flag_target(elt)
        elif isinstance(node, ast.Starred):
            _flag_target(node.value)

    for node in ast.walk(tree):
        # Function / lambda parameters.
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            args = node.args
            for arg in (
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ):
                if arg is not None:
                    _flag(arg.arg, arg.lineno)
        # Plain assignment targets: ``df = ...``.
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _flag_target(target)
        # Augmented assignment (``df += ...``), annotated assignment /
        # dataclass field (``df: pl.DataFrame``), or walrus (``(df := ...)``).
        elif isinstance(node, ast.AugAssign | ast.AnnAssign | ast.NamedExpr):
            _flag_target(node.target)

    return hits


@pytest.mark.parametrize(
    "path", _python_modules(), ids=lambda p: str(p.relative_to(FACTRIX_DIR))
)
def test_no_bare_df_identifier(path: pathlib.Path) -> None:
    """No ``factrix/`` module may define a bare ``df`` / ``_df`` identifier."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = _banned_definitions(tree)
    assert not hits, (
        f"{path}: bare ``df``/``_df`` identifier(s) are banned — rename to "
        f"``data`` (or a semantic noun). See docs/development/architecture.md "
        f"'Naming: data vs df_*'.\n  " + "\n  ".join(hits)
    )
