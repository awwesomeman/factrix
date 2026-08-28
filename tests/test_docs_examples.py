"""Executes the ``python`` fenced examples on every ``docs/**/*.md`` page.

``test_docs_pages.py`` checks that the names a page mentions resolve; this
test checks that the code a page shows actually *runs* against the current
API. The gap between the two is where the #845 rename leaked: a page can
reference only live symbols and still call them with a keyword that no
longer exists, and nothing short of executing the block catches that.

Execution contract (matches the "Runnable" / "Illustrative" split in
``docs/development/contributing.md``):

* Every ``python`` fence on a page runs, in order, in one shared namespace —
  later blocks may build on earlier ones, as the prose reads.
* A block that raises ``NameError`` is **illustrative** by construction — it
  leans on an unbound placeholder such as ``panel_large`` or ``your_df`` to
  communicate intent — and is skipped. Nothing else is: a ``TypeError`` from
  a stale keyword, an ``AttributeError`` from a removed helper, a
  ``KeyError`` from a renamed metadata key, or a polars column error from an
  outdated schema all fail the page.
* Blocks must be side-effect free (no file I/O, no plotting); stdout and
  warnings are swallowed so a noisy example is not a failing one.

``docs/plans/**`` is excluded (fossilised planning artifacts, also excluded
from the published site). ``docs/examples/*.md`` are generated from the
notebooks under ``examples/``; fix the notebook and re-render rather than the
page (``test_example_notebook_docs.py`` pins that).
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import textwrap
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from tests._doc_validation import docs_page_paths, python_fences


@dataclass(frozen=True)
class _BlockFailure:
    index: int
    line: int
    error: str


@pytest.fixture
def _isolated_metric_registry() -> Iterator[None]:
    """Undo any metric registration a page performs.

    ``docs/guides/custom-metrics.md`` runs ``@metric`` / ``register()`` for
    real, which writes into the process-global registry, attaches the class to
    the ``factrix.metrics`` namespace and invalidates the discovery caches.
    Left in place, a phantom ``__docs_example__`` metric leaks into every later
    test in the session — 94 downstream failures on the first run, plus a
    bogus row in the tracked ``docs/reference/_generated_metric_*.md`` files
    whenever something regenerates them. Snapshot before, restore after, and
    clear the same caches ``register()`` clears so the index rebuilds clean.
    """
    import factrix._dag as dag
    import factrix._metric_index as index
    import factrix.metrics as metrics_pkg
    from factrix.metrics._registry import REGISTRY

    registry_before = dict(REGISTRY)
    namespace_before = set(vars(metrics_pkg))
    try:
        yield
    finally:
        for name in [n for n in REGISTRY if n not in registry_before]:
            del REGISTRY[name]
        for name in set(vars(metrics_pkg)) - namespace_before:
            delattr(metrics_pkg, name)
        index._all_specs.cache_clear()
        index.public_specs.cache_clear()
        index._first_party_spec_by_name.cache_clear()
        dag._registry_callable_table.cache_clear()


def _run_page(text: str) -> tuple[list[_BlockFailure], int, int]:
    """Run a page's fences in order; return (failures, n_run, n_illustrative)."""
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    failures: list[_BlockFailure] = []
    n_run = n_illustrative = 0
    for index, (line, source) in enumerate(python_fences(text), start=1):
        code_text = textwrap.dedent(source)
        try:
            code = compile(code_text, f"<docs block {index} @ line {line}>", "exec")
        except SyntaxError as exc:
            failures.append(_BlockFailure(index, line, f"SyntaxError: {exc.msg}"))
            continue
        sink = io.StringIO()
        try:
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                warnings.simplefilter("ignore")
                exec(code, namespace)
        except NameError:
            # Unbound placeholder → illustrative fragment; see module docstring.
            n_illustrative += 1
            continue
        except Exception as exc:  # any other error is drift, not illustration
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            failures.append(
                _BlockFailure(index, line, f"{type(exc).__name__}: {first_line}")
            )
            continue
        n_run += 1
    return failures, n_run, n_illustrative


# README.md is the one authored page outside docs/ with a python fence; the
# quickstart it carries is the first snippet any user runs.
_PAGES = [pathlib.Path("README.md"), *docs_page_paths()]


@pytest.mark.usefixtures("_isolated_metric_registry")
@pytest.mark.parametrize("path", _PAGES, ids=lambda p: p.as_posix())
def test_page_examples_execute(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not python_fences(text):
        pytest.skip("no python fences on this page")
    failures, _n_run, _n_illustrative = _run_page(text)
    assert not failures, (
        f"python examples in {path.as_posix()} no longer run against the "
        "current API (blocks that only reference unbound placeholder names "
        "are skipped as illustrative; everything else must execute):\n  "
        + "\n  ".join(f"block {f.index} (line {f.line}): {f.error}" for f in failures)
    )
