"""Executes the ``python`` fenced examples on every authored markdown page.

``test_docs_pages.py`` checks that the names a page mentions resolve; this
test checks that the code a page shows actually *runs* against the current
API. The gap between the two is where a rename leaks: a page can reference
only live symbols and still call them with a keyword that no longer exists,
and nothing short of executing the block catches that.

Execution contract (matches the "Runnable" / "Illustrative" split in
``docs/development/contributing.md``):

* Every ``python`` fence on a page runs, in order, in one shared namespace —
  later blocks may build on earlier ones, as the prose reads. Any exception
  fails the page, ``NameError`` included.
* A block that is *declared* illustrative — ``` ```python title="Illustrative"
  ``` — is compiled but not executed. Declaring it is the only way out: an
  undeclared block that leans on an unbound placeholder fails like any other
  drift, so the decision is written down on the page rather than inferred
  from the exception type at runtime.
* Blocks must be side-effect free (no file I/O, no plotting); stdout and
  warnings are swallowed so a noisy example is not a failing one.

Coverage is auditable from the test output: ``test_illustrative_block_compiles``
is parametrized one case per declared-illustrative block, so the default ``-v``
run prints the full inventory of what was *not* executed, page and line
included. Those blocks are not unchecked — ``test_docs_pages.py`` resolves the
``factrix.*`` symbols they name, which is what the two tests being
complementary buys.

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

from tests._doc_validation import PythonFence, docs_page_paths, python_fences


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


def _compile(fence: PythonFence, index: int) -> object | None:
    """Compile one fence; return the code object, or ``None`` on ``SyntaxError``."""
    return compile(
        textwrap.dedent(fence.body),
        f"<docs block {index} @ line {fence.line}>",
        "exec",
    )


def _run_page(text: str) -> list[_BlockFailure]:
    """Run a page's executable fences in order, in one shared namespace."""
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    failures: list[_BlockFailure] = []
    for index, fence in enumerate(python_fences(text), start=1):
        if fence.illustrative:
            continue
        try:
            code = _compile(fence, index)
        except SyntaxError as exc:
            failures.append(_BlockFailure(index, fence.line, f"SyntaxError: {exc.msg}"))
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
        except Exception as exc:  # any error is drift, not illustration
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            failures.append(
                _BlockFailure(index, fence.line, f"{type(exc).__name__}: {first_line}")
            )
    return failures


# README.md is the one authored page outside docs/ with a python fence; the
# quickstart it carries is the first snippet any user runs, and it lives under
# the same rule as the docs pages — including the marker, which GitHub's
# renderer ignores as trailing info-string text.
_PAGES = [pathlib.Path("README.md"), *docs_page_paths()]

# One case per declared-illustrative block, so the inventory of what is *not*
# executed is printed by every ``-v`` run instead of being invisible.
_ILLUSTRATIVE_BLOCKS = [
    (path, index, fence)
    for path in _PAGES
    for index, fence in enumerate(
        python_fences(path.read_text(encoding="utf-8")), start=1
    )
    if fence.illustrative
]


@pytest.mark.usefixtures("_isolated_metric_registry")
@pytest.mark.parametrize("path", _PAGES, ids=lambda p: p.as_posix())
def test_page_examples_execute(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    fences = python_fences(text)
    if not fences:
        pytest.skip("no python fences on this page")
    if all(fence.illustrative for fence in fences):
        pytest.skip("every python fence on this page is declared illustrative")
    failures = _run_page(text)
    assert not failures, (
        f"python examples in {path.as_posix()} no longer run against the "
        "current API (a block that cannot run as written must declare itself "
        'with ```python title="Illustrative"`):\n  '
        + "\n  ".join(f"block {f.index} (line {f.line}): {f.error}" for f in failures)
    )


@pytest.mark.parametrize(
    ("path", "index", "fence"),
    _ILLUSTRATIVE_BLOCKS,
    ids=[
        f"{path.as_posix()}:block {index} @ line {fence.line}"
        for path, index, fence in _ILLUSTRATIVE_BLOCKS
    ],
)
def test_illustrative_block_compiles(
    path: pathlib.Path, index: int, fence: PythonFence
) -> None:
    """A declared-illustrative block is not executed, but must still parse.

    The test ids are the audit trail: they name every block the executing
    test skipped, so removing a block from the run is a visible diff on this
    file's output rather than a silent gap.
    """
    try:
        _compile(fence, index)
    except SyntaxError as exc:
        pytest.fail(
            f"illustrative block {index} (line {fence.line}) in "
            f"{path.as_posix()} does not parse: {exc.msg}"
        )
