"""Validate every package ``docs_path`` against the rendered MkDocs site.

Source-level checks cannot reproduce mkdocstrings' fully qualified ids or
MkDocs' heading normalization reliably. Build the real site once, then verify
that every literal API path in ``factrix/**/*.py`` resolves to a page and, when
present, an HTML id. Dynamic multi-factor paths are generated through their
shared helper and included explicitly.
"""

from __future__ import annotations

import ast
import importlib.metadata
import pathlib
from html.parser import HTMLParser

import pytest
from factrix._errors import _api_docs_path

pytest.importorskip("mkdocs", reason="docs-path validation requires the docs extra")
try:
    importlib.metadata.version("mkdocs-material")
    importlib.metadata.version("mkdocstrings-python")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("docs-path validation requires the docs extra", allow_module_level=True)
build = pytest.importorskip("mkdocs.commands.build").build
load_config = pytest.importorskip("mkdocs.config").load_config

PACKAGE_ROOT = pathlib.Path("factrix")
#: ``factrix.stats`` functions whose validators build their docs anchor from
#: ``_STATS_DOCS_TEMPLATE`` (one shared validator serves several functions), so
#: the literal path never appears in the source for the sweep below to find.
_STATS_DOCS_FUNCTIONS = (
    "bhy_adjust",
    "bhy_adjusted_p",
    "holm_adjusted_p",
    "romano_wolf_adjusted_p",
)
_DYNAMIC_DOCS_FUNCTIONS = (
    "compare",
    "bhy",
    "bhy_across_metrics",
    "partial_conjunction",
    "partial_conjunction_across_metrics",
    "bhy_hierarchical",
)


def _literal_docs_paths() -> set[str]:
    paths: set[str] = set()
    for source in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        paths.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("api/")
            and "{" not in node.value
        )
    return paths


DOCS_PATHS = sorted(
    _literal_docs_paths()
    | {_api_docs_path(name) for name in _DYNAMIC_DOCS_FUNCTIONS}
    | {f"api/stats#factrix.stats.{name}" for name in _STATS_DOCS_FUNCTIONS}
)


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        self.ids.update(value for key, value in attrs if key == "id" and value)


@pytest.fixture(scope="module")
def built_site(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    site_dir = tmp_path_factory.mktemp("docs-path-site")
    config = load_config(config_file="mkdocs.yml", site_dir=str(site_dir))
    build(config)
    return site_dir


def test_docs_paths_found() -> None:
    """Guard against a silently empty or over-narrow source sweep."""
    assert len(DOCS_PATHS) >= 25
    assert any("#" not in path for path in DOCS_PATHS)
    assert any("#" in path for path in DOCS_PATHS)


@pytest.mark.parametrize("docs_path", DOCS_PATHS)
def test_docs_path_resolves_in_built_site(
    docs_path: str, built_site: pathlib.Path
) -> None:
    """Every path resolves to a rendered page and optional HTML anchor."""
    page_path, separator, anchor = docs_path.partition("#")
    html_path = built_site / page_path / "index.html"
    assert html_path.is_file(), f"{docs_path!r} points to missing page {html_path}"
    if not separator:
        return

    parser = _IdCollector()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert anchor in parser.ids, (
        f"{docs_path!r} points to a missing anchor; rendered ids include "
        f"{sorted(parser.ids)}"
    )
