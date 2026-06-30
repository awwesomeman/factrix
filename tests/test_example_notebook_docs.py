"""Drift guard for docs example pages rendered from notebooks."""

from __future__ import annotations

from scripts.mkdocs_hooks.render_example_notebooks import (
    iter_example_notebooks,
    output_path_for,
    render_notebook,
)


def test_generated_example_docs_match_notebook_sources() -> None:
    """``docs/examples/*.md`` must match the repo-root notebook SSOT."""
    notebooks = iter_example_notebooks()
    assert notebooks, "Expected at least one example notebook under examples/."

    stale: list[str] = []
    missing: list[str] = []
    for notebook_path in notebooks:
        out_path = output_path_for(notebook_path)
        if not out_path.exists():
            missing.append(out_path.as_posix())
            continue
        if out_path.read_text(encoding="utf-8") != render_notebook(notebook_path):
            stale.append(out_path.as_posix())

    assert not missing, "Missing generated example docs:\n  " + "\n  ".join(missing)
    assert not stale, (
        "Generated example docs are stale. Run "
        "'python scripts/mkdocs_hooks/render_example_notebooks.py'.\n  "
        + "\n  ".join(stale)
    )
