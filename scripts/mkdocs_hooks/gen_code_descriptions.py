"""Build-time generator for the `WarningCode` table.

Renders a Markdown table to `docs/reference/_generated_warning_codes.md`
from the description dictionary in `factrix._codes`. The descriptions
are the SSOT for trigger / meaning text; this hook surfaces them on the
user-facing `reference/warning-codes.md` page without manual sync.

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._codes import WarningCode

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUT_DIR = _REPO_ROOT / "docs" / "reference"


def _render_table(header: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"| {header} | Trigger / meaning |\n|---|---|\n"]
    for code, desc in rows:
        desc_inline = desc.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{code}` | {desc_inline} |\n")
    return "".join(lines)


def generate() -> None:
    warning_rows = [(m.value, m.description) for m in WarningCode]

    (_OUT_DIR / "_generated_warning_codes.md").write_text(
        _render_table("WarningCode", warning_rows), encoding="utf-8"
    )
    print(f"gen_code_descriptions: wrote {len(warning_rows)} WarningCode row(s)")


def on_pre_build(config: object) -> None:
    generate()


if __name__ == "__main__":
    generate()
