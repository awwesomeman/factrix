"""Build-time generator for `WarningCode` / `InfoCode` tables.

Renders two Markdown tables to `docs/reference/_generated_*_codes.md`
from the description dictionaries in `factrix._codes`. The descriptions
are the SSOT for trigger / meaning text; this hook surfaces them on the
user-facing `reference/warning-codes.md` page without manual sync.

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    The ``on_pre_build(config)`` function is called by MkDocs before
    each build.
"""

from __future__ import annotations

import pathlib

from factrix._codes import InfoCode, WarningCode

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
    info_rows = [(m.value, m.description) for m in InfoCode]

    (_OUT_DIR / "_generated_warning_codes.md").write_text(
        _render_table("WarningCode", warning_rows), encoding="utf-8"
    )
    (_OUT_DIR / "_generated_info_codes.md").write_text(
        _render_table("InfoCode", info_rows), encoding="utf-8"
    )
    print(
        f"gen_code_descriptions: wrote {len(warning_rows)} WarningCode / "
        f"{len(info_rows)} InfoCode row(s)"
    )


def on_pre_build(config: object) -> None:
    generate()


if __name__ == "__main__":
    generate()
