# `scripts/` — repo automation

Build-time helpers and one-off utilities. Sub-folders group by **who
runs them**, not by what they touch.

## Layout

| Path | Who runs | Purpose |
|------|----------|---------|
| [`mkdocs_hooks/`](mkdocs_hooks/) | mkdocs (`hooks:` in `mkdocs.yml`) | Build-time content generation for the docs site. Triggered automatically before every `mkdocs build` / `mkdocs serve`. Each script also has a `__main__` guard so it can be run directly for debugging (`uv run python scripts/mkdocs_hooks/<name>.py`), but that path is not part of the normal workflow. |
| `scripts/` (root) | humans (CLI), one-off | Manual utilities — release helpers, ad-hoc data fixes, anything outside the mkdocs lifecycle. Empty at present; reserved for future additions. |

## Adding a new script

- **It runs as part of `mkdocs build`** → put it in `mkdocs_hooks/` and
  register it in `mkdocs.yml` under `hooks:`. The hook entry should
  carry a leading comment describing input → output.
- **It is invoked by a human or by CI separately from mkdocs** → put
  it at the `scripts/` root. Include a docstring and a `__main__`
  guard so it is runnable as `uv run python scripts/<name>.py`.

The split exists so `mkdocs.yml` is not the only place documenting
which scripts run automatically — drop a script in the wrong folder
and it either fails to fire (hook not registered) or fires twice
(human invocation hitting a build hook).
