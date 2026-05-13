---
title: Contributing to factrix
---

This document describes the factrix development workflow. factrix is
currently a private single-author repo, so this guide covers
**development modes and pitfalls** rather than OSS contributor
conventions (licensing / DCO / CLA).

---

## 1. Two development modes

### Mode A — Standalone development (recommended for most cases)

Clone the factrix repo directly, isolated venv, fastest cycle:

```bash
git clone https://github.com/awwesomeman/factrix.git
cd factrix
uv sync
uv run pytest        # confirm baseline is green
```

**When to use**: new feature / large refactor / SemVer bump / release.
Isolated environment, no risk of accidentally touching downstream
research, shortest test cycle.

### Mode B — In-workspace development (via submodule)

Edit from the downstream workspace (`factor-analysis`) under
`external/factorlib/`—you can change factrix and observe the effect in
a real research notebook simultaneously:

!!! note "Submodule path"
    The submodule directory name on disk depends on the parent workspace's
    `.gitmodules` setting; the actual path is currently
    **`external/factorlib/`**. Once the parent workspace renames the
    submodule path, it will become `external/factrix/`. Commands below
    follow the current on-disk path.

```bash
cd ~/Desktop/dst/code/factor-analysis
cd external/factorlib
# edit factrix source
uv run pytest
# back to workspace and run the notebook for end-to-end verification
cd ../..
uv run jupyter notebook
```

**When to use**:
- Debugging a "bug that only appears in the research environment"—real
  data context required to reproduce
- Changing an API for a known downstream need, with immediate verification
- Small tweaks (< 10 lines) that don't justify Mode A's setup overhead

**Caution**: Mode B has three critical pitfalls; see §4 below.

---

## 2. Environment setup

factrix uses **uv** to manage the Python venv and lockfile.
`pyproject.toml` and `uv.lock` are the sole authority—do not use
`pip install` to drop anything directly into `.venv/`.

Python is pinned to **3.12+** (defined by `requires-python` in
`pyproject.toml`).

### Dependencies and extras
Use `--extra` to add or drop modules per development needs:

```bash
uv sync                              # core only (polars, numpy, pandera)
uv sync --extra dev                  # +pytest, commitizen, etc. (required to write code)
uv sync --extra charts               # +plotly for charts
uv sync --all-extras                 # charts + mlflow + jupyter (feature extras)
```

!!! note "`dev` extra is separate from `all`"
    The `dev` extra is **not** part of `all` (toolchain and feature extras
    are deliberately separated), so `--all-extras` does **not** install
    pytest / commitizen. Developers should use:

    ```bash
    uv sync --all-extras --extra dev   # feature extras + dev tools in one shot
    ```

### Common environment commands
```bash
uv run <cmd>         # run inside the venv (e.g. uv run pytest)
uv add <pkg>         # add dep, sync pyproject + uv.lock
uv lock --upgrade    # upgrade lock to the latest within current pyproject constraints
```

### Git hooks (pre-push CHANGELOG check)

`.githooks/` at the repo root holds version-controlled hook scripts.
Run once after a fresh clone:

```bash
python scripts/setup_dev.py
```

The script sets `git config core.hooksPath .githooks`. Idempotent —
re-running is a no-op; aborts (non-zero exit) if `core.hooksPath` is
already pointed at a different path so a contributor's dotfiles-
managed hook surface is not silently overwritten. The setting is
per-clone and local; it does not propagate across machines, so every
clone / new machine must run the script once. `git worktree`
instances share `.git/config` with their primary clone, so one run at
the primary clone covers every worktree under it. After activation,
all hooks below run automatically; no per-hook install needed.

`pre-commit` — when staged changes include `*.py`, runs `ruff check`
+ `ruff format --check` (mirrors CI's `lint` job). Fail → blocks the
commit; fix-then-commit, or use `git commit --no-verify` to bypass.
Scoped to staged files, so commits touching only YAML / Markdown / .txt
don't trigger.

`pre-push` — when the push includes `chore(release): vX.Y.Z` (i.e. the
release commit produced by `cz bump`), checks that the `## vX.Y.Z`
section in `CHANGELOG.md` has ≥ 25 non-blank lines. Below threshold →
blocks the push, forcing you to add WHY narrative (BREAKING migration,
behavioural direction, motivation) before pushing. To bypass:
`git push --no-verify`.

Adjust the threshold (per-shell):

```bash
CHANGELOG_MIN_LINES=10 git push
```

Rationale: see §9 "Release workflow" on the limits of cz's
auto-CHANGELOG.

---

## 3. Development cycle (branch → test → commit → push → PR)

```bash
# 1. Branch off main (do not commit directly to main)
git checkout main && git pull
git checkout -b <type>/<short-desc>     # e.g. feat/redundancy-heatmap

# 2. Develop + run tests frequently
# ...edit...
uv run pytest                            # must be all green before commit
uv run pytest tests/test_<file>.py -v   # focus on a single module for fast iteration

# 3. Commit (Conventional Commits + interactive generation)
git add <specific-files>                 # avoid -A
cz commit -- -s                          # Commitizen produces standard format and appends sign-off

# 4. Push + open PR
git push origin feat/redundancy-heatmap
gh pr create --title "..." --body "..."

# 5. After CI is green, merge (currently solo → squash-merge or rebase-merge)
gh pr merge --squash
```

!!! warning "Do not run `cz bump` after merge"
    Versions and tags follow the release-train cadence (see §9) — a single
    release fires after several PRs accumulate. Each PR writes its own
    changes into the `## [Unreleased]` section of `CHANGELOG.md` (under
    `### Added` / `### Changed` / `### Fixed` / `### Migration` subsections
    as applicable); at release time, `cz bump --changelog` freezes that
    section into the next version heading.

**CHANGELOG formatting**: paragraphs and bullets are not hard-wrapped — each paragraph is one line, each bullet one line. The 72-char wrap convention applies to commit messages, not to CHANGELOG prose; GitHub Release notes treat single newlines as `<br>`, so source-level wrapping leaks into the rendered output. Aligns with Polars / ruff / Pydantic.

### Branch naming

`<type>/<short-desc>`, all lowercase with hyphens:

- `feat/...` — new feature
- `fix/...` — bug fix
- `refactor/...` — refactor (no behavioural change)
- `docs/...` — documentation
- `chore/...` — packaging / CI / lockfile maintenance

### Commit message

The project follows **Commitizen** + **Conventional Commits**.
**Always use `cz commit` for interactive commits**—it validates title
length and other rules automatically.

**Reminders**:
- Description length `< 50 chars`
- Body uses `-` bullets, recording only "why" + "what"—do not restate
  the diff; aim for `< 72 chars` per line
- No AI co-author signature, no emoji, no trailing period
- Pass `-s` via `cz commit -- -s` to append Signed-off-by

#### Changing the `Signed-off-by` name and email

`git commit -s` reads `user.name` and `user.email` from your Git
config. To change the signature, run:

```bash
# project-local only
git config user.name "Your New Name"
git config user.email "your-new-email@example.com"

# global (default for all projects)
git config --global user.name "Your New Name"
git config --global user.email "your-new-email@example.com"
```

---

## 4. Three critical pitfalls of Mode B

### G1. Submodule = detached HEAD (the biggest trap)

When entering `external/factrix/` from the workspace, the submodule
defaults to **detached HEAD**. Commits made on a detached HEAD
**belong to no branch**—the next `git submodule update --remote`
silently overwrites them and the commits vanish (reflog does not
record them).

```bash
# Way to lose commits (incorrect)
cd external/factrix
# ...edit...
git commit -am "feat: xxx"              # detached — commit has no home
git submodule update --remote           # GONE

# Correct: branch first
cd external/factrix
git checkout -b feat/xxx                # create branch
# ...edit...
git commit -am "feat: xxx"
git push origin feat/xxx
gh pr create                            # open PR in the factrix repo
```

### G2. Editable install has Jupyter caveats

Workspace `uv sync` installs factrix in editable mode, so source
changes in the submodule reflect in `import factrix` immediately—but
Jupyter kernels have exceptions:

- Function body changes → caught by `%autoreload 2`
- `__init__.py` imports / dataclass definitions / module-level
  constants → must **restart the kernel**

When in doubt, restart the kernel—the safest option.

### G3. Workspace does not auto-track factrix updates

After a PR merges into factrix main, the workspace's submodule pointer
**does not auto-bump**. This is a feature, not a bug—each workspace
commit binds an explicit factrix SHA, keeping research results
reproducible.

Manual bump:

```bash
cd external/factrix && git fetch && git checkout main && git pull
cd ../.. && git add external/factrix
git commit -m "chore: bump factrix to <short-sha>: <why>"
```

---

## 5. Submodule sync reference

Command index for Mode B and consumer-workspace daily ops. The cheat
sheet covers 90% of cases; read the mental model and scenarios below
only when the cheat sheet is unclear.

### 5.1 Cheat sheet

| Goal | Command |
|------|---------|
| See the workspace-pinned SHA | `git submodule status` |
| See the submodule's actual HEAD | `cd external/factrix && git rev-parse --short HEAD` |
| Pull factrix main latest into actual | `git submodule update --remote` |
| Switch to a specific tag | `cd external/factrix && git fetch --tags && git checkout vX.Y.Z` |
| Freeze actual into the pin | `git add external/factrix && git commit -m "chore: bump..."` |
| Discard actual changes, reset to pin | `git submodule update` |
| Initialise submodule on fresh clone | `git submodule update --init --recursive` |

### 5.2 Mental model

The workspace has a `pin` (the SHA recorded in the workspace commit);
the submodule has its own `actual HEAD` (the SHA actually checked out).
The two can differ—`git status` will show the submodule as `modified`.
**Python editable install reads the actual HEAD**, so the imported
version follows actual, not pin.

### 5.3 Common scenarios

- **Editing factrix and syncing back to workspace**: follow §3 + §4
  (dev workflow), then freeze with cheat sheet row 5
- **Someone else pushed factrix; I need to catch up**: cheat sheet rows
  3 + 5
- **Pinning to a tag (recommended, see §9)**: cheat sheet rows 4 + 5
- **Submodule looks broken after switching branches**: reset via cheat
  sheet row 6

### 5.4 When confused, do two things first

1. `git submodule status` — see the pin
2. `cd external/factrix && git rev-parse --short HEAD` — see the actual

If both SHAs match, you're clean; if not, decide whether to bump the
pin (row 5) or reset actual (row 6).

---

## 6. Docs sync boundary (Source of Truth)

After editing `factrix/`, the table below tells you which `docs/`
files auto-update and which need manual maintenance.

### Auto-sync pipelines

| Source (SSOT) | Docs target | Mechanism |
|---|---|---|
| `factrix/**/*.py` docstrings | `:::` directives in `docs/api/**/*.md` | mkdocstrings plugin |
| `Matrix-row:` in `factrix/metrics/*.py` | `docs/reference/_generated_metric_matrix.md` | hook: `scripts/mkdocs_hooks/gen_metric_matrix.py` |
| `factrix/llms*.txt` | site root `llms*.txt` | hook: `scripts/mkdocs_hooks/sync_llms_txt.py` |

#### Docstring `Examples:` — runnable, copy-paste ready

Every user-facing function reachable from the API Reference nav
(`evaluate`, `run_metrics`, `by_slice`, `slice_pairwise_test`,
`slice_joint_test`, `multi_factor.{bhy, partial_conjunction,
bhy_hierarchical}`, `compare`, `suggest_config`, `list_metrics`,
`list_estimators`) carries an `Examples:` block in its docstring.
The docstring is the single source of truth — `.md` pages do not
duplicate runnable examples, only document things the example
cannot show (output schemas, attribute tables, semantic intent).

Convention:

- Section header is `Examples:` (plural, Google style).
- Source uses `>>>` doctest prompt syntax. The Material copy button
  is configured (via `docs/javascripts/copy-strip-pycon.js`) to
  strip `>>>` prompts and expected-output lines from the clipboard
  payload, so readers can copy a `pycon` block straight into a
  REPL or script.
- Show **call shape**, not fragile output. Lines starting with
  `>>>` carry the educational value; expected-output lines (lines
  without `>>>`) are reserved for values stable across BLAS / numpy
  / polars / Python versions.
- Avoid concrete floating-point values, DataFrame / array reprs, or
  multi-line text reprs as expected output. If a value must be
  shown, prefer structural facts (enum value, integer length,
  boolean from `isinstance`).
- Setup is self-contained — construct any required panel inline via
  `fx.datasets.make_cs_panel(...)` + `compute_forward_return(...)`
  so the snippet runs verbatim with no surrounding fixtures.

Examples blocks are CI-verified. The `doctest` job in
`.github/workflows/test.yml` runs `pytest --doctest-modules
factrix/` on every PR; option flags (`ELLIPSIS`,
`NORMALIZE_WHITESPACE`) live in `[tool.pytest.ini_options]
doctest_optionflags` so individual Examples carry no
`# doctest:` directives. A renamed symbol, changed signature,
or dropped import that breaks an Example surfaces on the same
PR as the change.

Page-level demo admonitions (`## Worked example`, `!!! example`
blocks) are reserved for end-to-end demos that intentionally show
something the docstring cannot — typically a longer
synthetic-panel walk-through with `profile.diagnose()` output or a
cross-cell config recipe table. They do not echo the docstring
example.

#### Examples — markdown SSOT + optional runnable mirror

`docs/examples/*.md` are hand-authored; markdown is the SSOT. The `examples/*.ipynb` files at repo root are *optional* runnable mirrors for users who want to step through a recipe interactively — they are not the source for the rendered site and not required.

New example convention:

- Write `docs/examples/<name>.md` directly. Match the shape of the two shipping recipes — frontmatter title, narrative blocks (`Factor type` / `Use this when` / `What it tests` / `Output to read`), numbered step sections, fenced code blocks with illustrative outputs in adjacent `text` / `json` fences.
- Do **not** print `fx.__version__` or include trailing `print("<name>: ok")` smoke tests in the code blocks. Outputs in markdown are illustrative literals; pinning a version line invites drift on every release.
- If interactive execution matters, also commit a parallel `examples/<name>.ipynb`. Link it from the markdown page header (`Runnable notebook: …`). Notebook drift is an independent maintenance debt — markdown wins on disagreement.

### Docs that still need manual maintenance

- `docs/api/**/*.md` (30 files): each contains a hand-written 2–5 line
  narrative intro plus a `:::` directive; new public metrics require a
  new file plus a `nav:` entry in `mkdocs.yml`
- `docs/getting-started/`, `docs/guides/`, `docs/development/`, the
  homepage `index.md`, and per-section `index.md`: pure narrative
- `docs/reference/metric-pipelines.md`, `statistical-methods.md`,
  `warning-codes.md`, `bibliography.md`: narrative roll-ups (not
  matrices)

### Nav structure conventions

The mkdocs nav follows a **pure-label** policy: every group label in
`mkdocs.yml` is a non-clickable organising heading; every entry with
content is an explicit leaf with a sidebar label. `navigation.indexes`
is intentionally disabled so the visual contract is "label = heading,
leaf = page, never both".

When adding a new section or hub page:

- Group labels carry no page reference. They look like `- Concepts:`
  followed by an indented list of leaves.
- Hub or overview pages (e.g. `api/metrics/index.md`) appear as an
  explicit leaf with label `Overview` as the first child of their
  group: `- Overview: api/metrics/index.md`.
- Leaf labels are unique within their sidebar branch — never reuse the
  parent group's name as a leaf label.

This keeps `mkdocs build --strict` green and the sidebar legible without
clicking through to discover affordance.

---

## 7. Drift management

Documentation, generated reference material, and example notebooks drift
away from the code they describe. The project's working policy:
**automate symbol-level drift, leave narrative/conceptual drift to
review, and run a manual half-hour pass at every release-train cut.**
The framing below is heuristic — when a new "should I add a test for
this drift?" question arises, judge it by whether the cost of
automating exceeds the cost of missing.

### 7.1 Automated drift checks

Enforced by tests and CI — a regression fails the PR build.

| Drift class | Enforced by |
|---|---|
| Generated docs freshness (metric matrix, registry cells, examples sync) | `tests/test_docs_matrix.py`, `tests/test_docs_registry_cells.py`, plus the `git diff --exit-code` step in `.github/workflows/docs-deploy-dev.yml` |
| Public-surface mention coverage in `factrix/llms-full.txt` | `tests/test_docs_llms.py` |
| Public-surface mention coverage across all docs pages | `tests/test_docs_pages.py` |
| README quickstart end-to-end | `tests/test_readme_quickstart.py` |
| mkdocs nav / link integrity | `uv run mkdocs build --strict` (run in `.github/workflows/docs-deploy-dev.yml`) |
| Type-checking gate (`mypy factrix`) | `uv run mypy factrix` (lint job in `.github/workflows/test.yml`) |

### 7.2 Drift left to human review

Deliberately not automated, because machine-judgement cost > miss cost:

- **Architecture narrative vs current code design**
  (`docs/development/architecture.md`) — accuracy hinges on whether the
  prose still captures the *spirit* of the design; a string match
  cannot judge that.
- **Conceptual / explanatory text in guides**
  (`docs/guides/*.md`, `docs/getting-started/*.md`) — wording quality
  and pedagogical ordering are reviewer calls; a stale phrasing often
  parses fine.
- **Editorial choices in `factrix/llms-full.txt`** — depth of context,
  ordering, and what to omit are agent-UX decisions, not symbol
  coverage (which §7.1 already enforces).

Rule of thumb: if the drift can be detected by a string match or a
function call, automate it; if catching it requires reading prose for
meaning, leave it to release-train review.

### 7.3 Release-train drift audit

Before running `cz bump --changelog` (see §9), run this checklist on
`main`:

```bash
# 1. Search for known-deprecated symbol names that may have leaked back in.
#    Extend the pattern per release with names retired since the last tag.
git grep -nE 'q1_q5_spread'

# 2. Full test suite — covers every check listed in §7.1.
uv run pytest -q

# 3. Strict docs build — surfaces broken nav, links, and generated-file drift.
uv run mkdocs build --strict

# 4. Public-surface coverage spot-check (also run by step 2; explicit run
#    is cheap and isolates failures).
uv run pytest tests/test_docs_llms.py tests/test_docs_pages.py -q

# 5. Skim the [Unreleased] CHANGELOG section for stale paths / kwargs
#    that drifted since the entry was written.
sed -n '/## \[Unreleased\]/,/^## /p' CHANGELOG.md
```

A failure on any step is a release blocker — fix on `main` (or revert
the offending PR) before bumping.

### 7.4 Type-checking conventions

`uv run mypy factrix` is enforced in CI. Three recurring patterns:

- Polars scalar aggregations (`.median() / .mean() / .std() / .quantile() / .item()`) annotate as a broad union; suppress per call site with `# type: ignore[arg-type]`. `warn_unused_ignores = true` self-cleans these if stubs improve.
- Polars schema dicts typed `dict[str, pl.DataType]` need **instances** (`pl.String()`), not class references (`pl.String`). Both work at runtime; only instances satisfy the annotation.
- scipy / pandas: routed through `[[tool.mypy.overrides]] ignore_missing_imports = true`. New stub-less third-party deps must extend that override.

For hand-written nested dicts (e.g. per-regime / per-horizon stats), prefer a `TypedDict` over scattered suppressions — those errors point at a typing gap, not a Polars one.

### 7.5 Design proposals — use issues, not files

New design proposals go in a **GitHub issue** (label: `design`), not a markdown file under `docs/plans/`. Issues give threaded discussion, edit history, cross-links to PRs / commits, and zero file-maintenance overhead. The `docs/plans/archive/` directory is the frozen pre-v0.10 plan corpus — read-only history; never add to it.

Exceptions where a file-form plan still earns its keep:

- Multi-thousand-line specs with heavy LaTeX / diagrams that strain GitHub markdown
- Plans that go through ≥3 numbered revisions where commit history of the file itself is the record

In those cases, file under `docs/plans/active/<short-slug>.md` and open a tracking issue that links to it. Once shipped or superseded, the same PR that lands the work moves the file to `docs/plans/archive/`.

---

## 8. Testing rules

### Synthetic fixtures only

**factrix tests must not load real market data**—every fixture must be
constructed programmatically in `tests/conftest.py` or the test module
itself, using numpy / polars. This invariant lets tests run in any
environment (CI / new machine / fresh clone) and prevents the repo
from being polluted with data.

Pattern (see `tests/conftest.py`):

```python
@pytest.fixture
def clean_panel():
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "date": [...],
        "asset_id": [...],
        "factor_value": rng.normal(size=n),
        "forward_return": rng.normal(size=n) * 0.01,
    })
```

### New features require tests

A new metric / Profile field / API parameter must come with a matching
test in the same PR. The PR reviewer (you + Claude, today) should
block the PR if tests are missing.

### CI must be green

`.github/workflows/test.yml` runs the full pytest suite on every push
/ PR. **A red PR must not merge**—fix first, then continue.

### Docstring style boundary — Google sections, ruff for everything else

The project's style policy is split between two complementary but distinct conventions; conflating them invites drift.

- **Code formatting and structure** (line length, naming, imports, indentation, formatter tool) follows PEP 8 and the Black / ruff defaults configured in `pyproject.toml [tool.ruff]`. The selector set (`E/W/F/I/B/UP/SIM/RUF`) and 88-character line length there are the source of truth.
- **Docstring format** follows Google docstring style — section headers `Args:` / `Returns:` / `Raises:` / `Warns:` / `Notes:` / `Examples:` / `References:` — because the mkdocstrings python handler is configured for `docstring_style: google` (see `mkdocs.yml`). Section ordering is `Args` → `Returns` → `Raises` → `Warns` → `Notes` → `Examples` → `References`. Section names are plural (`Examples:` not `Example:`, `Notes:` not `Note:`, `Warns:` not `Warning:`); the mkdocstrings handler accepts both, but plural-only keeps rendered admonitions consistent.
- **The Google Python Style Guide as a whole is not adopted.** Its 80-character line limit, single-quote string preference, and yapf formatter conflict with the ruff configuration above and do not apply. Only the docstring section convention is taken from Google.

NumPy-style underline sections (`Parameters\n----------`) and Sphinx field lists (`:param x:` / `:returns:`) are not parsed by the Google handler and render as plain prose under generic headings. Convert on sight.

### Markdown code-block intent layers — runnable vs illustrative

Code blocks under `docs/api/**/*.md` carry two distinct intents; verify which layer a block belongs to before editing.

- **Runnable** — `pycon` blocks injected from docstring `Examples:` via mkdocstrings autodoc. Self-contained imports, no unbound names, no fragile output; the rendered page exposes a copy button that strips `>>>` and expected-output lines, so blocks must remain paste-ready Python.
- **Illustrative** — hand-authored `python` fenced blocks that use unbound names (e.g. `panel_large`, `regime_labels`) to communicate semantic intent, plus ASCII / DataFrame layouts that document output schema. Deliberately not runnable; visual lookup value beats setup faithfulness. Do not "fix" these into runnable form — confirm the intent first.

### Metric docstring style

Docstrings in `factrix/metrics/*.py` are the **authoritative source**
for each metric's "exact formula / algorithm". Overall doc partitioning
(what goes where) is in the README "Where to look next" table—single
source, not restated here. Format:

1. **TL;DR first line** — one sentence describing what the metric
   measures, optionally with a short formula
   (e.g. `IC_IR = mean(IC) / std(IC).`)
2. **Formula**:
   - **Single-line formula** → inline, format `value = <expr>`
   - **Multi-line algorithms / sandwich SE / non-trivial procedures** →
     `Formula:` block with indented equations, so anyone scanning
     `help()` gets readable display math
3. **Args / Returns** blocks (Google-style)
4. **Short-circuit conditions** as a final paragraph (which inputs
   short-circuit to NaN, what `metadata["reason"]` reports)
5. Paper citations under a `References:` block (optional; only complex
   methods need it, simple diagnostics don't)

Examples (inline formula): `ts_beta.ts_beta_sign_consistency`
Examples (Formula block): `fama_macbeth.pooled_ols`,
`_helpers._sample_non_overlapping`

### LLM agent reference sync

The SSOT lives in `factrix/llms.txt` and `factrix/llms-full.txt`
(shipped with the wheel; an mkdocs hook mirrors them to the site root
at build time). Their content overlaps with the mkdocs site partially
but **neither is the SSOT for the other**—agents need density and
humans need progressive disclosure; the structural targets are
mutually exclusive, so both tracks are maintained.

When any of the following ships, sync `factrix/llms*.txt` in the same
PR (no CI gate):

- Additions / removals to `factrix/__init__.py` `__all__`
- Public API signature changes (factory, `evaluate`, `bhy`,
  `FactorProfile`)
- `WarningCode` / `InfoCode` / `StatCode` additions, renames, or
  description rewrites
- Mode dispatch rules or canonical panel schema changes

PR self-check: run all three code blocks, `uv run mkdocs build
--strict` clean, `tiktoken` cl100k count < 8000.

### Docs callout conventions

Use mkdocs-material admonitions to elevate content that breaks from the
surrounding flow (different audience or different urgency). Default to
plain prose; reach for a callout only when the elevation earns it.

- `!!! abstract "Answers"` — top-of-page scope statement on reference pages.
- `!!! warning` — gotcha / data-validity precondition the reader must check first; surface invariants whose violation breaks the analysis silently.
- `!!! note` — orthogonal context that helps but isn't required (e.g. mode-axis edge cases).
- `!!! info` — contract / convention block (e.g. event-study contracts, TS-mode conventions).
- `!!! example` — minimal worked code that surrounding prose references.
- `??? note "..."` (collapsible) — long content for a subset of readers (derivations, full enum tables).
- `> **Input contract** — …` (blockquote, two lines) — appears only on raw-panel `(panel, cfg)` entry points (`docs/api/evaluate.md`, `docs/api/run-metrics.md`), placed between the frontmatter and the autodoc block. Format: one short sentence naming the four-column floor + a link to [Panel schema](../api/panel-schema.md). Other API pages consume pre-computed artefacts (`FactorProfile` / `Survivors` / `MetricsBundle`) and do not carry the callout.

Apply opportunistically: when you touch a page for any other reason and a paragraph already qualifies, hoist it. Do not retrofit pages just to add admonitions.

### Autodoc target — top-level path for `__all__` symbols

For each `::: <target>` directive in `docs/api/`, the target dotted path matches the symbol's canonical user-facing import:

- Symbol in `factrix.__all__` → top-level path (`::: factrix.evaluate`, `::: factrix.by_slice`, `::: factrix.SliceResult`). Do not target the submodule that physically defines it (e.g. `factrix.slicing.dispatcher` with `members: [by_slice]`) — submodule-target with member filter renders the *submodule* as the page h1 and buries the documented symbol below.
- Symbol reached only via a submodule path → submodule path (`::: factrix.preprocess.compute_forward_return`, `::: factrix.metrics.ic` with `members: [ic, compute_ic, ic_newey_west]`, `::: factrix.datasets.make_cs_panel`). The submodule path is the canonical import.

mkdocstrings cross-references (`[X][factrix.<...>.X]`) and intra-doc anchor links (`page.md#factrix.<...>.X`) follow the same rule — the path inside the brackets matches the autodoc target. Changing one without the other breaks the cross-ref.

### Autodoc options — globals + per-block deviations

`mkdocs.yml` carries the page-primary defaults for mkdocstrings (`show_root_heading: true`, `show_root_full_path: true`, `show_root_toc_entry: true`, `heading_level: 1`, `separate_signature: true`, `show_signature_annotations: true`, `show_source: false`, `merge_init_into_class: true`, `docstring_style: google`). Every `::: factrix.<X>` block in `docs/api/**` inherits these.

Per-block `options:` should carry **only deviations** from the globals. Common deviations:

- Secondary block on a page (e.g. `Survivors` on `bhy.md`, individual error classes on `errors.md`): `show_root_toc_entry: false`, `heading_level: 3` or `4`.
- Dataclass page where the class name is already in the frontmatter title: `show_root_heading: false`.
- Module-level block with a curated function list: `members: [...]`, optionally `show_root_members_full_path: true`.

Bare `::: factrix.<X>` is the canonical form for page-primary function / dataclass blocks; do not duplicate globals.

### Page-shape conventions — when to add `## Use cases` / `## Worked example`

These two sections appear on pages whose primary purpose is to show the reader *how to call the API*. They are content shapes for workflow-oriented pages — not a universal requirement.

- **Expected on callable entry points.** Function pages under `docs/api/` whose page subject is a callable the user invokes directly. Includes the entry-point callables (`evaluate`, `run_metrics`, `bhy`, `partial_conjunction`, `bhy_hierarchical`, `by_slice`, `slice_pairwise_test` / `slice_joint_test`, `compare`, `list_metrics`, `list_estimators`, `suggest_config`, `preprocess.compute_forward_return`), and every metric page under `docs/api/metrics/` (each documents one or more callables).
- **Not expected** on:
    - Dataclass / container pages (`factor-profile.md`, `metric-output.md`, `metrics-bundle.md`) — these describe a return type, not a workflow.
    - Reference / taxonomy / hub pages (`errors.md`, `decision-tree.md`, `panel-schema.md`, `api/index.md`, `multi-horizon.md`, `identity.md`, `estimator-alternatives.md`, `metrics/index.md`, the cell-grouped metrics index pages) — content shape is a table or a concept, not a call.
    - Namespace / module pages (`stats.md`, `datasets.md`) — content shape is a catalogue of members.

A page that legitimately does not need these sections carries no marker — silence is the policy default. Pages in the "expected" category that currently lack the sections are accepted as a backfill debt rather than a defect.

### Terminology — functional names, not stage labels

Code (`factrix/**/*.py` docstrings + module headers) and published docs (`docs/**/*.md` excluding `docs/plans/`) describe behaviour by **what a function does**, not by **which planning tier it lives in**. Stage labels — `Layer-A` / `Layer-B`, `first / second layer`, `Phase 1` / `P1`, `curated wrapper`, `dispatcher vs wrapper` as a tier pair — belong only to GitHub issues / labels / milestones, where they can be renamed as the roadmap shifts.

Reason: planning labels drift on the issue tracker (`P1` → `P0` after triage, `Phase 1` → `v1` after milestone rename, `Layer-B` → `slice-test function` after the feature lands), but a docstring or `architecture.md` paragraph is bound to a release. The two timescales pull apart; the docstring becomes wrong without ever being touched. Additionally, AI agents reading a body that says `P1 contract` will copy that label into downstream files and amplify the drift.

The slicing subsystem is the worked example of the rule:

| Stage-label phrasing (avoid) | Functional phrasing (use) |
|---|---|
| `Layer-A` / first-layer dispatcher | **slice dispatcher** — describes partitioning by label + applying a metric per slice (`by_slice`) |
| `Layer-B` / second-layer / curated wrapper (inference path) | **slice-test function** / **inference function** — describes the cross-slice estimator + multiple-testing pipeline (`slice_pairwise_test` / `slice_joint_test`) |
| `Layer-B` Estimator | **slice-test Estimator** — Estimators consumed by the slice-test functions (`WaldNWCluster` / `WaldTwoWayCluster` / `BlockBootstrap`) |
| metric-specific `regime_<metric>` curated wrapper | **legacy metric-specific wrapper** (when describing removed surface area); for the current path, name `by_slice` + the inference function directly |
| `SliceResult.to_frame()` renderer layer | **renderer** — container-side method; no separate tier implied |

The rule is functional, not lexical — `dispatcher`, `function`, and `wrapper` are fine on their own when they describe what the function does. It is the **pairing** as a tier label (`dispatcher` vs `curated wrapper` as the two levels of the slicing system) that drifts; the same word as a behavioural noun is stable. Mention an issue number (`#176`) when the docstring needs to point at a specification, instead of `Layer-B (#176)` which encodes a label that will not survive.

#### Two-register convention: "verb" vs "function" / "entry point"

User-facing surface (`docs/**/*.md`, README, docstrings, CHANGELOG) uses **function** when referring to one specific callable, and **entry point** when referring to the set of public callables (the seven that appear in nav under "Entry points"). Design-issue bodies and RFC comments may keep **verb** as RFC vocabulary — that register is internal to design discussion and does not propagate to user docs. When sweeping prose from a design issue into a guide or docstring, translate `verb` → `function` (or rephrase to name the specific callable) as part of the move.

---

## 9. Versioning and release (SemVer & Release)

factrix is currently in **pre-1.0** (v0.x.x)—the public API **may
break in MINOR bumps**. Consumers (e.g. the `factor-analysis`
workspace) should pin via **git submodule SHA**, not version range,
until 1.0.0 stabilises.

**The project uses Commitizen for fully automated bump and changelog
generation, paired with the release-train cadence: PRs merge whenever,
but releases (bump + tag) are scheduled independently.**

### Release cadence — release train

PRs and releases are decoupled:

- **PR cadence**: high-frequency, atomic. Do **not** bump or tag after
  merge.
- **Release cadence**: low-frequency, aggregated. A release fires when
  any of the following holds:
  - ≥ 3 user-facing `feat:` / `fix:` accumulated
  - ≥ 2 weeks since the last tag
  - An urgent downstream-workspace bug fix (a one-off PATCH may ship
    immediately)
  - A named version is needed for a person / demo

Each PR writes its own WHY narrative into the `## [Unreleased]`
section of `CHANGELOG.md` (under `### Added` / `### Changed` /
`### Fixed` / `### Migration` subsections). At release time, no
narrative reconstruction is needed—`cz bump --changelog` freezes the
section into the next version heading.

### Release workflow

```bash
# 1. On main, ensure latest
git checkout main && git pull

# 2. Verify CI is green and local pytest passes
uv run pytest

# 3. Auto-bump and tag
# cz derives the level from commits since the last tag (feat=MINOR, fix=PATCH),
# renames [Unreleased] to the new version heading, adds a fresh empty
# [Unreleased], updates pyproject.toml, and auto-commits + tags.
cz bump --changelog

# 4. (Optional) Manually polish the release section — fill in BREAKING
#    migration / direction / motivation to ≥ 25 non-blank lines, otherwise
#    the pre-push hook blocks (see §2). After polishing, amend the release
#    commit and re-tag:
git commit --amend --no-edit
git tag -d v<X.Y.Z> && git tag v<X.Y.Z>

# 5. Push
git push origin main
git push origin v<X.Y.Z>

# 6. Bump the workspace submodule
cd ~/Desktop/dst/code/factor-analysis
cd external/factrix && git fetch && git checkout v<X.Y.Z>
cd ../.. && git add external/factrix
git commit -m "chore: bump factrix to v<X.Y.Z>"
git push
```

### BC change reminders

Example: `q1_q5_spread → long_short_spread` (this rename actually
occurred in workspace history). This kind of rename is a BC change.
When using `cz commit`, the developer must select Breaking Change and
**explicitly write the migration path** in the prompt (old → new
names, affected fields). This way, downstream workspaces find the
upgrade guide directly in the auto-generated CHANGELOG.

### Workspace pins to tags, not main

Downstream research workspaces should generally **pin to a tag** (not
main HEAD), so each workspace commit corresponds to a clear factrix
version and remains reproducible.

Main HEAD is used only temporarily during Mode B development (debug
flow); once finished, merge back into factrix main, tag, and let the
workspace bump to the tag.

---

## 10. Architecture / design decisions

Before a new feature or large change, read:

- `docs/development/architecture.md` — current snapshot of the package
  (positioning, public API, factor types, Profile contract, artifacts,
  invariants)
- `CHANGELOG.md` — historical BC changes and caveats

`docs/development/architecture.md` describes the **current state**, not
the design history. For "why is it designed this way" process records,
see the `awwesomeman/factor-analysis` workspace under
`docs/spike_*.md` / `docs/refactor_*.md` (pre-extraction history is
preserved there).

---

## 11. Asking questions / decision communication

Self-use repo for the author + AI agents, so there is no issue
template / discussion board. Decision-record channels:

- **Small changes**: PR description spells out the why and any BC
- **Large changes / architectural decisions**: add a `spike_*.md`
  design doc under the workspace repo's `docs/` (matching historical
  spikes), and reference it from the factrix PR
- **Invariant-level rule changes**: update the `Invariants` section in
  `docs/development/architecture.md`

---

## 12. Style — single language

All issue / PR titles + bodies, commit messages, CHANGELOG entries,
and any content under `factrix/` and `docs/` (excluding `docs/plans/`)
is written in **English**. Internal planning notes under `docs/plans/`
may remain bilingual; they are excluded from the published mkdocs site
via `mkdocs.yml` `exclude_docs`.

Rationale: mixed-language content fragments full-text search
(`gh issue list`, GitHub search, `git log`), splits visual flow in
notification emails, and burdens readers who must context-switch
mid-paragraph. The choice of language is not prescribed by tooling—the
repo settled on English to align with the public docstring surface,
README, and CHANGELOG.

---

## 13. Licensing and contribution terms

This project is released under the [Apache License 2.0](https://github.com/awwesomeman/factrix/blob/main/LICENSE).

**Inbound = Outbound**: per Apache-2.0 §5, any content you submit to
this repo (PRs, patches, issue-attached code, etc.) is **deemed
licensed back to the project under the same Apache-2.0 terms**, unless
you explicitly state otherwise at submission time. Before opening a
PR, confirm:

- You hold the right to license that code (self-authored, or sourced
  under an Apache-2.0-compatible licence)
- When citing third-party code, mark the original licence in the file
  header or PR description
- Patent-encumbered algorithms (e.g. methods you or your organisation
  hold patents on) must be disclosed proactively in the PR description

Code under strong copyleft (GPL/AGPL etc.) is not accepted into the
main code tree, since it would propagate licence obligations onto
downstream users.
