---
title: Documentation conventions
---

This page defines how factrix documentation is organised, generated, and
reviewed. The project style takes precedence; otherwise prefer clear, direct
technical prose and separate pages by the reader's task.

## Information architecture

The navigation classifies content by how it is read, not by its source folder:

| Page type | Reader task | Examples |
|---|---|---|
| Concepts | Understand the mental model or scope | Concepts, Where factrix fits, Panel vs timeseries |
| How-to | Complete a specific task | Preparing data, Choosing a metric, Reading results |
| Example | Follow an end-to-end runnable workflow | Multi-factor screening, Stock factor evaluation |
| Method reference | Check assumptions, formulas, conventions, or calibration | Statistical methods, Inference calibration |
| Lookup reference | Scan a table or reverse index | Metric applicability, Stat keys, Warning codes |
| API reference | Look up a public symbol's signature and semantics | `evaluate`, `ic`, `EvaluationResult` |
| Development | Change or release the project | Architecture, Contributing, this page |

Do not move a file merely to match its navigation section. Published URLs are
versioned and may be linked from notebooks, issues, and search indexes;
`mkdocs.yml` is the editorial layer.

Keep the navigation shallow. Group labels are non-clickable headings, and hub
pages appear as an explicit `Overview` leaf. Page and nav titles use sentence
case. Public identifiers keep their exact spelling (`fm_beta`, not
`fama_macbeth`); expand uncommon acronyms in body prose, not sidebar labels.

## Sources of truth

| Source | Published target | Synchronisation |
|---|---|---|
| Public Python docstrings | `docs/api/**/*.md` `:::` blocks | mkdocstrings at build time |
| `factrix._metric_index.public_specs()` | Generated metric matrix and name index | MkDocs hooks |
| `WarningCode.description` | Generated warning-code descriptions | MkDocs hook |
| `examples/*.ipynb` | `docs/examples/*.md` | Notebook-render hook |
| `factrix/llms*.txt` | Site-root LLM snapshots | MkDocs hook |
| `CHANGELOG.md` | `docs/development/changelog.md` | Snippet include |

Do not hand-edit generated tables or example Markdown. Edit the registry,
warning description, notebook, or root changelog, then regenerate through a
strict docs build. Generated Markdown stays committed so repository readers do
not need MkDocs to see it.

Narrative pages, API-page introductions, navigation entries, and cross-links
remain manually maintained. A code change is not complete until those pages
match the public behaviour.

## API and docstring layers

Docstrings are the callable-level contract. They explain parameters, return
values, raised errors, statistical notes, examples, and citations needed to
use that symbol correctly. API Markdown pages add only page-level orientation,
use cases, or links; do not restate the generated parameter schema.

Use Google-style sections in this order when present:

1. Summary and body prose.
2. `Args`, `Returns`, and `Raises`.
3. `Notes` for assumptions, statistical contracts, and non-obvious design
   choices.
4. `References`.
5. `Examples`.

A function must remain understandable without following a citation. Put the
claim and its practical implication in `Notes`, then cite the source. Use a
short authored bibliography label that resolves to
[`bibliography.md`](../reference/bibliography.md); the bibliography is a
catalog, while the docstring remains the local source of behavioural truth.

Module docstrings provide navigation and family-wide conventions. Do not copy
an entire callable contract into both the module and function. When a rule is
shared across many modules, document it once on a reference page and link to
it from the local docstrings.

## Examples and code blocks

Python blocks are executable by default. They must run in document order with
the current public API and use deterministic synthetic data where randomness
matters.

Use an illustrative block only when external state or a deliberately omitted
object prevents execution:

````markdown
```python title="Illustrative"
# Sketch that depends on caller-owned data.
```
````

Illustrative code still has to parse. Prefer a runnable example over a sketch;
do not mark stale code illustrative merely to bypass validation.

Notebook examples are the source of truth for multi-step tutorials. Keep short
single-call examples in docstrings or API pages and link to the notebook for a
complete workflow.

## Writing style

- Lead with the reader's outcome or the contract, then explain why.
- Keep one main idea per paragraph and put its distinguishing information in
  the first sentence.
- Use descriptive headings and a consistent heading hierarchy.
- Prefer direct verbs and concrete nouns over roadmap or implementation-tier
  labels.
- Say **periods** for contract-level horizons, windows, and lags. Calendar
  cadences belong only in clearly labelled examples.
- Define a specialised acronym on first use. Keep universal API identifiers
  and literal code names exact.
- Use lists for steps and parallel alternatives; use tables for repeated field
  comparisons, not ordinary prose.
- Use meaningful link text. Avoid “click here,” bare paths, and links whose
  label does not describe the destination.
- Avoid superlatives and competitor claims that require continuous market
  monitoring. State verifiable scope differences instead.

Long lookup catalogs are acceptable when their headings and generated indexes
support scanning. Long narrative pages are a signal to separate concepts,
procedures, and evidence—not a rule to split at a fixed line count.

## Metric and statistical pages

Per-metric API pages answer:

- what the metric estimates;
- what input and cell it accepts;
- the inference unit and null hypothesis, when present;
- how to read `value`, `stat`, `p_value`, `n_obs`, metadata, and warnings;
- where the cross-cutting method and calibration evidence live.

Keep general estimator contracts in
[Statistical methods](../reference/statistical-methods.md) and empirical null
measurements in
[Inference calibration](../reference/inference-calibration.md). Do not turn a
method reference into a chronological lab notebook; retain the measured range,
null design, known limit, and producing test, and leave obsolete intermediate
attempts in git history or the design issue.

## Drift checks

Run the checks that match the edited layer:

```bash
uv run python scripts/mkdocs_hooks/gen_metric_matrix.py
uv run python scripts/mkdocs_hooks/gen_metric_name_index.py
uv run python scripts/mkdocs_hooks/gen_code_descriptions.py
uv run python scripts/mkdocs_hooks/render_example_notebooks.py
uv run pytest tests/test_docs_pages.py tests/test_docs_paths.py
uv run pytest tests/test_docs_examples.py tests/test_docs_bibliography.py
uv run mkdocs build --strict
```

The full test suite also checks public symbols, generated matrices, notebook
drift, warning descriptions, bibliography anchors, metric API members, and LLM
snapshots. A strict MkDocs build catches rendered navigation and anchor errors
that source-only checks cannot reproduce.

## Review checklist

- Does each page serve one reader task?
- Is information repeated because two entry points need a short orientation,
  or because two files are accidentally acting as the source of truth?
- Are assumptions, inference unit, defaults, and finite-sample limits stated
  without overclaiming?
- Do links use stable relative paths and meaningful labels?
- Are long tables lookup material, and are long narratives split by purpose?
- Do examples, generated files, API pages, and docstrings match the current
  code?
