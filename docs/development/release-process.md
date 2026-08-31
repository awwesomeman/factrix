---
title: Release process
---

Releases are cut from `main` after the selected pull requests are reviewed and
merged. A feature or fix PR does not bump the version, create a tag, or edit a
release solely because it landed.

## Pre-1.0 policy

Until `v1.0.0`, public API changes may ship in a minor release. Consumers
should pin an exact version or tag.

Published docs describe the current public surface. Do not add historical
“added in” or “removed in” prose to guides, references, or docstrings; put the
reviewable reason and migration path in the issue and pull request.

`CHANGELOG.md` remains a policy note and index of pre-1.0 GitHub releases.
Detailed maintained entries start at `v1.0.0`; do not reconstruct earlier
entries unless each statement is audited against its tag.

## Release cadence

Pull requests are small and frequent; releases aggregate them. Cut a release
when the accumulated user-facing change justifies a new version, a named build
is needed, or an urgent downstream fix cannot wait. The maintainer chooses the
release point after reviewing the merged change set.

For pre-1.0 versions:

- Use a minor bump for a planned batch that changes or adds public behaviour.
- Use a patch bump for a compatible correction to the current minor line.
- Keep the version, tag, GitHub release, and published docs label identical.

## Release audit

Start from a clean, current `main` and install every declared extra:

```bash
git switch main
git pull --ff-only origin main
uv sync --frozen --all-extras
```

Then run the checks mirrored by CI:

```bash
uv run pre-commit run --all-files --hook-stage pre-commit
uv run mypy factrix
uv run pytest
uv run pytest --doctest-modules factrix
uv run mkdocs build --strict
uv build
```

Before bumping, also review:

- removed or renamed symbols and their migration path;
- docs and LLM snapshots for retired terminology;
- generated metric, warning, and example pages for drift;
- optional pandas/pyarrow adapter coverage;
- the exact commits since the previous tag and their Conventional Commit
  types.

Do not reuse a removal-search pattern from an older release. Build it from the
current change set so the audit remains signal rather than accumulated noise.

## Bump, tag, and publish

Commitizen derives the bump from commits since the previous tag and updates the
configured version locations:

```bash
uv run cz bump
git push origin main --follow-tags
```

Inspect the generated release commit and tag before pushing. The tag must point
to the release commit on `main`; do not create a release branch solely for the
bump.

Create the GitHub release from the pushed tag and publish the matching version
of the docs. Release notes should lead with behavioural changes and migration
information, not a raw commit list.

From `v1.0.0` onward, maintain `## [Unreleased]` and freeze it into a versioned
section at release time. Changelog entries link to the pull request because it
contains the reviewed diff and rationale.

## Breaking changes

A breaking change must state:

- the old and new public names or behaviour;
- affected inputs, outputs, and serialized fields;
- a concrete migration example;
- whether the change alters calculation, inference, or only presentation.

Select the breaking-change option in the Conventional Commit workflow so the
release tool can derive the correct version. Before `v1.0.0`, the PR is the
primary migration record; after `v1.0.0`, carry the same concise guidance into
the maintained changelog and GitHub release.
