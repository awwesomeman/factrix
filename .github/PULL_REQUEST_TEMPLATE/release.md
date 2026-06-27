<!-- Release PR template. Select via ?template=release.md on the PR
     compare URL, or copy this body into the PR description manually. -->

## Summary

- Release **vX.Y.Z** — <one-line theme>
- Pre-1.0: CHANGELOG remains a policy note + GitHub-release index; PR body carries the WHY / migration narrative.
- v1.0.0+: CHANGELOG `[Unreleased]` section moved under `## vX.Y.Z (YYYY-MM-DD)` and polished per `release-management/SKILL.md` §3.

## Release checklist

- [ ] Version bump ran on `main` (not on a feature branch)
- [ ] Pre-1.0: historical release index updated if a GitHub Release will exist
- [ ] v1.0.0+: CHANGELOG entries polished (Keep-a-Changelog headings, WHY added, breaking-change migration notes)
- [ ] `pyproject.toml:version` + `factrix/__init__.py:__version__` updated by `cz` (verify the diff)
- [ ] Annotated tag created and pushed via `git push origin main --follow-tags`
- [ ] GitHub Release drafted from the tag

## Notes

<!-- Any non-obvious context: deferred work, etc. -->
