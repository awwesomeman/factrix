<!-- Release PR template. Select via ?template=release.md on the PR
     compare URL, or copy this body into the PR description manually. -->

## Summary

- Release **vX.Y.Z** — <one-line theme>
- CHANGELOG `[Unreleased]` section moved under `## vX.Y.Z (YYYY-MM-DD)` and polished per `release-management/SKILL.md` §3.

## Release checklist

- [ ] `cz bump --changelog` ran on `main` (not on a feature branch)
- [ ] CHANGELOG entries polished (Keep-a-Changelog headings, WHY added, breaking-change migration notes)
- [ ] `pyproject.toml:version` + `factrix/__init__.py:__version__` updated by `cz` (verify the diff)
- [ ] Annotated tag created and pushed via `git push origin main --follow-tags`
- [ ] GitHub Release drafted from the tag

## Notes

<!-- Any non-obvious context: deferred work, etc. -->
