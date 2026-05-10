"""One-time local dev setup — point Git at the repo-tracked hooks.

Sets ``core.hooksPath`` to ``.githooks`` so the version-controlled
pre-commit / pre-push scripts run automatically. The setting is
per-clone and does not travel with ``git clone``; ``git worktree``
instances share ``.git/config`` with the primary clone, so one run
covers every worktree under it.

Idempotent: re-running is a no-op. Aborts (non-zero exit) if
``core.hooksPath`` is already set to a different path so a
contributor's dotfiles-managed hook surface is not silently
overwritten.

Usage::

    python scripts/setup_dev.py
"""

from __future__ import annotations

import subprocess
import sys

_TARGET_PATH = ".githooks"


def _run_git(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _git_config_get(key: str) -> str | None:
    """Return the value of a git config key, or ``None`` if unset."""
    result = _run_git("config", "--get", key)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> int:
    try:
        repo_check = _run_git("rev-parse", "--git-dir")
    except FileNotFoundError:
        print(
            "[setup_dev] ERROR: 'git' not found in PATH. "
            "Install Git, or run this script from a shell where git is available.",
            file=sys.stderr,
        )
        return 1

    if repo_check.returncode != 0:
        print(
            "[setup_dev] ERROR: not inside a Git working tree. "
            "Run this script from the factrix clone root.",
            file=sys.stderr,
        )
        return 1

    current = _git_config_get("core.hooksPath")

    if current == _TARGET_PATH:
        print(
            f"[setup_dev] core.hooksPath already set to {_TARGET_PATH!r}; nothing to do."
        )
        return 0

    if current is not None:
        print(
            f"[setup_dev] ERROR: core.hooksPath is already set to {current!r}, "
            f"not {_TARGET_PATH!r}.",
            file=sys.stderr,
        )
        print(
            "[setup_dev] Refusing to overwrite a contributor-managed hook path. "
            "If you want this clone to use the repo hooks, run "
            "`git config --unset core.hooksPath` then re-run this script.",
            file=sys.stderr,
        )
        return 1

    _run_git("config", "core.hooksPath", _TARGET_PATH, check=True)
    print(
        f"[setup_dev] core.hooksPath -> {_TARGET_PATH}. "
        "Hooks in .githooks/ are now active for this clone "
        "(and any worktree sharing its .git/config)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
