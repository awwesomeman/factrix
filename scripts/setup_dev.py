"""One-time local dev setup — install the pre-commit framework hooks.

Runs ``pre-commit install`` for the pre-commit / commit-msg / pre-push
stages so the hooks declared in ``.pre-commit-config.yaml`` fire
automatically. The installation is per-clone and does not travel with
``git clone``; ``git worktree`` instances share ``.git/hooks`` with the
primary clone, so one run covers every worktree under it.

Idempotent: re-running just rewrites the same hook shims.

Migration: earlier setups pointed ``core.hooksPath`` at the tracked
``.githooks`` directory, which is gone. That setting would shadow the
``.git/hooks`` shims pre-commit installs, so it is unset here. A
``core.hooksPath`` pointing anywhere else is a contributor-managed hook
surface — the script aborts rather than overwrite it.

Usage::

    python scripts/setup_dev.py
"""

from __future__ import annotations

import subprocess
import sys

_LEGACY_HOOKS_PATH = ".githooks"
_HOOK_TYPES = ("pre-commit", "commit-msg", "pre-push")


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

    hooks_path = _git_config_get("core.hooksPath")

    if hooks_path == _LEGACY_HOOKS_PATH:
        _run_git("config", "--unset", "core.hooksPath", check=True)
        print(
            f"[setup_dev] unset core.hooksPath (was {_LEGACY_HOOKS_PATH!r}). "
            "The tracked hook scripts moved under the pre-commit framework; "
            "leaving the old path set would shadow the hooks installed below."
        )
    elif hooks_path is not None:
        print(
            f"[setup_dev] ERROR: core.hooksPath is set to {hooks_path!r}, which "
            "shadows the hooks pre-commit installs into .git/hooks.",
            file=sys.stderr,
        )
        print(
            "[setup_dev] Refusing to overwrite a contributor-managed hook path. "
            "If you want this clone to use the repo hooks, run "
            "`git config --unset core.hooksPath` then re-run this script.",
            file=sys.stderr,
        )
        return 1

    install_cmd = [
        "uv",
        "run",
        "pre-commit",
        "install",
        *[arg for hook_type in _HOOK_TYPES for arg in ("--hook-type", hook_type)],
    ]
    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            "[setup_dev] ERROR: 'uv' not found in PATH. "
            "Install uv (https://docs.astral.sh/uv/), then re-run this script.",
            file=sys.stderr,
        )
        return 1

    if result.returncode != 0:
        print(
            f"[setup_dev] ERROR: {' '.join(install_cmd)} failed.",
            file=sys.stderr,
        )
        print(result.stdout + result.stderr, file=sys.stderr)
        print(
            "[setup_dev] Install the dev tooling first: uv sync --extra dev",
            file=sys.stderr,
        )
        return 1

    print(result.stdout.strip())
    print(
        f"[setup_dev] pre-commit hooks installed ({', '.join(_HOOK_TYPES)}) "
        "for this clone (and any worktree sharing its .git/hooks)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
