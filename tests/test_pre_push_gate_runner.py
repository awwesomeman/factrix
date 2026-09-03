"""A gate that could not run must not be reported as a gate that failed.

#1048: ``scripts/hooks/pre-push`` shells out to ``uv run mkdocs`` / ``uv run
mypy``. From a git worktree ``uv run`` resolves no environment — it exits with
``Failed to spawn: mkdocs`` — and the hook printed ``pre-push blocked: mkdocs
build --strict failed`` although mkdocs never started. That is the repo's own
"report what ran, not what was asked for" rule inverted, on its own tooling,
and its practical cost is a standing ``--no-verify`` habit that disables every
other gate along with the one that could not run.

The tests drive the real hook under a stub ``uv`` so the three outcomes stay
distinguishable: could not run (skip, non-blocking), ran and failed (blocked),
ran and passed (silent).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess

import pytest

HOOK = pathlib.Path("scripts/hooks/pre-push").resolve()

#: What ``uv run`` prints from a worktree with no synced environment.
_SPAWN_ERROR = "error: Failed to spawn: `mkdocs`\\n  Caused by: program not found"

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="POSIX shell not available")


def _write_stub(path: pathlib.Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _repo(tmp_path: pathlib.Path, touched: str) -> tuple[pathlib.Path, str, str]:
    """A one-commit repo whose commit touches ``touched``, plus the ref pair."""
    repo = tmp_path / "repo"
    (repo / pathlib.Path(touched).parent).mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    run("add", "seed.txt")
    run("commit", "-qm", "seed")
    base = run("rev-parse", "HEAD").stdout.strip()
    (repo / touched).write_text("x = 1\n", encoding="utf-8")
    run("add", touched)
    run("commit", "-qm", "touch the gated path")
    head = run("rev-parse", "HEAD").stdout.strip()
    return repo, base, head


def _run_hook(
    repo: pathlib.Path, base: str, head: str, stub_dir: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "PRE_COMMIT_FROM_REF": base,
        "PRE_COMMIT_TO_REF": head,
    }
    assert bash is not None
    return subprocess.run(
        [bash, str(HOOK)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("gate", "touched"),
    [("mkdocs", "docs/page.md"), ("mypy", "factrix/mod.py")],
)
def test_a_gate_that_cannot_run_is_not_reported_as_a_failed_gate(
    tmp_path: pathlib.Path, gate: str, touched: str
) -> None:
    """The #1048 defect: no environment can run the tool, so it never ran."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _write_stub(stub_dir / "uv", f'printf "{_SPAWN_ERROR}\\n" >&2; exit 1')

    repo, base, head = _repo(tmp_path, touched)
    result = _run_hook(repo, base, head, stub_dir)

    assert result.returncode == 0, (
        f"an unrunnable gate must not block the push:\n{result.stderr}"
    )
    assert "blocked" not in result.stderr, (
        f"the hook claimed a gate failed that never ran:\n{result.stderr}"
    )
    # The message has to say which gate was skipped and that CI still covers it,
    # as one statement: a bare "skipped" reads as "nothing to do here".
    assert gate in result.stderr
    assert "did not run" in result.stderr
    assert "CI still checks" in result.stderr


def test_a_gate_that_ran_and_failed_still_blocks(tmp_path: pathlib.Path) -> None:
    """False-positive guard: the skip path must not swallow a real failure."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    # Resolvable — the probe succeeds — but the build itself fails.
    _write_stub(
        stub_dir / "uv",
        'case "$*" in *--version*) exit 0 ;; *) echo "boom" >&2; exit 1 ;; esac',
    )

    repo, base, head = _repo(tmp_path, "docs/page.md")
    result = _run_hook(repo, base, head, stub_dir)

    assert result.returncode == 1
    assert "blocked" in result.stderr
    assert "did not run" not in result.stderr


def test_a_gate_that_ran_and_passed_says_nothing(tmp_path: pathlib.Path) -> None:
    """The other false-positive guard: a healthy environment is not a skip."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _write_stub(stub_dir / "uv", "exit 0")

    repo, base, head = _repo(tmp_path, "docs/page.md")
    result = _run_hook(repo, base, head, stub_dir)

    assert result.returncode == 0
    assert "blocked" not in result.stderr
    assert "did not run" not in result.stderr
