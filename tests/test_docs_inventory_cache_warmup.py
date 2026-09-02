"""The inventory warm-up must survive a blip without hiding a real failure.

Under ``--strict`` a failed intersphinx fetch aborts the docs build, and
``mkdocstrings`` makes exactly one attempt. #1037: a ``Connection reset by
peer`` from ``docs.pola.rs`` failed a pull request whose content was fine.

These pin the three behaviours that make retrying safe rather than merely
quieter — recover from a blip, refuse to retry a server verdict, and stay
non-fatal so ``mkdocstrings`` reports whatever it finds in its own terms.
"""

from __future__ import annotations

import pathlib
import urllib.error

import pytest
from scripts.mkdocs_hooks.warm_inventory_cache import (
    _ATTEMPTS,
    _CACHE_DURATION,
    _download_with_retry,
    warm,
)

# The hook itself needs nothing beyond the standard library, but exercising it
# needs the docs extras. The dependency-floor CI job installs neither, so skip
# the module there rather than dragging mkdocs into the declared floor.
yaml = pytest.importorskip("yaml", reason="docs extra not installed")
pytest.importorskip("mkdocstrings", reason="docs extra not installed")

MKDOCS_YML = pathlib.Path("mkdocs.yml")


@pytest.fixture
def downloader(monkeypatch):
    """Replace the network call mkdocstrings would make; count attempts."""
    calls: list[str] = []

    def install(behaviour):
        def fake(url: str) -> bytes:
            calls.append(url)
            return behaviour(len(calls))

        monkeypatch.setattr(
            "mkdocstrings._internal.handlers.base._download_url_with_gz", fake
        )
        return calls

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    return install


def _reset_error(_attempt: int) -> bytes:
    """The exact failure #1037 observed."""
    raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))


def test_recovers_from_a_transient_reset(downloader) -> None:
    """One blip then success — the shape #1037 actually hit.

    The second attempt is hard-coded rather than derived from ``_ATTEMPTS``:
    a test that reads the constant it exists to guard passes when the
    constant drops to 1, which is precisely the regression to catch.
    """

    def behaviour(attempt: int) -> bytes:
        if attempt == 1:
            return _reset_error(attempt)
        return b"inventory"

    calls = downloader(behaviour)
    assert _download_with_retry("https://example.invalid/objects.inv") == b"inventory"
    assert len(calls) == 2


def test_does_not_retry_a_server_verdict(downloader) -> None:
    """A 404 is a mistyped URL, not a blip; retrying would only hide it slowly."""

    def behaviour(_attempt: int) -> bytes:
        raise urllib.error.HTTPError(
            "https://example.invalid/typo.inv", 404, "Not Found", {}, None
        )

    calls = downloader(behaviour)
    with pytest.raises(urllib.error.HTTPError):
        _download_with_retry("https://example.invalid/typo.inv")
    assert len(calls) == 1


def test_gives_up_after_the_declared_attempts(downloader) -> None:
    assert _ATTEMPTS > 1, "one attempt is the behaviour this hook replaces"
    calls = downloader(_reset_error)
    with pytest.raises(urllib.error.URLError):
        _download_with_retry("https://example.invalid/objects.inv")
    assert len(calls) == _ATTEMPTS


def test_warm_up_is_not_itself_fatal(downloader, capsys) -> None:
    """A dead host must leave the real error to mkdocstrings, not pre-empt it.

    Raising here would move the failure into the hook and lose the handler
    context, while still failing the build — no gain, worse message.
    """
    downloader(_reset_error)
    warm(["https://example.invalid/objects.inv"])
    assert "unavailable" in capsys.readouterr().out


def test_urls_come_from_mkdocs_yml_not_a_second_list() -> None:
    """A new inventory in mkdocs.yml is warmed without editing the hook."""
    configured = (
        yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8").replace("!!python/", "#"))
        or {}
    )
    plugins = configured.get("plugins") or []
    handler = next(
        entry["mkdocstrings"]["handlers"]["python"]
        for entry in plugins
        if isinstance(entry, dict) and "mkdocstrings" in entry
    )
    declared = handler["inventories"]
    assert declared, "mkdocs.yml should declare at least one inventory"

    hook_source = pathlib.Path(
        "scripts/mkdocs_hooks/warm_inventory_cache.py"
    ).read_text(encoding="utf-8")
    for url in declared:
        assert url not in hook_source, (
            f"{url} is hard-coded in the hook; it must be read from mkdocs.yml "
            "so a new inventory is warmed automatically."
        )


def test_cache_duration_matches_what_mkdocstrings_asks_for() -> None:
    """A shorter warm-up TTL would warm the cache and be re-downloaded anyway."""
    import datetime
    import inspect

    from mkdocstrings._internal.handlers import base

    source = inspect.getsource(base)
    assert "def _download_inventories" in source, (
        "mkdocstrings no longer downloads inventories through this module; "
        "re-check what _CACHE_DURATION has to match."
    )
    assert "datetime.timedelta(days=1)" in source, (
        "mkdocstrings' cache duration changed; _CACHE_DURATION must follow it "
        "or the warmed entry will be considered stale."
    )
    assert datetime.timedelta(days=1) == _CACHE_DURATION


def test_hook_is_registered_before_the_generators() -> None:
    """It must run in the build, and first — the others do not need the network."""
    hooks = (
        yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8").replace("!!python/", "#"))
        or {}
    ).get("hooks") or []
    assert "scripts/mkdocs_hooks/warm_inventory_cache.py" in hooks
