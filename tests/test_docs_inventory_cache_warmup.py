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
import re
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

#: Workflows whose jobs run ``mkdocs build`` and therefore need the cache.
_WORKFLOWS = (
    pathlib.Path(".github/workflows/docs-deploy-dev.yml"),
    pathlib.Path(".github/workflows/link-check.yml"),
)

#: The path the CI cache step declares. It has to be the directory
#: ``download_and_cache_url`` actually writes to on the runner, which is
#: ``platformdirs.user_cache_dir("mkdocs")`` — Linux ``~/.cache/mkdocs``.
_CI_CACHE_PATH = "~/.cache/mkdocs"


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


def _mkdocs_yml() -> dict:
    return (
        yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8").replace("!!python/", "#"))
        or {}
    )


def _declared_inventories() -> list[str]:
    """Inventory URLs as written in ``mkdocs.yml``."""
    handler = next(
        entry["mkdocstrings"]["handlers"]["python"]
        for entry in _mkdocs_yml().get("plugins") or []
        if isinstance(entry, dict) and "mkdocstrings" in entry
    )
    declared = handler["inventories"]
    assert declared, "mkdocs.yml should declare at least one inventory"
    return declared


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


def _http_error(code: int, reason: str):
    def behaviour(_attempt: int) -> bytes:
        raise urllib.error.HTTPError(
            "https://example.invalid/objects.inv", code, reason, {}, None
        )

    return behaviour


@pytest.mark.parametrize(
    ("code", "reason"),
    [(404, "Not Found"), (403, "Forbidden"), (410, "Gone"), (400, "Bad Request")],
)
def test_does_not_retry_a_verdict_about_the_url(downloader, code, reason) -> None:
    """A 4xx names the URL as wrong; retrying only hides a typo more slowly."""
    calls = downloader(_http_error(code, reason))
    with pytest.raises(urllib.error.HTTPError):
        _download_with_retry("https://example.invalid/objects.inv")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (429, "Too Many Requests"),
        (500, "Internal Server Error"),
        (502, "Bad Gateway"),
        (503, "Service Unavailable"),
        (504, "Gateway Timeout"),
    ],
)
def test_retries_a_verdict_about_the_server(downloader, code, reason) -> None:
    """A docs CDN answering 5xx is the unreachable host #1037 is about.

    It cannot conceal a mistyped URL — a typo produces 404/410, not 502 — so
    the concern that keeps 4xx on one attempt does not apply here.
    """
    calls = downloader(_http_error(code, reason))
    with pytest.raises(urllib.error.HTTPError):
        _download_with_retry("https://example.invalid/objects.inv")
    assert len(calls) == _ATTEMPTS


def test_recovers_from_a_transient_gateway_error(downloader) -> None:
    """The 5xx equivalent of the reset: one bad response, then success."""

    def behaviour(attempt: int) -> bytes:
        if attempt == 1:
            raise urllib.error.HTTPError(
                "https://example.invalid/objects.inv",
                503,
                "Service Unavailable",
                {},
                None,
            )
        return b"inventory"

    calls = downloader(behaviour)
    assert _download_with_retry("https://example.invalid/objects.inv") == b"inventory"
    assert len(calls) == 2


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


def test_config_walk_finds_the_urls_mkdocs_actually_resolves() -> None:
    """The walk must fail loudly, not return ``[]``.

    ``_inventory_urls`` reads ``plugins.mkdocstrings.handlers.python
    .inventories``. mkdocstrings has renamed that key once already — ``import``
    became ``inventories`` — and if it moves again the walk returns an empty
    list, the hook warms nothing, every other test still passes and the build
    silently loses this protection. Assert against the real resolved config,
    the way ``test_cache_duration_matches_what_mkdocstrings_asks_for`` asserts
    against mkdocstrings' own source.
    """
    from mkdocs.config import load_config
    from scripts.mkdocs_hooks.warm_inventory_cache import _inventory_urls

    declared = _declared_inventories()
    found = _inventory_urls(load_config(str(MKDOCS_YML)))

    assert found, (
        "_inventory_urls found no inventories in the resolved config — the "
        "handler config key has moved and the hook is warming nothing."
    )
    assert set(found) == set(declared)


def test_config_walk_is_not_silently_empty_on_a_renamed_key() -> None:
    """Renaming the key must change the result, so the pin above has teeth."""
    from mkdocs.config import load_config
    from scripts.mkdocs_hooks.warm_inventory_cache import _inventory_urls

    config = load_config(str(MKDOCS_YML))
    handler = config.plugins["mkdocstrings"].config["handlers"]["python"]
    handler["import"] = handler.pop("inventories")

    assert _inventory_urls(config) == [], (
        "the walk no longer keys off 'inventories'; update it and the pin "
        "above together."
    )


def test_urls_come_from_mkdocs_yml_not_a_second_list() -> None:
    """A new inventory in mkdocs.yml is warmed without editing the hook."""
    declared = _declared_inventories()
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


def test_ci_cache_path_is_the_one_mkdocs_writes_to() -> None:
    """#1042: the workflows hard-code a path; pin it to the library's own.

    The chain is ``mkdocs.utils.cache.download_and_cache_url`` ->
    ``mkdocs_get_deps.cache.download_and_cache_url`` ->
    ``platformdirs.user_cache_dir(<appname>)``. Both halves have to come from
    the library: writing the appname here would pin ``platformdirs``' layout
    for a literal string, and a rename to ``"mkdocs-get-deps"`` upstream would
    leave this passing while the CI cache covered nothing — the exact failure
    this exists to prevent.
    """
    import inspect

    pytest.importorskip("platformdirs")
    import mkdocs.utils.cache
    import mkdocs_get_deps.cache
    from platformdirs.unix import Unix

    assert "mkdocs_get_deps" in inspect.getsource(mkdocs.utils.cache), (
        "mkdocs no longer delegates its URL cache to mkdocs_get_deps; "
        "re-derive where the inventories are written."
    )
    source = inspect.getsource(mkdocs_get_deps.cache)
    appnames = re.findall(r"user_cache_dir\(\s*[\"']([^\"']+)[\"']", source)
    assert len(appnames) == 1, (
        f"expected exactly one user_cache_dir appname in mkdocs_get_deps."
        f"cache; found {appnames}. The cache location has been restructured."
    )

    # The runners are ubuntu-latest, so the Unix implementation is the one
    # that decides. Asserted rather than assumed: this host is Windows and
    # resolves the same appname somewhere else entirely.
    linux_path = Unix(appname=appnames[0]).user_cache_dir
    assert linux_path.replace("\\", "/").endswith(_CI_CACHE_PATH.lstrip("~")), (
        f"mkdocs_get_deps caches under {linux_path!r} on Linux; the workflows "
        f"cache {_CI_CACHE_PATH} and would cover nothing."
    )


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_every_mkdocs_build_job_caches_the_inventories(workflow: pathlib.Path) -> None:
    """A job that builds without the cache fetches on every run.

    Checks position too: restoring after the build would cache nothing the
    build could have used.
    """
    jobs = (yaml.safe_load(workflow.read_text(encoding="utf-8")) or {})["jobs"]
    building = {
        name: spec
        for name, spec in jobs.items()
        if any("mkdocs build" in str(step.get("run", "")) for step in spec["steps"])
    }
    assert building, f"{workflow} no longer runs mkdocs build; drop this check"

    for name, spec in building.items():
        steps = spec["steps"]

        # ``~/.cache/mkdocs`` is the Linux layout. On macOS the same appname
        # resolves to ``~/Library/Caches/mkdocs``, so a job moved to a macOS
        # runner would cache a directory nothing writes to — the step would
        # still be there, still green, and cover nothing.
        assert "ubuntu" in str(spec.get("runs-on", "")), (
            f"{workflow}::{name} runs on {spec.get('runs-on')!r}; "
            f"{_CI_CACHE_PATH} is the Linux cache layout and would not be the "
            "directory mkdocs writes to there."
        )

        cache = [
            i
            for i, step in enumerate(steps)
            if step.get("uses", "").startswith("actions/cache")
            and str(step.get("with", {}).get("path", "")) == _CI_CACHE_PATH
        ]
        assert cache, f"{workflow}::{name} builds docs without the inventory cache"

        build = next(
            i
            for i, step in enumerate(steps)
            if "mkdocs build" in str(step.get("run", ""))
        )
        assert cache[0] < build, (
            f"{workflow}::{name} restores the cache after the build that needs it"
        )


def test_hook_is_registered_before_the_generators() -> None:
    """It must run in the build, and first.

    The position is the assertion, not decoration: warming after a generator
    has already failed is warming after the build is lost. An earlier version
    checked membership only while the name promised order — the shape closed
    twice in #1040.
    """
    hooks = _mkdocs_yml().get("hooks") or []
    warm_up = "scripts/mkdocs_hooks/warm_inventory_cache.py"

    assert warm_up in hooks
    assert hooks.index(warm_up) == 0, (
        f"the warm-up must be the first hook; it is at index "
        f"{hooks.index(warm_up)} of {hooks}"
    )
