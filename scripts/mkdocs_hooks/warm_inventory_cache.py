"""Pre-warm the intersphinx inventory cache so one blip cannot fail the build.

``mkdocstrings`` downloads each configured inventory through
``mkdocs.utils.cache.download_and_cache_url`` with a one-day cache, and a
failed fetch is fatal under ``--strict``. It is a single attempt: on
``docs-deploy-dev`` run 33599419309 a ``Connection reset by peer`` from
``docs.pola.rs`` failed a pull request whose content was fine, and a rerun went
green (#1037).

This hook fetches the same URLs into the same cache first, retrying a
connection-level failure. Because it calls the same cache function, a warmed
entry is what ``mkdocstrings`` finds a moment later — it never reaches the
network at all. When every attempt fails the hook stays quiet about it beyond a
warning and lets ``mkdocstrings`` proceed, so a genuinely wrong or permanently
unreachable URL still fails the build with its own error. ``--strict`` keeps
its meaning; only the number of chances a transient blip gets changes.

The URLs are read from ``mkdocs.yml`` rather than repeated here, so adding an
inventory there warms it automatically.

Usage (manual)::

    python scripts/mkdocs_hooks/warm_inventory_cache.py

MkDocs hook usage (automatic, via ``hooks:`` in mkdocs.yml)::

    ``on_config(config)`` runs before mkdocstrings resolves its handlers.
"""

from __future__ import annotations

import datetime
import time
import urllib.error
from typing import Any

#: Matches the ``cache_duration`` mkdocstrings passes, so a warmed entry is
#: still fresh when it looks. A shorter value here would warm the cache and
#: then let mkdocstrings re-download anyway.
_CACHE_DURATION = datetime.timedelta(days=1)

#: Attempts per URL, and the pause before each retry. Small on purpose: this
#: runs on every local build too, and an offline developer should wait a
#: couple of seconds, not a minute, before the build continues without it.
_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 3.0)


def _inventory_urls(config: Any) -> list[str]:
    """Inventory URLs configured for the mkdocstrings python handler."""
    plugin = config.plugins.get("mkdocstrings")
    if plugin is None:
        return []
    handlers = plugin.config.get("handlers") or {}
    python = handlers.get("python") or {}
    return [
        entry if isinstance(entry, str) else entry["url"]
        for entry in python.get("inventories") or []
    ]


def _download_with_retry(url: str) -> bytes:
    """``mkdocs``' downloader, retried on a connection-level failure.

    Only ``URLError`` is retried. An ``HTTPError`` — 404 on a mistyped URL,
    403, 500 — is a server verdict rather than a blip, and is raised on the
    first attempt so a wrong URL is not hidden behind three slow tries.
    """
    from mkdocstrings._internal.handlers.base import _download_url_with_gz

    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            return _download_url_with_gz(url)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as error:
            last = error
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    assert last is not None
    raise last


def warm(urls: list[str]) -> None:
    """Fetch each URL into the cache mkdocstrings reads, retrying blips."""
    from mkdocs.utils.cache import download_and_cache_url

    for url in urls:
        try:
            download_and_cache_url(url, _CACHE_DURATION, download=_download_with_retry)
        except Exception as error:
            # Deliberately not fatal: mkdocstrings is about to try the same
            # URL and will report the real failure in its own terms. Failing
            # here would only move the error and lose that context.
            print(f"warm_inventory_cache: {url} unavailable ({error})")
        else:
            print(f"warm_inventory_cache: cached {url}")


def on_config(config):
    """MkDocs ``on_config`` hook — runs before mkdocstrings downloads."""
    warm(_inventory_urls(config))
    return config


if __name__ == "__main__":
    from mkdocs.config import load_config

    warm(_inventory_urls(load_config("mkdocs.yml")))
