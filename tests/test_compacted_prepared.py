"""RuntimeError guarantees for _CompactedPrepared dunders.

In compact=True mode, Artifacts.prepared is replaced by a sentinel that
must refuse every access path. Slot-based dunders bypass ``__getattr__``
— each one has to be overridden explicitly, and it's easy for a future
edit to drop one silently. These tests lock every container / truth-
testing / iteration dunder raises RuntimeError with the prepared-dropped
diagnostic.
"""

from __future__ import annotations

import pytest

from factorlib.evaluation._protocol import _COMPACTED_PREPARED


class TestCompactedPreparedDunders:
    def test_bool_raises(self):
        with pytest.raises(RuntimeError, match="compact mode"):
            bool(_COMPACTED_PREPARED)

    def test_len_raises(self):
        with pytest.raises(RuntimeError, match="compact mode"):
            len(_COMPACTED_PREPARED)

    def test_iter_raises(self):
        with pytest.raises(RuntimeError, match="compact mode"):
            iter(_COMPACTED_PREPARED)

    def test_getitem_raises(self):
        with pytest.raises(RuntimeError, match="compact mode"):
            _COMPACTED_PREPARED["asset_id"]

    def test_contains_raises(self):
        with pytest.raises(RuntimeError, match="compact mode"):
            "asset_id" in _COMPACTED_PREPARED

    def test_attr_access_raises_with_name(self):
        # __getattr__ should name the missed attribute in the message
        # so callers see "Cannot access 'prepared.columns'" rather than
        # an opaque sentinel message.
        with pytest.raises(RuntimeError, match="prepared.columns"):
            _COMPACTED_PREPARED.columns

    def test_repr_does_not_raise(self):
        # repr is called by many debugging paths (incl. pytest assertion
        # rewriting); it must stay safe so a test failure report doesn't
        # turn into a second exception.
        assert "CompactedPrepared" in repr(_COMPACTED_PREPARED)
