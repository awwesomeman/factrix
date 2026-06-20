"""Flat ``Warning`` dataclass + source filter pattern."""

from __future__ import annotations

from factrix import Warning, WarningCode


def _sample_warnings() -> list[Warning]:
    return [
        Warning(code=WarningCode.FEW_ASSETS, source="ic", message="n=8"),
        Warning(code=WarningCode.FEW_ASSETS, source="fm_beta", message=""),
        Warning(
            code=WarningCode.SERIAL_CORRELATION_DETECTED, source=None, message="bundle"
        ),
    ]


def test_default_source_and_message():
    w = Warning(code=WarningCode.FEW_ASSETS)
    assert w.source is None
    assert w.message == ""


def test_filter_by_source_metric_name():
    warnings = _sample_warnings()
    per_metric = [w for w in warnings if w.source == "ic"]
    assert len(per_metric) == 1
    assert per_metric[0].code is WarningCode.FEW_ASSETS


def test_filter_bundle_level_with_source_none():
    warnings = _sample_warnings()
    bundle_level = [w for w in warnings if w.source is None]
    assert len(bundle_level) == 1
    assert bundle_level[0].code is WarningCode.SERIAL_CORRELATION_DETECTED


def test_group_by_source():
    warnings = _sample_warnings()
    by_source: dict[str | None, list[Warning]] = {}
    for w in warnings:
        by_source.setdefault(w.source, []).append(w)
    assert set(by_source) == {"ic", "fm_beta", None}


def test_frozen_dataclass_equality_and_hashable():
    a = Warning(code=WarningCode.FEW_ASSETS, source="ic", message="m")
    b = Warning(code=WarningCode.FEW_ASSETS, source="ic", message="m")
    assert a == b
    assert hash(a) == hash(b)
