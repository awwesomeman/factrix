"""Statistical significance gate for event signals.

Passes if CAAR t-stat OR BMP SAR z-stat OR event hit_rate z-stat >= threshold.
Any single path passing is sufficient — parallels the CS gate (IC OR spread)
and macro_panel gate (FM β OR Pooled β).

Default threshold = 2.0 (95% confidence).

Customize via ``functools.partial(event_significance_gate, threshold=3.0)``.
"""

from __future__ import annotations

from factorlib.evaluation._protocol import Artifacts, GateResult
from factorlib.metrics.caar import caar as caar_metric, bmp_test
from factorlib.metrics.event_quality import event_hit_rate


def event_significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """Gate 1: CAAR t-stat OR BMP z-stat OR hit_rate z-stat >= threshold.

    Args:
        artifacts: Pre-computed pipeline artifacts. Must contain
            ``caar_series`` in intermediates.
        threshold: Minimum |stat| for significance (default 2.0).

    Returns:
        GateResult with detail containing all stats and which path(s) passed.
    """
    ret_col = (
        "abnormal_return" if "abnormal_return" in artifacts.prepared.columns
        else "forward_return"
    )

    caar_series = artifacts.get("caar_series")
    caar_result = caar_metric(caar_series, forward_periods=artifacts.config.forward_periods)
    caar_tstat = caar_result.stat or 0.0

    bmp_result = bmp_test(
        artifacts.prepared, return_col=ret_col,
        forward_periods=artifacts.config.forward_periods,
    )
    bmp_zstat = bmp_result.stat or 0.0

    hit_result = event_hit_rate(artifacts.prepared, return_col=ret_col)
    hit_zstat = hit_result.stat or 0.0

    via: list[str] = []
    if abs(caar_tstat) >= threshold:
        via.append("CAAR")
    if abs(bmp_zstat) >= threshold:
        via.append("BMP")
    if abs(hit_zstat) >= threshold:
        via.append("hit_rate")

    return GateResult(
        name="event_significance",
        status="PASS" if via else "FAILED",
        detail={
            "caar_tstat": caar_tstat,
            "bmp_zstat": bmp_zstat,
            "hit_rate_zstat": hit_zstat,
            "hit_rate": hit_result.value,
            "threshold": threshold,
            "via": via,
        },
    )
