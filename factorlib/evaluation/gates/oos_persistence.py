"""Out-of-sample persistence gate (shared across types).

Passes if multi-split OOS decay ratio >= threshold and no sign flip.
Default threshold = 0.5 (McLean & Pontiff 2016 average ~32%, so 50% is strict).

Customize via ``functools.partial(oos_persistence_gate, decay_threshold=0.3)``.
"""

from __future__ import annotations

from factorlib.evaluation._protocol import Artifacts, GateResult
from factorlib.metrics.oos import multi_split_oos_decay


def oos_persistence_gate(
    artifacts: Artifacts,
    *,
    survival_threshold: float = 0.5,
    value_key: str = "ic_values",
) -> GateResult:
    """Gate 2: multi-split OOS survival >= threshold, no sign flip.

    Args:
        artifacts: Pre-computed pipeline artifacts.
        survival_threshold: Minimum median survival ratio for PASS
            (default 0.5). survival = |mean_OOS| / |mean_IS|.
        value_key: Artifacts key for the (date, value) series to test.
            Defaults to "ic_values" (cross-sectional).
            Use "beta_values" for macro_panel.

    Returns:
        GateResult with PASS or VETOED status.
    """
    oos_result = multi_split_oos_decay(
        artifacts.get(value_key), survival_threshold=survival_threshold,
    )

    return GateResult(
        name="oos_persistence",
        status=oos_result.status,
        detail={
            "survival_ratio": oos_result.survival_ratio,
            "sign_flipped": oos_result.sign_flipped,
            "survival_threshold": survival_threshold,
            "per_split": [
                {
                    "is_ratio": s.is_ratio,
                    "mean_is": s.mean_is,
                    "mean_oos": s.mean_oos,
                    "survival_ratio": s.survival_ratio,
                    "sign_flipped": s.sign_flipped,
                }
                for s in oos_result.per_split
            ],
        },
    )
