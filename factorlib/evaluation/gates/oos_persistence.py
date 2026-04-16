"""Gate 2: Out-of-sample persistence.

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
    decay_threshold: float = 0.5,
    value_key: str = "ic_values",
) -> GateResult:
    """Gate 2: multi-split OOS decay >= threshold, no sign flip.

    Args:
        artifacts: Pre-computed pipeline artifacts.
        decay_threshold: Minimum median decay ratio for PASS (default 0.5).
        value_key: Artifacts key for the (date, value) series to test.
            Defaults to "ic_values" (cross-sectional).
            Use "beta_values" for macro_panel.

    Returns:
        GateResult with PASS or VETOED status.
    """
    oos_result = multi_split_oos_decay(
        artifacts.get(value_key), decay_threshold=decay_threshold,
    )

    return GateResult(
        name="oos_persistence",
        status=oos_result.status,
        detail={
            "decay_ratio": oos_result.decay_ratio,
            "sign_flipped": oos_result.sign_flipped,
            "decay_threshold": decay_threshold,
            "per_split": [
                {
                    "is_ratio": s.is_ratio,
                    "mean_is": s.mean_is,
                    "mean_oos": s.mean_oos,
                    "decay_ratio": s.decay_ratio,
                    "sign_flipped": s.sign_flipped,
                }
                for s in oos_result.per_split
            ],
        },
    )
