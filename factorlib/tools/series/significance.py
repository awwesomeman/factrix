"""Statistical significance tools for any numeric series.

Provides t-statistic computation and significance markers (★●○).
Operates on raw numeric arrays — agnostic to what the series represents.
"""

from __future__ import annotations

import numpy as np

from factorlib.tools._typing import EPSILON, DDOF


def calc_t_stat(mean: float, std: float, n: int) -> float:
    """Compute t-statistic with EPSILON guard against near-zero std.

    Args:
        mean: Sample mean.
        std: Sample standard deviation (ddof=1).
        n: Sample size.

    Returns:
        t-statistic, or 0.0 if std is near-zero or n ≤ 0.
    """
    if std > EPSILON and n > 0:
        return float(mean / (std / np.sqrt(n)))
    return 0.0


def t_stat_from_array(values: np.ndarray) -> float:
    """Convenience: compute t-stat directly from a 1-D array.

    Args:
        values: 1-D numeric array with at least 2 elements.

    Returns:
        t-statistic of the mean, or 0.0 if insufficient data.
    """
    if len(values) < 2:
        return 0.0
    return calc_t_stat(
        float(np.mean(values)),
        float(np.std(values, ddof=DDOF)),
        len(values),
    )


def significance_marker(t_stat: float | None) -> str:
    """Map t-stat to visual significance marker.

    | Marker | Condition     | Meaning                         |
    |:------:|---------------|---------------------------------|
    |   ★    | t ≥ 3.0       | Highly significant (Harvey-strict) |
    |   ●    | 2.0 ≤ t < 3.0 | Significant (95% CI)            |
    |   ○    | t < 2.0       | Not significant                 |

    Returns:
        One of "★", "●", "○".
    """
    if t_stat is None:
        return "○"
    abs_t = abs(t_stat)
    if abs_t >= 3.0:
        return "★"
    if abs_t >= 2.0:
        return "●"
    return "○"
