"""Multiple-testing adjustments.

Benjamini-Yekutieli (2001) step-up procedure for FDR control under
arbitrary dependence among the tests. The BHY correction factor
``c(m) = sum_{i=1..m} 1/i`` makes the procedure conservative enough
to handle dependent tests (as opposed to BH 1995 which requires
independence or PRDS).

Rejection rule: order p-values ``p_(1) <= p_(2) <= ... <= p_(m)``.
Reject ``H_(k)`` for all ``k <= k_max`` where
``k_max = max {k : p_(k) <= k / (m * c(m)) * alpha}``.

Adjusted-p mapping: ``p_adj_(k) = min_{j >= k} (m * c(m) / j) * p_(j)``,
clipped at 1. Guarantees monotonicity in ranked order.

``n_tests`` kwarg: when the caller pre-filtered the candidate family
(e.g. "1000 candidates → only the top 50 p-values reach BHY"), pass
``n_tests=1000`` so ``m`` reflects the true family size. ``k`` still
ranges over the submitted p's; the unsubmitted ``m - len(p)`` candidates
are implicitly not rejected. Default (``None``) reproduces single-stage
BHY where ``m = len(p_values)``.

References:
    Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False
    Discovery Rate in Multiple Testing under Dependency."
    Annals of Statistics 29(4), 1165-1188.

    Benjamini, Y. & Heller, R. (2008). "Screening for partial conjunction
    hypotheses." Biometrics 64(4), 1215-1222. — partial_conjunction_p

    Simes, R. J. (1986). "An improved Bonferroni procedure for multiple
    tests of significance." Biometrika 73(3), 751-754. — simes_p

    Yekutieli, D. (2008). "Hierarchical false discovery rate-controlling
    methodology." JASA 103(481), 309-316. — Simes as group representative
    in hierarchical FDR procedures.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt


# WHY: lru_cache — multiple_testing_correct() calls bhy_adjust() and
# bhy_adjusted_p() back-to-back with the same n; without memoization the
# harmonic sum is recomputed. Cache is keyed by int m (small integer set),
# so memory is bounded.
@functools.lru_cache(maxsize=128)
def _bhy_correction_factor(m: int) -> float:
    """c(m) = sum_{i=1..m} 1/i — conservative dependence adjustment."""
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def _resolve_m(n_submitted: int, n_tests: int | None) -> int:
    """Validate n_tests and return the BHY denominator m.

    ``n_tests < n_submitted`` would mean the caller is claiming the
    candidate family is smaller than the submitted set — incoherent.
    Callers of this helper are already past the ``n_submitted == 0``
    early-return, so ``n_tests < 1`` is caught by the same check.
    """
    if n_tests is None:
        return n_submitted
    if n_tests < n_submitted:
        raise ValueError(
            f"n_tests ({n_tests}) must be >= len(p_values) ({n_submitted}). "
            f"BHY assumes submitted p-values are a subset of the full "
            f"candidate family; a smaller n_tests is incoherent."
        )
    return int(n_tests)


def bhy_adjust(
    p_values: npt.ArrayLike,
    fdr: float = 0.05,
    *,
    n_tests: int | None = None,
) -> np.ndarray:
    """BHY step-up rejection mask.

    Args:
        p_values: 1-D array of p-values in [0, 1]. Each must come from
            the same test family (e.g. all IC p-values or all CAAR
            p-values); the ProfileSet wrapper enforces this via the
            P_VALUE_FIELDS whitelist.
        fdr: Target false discovery rate (default 0.05).
        n_tests: Full candidate family size for two-stage screening. If
            caller already pre-filtered from a larger pool (e.g. 1000
            candidates → 50 submitted), pass the pre-filter size here.
            Must be ``>= len(p_values)``. ``None`` (default) uses
            ``len(p_values)``, i.e. single-stage BHY.

    Returns:
        Boolean mask of length ``len(p_values)`` — True where the null
        is rejected. Order matches the input.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("bhy_adjust: p_values must all lie in [0, 1].")
    if not (0 < fdr < 1):
        raise ValueError(f"bhy_adjust: fdr must be in (0, 1); got {fdr}.")

    m = _resolve_m(n, n_tests)
    c_m = _bhy_correction_factor(m)
    order = np.argsort(p)
    sorted_p = p[order]
    k_vec = np.arange(1, n + 1)
    # k ranges over submitted p's; denominator m reflects the full
    # candidate family (may exceed n when caller pre-filtered).
    crits = k_vec / (m * c_m) * fdr
    passing_sorted = sorted_p <= crits

    out = np.zeros(n, dtype=bool)
    if not np.any(passing_sorted):
        return out

    # Step-up: reject everything up to the largest k that passes.
    k_max = int(np.max(np.where(passing_sorted)[0]))
    mask_sorted = np.zeros(n, dtype=bool)
    mask_sorted[: k_max + 1] = True
    out[order] = mask_sorted
    return out


def bhy_adjusted_p(
    p_values: npt.ArrayLike,
    *,
    n_tests: int | None = None,
) -> np.ndarray:
    """Per-hypothesis BHY-adjusted p-values (clipped at 1).

    Formula: scale p_(k) by ``(m * c(m)) / k`` then cummin from the
    right to enforce monotonicity in ranked order. Gives a stable
    per-factor "how significant under FDR control" number.

    ``n_tests`` follows the same contract as ``bhy_adjust`` — pass the
    pre-filter size when the submitted p's are survivors of a larger
    candidate family.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=float)

    m = _resolve_m(n, n_tests)
    c_m = _bhy_correction_factor(m)
    order = np.argsort(p)
    sorted_p = p[order]
    k_vec = np.arange(1, n + 1)
    scaled = (m * c_m / k_vec) * sorted_p
    # Cummin from the right → monotone non-decreasing in rank order.
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    np.minimum(adj_sorted, 1.0, out=adj_sorted)

    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def simes_p(p_values: npt.ArrayLike) -> float:
    """Simes (1986) global-null p-value for a group of tests.

    Formula: ``p_Simes = min_{k=1..m} (m / k) * p_((k))`` where
    ``p_((k))`` is the ``k``-th smallest of the ``m`` p-values
    (1-indexed). Tests the global null "all ``m`` nulls hold" against
    "at least one alternative is true"; valid under independence and
    PRDS.

    Yekutieli (2008) uses Simes as the default group representative
    in hierarchical FDR procedures — it dominates Bonferroni
    (``m * min(p)``) and preserves group-level FDR control when fed
    to an outer BHY step-up.

    Args:
        p_values: 1-D array of ``m`` p-values for one group. ``m >= 1``.

    Returns:
        The Simes combined p-value, clipped to ``[0, 1]``.

    Raises:
        ValueError: ``len(p_values) == 0`` (Simes is undefined on an
            empty group).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        raise ValueError("simes_p: p_values must be non-empty.")
    sorted_p = np.sort(p)
    k_vec = np.arange(1, m + 1)
    return float(min(np.min((m / k_vec) * sorted_p), 1.0))


def partial_conjunction_p(
    p_values: npt.ArrayLike,
    *,
    min_pass: int,
) -> float:
    """Bonferroni-style partial conjunction p-value (Benjamini-Heller 2008).

    Tests ``H_0^{k/m}``: at most ``k - 1`` of the ``m`` alternatives are
    true, against ``H_1^{k/m}``: at least ``k`` are true. The combined
    p-value is

        ``p_PC = min(1, (m - k + 1) * p_((k)))``

    where ``p_((k))`` is the ``k``-th smallest of the ``m`` p-values
    (1-indexed). ``k = m`` reduces to ``max(p)`` (full conjunction);
    ``k = 1`` reduces to Bonferroni-corrected ``min(p)``.

    Args:
        p_values: 1-D array of m per-condition p-values for a single
            hypothesis (e.g. one factor across m universes).
        min_pass: ``k`` — the minimum number of conditions required to
            be significant. Must satisfy ``1 <= min_pass <= m``.

    Returns:
        The PC p-value, clipped to ``[0, 1]``.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        raise ValueError("partial_conjunction_p: p_values must be non-empty.")
    if not 1 <= min_pass <= m:
        raise ValueError(
            f"partial_conjunction_p: min_pass ({min_pass}) must satisfy "
            f"1 <= min_pass <= m ({m})."
        )
    sorted_p = np.sort(p)
    pc = (m - min_pass + 1) * float(sorted_p[min_pass - 1])
    return min(pc, 1.0)
