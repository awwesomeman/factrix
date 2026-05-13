"""Family-wise error rate (FWER) adjustments — Holm / Bonferroni / Romano-Wolf.

Sister module to public ``factrix.stats.multiple_testing`` (Benjamini-
Yekutieli FDR control). The procedures here control the *family-wise*
error rate — probability of at least one false rejection — and target
the slice-test setting where a small number of hypotheses
(per-slice contrasts vs a baseline) are tested simultaneously.

- **Bonferroni** — single-step ``p_adj_k = min(m * p_k, 1)``. Controls
  FWER under any dependence; uniformly the most conservative.
- **Holm step-down** (Holm 1979) — uniformly dominates Bonferroni under
  the same dependence assumptions; gains power by sequentially
  rejecting ordered p-values.
- **Romano-Wolf step-down** (Romano & Wolf 2005) — bootstrap-based
  step-down that exploits the *joint* dependence structure of the
  test statistics. Needs the user to supply a bootstrap distribution
  (under H0); in return delivers tight FWER control even when tests
  share a common shock (universe pairwise IC, factor-portfolio
  contrasts, etc.).

The choice between Holm and Romano-Wolf is the caller's: time-disjoint
slices (e.g. regimes) work fine under Holm; date-shared slices
(universe pairwise) leave significant power on the table without
Romano-Wolf. The function-side fallback that picks between the two
lives in #176 — this module does not encode a default.

References:
    - Holm, S. (1979). "A simple sequentially rejective multiple test
      procedure." Scandinavian Journal of Statistics, 6(2), 65–70.
    - Romano, J. P. & Wolf, M. (2005). "Stepwise multiple testing as
      formalized data snooping." Econometrica, 73(4), 1237–1282.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def _validate_p(p_values: Sequence[float] | npt.ArrayLike) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D; got shape {p.shape}.")
    if p.size and not np.all((p >= 0) & (p <= 1)):
        raise ValueError("p_values must all lie in [0, 1].")
    return p


def bonferroni(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """Bonferroni-adjusted p-values: ``p_adj_k = min(m * p_k, 1)``.

    Single-step, strong FWER control under arbitrary dependence.
    Uniformly the most conservative of the three procedures here;
    used as a baseline / sanity-check sibling to Holm.
    """
    p = _validate_p(p_values)
    m = len(p)
    if m == 0:
        return []
    return list(np.minimum(m * p, 1.0))


def holm_step_down(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """Holm (1979) step-down adjusted p-values.

    Order ``p_(1) ≤ p_(2) ≤ … ≤ p_(m)`` and set
    ``p_adj_(k) = max_{j ≤ k} (m - j + 1) * p_(j)``, clipped at 1.
    The cummax enforces monotonicity: a smaller raw p never gets a
    larger adjusted p in rank order — required so the rejection set
    ``{k : p_adj_k ≤ α}`` is closed downward in significance.

    Strong FWER control under arbitrary dependence; uniformly
    dominates Bonferroni (each adjusted p is ≤ Bonferroni's).
    """
    p = _validate_p(p_values)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    factors = np.arange(m, 0, -1, dtype=float)
    scaled = factors * sorted_p
    adj_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return list(out)


def romano_wolf(
    statistics: Sequence[float] | npt.ArrayLike,
    bootstrap_distribution: npt.ArrayLike,
    *,
    one_sided: bool = False,
) -> list[float]:
    """Romano-Wolf (2005) step-down adjusted p-values.

    Args:
        statistics: Observed test statistics ``t_1, …, t_m`` (e.g.
            studentized contrasts). Sign convention: large positive
            ``t_k`` favours rejection. For two-sided tests
            ``one_sided=False`` operates on ``|t_k|`` internally.
        bootstrap_distribution: ``(B, m)`` array of bootstrap test
            statistics drawn under H0 (caller is responsible for
            null centring — typically subtract the sample statistic
            from each bootstrap replicate before passing in).
        one_sided: If ``True``, reject only on the positive tail
            (suitable when the alternative is signed, e.g. "long
            portfolio outperforms short"). If ``False`` (default),
            two-sided via ``|t_k|``.

    Returns:
        Adjusted p-values in input order, each in ``[0, 1]``.

    The step-down critical sequence is built from the *max* of the
    bootstrap distribution restricted to the not-yet-rejected
    hypotheses, so the dependence structure (universe co-movement,
    common-factor exposure) shrinks the multiplicity penalty
    automatically — Bonferroni / Holm assume worst-case dependence
    and over-correct in this regime.
    """
    t = np.asarray(statistics, dtype=float)
    boot = np.asarray(bootstrap_distribution, dtype=float)
    m = len(t)
    if m == 0:
        return []
    if boot.ndim != 2 or boot.shape[1] != m:
        raise ValueError(
            f"bootstrap_distribution must have shape (B, {m}); got {boot.shape}."
        )
    if boot.shape[0] < 1:
        raise ValueError("bootstrap_distribution must have at least 1 resample.")

    # Two-sided default: collapse to absolute values up front so the
    # max-over-remaining is computed on the right scale.
    if one_sided:
        t_use = t
        boot_use = boot
    else:
        t_use = np.abs(t)
        boot_use = np.abs(boot)

    # Most-significant first: order by descending observed statistic.
    desc_order = np.argsort(-t_use)
    p_adj_desc = np.empty(m, dtype=float)
    remaining = list(desc_order)
    for j, k in enumerate(desc_order):
        # max over the not-yet-rejected hypotheses, per bootstrap row.
        max_remaining = boot_use[:, remaining].max(axis=1)
        # Empirical p with +1 / (B+1) smoothing — keeps p strictly > 0
        # so log-scale plotting / multi-stage stacks don't crash on a
        # zero-count bootstrap tail. Standard Davison-Hinkley convention.
        b = boot_use.shape[0]
        p_adj_desc[j] = (np.sum(max_remaining >= t_use[k]) + 1.0) / (b + 1.0)
        remaining = remaining[1:]

    # Enforce monotonicity in descending-significance order: an earlier
    # (more significant) rejection cannot have a larger adjusted p than
    # a later one. Cummax over the sequence.
    p_adj_desc = np.maximum.accumulate(p_adj_desc)
    p_adj_desc = np.minimum(p_adj_desc, 1.0)

    out = np.empty(m, dtype=float)
    out[desc_order] = p_adj_desc
    return list(out)
