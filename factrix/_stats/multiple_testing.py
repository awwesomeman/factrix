"""Family-wise error rate (FWER) adjustments — Holm / Bonferroni / Romano-Wolf.

The public Holm and Romano-Wolf adjustments live in
``factrix.stats.multiple_testing``; this module retains the Bonferroni baseline
and compatibility wrappers for the former private call sites. These procedures
control the *family-wise* error rate — probability of at least one false
rejection — and target the slice-test setting where a small number of
hypotheses are tested simultaneously.

- **Bonferroni** — single-step ``p_adj_k = min(m * p_k, 1)``. Controls
  FWER under any dependence; uniformly the most conservative.
- **Holm step-down** ([Holm (1979)][holm-1979]) — uniformly dominates Bonferroni under
  the same dependence assumptions; gains power by sequentially
  rejecting ordered p-values.
- **Romano-Wolf step-down** ([Romano-Wolf (2005)][romano-wolf-2005]) — bootstrap-based
  step-down that exploits the *joint* dependence structure of the
  test statistics. Needs the user to supply a bootstrap distribution
  (under H0); in return delivers tight FWER control even when tests
  share a common shock (universe pairwise information coefficient (IC), factor-portfolio
  contrasts, etc.).

The choice between Holm and Romano-Wolf is the caller's: time-disjoint
slices (e.g. regimes) work fine under Holm; date-shared slices
(universe pairwise) leave significant power on the table without
Romano-Wolf. The function-side fallback that picks between the two
lives in the slice-test functions — this module does not encode a default.

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

from factrix.stats.multiple_testing import holm_adjusted_p, romano_wolf_adjusted_p


def _validate_p(p_values: Sequence[float] | npt.ArrayLike) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D; got shape {p.shape}.")
    if p.size and not np.all((p >= 0) & (p <= 1)):
        raise ValueError("p_values must all lie in [0, 1].")
    return p


def bonferroni(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """Bonferroni-adjusted p-values: ``p_adj_k = min(m * p_k, 1)``.

    Single-step, strong family-wise error rate (FWER) control under arbitrary dependence.
    Uniformly the most conservative of the three procedures here;
    used as a baseline / sanity-check sibling to Holm.
    """
    p = _validate_p(p_values)
    m = len(p)
    if m == 0:
        return []
    return list(np.minimum(m * p, 1.0))


def holm_step_down(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """[Holm (1979)][holm-1979] step-down adjusted p-values.

    Order ``p_(1) ≤ p_(2) ≤ … ≤ p_(m)`` and set
    ``p_adj_(k) = max_{j ≤ k} (m - j + 1) * p_(j)``, clipped at 1.
    The cummax enforces monotonicity: a smaller raw p never gets a
    larger adjusted p in rank order — required so the rejection set
    ``{k : p_adj_k ≤ α}`` is closed downward in significance.

    Strong family-wise error rate (FWER) control under arbitrary dependence; uniformly
    dominates Bonferroni (each adjusted p is ≤ Bonferroni's).
    """
    return list(holm_adjusted_p(p_values))


def romano_wolf(
    statistics: Sequence[float] | npt.ArrayLike,
    bootstrap_distribution: npt.ArrayLike,
    *,
    one_sided: bool = False,
) -> list[float]:
    """[Romano-Wolf (2005)][romano-wolf-2005] step-down adjusted p-values.

    The step-down critical sequence is built from the *max* of the
    bootstrap distribution restricted to the not-yet-rejected
    hypotheses, so the dependence structure (universe co-movement,
    common-factor exposure) shrinks the multiplicity penalty
    automatically — Bonferroni / Holm assume worst-case dependence
    and over-correct in this regime.

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
    """
    return list(
        romano_wolf_adjusted_p(
            statistics,
            bootstrap_distribution,
            one_sided=one_sided,
        )
    )
