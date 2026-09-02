"""Regression lint for did-this-stage-run flags on short-circuit payloads.

A short circuit runs no kernel, so a flag reporting that a stage of the
inference *was applied* cannot be affirmative there without a documented
exception. ``predictive_beta``'s own short circuit is the illustration: it
neutralises ``stambaugh_adjusted`` to ``False`` and, one keyword later,
reports ``hac_applied`` as ``True``.

FX008 records the sites that legitimately stay affirmative — each with the
reason it does — and rejects any new one, the same migration-ledger shape as
FX007 in ``test_degenerate_guard_polarity``. It deliberately checks only the
mechanically decidable half of the short-circuit envelope: a key whose *name*
says a stage ran. Whether a value-bearing key such as ``mfe_mae_ratio`` was
neutralised is a semantic question the envelope documents but no lint can
settle.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections import Counter

PACKAGE_DIR = pathlib.Path("factrix")

#: Helpers that build a short-circuit ``MetricResult``. Keyword arguments to
#: these become ``metadata`` on a result that carries no estimate.
_SHORT_CIRCUIT_HELPERS = frozenset(
    {"_short_circuit_output", "_enforce_min_floor", "_enforce_scaled_floor"}
)

#: Keys whose name asserts that a stage of the *inference* ran. Deliberately a
#: suffix grammar rather than a hand-list: a new ``*_applied`` flag is exactly
#: the case this lint exists to catch. Data-preparation verbs (``*_lagged``)
#: stay out — ``quantile_spread_vw``'s ``weights_lagged`` echoes the
#: ``lag_weights`` parameter, so it describes the caller's configuration
#: rather than a stage of the withheld inference, and is affirmative on every
#: branch by design.
_STAGE_FLAG = re.compile(r"^[a-z0-9_]*_(applied|adjusted|corrected|flipped)$")

#: Sites that stay affirmative on purpose, with the reason each is sound.
#: A migration ledger, not a bug list — an entry leaves when the branch stops
#: needing it, and ``test_fx008_baseline_is_current`` fails if one goes stale.
_FX008_BASELINE: dict[tuple[str, str], str] = {
    (
        "factrix/metrics/predictive_beta.py",
        "hac_applied",
    ): (
        "no_amihud_hurvich_fit documents its inference keys as describing the "
        "attempt, not a result: har_lags is the bandwidth resolved for it and "
        "hac_applied the covariance branch it would have taken. Neutralising "
        "to False would make the pair identical to an h = 1 success, reusing "
        "a sentinel that already means something else."
    ),
}


def _flag_kwargs(path: pathlib.Path) -> list[tuple[str, str]]:
    """``(key, unparsed value)`` for every stage flag on a short-circuit call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in _SHORT_CIRCUIT_HELPERS:
            continue
        for keyword in node.keywords:
            if keyword.arg and _STAGE_FLAG.match(keyword.arg):
                found.append((keyword.arg, ast.unparse(keyword.value)))
    return found


def _affirmative_sites() -> Counter[tuple[str, str]]:
    """Stage flags passed a value that is not a literal ``False`` / ``None``."""
    sites: Counter[tuple[str, str]] = Counter()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.stem.startswith("llms"):
            continue
        for key, value in _flag_kwargs(path):
            if value in {"False", "None"}:
                continue
            sites[(path.as_posix(), key)] += 1
    return sites


def test_fx008_no_new_affirmative_stage_flag() -> None:
    """Lint FX008: a new stage flag must be neutral on a short circuit."""
    unexpected = {site for site in _affirmative_sites() if site not in _FX008_BASELINE}
    assert not unexpected, (
        "FX008: a short circuit runs no kernel, so a flag naming a stage of "
        "the inference cannot report that it was applied. Pass False (or "
        "None), or add the site to _FX008_BASELINE with the reason its "
        "branch documents the value as describing the attempt. New sites: "
        f"{sorted(unexpected)}"
    )


def test_fx008_baseline_is_current() -> None:
    """A baseline entry that no longer matches a site is stale."""
    stale = set(_FX008_BASELINE) - set(_affirmative_sites())
    assert not stale, (
        f"FX008: baseline entries no longer present in the source: {sorted(stale)}. "
        "Remove them so the ledger keeps describing the code."
    )


def test_fx008_detects_a_planted_violation(tmp_path: pathlib.Path) -> None:
    """The lint must fail on the shape it exists to reject."""
    module = tmp_path / "metric.py"
    module.write_text(
        """
def metric():
    return _short_circuit_output(
        "m", "no_fit", overlap_adjustment_applied=True, sign_flipped=False
    )
""",
        encoding="utf-8",
    )
    flags = dict(_flag_kwargs(module))

    assert flags["overlap_adjustment_applied"] == "True"
    assert flags["sign_flipped"] == "False"


def test_fx008_ignores_flags_on_normal_results() -> None:
    """Only short-circuit payloads are in scope; a real fit may report True."""
    module = pathlib.Path("factrix/metrics/predictive_beta.py")

    # The success path reports the correction it applied, affirmatively.
    assert '"stambaugh_adjusted": True' in module.read_text(encoding="utf-8")
    # ...and that emission is not a short-circuit call, so FX008 never sees it.
    assert ("stambaugh_adjusted", "True") not in _flag_kwargs(module)
