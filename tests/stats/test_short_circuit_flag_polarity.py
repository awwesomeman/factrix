"""Regression lint for did-this-stage-run flags on short-circuit payloads.

A short circuit runs no kernel, so a flag reporting that a stage of the
inference *was applied* cannot be affirmative there. ``predictive_beta`` is
what prompted this: its short circuit neutralised ``stambaugh_adjusted`` to
``False`` and, one keyword later, reported ``hac_applied`` as ``True`` — the
same claim about a different stage, decided two different ways inside one
call.

FX008 carries a baseline of documented exceptions in the migration-ledger
shape of FX007 in ``test_degenerate_guard_polarity``. The baseline is empty,
so the rule currently holds without carve-outs; the mechanism stays for a
future case that clears the envelope's bar.

It deliberately checks only the mechanically decidable half of the
short-circuit envelope: a key whose *name* says a stage ran. Whether a
value-bearing key such as ``mfe_mae_ratio`` was neutralised is a semantic
question the envelope documents and no lint can settle.
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
#: the case this lint exists to catch. It matches eight names across the
#: package today — ``hac_applied``, ``stambaugh_adjusted``,
#: ``kolari_pynnonen_applied``, ``calendar_time_se_applied``,
#: ``overlap_adjustment_applied``, ``event_clustering_adjusted``,
#: ``mean_adjusted``, ``sign_flipped`` — so the grammar has real reach beyond
#: the site that prompted it.
#:
#: Two verb families are excluded on purpose:
#:
#: - ``*_lagged`` is data preparation, not inference. ``quantile_spread_vw``'s
#:   ``weights_lagged`` echoes the ``lag_weights`` parameter, so it describes
#:   the caller's configuration and is affirmative on every branch by design.
#: - ``*_corrected`` matches ``ar1_phi_corrected``, which is a float — the
#:   bias-corrected AR(1) coefficient — not a did-this-run flag, and is
#:   legitimately attempt-side because the AR(1) fit precedes the bail.
#:   Including it would force a baseline entry for something that is not a
#:   flag the first time a short circuit reports it.
_STAGE_FLAG = re.compile(r"^[a-z0-9_]*_(applied|adjusted|flipped)$")

#: Sites that stay affirmative on purpose, with the reason each is sound.
#: A migration ledger, not a bug list — an entry leaves when the branch stops
#: needing it, and ``test_fx008_baseline_is_current`` fails if one goes stale.
#:
#: **Currently empty**, and that is the point: the rule holds everywhere
#: without a carve-out. Adding an entry needs the envelope's bar — that
#: neutralising the key is impossible or destroys information, not merely
#: less convenient — argued at the branch that adds it.
_FX008_BASELINE: dict[tuple[str, str], str] = {}


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


def test_fx008_docstring_count_matches_the_grammar() -> None:
    """The stated reach must equal the measured one.

    This shipped wrong once: the count stayed at ``nine`` after ``*_corrected``
    left the grammar, contradicting the list beside it. Measuring it here is
    the same rule this lint exists to enforce, one layer up — a stated number
    has to be the number the code produces.
    """
    matched = {
        node.value if isinstance(node, ast.Constant) else node.arg
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if not path.stem.startswith("llms")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _STAGE_FLAG.match(node.value)
        )
        or (
            isinstance(node, ast.keyword)
            and node.arg is not None
            and _STAGE_FLAG.match(node.arg)
        )
    }
    words = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    stated = words.get(len(matched))
    assert stated is not None, (
        f"grammar now matches {len(matched)} names; extend the word map."
    )
    own_source = pathlib.Path(__file__).read_text(encoding="utf-8")
    assert f"matches {stated} names" in own_source, (
        f"the grammar matches {len(matched)} names ({stated}); the comment on "
        f"_STAGE_FLAG states a different count."
    )
    # ...and every name the comment lists is one the grammar really matches.
    for name in matched:
        assert name in own_source, f"{name} matches but is not listed"


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
