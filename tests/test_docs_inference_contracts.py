"""Keep public regression-inference descriptions aligned with their consumers."""

import inspect
import re
from pathlib import Path

from factrix._codes import WarningCode
from factrix.metrics._registry import REGISTRY

ARCHITECTURE_DOCS = Path("docs/development/architecture.md")
SPANNING_SOURCE = Path("factrix/metrics/spanning.py")
SPANNING_DOCS = Path("docs/api/metrics/spanning.md")
PREDICTIVE_DOCS = Path("docs/api/metrics/predictive_beta.md")
CALIBRATION_DOCS = Path("docs/reference/inference-calibration.md")
RESULT_DOCS = Path("docs/api/evaluation-results.md")
STATS_DOCS = Path("docs/api/stats.md")
LLMS_FULL = Path("factrix/llms-full.txt")


def test_spanning_docs_delegate_to_the_scalar_har_contract() -> None:
    source = SPANNING_SOURCE.read_text(encoding="utf-8")
    docs = SPANNING_DOCS.read_text(encoding="utf-8")

    assert "_resolve_scalar_wald_hac" in source
    assert "statistical-methods.md#hac-families" in docs
    for stale_claim in ("auto_bartlett", "T - 1 - K", "floored at $h-1$"):
        assert stale_claim not in source
        assert stale_claim not in docs


def test_predictive_docs_do_not_claim_the_scalar_wald_variance_scale() -> None:
    docs = PREDICTIVE_DOCS.read_text(encoding="utf-8")

    assert "scalar HAR bandwidth and effective degrees of freedom" in docs
    assert "does not apply the\n    separate finite-sample variance scale" in docs
    assert "T / (T - L - 1)" not in docs


def test_serial_correlation_gloss_matches_persistence_calibration() -> None:
    docs = CALIBRATION_DOCS.read_text(encoding="utf-8")
    phi_point_six_row = next(
        line for line in docs.splitlines() if line.startswith("| 0.6 |")
    )
    _, plain_t, newey_west, bootstrap = (
        cell.strip().replace("–", "-")
        for cell in phi_point_six_row.strip("|").split("|")
    )
    gloss = WarningCode.SERIAL_CORRELATION_DETECTED.description

    assert f"{newey_west} (NW)" in gloss
    assert f"{bootstrap} (bootstrap)" in gloss
    assert f"{plain_t} (plain t)" in gloss


def test_result_and_multiplicity_docs_do_not_overpromise_calibration() -> None:
    result_docs = RESULT_DOCS.read_text(encoding="utf-8")
    stats_docs = STATS_DOCS.read_text(encoding="utf-8")
    llms_full = LLMS_FULL.read_text(encoding="utf-8")

    assert "The calibrated statistical p-value" not in result_docs
    assert "A valid number\n  does not by itself establish calibration" in result_docs
    assert "neither validate nor repair\nproducer calibration" in stats_docs
    assert "A finite value does not establish calibration" in llms_full


def _metrics_exposing_inference() -> set[str]:
    """Registered metrics whose public signature takes ``inference=``."""
    names = set()
    for name, metric in REGISTRY.items():
        target = getattr(metric, "_impl", None) or metric
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if "inference" in signature.parameters:
            names.add(name)
    return names


def test_architecture_lists_exactly_the_metrics_taking_inference() -> None:
    """The SSOT section names the selectable-inference family and nothing else.

    The same family is restated in prose on several pages; this pins the copy
    the other pages defer to, so adding ``inference=`` to a fifth metric fails
    here instead of leaving the architecture rule quietly stale.
    """
    section = ARCHITECTURE_DOCS.read_text(encoding="utf-8").split(
        "### Inference selection (`inference=`)"
    )[1]
    intro = section.split("This section is the SSOT")[0]
    named = set(re.findall(r"`([a-z_]+)`", intro)) & set(REGISTRY)

    assert named == _metrics_exposing_inference()
