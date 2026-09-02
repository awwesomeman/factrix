"""Keep public regression-inference descriptions aligned with their consumers."""

from pathlib import Path

from factrix._codes import WarningCode

SPANNING_SOURCE = Path("factrix/metrics/spanning.py")
SPANNING_DOCS = Path("docs/api/metrics/spanning.md")
PREDICTIVE_DOCS = Path("docs/api/metrics/predictive_beta.md")
CALIBRATION_DOCS = Path("docs/reference/inference-calibration.md")


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
