"""Guard MLflow reflection-based field logging.

``FactorTracker.log_profile`` iterates ``dataclasses.fields(profile)`` and
logs every float/int/bool as a metric. This test locks that contract so
new p-value fields added to profiles (``ts_beta_hac_p``, ``ic_nw_p``,
``bmp_p``, ...) are auto-logged without touching ``mlflow.py``.

If this test breaks, either the reflection loop regressed or a profile
field type changed in a way that bypasses the mlflow.log_metric path —
both warrant a conscious fix, not a silent drop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from factrix.evaluation.profiles import CrossSectionalProfile


mlflow = pytest.importorskip("mlflow")

from factrix.integrations.mlflow import FactorTracker  # noqa: E402


def _logged_metric_names(mock_log_metric: MagicMock) -> set[str]:
    return {call.args[0] for call in mock_log_metric.call_args_list}


def test_log_profile_reflects_every_pvalue_field(
    cs_profile_strong: CrossSectionalProfile,
) -> None:
    """Every ``PValue`` field on the profile must land as a MLflow metric.

    The reflection loop's contract: any float-valued dataclass field gets
    ``mlflow.log_metric(field_name, value)``. ``P_VALUE_FIELDS`` is the
    authoritative set of p-value names for a profile, so check each is
    reached.
    """
    with (
        patch("factrix.integrations.mlflow.mlflow.set_experiment"),
        patch("factrix.integrations.mlflow.mlflow.start_run") as mock_start,
        patch("factrix.integrations.mlflow.mlflow.set_tag"),
        patch("factrix.integrations.mlflow.mlflow.log_params"),
        patch("factrix.integrations.mlflow.mlflow.log_metric") as mock_log,
    ):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start.return_value.__enter__.return_value = mock_run

        tracker = FactorTracker(experiment_name="test")
        tracker.log_profile(cs_profile_strong)

    logged = _logged_metric_names(mock_log)
    for pfield in CrossSectionalProfile.P_VALUE_FIELDS:
        assert pfield in logged, (
            f"p-value field {pfield!r} was not auto-logged to MLflow. "
            f"Reflection loop in integrations/mlflow.py may have regressed."
        )
    # canonical_p is explicitly logged in addition to the reflected fields
    assert "canonical_p" in logged
