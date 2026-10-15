import numpy as np
import pytest

from jinyiquant.runner import _curve_diagnostics


def test_curve_reference_uses_training_mean_and_preserves_return_units():
    targets = np.array([0.01, 0.03])
    scores = np.array([0.0, 0.02])
    result = _curve_diagnostics(scores, targets, 0.0)
    assert result["mse"] == pytest.approx(0.0001)
    assert result["constant_mse"] == pytest.approx(0.0005)
    assert result["mse_over_constant"] == pytest.approx(0.2)
    assert result["prediction_std"] == pytest.approx(0.01)
    assert result["target_std"] == pytest.approx(0.01)
    assert _curve_diagnostics(np.ones(2), np.zeros(2), 0)["mse_over_constant"] is None
