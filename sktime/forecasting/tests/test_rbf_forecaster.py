import pytest

from sktime.forecasting.rbf_forecaster import RBFForecaster
from sktime.tests.test_switch import run_test_for_class

__author__ = ["aryanyk"]


@pytest.mark.skipif(
    not run_test_for_class(RBFForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hidden_layers_param_roundtrip_none():
    """`hidden_layers=None` should remain unchanged on estimator params."""
    f = RBFForecaster(hidden_layers=None)

    assert f.hidden_layers is None
    assert f.get_params()["hidden_layers"] is None


@pytest.mark.skipif(
    not run_test_for_class(RBFForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hidden_layers_param_roundtrip_user_value():
    """User-provided hidden layer config should round-trip through params."""
    f = RBFForecaster(hidden_layers=[32, 16])

    assert f.hidden_layers == [32, 16]
    assert f.get_params()["hidden_layers"] == [32, 16]


@pytest.mark.skipif(
    not run_test_for_class(RBFForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hidden_layers_resolution_default_and_copy():
    """Resolved hidden layer config should default and return a defensive copy."""
    f_default = RBFForecaster(hidden_layers=None)
    resolved_default = f_default._get_hidden_layers()

    assert resolved_default == [64, 32]
    assert isinstance(resolved_default, list)

    f_user = RBFForecaster(hidden_layers=[32, 16])
    resolved_user = f_user._get_hidden_layers()
    resolved_user.append(8)

    # mutation of resolved list must not mutate constructor param storage
    assert f_user.hidden_layers == [32, 16]
