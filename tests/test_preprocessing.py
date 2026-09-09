import numpy as np
import pytest
from src.preprocessing import mean_flux, scale_tau, fit_tau_scale, mean_flux_target

def test_mean_flux_avg_exp_minus_tau():
    tau = np.array([0.0, np.log(2.0), np.log(4.0)])
    expected = (1.0 + 0.5 + 0.25) / 3.0
    
    assert mean_flux(tau) == pytest.approx(expected)
    
def test_scale_tau_not_modify_input():
    tau = np.array([0.2, 0.5, 1.0])
    original = tau.copy()
    
    scaled_tau = scale_tau(tau, factor=2.0)
    
    np.testing.assert_allclose(scaled_tau, [0.4, 1.0, 2.0])
    np.testing.assert_array_equal(tau, original)
    
def test_fit_tau_scale_matches_target_mean_flux():
    tau = np.full((2, 3), 0.5)  # 2x3 array of tau values
    target = 0.4
    
    factor = fit_tau_scale(tau, target, tol=1e-12)
    
    expected = -np.log(target) / 0.5
    assert factor == pytest.approx(expected, rel=1e-10)
    assert mean_flux(scale_tau(tau, factor)) == pytest.approx(target, abs=1e-12)
    
def test_fit_tau_scale_handles_broad_tau_range():
    tau = np.array([0.0, 1000.0])  # Very broad range of tau values
    target = 0.999
    
    factor = fit_tau_scale(tau, target, tol=1e-10)
    
    assert factor >= 0.0
    assert mean_flux(scale_tau(tau, factor)) == pytest.approx(target, abs=1e-10)
    
@pytest.mark.parametrize("tau", [
    [],
    [np.nan],
    [np.inf],
    [-0.1],
])
def test_mean_flux_invalid_inputs(tau):
    with pytest.raises(ValueError):
        mean_flux(tau)

@pytest.mark.parametrize("target", [
    -0.1,
    1.1,
    0.0,
    np.nan,
    np.inf,
])
def test_fit_tau_scale_invalid_target(target):
    tau = np.array([0.2, 0.5, 1.0])
    
    with pytest.raises(ValueError):
        fit_tau_scale(tau, target)
        
@pytest.mark.parametrize("factor", [-1.0, np.nan, np.inf])
def test_scale_tau_invalid_factor(factor):
    tau = np.array([0.2, 0.5, 1.0])
    
    with pytest.raises(ValueError):
        scale_tau(tau, factor)
        
def test_fit_tau_scale_returns_zero_for_target_one():
    tau = np.array([0.2, 0.5, 1.0])
    
    assert fit_tau_scale(tau, target=1.0) == 0.0
    
def test_fit_tau_scale_rejects_unrealistic_target():
    tau = np.array([0.0, 0.0, 1.0, 2.0])
    
    with pytest.raises(ValueError, match="unrealistic target"):
        fit_tau_scale(tau, target=0.4)
        
def test_mean_flux_target_weighs_models_equally():
    tau_a = np.array([0.0])
    tau_b = np.full(3, np.log(2.0))  # 3 pixels with tau = log(2) or flux = 0.5
    
    target = mean_flux_target([tau_a, tau_b])
    
    assert target == pytest.approx(0.75) 
    
def test_mean_flux_target_with_empty_model():
    tau_a = np.array([0.0])
    tau_b = np.array([])  # Empty model
    
    with pytest.raises(ValueError, match="must not be empty"):
        mean_flux_target([tau_a, tau_b])