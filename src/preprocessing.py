""" Preprocessing module for Lyman-alpha forest data. """

import numpy as np

def _as_tau(tau):
    """Convert tau to a validated NumPy array."""
    tau = np.asarray(tau, dtype=np.float64)

    if tau.size == 0:
        raise ValueError("tau must not be empty")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must contain only finite values")
    if np.any(tau < 0.0):
        raise ValueError("tau must be non-negative")

    return tau

def mean_flux(tau):
    """
    Return the mean transmitted flux <F> = <exp(-tau)> for a given array of optical depths tau.
    """
    tau = _as_tau(tau)
    return float(np.mean(np.exp(-tau)))

def scale_tau(tau, factor):
    """
    Scale the optical depth tau by a given factor.
    
    Parameters:
    tau (array-like): Array of optical depths.
    factor (float): Scaling factor.
    
    Returns:
    np.ndarray: Scaled optical depths.
    """
    tau = _as_tau(tau)
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be a finite value")
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    return tau * factor

def fit_tau_scale(tau, target, tol=1e-4, max_iter=1000):
    """
    Fit a scaling factor for tau such that the mean transmitted flux matches the target value.
    
    Parameters:
    tau (array-like): Array of optical depths.
    target (float): Target mean transmitted flux <F>.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    float: Scaling factor for tau.
    """
    tau = _as_tau(tau)
    target = float(target)
    
    if not (0.0 < target <= 1.0) or not np.isfinite(target):
        raise ValueError("target must be a finite value in the range (0, 1]")
    
    if target == 1.0:
        return 0.0  # scaling factor of 0 yields mean flux of 1
    
    floor_flux = float(np.mean(tau == 0.0))
    
    if floor_flux >= target:
        raise ValueError("unrealistic target: mean flux cannot be achieved with given tau values because a significant fraction of optical depth pixels are zero, leading to a floor mean flux of {:.6f}".format(floor_flux))

    scale = 0.01

    for _ in range(max_iter):
        flux = np.exp(-scale * tau)
        err = float(np.mean(flux)) - target

        if abs(err) <= tol:
            return scale

        denominator = float(np.mean(tau * flux))
        if denominator == 0.0:
            raise ValueError("Newton-Raphson denominator is zero")

        new_scale = scale + err / denominator
        
        if new_scale <= 0.0:
            new_scale = scale / 2.0

        scale = new_scale

    raise RuntimeError("Newton-Raphson did not converge")

def mean_flux_target(tau_sets):
    """Return the mean flux across model datasets, weighted equally accross all model
    datasets.

    Args:
        tau_sets (array-like): A list of arrays of optical depths for different models.
    """
    fluxes = [mean_flux(tau) for tau in tau_sets]
    
    if not fluxes:
        raise ValueError("tau_sets must not be empty")
    
    return float(np.mean(fluxes))