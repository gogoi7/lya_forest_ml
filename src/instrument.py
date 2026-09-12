"""Instrument-response functions for simulated Ly-alpha forest spectra."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_cos_gaussian(flux, dv_sim, sigma_kms=7.96):
    """
    Apply a periodic Gaussian approximation of the COS LSF to the input flux array.

    Parameters:
    flux :array-like
        Input flux array (1D or 2D) to which the COS LSF will be applied.
    dv_sim :float
        Pixel width of the simulation pixel in km/s. This is used to convert the Gaussian LSF width from km/s to pixels.
    sigma_kms :float, optional
        Standard deviation of the Gaussian LSF in km/s. Default is 7.96 km/s, which corresponds to the COS LSF resolving power of R ~ 18,000 at 1216 Å.

    Returns:
    flux_smoothed :np.ndarray
        Flux array after applying the COS LSF. The shape is the same as the input flux array.
    dv_out :float
        output pixel width in km/s, which is the same as dv_sim because the function does not perform any resampling or rebinning of the flux array.
    """
    flux = np.asarray(flux, dtype=np.float64)
    dv_sim = float(dv_sim)
    sigma_kms = float(sigma_kms)

    if not flux.ndim in (1, 2):
        raise ValueError("flux must be a 1D or 2D array")

    if flux.shape[-1] < 2:
        raise ValueError("flux must have at least two pixels")

    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values")

    if not np.isfinite(dv_sim) or dv_sim <= 0.0:
        raise ValueError("dv_sim must be a positive finite value")

    if not np.isfinite(sigma_kms) or sigma_kms <= 0.0:
        raise ValueError("sigma_kms must be a positive finite value")

    # Convert sigma from km/s to pixels
    sigma_pix = sigma_kms / dv_sim

    # Apply the Gaussian filter
    flux_smoothed = gaussian_filter1d(flux, sigma=sigma_pix, axis=-1, mode='wrap', truncate=4.0)

    return flux_smoothed, dv_sim  # output pixel width is the same as input since we are not resampling