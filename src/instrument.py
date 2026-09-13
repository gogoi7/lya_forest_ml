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

def resample_flux(flux, dv_in, n_pixels_out):
    """
    Resample the input flux array to a new number of pixels using linear interpolation.

    Parameters:
    flux :array-like
        Input flux array (1D or 2D) to be resampled.
    dv_in :float
        Pixel width of the input flux array in km/s.
    n_pixels_out :int
        Desired number of pixels in the output flux array.

    Returns:
    flux_out :np.ndarray
        Resampled flux array with shape (n_pixels_out,) for 1D input or (n_samples, n_pixels_out) for 2D input.
    dv_out :float
        Pixel width of the output flux array in km/s, calculated as (dv_in * n_pixels_in / n_pixels_out).

    Notes
    -----
    This preserves the total velocity length and integrated flux.
    Apply instrumentel smoothing (e.g., COS LSF) before resampling to avoid aliasing.
    """
    flux = np.asarray(flux, dtype=np.float64)
    dv_in = float(dv_in)

    if flux.ndim not in (1, 2):
        raise ValueError("flux must be a 1D or 2D array")

    if flux.size == 0 or flux.shape[-1] < 2:
        raise ValueError("flux must have at least two pixels")

    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values")

    if not np.isfinite(dv_in) or dv_in <= 0.0:
        raise ValueError("dv_in must be a positive finite value")

    if (
        not isinstance(n_pixels_out, (int, np.integer))
        or isinstance(n_pixels_out, (bool, np.bool_))
        or n_pixels_out < 2
    ):
        raise ValueError("n_pixels_out must be a positive integer >= 2")

    n_pixels_out = int(n_pixels_out)
    n_pixels_in = flux.shape[-1]

    velocity_length = n_pixels_in * dv_in
    dv_out = velocity_length / n_pixels_out

    # Create the new velocity grid for the output flux
    edges_in = np.arange(n_pixels_in + 1, dtype=np.float64) * dv_in
    edges_out = np.linspace(0.0, velocity_length, n_pixels_out + 1)

    # Convert flux to 2d if it's 1d for consistent processing
    flux_2d = np.atleast_2d(flux)
    flux_out = np.empty((flux_2d.shape[0], n_pixels_out), dtype=np.float64)

    for i, spectrum in enumerate(flux_2d):
        # Cumulative integral at the edges of the input pixels
        integral_in = np.zeros(n_pixels_in + 1, dtype=np.float64)
        integral_in[1:] = np.cumsum(spectrum) * dv_in

        # Interpolate the cumulative integral to the new edges
        integral_out = np.interp(edges_out, edges_in, integral_in)

        # Compute the output flux by differentiating the cumulative integral
        flux_out[i] = np.diff(integral_out) / dv_out

    if flux.ndim == 1:
        return flux_out[0], dv_out

    return flux_out, dv_out