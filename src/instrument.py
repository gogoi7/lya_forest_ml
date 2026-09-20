"""Instrument-response functions for simulated Ly-alpha forest spectra."""
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.interpolate import PchipInterpolator

def apply_cos_gaussian(
    flux,
    dv_sim,
    sigma_kms=7.96,
    truncate=4.0,
):
    """
    Apply a periodic Gaussian LSF without changing the pixel grid.

    Parameters
    ----------
    flux : array_like
        One spectrum or an array shaped (n_spectra, n_pixels).
    dv_sim : float
        Input pixel width in km/s.
    sigma_kms : float
        Standard deviation of the Gaussian in km/s.
    truncate : float
        Kernel extent on each side, in Gaussian standard deviations.

    Returns
    -------
    flux_smoothed : np.ndarray
        Smoothed flux with the same shape as the input.
    dv_out : float
        Unchanged pixel width in km/s.
    """
    flux = np.asarray(flux, dtype=np.float64)
    dv_sim = float(dv_sim)
    sigma_kms = float(sigma_kms)
    truncate = float(truncate)

    if flux.ndim not in (1, 2):
        raise ValueError("flux must be a 1D or 2D array")

    if flux.shape[-1] < 2:
        raise ValueError("flux must contain at least two pixels")

    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values")

    if not np.isfinite(dv_sim) or dv_sim <= 0.0:
        raise ValueError("dv_sim must be a positive finite value")

    if not np.isfinite(sigma_kms) or sigma_kms <= 0.0:
        raise ValueError("sigma_kms must be a positive finite value")

    if not np.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be a positive finite value")

    sigma_pix = sigma_kms / dv_sim

    flux_smoothed = gaussian_filter1d(
        flux,
        sigma=sigma_pix,
        axis=-1,
        mode="wrap",
        truncate=truncate,
    )

    return flux_smoothed, dv_sim

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

def load_cos_lsf(path):
    """Load an STScI FUV LSF table and normalize each profile.

    Returns
    -------
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Wavelength of each tabulated profile, in Angstrom.
    kernels : np.ndarray, shape (321, n_wavelengths)
        Nonnegative LSF weights, with each column summing to one.
    """
    table = np.loadtxt(path, dtype=np.float64, ndmin=2)

    if table.shape[0] != 322 or table.shape[1] < 2:
        raise ValueError(
            "Expected a wavelength header followed by 321 LSF rows"
        )

    if not np.all(np.isfinite(table)):
        raise ValueError("LSF table must contain only finite values")

    wavelengths = table[0]
    kernels = table[1:]

    if np.any(wavelengths <= 0.0) or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("LSF wavelengths must be positive and increasing")

    if np.any(kernels < 0.0):
        raise ValueError("LSF weights must be nonnegative")

    column_sums = kernels.sum(axis=0)

    # Reject incorrectly scaled tables; allow small tabulation errors.
    if not np.allclose(column_sums, 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("Each LSF profile must already sum approximately to one")

    return wavelengths, kernels / column_sums[None, :]

def resample_lsf(kernel, dv_in, dv_out):
    """Resample centered LSF weights onto a new velocity grid.

    Pixel spacings are in km/s. The middle sample represents zero
    velocity. Treats input values as weights in native pixel bins.
    Returns normalized, odd-length weights covering the full input.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    dv_in = float(dv_in)
    dv_out = float(dv_out)

    if kernel.ndim != 1 or kernel.size < 3 or kernel.size % 2 == 0:
        raise ValueError("kernel must be a 1D array with an odd length >= 3")

    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("kernel must contain finite, nonnegative weights")

    total = kernel.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("kernel must have a positive finite total weight")

    if not np.isfinite(dv_in) or dv_in <= 0.0:
        raise ValueError("dv_in must be positive and finite")

    if not np.isfinite(dv_out) or dv_out <= 0.0:
        raise ValueError("dv_out must be positive and finite")

    kernel = kernel / total
    if dv_in == dv_out:
        return kernel.copy()

    # Native bin edges and cumulative weight at those edges.
    half_in = kernel.size // 2
    edges_in = (np.arange(kernel.size + 1) - half_in - 0.5) * dv_in
    cumulative = np.concatenate(([0.0], np.cumsum(kernel)))
    cumulative /= cumulative[-1]

    # Choose an odd number of output bins covering all native bins.
    half_out = int(np.ceil(edges_in[-1] / dv_out - 0.5))
    edges_out = (
        np.arange(2 * half_out + 2) - half_out - 0.5
    ) * dv_out

    interpolate_cdf = PchipInterpolator(
        edges_in, cumulative, extrapolate=False
    )

    # Outside the native support, cumulative weight stays at 0 or 1.
    cumulative_out = interpolate_cdf(
        np.clip(edges_out, edges_in[0], edges_in[-1])
    )
    resampled = np.diff(cumulative_out)

    if np.any(resampled < -1e-14):
        raise ValueError("Resampling produced negative LSF weights")

    # Remove only tiny negative roundoff errors.
    resampled = np.maximum(resampled, 0.0)
    return resampled / resampled.sum()

def apply_cos_lsf(flux, dv_sim, kernel):
    """Convolve complete periodic mock sightlines with a tabulated LSF.

    The kernel must be sampled at dv_sim, with zero velocity at its
    middle element. Returns (smoothed_flux, dv_sim), preserving the
    input shape and pixel grid.
    """
    flux = np.asarray(flux, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    dv_sim = float(dv_sim)

    if flux.ndim not in (1, 2) or flux.size == 0 or flux.shape[-1] < 2:
        raise ValueError(
            "flux must be a nonempty 1D or 2D array with >= 2 pixels"
        )

    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values")

    if not np.isfinite(dv_sim) or dv_sim <= 0.0:
        raise ValueError("dv_sim must be positive and finite")

    if kernel.ndim != 1 or kernel.size == 0 or kernel.size % 2 == 0:
        raise ValueError("kernel must be a nonempty 1D array of odd length")

    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("kernel must contain finite, nonnegative weights")

    total = kernel.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("kernel must have a positive finite total weight")

    smoothed = convolve1d(
        flux,
        weights=kernel / total,
        axis=-1,
        mode="wrap",
        origin=0,
    )

    return smoothed, dv_sim