import numpy as np


def rest_ew(flux, velocity_edges, galaxy_velocity, half_window):
    """Rest-frame Ly-alpha EW in Angstroms; all velocities are in km/s.

    flux is (n_pixels,) or (n_rays, n_pixels).
    velocity_edges contains n_pixels + 1 increasing edges for one full box.
    galaxy_velocity is a scalar or one value per ray, in the same frame.
    Each pixel is treated as having constant flux within its edges.
    """
    flux = np.asarray(flux, dtype=float)
    edges = np.asarray(velocity_edges, dtype=float)
    half_window = float(half_window)

    if flux.ndim not in (1, 2):
        raise ValueError("flux must be a spectrum or a 2D array of spectra")
    if edges.shape != (flux.shape[-1] + 1,) or len(edges) < 2:
        raise ValueError("Provide one more edge than flux pixels")
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Velocity edges must be finite and increasing")

    single_spectrum = flux.ndim == 1
    flux = np.atleast_2d(flux)
    centers = np.broadcast_to(
        np.asarray(galaxy_velocity, dtype=float), (len(flux),)
    )
    if not np.isfinite(flux).all() or not np.isfinite(centers).all():
        raise ValueError("Flux and galaxy velocities must be finite")

    box_velocity = edges[-1] - edges[0]
    if not 0 < half_window <= box_velocity / 2:
        raise ValueError("Window must fit within one box period")

    centers = (centers - edges[0]) % box_velocity + edges[0]
    window_left = centers[:, None] - half_window
    window_right = centers[:, None] + half_window

    weights = np.zeros_like(flux)
    for shift in (-box_velocity, 0.0, box_velocity):
        left = np.maximum(edges[:-1] + shift, window_left)
        right = np.minimum(edges[1:] + shift, window_right)
        weights += np.maximum(right - left, 0.0)

    ew = (1215.67 / 299792.458) * np.sum((1 - flux) * weights, axis=1)
    return ew[0] if single_spectrum else ew