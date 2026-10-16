"""Reason flags for COS contamination masks."""

from enum import IntFlag

import numpy as np


class MaskReason(IntFlag):
    NONE = 0
    BAD_WAVE = 1
    BAD_FLUX = 2
    BAD_ERROR = 4
    NO_COVERAGE = 8
    AIRGLOW = 16
    GALACTIC = 32
    METAL = 64

    INVALID = BAD_WAVE | BAD_FLUX | BAD_ERROR | NO_COVERAGE


def invalid_flags(wavelength, flux, error):
    """Flag invalid native pixels while preserving overlapping reasons.

    Finite negative flux remains valid. NO_COVERAGE is reserved for
    missing wavelength coverage discovered during resampling.
    """
    wavelength, flux, error = [
        np.asarray(array, dtype=np.float64)
        for array in (wavelength, flux, error)
    ]

    if any(array.ndim != 1 for array in (wavelength, flux, error)):
        raise ValueError("Inputs must be 1D arrays")

    if not (wavelength.shape == flux.shape == error.shape):
        raise ValueError("Input shapes must match")

    flags = np.zeros(wavelength.shape, dtype=np.uint16)

    flags[
        ~np.isfinite(wavelength) | (wavelength <= 0)
    ] |= int(MaskReason.BAD_WAVE)

    flags[
        ~np.isfinite(flux)
    ] |= int(MaskReason.BAD_FLUX)

    flags[
        ~np.isfinite(error) | (error <= 0)
    ] |= int(MaskReason.BAD_ERROR)

    return flags

C_KMS = 299792.458


def line_flags(wavelength, centers_A, reason, half_width_kms=150.0):
    """Flag pixel centers within fixed log-velocity windows.

    Wavelengths and line centers must use the same wavelength frame.
    Returns a new flags array without changing the input.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    centers_A = np.asarray(centers_A, dtype=np.float64)
    half_width_kms = float(half_width_kms)

    if wavelength.ndim != 1 or centers_A.ndim != 1:
        raise ValueError("Wavelength and centers must be 1D arrays")

    if np.any(~np.isfinite(centers_A)) or np.any(centers_A <= 0):
        raise ValueError("Line centers must be positive and finite")

    if not np.isfinite(half_width_kms) or half_width_kms <= 0:
        raise ValueError("Window half-width must be positive and finite")

    reason = MaskReason(reason)
    if reason not in (
        MaskReason.AIRGLOW,
        MaskReason.GALACTIC,
        MaskReason.METAL,
    ):
        raise ValueError("Expected one contamination reason")

    flags = np.zeros(wavelength.shape, dtype=np.uint16)
    factors = np.exp(
        np.array([-half_width_kms, half_width_kms]) / C_KMS
    )

    for center in centers_A:
        left, right = center * factors
        selected = (wavelength >= left) & (wavelength <= right)
        flags[selected] |= int(reason)

    return flags