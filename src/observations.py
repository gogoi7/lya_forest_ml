"""Load and process observational data"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Spectrum:
    """Class representing a single spectrum observation. Wavelengths are in Angstroms, fluxes are normalized to the continuum, and errors are 1-sigma uncertainties in the fluxes."""
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray

    @property
    def valid(self):
        """Pixels with finite fluxes and errors."""
        return np.isfinite(self.flux) & np.isfinite(self.error) & (self.error > 0.0)

def load_spectrum(path):
    """Load a spectrum from a file. The file should contain three columns: wavelength, normalized flux, and error."""
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)

    if data.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (wavelength, flux, error) in the spectrum file, got {data.shape[1]}")

    if data.shape[0] < 2:
        raise ValueError("Spectrum file must contain at least two data pixels")

    wavelength, flux, error = data.T

    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("Wavelengths must be finite and positive")

    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("Wavelengths must be strictly increasing")

    return Spectrum(wavelength=wavelength, flux=flux, error=error)