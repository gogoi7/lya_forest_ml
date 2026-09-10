import numpy as np
from scipy.signal import find_peaks
from scipy import stats

def extract_features(X, dv):
    """
    Extracts physical and statistical features from spectra.

    Parameters:
    X (np.ndarray): Array of spectra (fluxes), shape (n_samples, n_pixels).
    dv (float): Velocity width of a pixel in km/s.

    Returns:
    np.ndarray: Array of features, shape (n_samples, n_features).
    """
    n_samples, n_pixels = X.shape
    features = []

    for i in range(n_samples):
        flux = X[i]

        # 1. Flux Statistics
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        min_flux = np.min(flux)
        # max_flux = np.max(flux) # Removed as per feedback (always ~1)

        # 4. Distribution Shape
        skewness = stats.skew(flux)
        kurtosis = stats.kurtosis(flux)

        # 6. Percentiles
        percentiles = np.percentile(flux, [5, 25, 50, 75, 95])

        # 7. Absorption Lines (using find_peaks on 1 - flux)
        # Prominence and width help filter noise
        inverted_flux = 1 - flux
        peaks, properties = find_peaks(inverted_flux, height=0.05, prominence=0.01, width=1)

        num_lines = len(peaks)

        if num_lines > 0:
            line_depths = properties['peak_heights']
            line_widths = properties['widths'] * dv # Convert width to km/s (FWHM approx)

            mean_line_depth = np.mean(line_depths)
            max_line_depth = np.max(line_depths)
            mean_line_width = np.mean(line_widths)
            max_line_width = line_widths[np.argmax(line_depths)] # Width of deepest line
        else:
            mean_line_depth = 0
            max_line_depth = 0
            mean_line_width = 0
            max_line_width = 0

        # 12. Total Equivalent Width (Sum of (1-F) * dv)
        # This is an approximation integrating over the whole spectrum
        total_ew = np.sum(1 - flux) * dv

        # 13. Column Density Proxy (Sum of optical depth)
        # tau = -ln(F). small offset to avoid log(0)
        tau = -np.log(flux + 1e-10)
        total_col_density_proxy = np.sum(tau)

        # Combine all features
        sample_features = [
            mean_flux, std_flux, min_flux,
            skewness, kurtosis,
            *percentiles,
            num_lines,
            mean_line_depth, max_line_depth,
            mean_line_width, max_line_width,
            total_ew,
            total_col_density_proxy
        ]

        features.append(sample_features)

    return np.array(features)

if __name__ == "__main__":
    # Test block
    print("Testing feature extraction...")
    dummy_data = np.random.rand(10, 2000)
    dv = 10.0 # dummy velocity width
    feats = extract_features(dummy_data, dv)


def extract_power_spectrum(X, dv, n_bins=20):
    """
    Extracts binned 1D Flux Power Spectrum P(k) (Log-spaced bins).
    Total features: n_bins (default 20).
    """
    n_samples, n_pixels = X.shape
    mean_F = np.mean(X, axis=1, keepdims=True)
    delta_F = (X / mean_F) - 1

    fft_flux = np.fft.rfft(delta_F, axis=1)
    L = n_pixels * dv
    P_raw = (L / (n_pixels**2)) * np.abs(fft_flux)**2

    k_raw = 2 * np.pi * np.fft.rfftfreq(n_pixels, d=dv)

    # Binning logic
    k_min = k_raw[1] # Skip DC
    k_max = k_raw[-1]
    bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)

    P_binned = np.zeros((n_samples, n_bins))

    for i in range(n_bins):
        mask = (k_raw >= bins[i]) & (k_raw < bins[i+1])
        if np.sum(mask) > 0:
            P_binned[:, i] = np.mean(P_raw[:, mask], axis=1)
        else:
            P_binned[:, i] = 0 # Empty bin

    return P_binned

def extract_bispectrum_features(X, dv):
    """
    Compact Bispectrum Proxy: Just the Skewness/Kurtosis of the flux gradient.
    Total features: 3 (std, skew, kurt of dF/dx).
    """
    n_samples, n_pixels = X.shape
    features = []

    for i in range(n_samples):
        flux = X[i]
        dF = np.gradient(flux)
        features.append([np.std(dF), stats.skew(dF), stats.kurtosis(dF)])

    return np.array(features)

def extract_wavelet_features(X, dv):
    """
    Compact Wavelet Features: Just the Energy (Variance) at each scale.
    Total features: ~6 (for 5 levels + approximation).
    """
    import pywt
    n_samples, n_pixels = X.shape
    features = []
    wavelet = 'db4'
    level = 5

    for i in range(n_samples):
        flux = X[i]
        coeffs = pywt.wavedec(flux, wavelet, level=level)
        # Just Energy: sum(c^2)/N
        energies = [np.sum(c**2)/len(c) for c in coeffs]
        features.append(energies)

    return np.array(features)

def extract_combined_features(X, dv):
    """
    Compact Feature Set (<50).
    Stats (~15) + P(k) (20) + Bispectrum (3) + Wavelets (6) ~= 44 features.
    """
    print("Extracting P(k) (20 bins)...")
    pk_feats = extract_power_spectrum(X, dv, n_bins=20)

    print("Extracting Stats...")
    stats_feats = extract_features(X, dv)

    print("Extracting Bispectrum Proxy (3)...")
    bi_feats = extract_bispectrum_features(X, dv)

    print("Extracting Wavelets (Energy only)...")
    wav_feats = extract_wavelet_features(X, dv)

    return np.concatenate([stats_feats, pk_feats, bi_feats, wav_feats], axis=1)

def bundle_features(features, bundles):
    """
    Bundle features according to the provided bundle indices.

    Parameters:
    features (np.ndarray): Array of features, shape (n_samples, n_features).
    bundles (np.ndarray): Array of bundle indices, shape (n_bundles, bundle_size).

    Returns:
    np.ndarray: mean and std of features for each bundle, shape (n_bundles, n_features * 2).
    """
    features = np.asarray(features)
    bundles = np.asarray(bundles)

    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    if bundles.ndim != 2 or bundles.size == 0:
        raise ValueError("bundles must be a 2D array with at least one bundle")

    if not np.issubdtype(bundles.dtype, np.integer):
        raise ValueError("bundles must contain integer indices")

    if np.any(bundles < 0) or np.any(bundles >= features.shape[0]):
        raise ValueError("bundles contain out-of-bounds indices")

    grouped_features = features[bundles]  # shape: (n_bundles, bundle_size, n_features)
    mean_features = np.mean(grouped_features, axis=1)
    std_features = np.std(grouped_features, axis=1)

    return np.concatenate([mean_features, std_features], axis=1)