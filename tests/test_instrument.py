import numpy as np
import pytest

from src.instrument import apply_cos_gaussian, resample_flux

TARGET_K_VALUES = [0.02, 0.05, 0.10, 0.20, 0.30]

def test_gaussian_lsf_preserves_constant_flux():
    flux = np.ones((2, 2499), dtype=np.float64)

    flus_smoothed, dv_out = apply_cos_gaussian(flux, dv_sim=1.0004, sigma_kms=7.96)

    np.testing.assert_allclose(flus_smoothed, flux, atol=1e-14)
    assert dv_out == pytest.approx(1.0004)

def test_gaussian_lsf_wraps_across_periodic_boundary():
    flux = np.zeros(2499, dtype=np.float64)
    flux[0] = 1.0  # Set the first pixel to 1

    flux_smoothed, _ = apply_cos_gaussian(flux, dv_sim=1.0004, sigma_kms=7.96)

    assert flux_smoothed[-1] > 0.0  # The last pixel should have a non-zero value due to wrapping
    assert np.sum(flux_smoothed) == pytest.approx(1.0, abs=1e-14)  # Total flux should be preserved

@pytest.mark.parametrize("target_k", TARGET_K_VALUES)
def test_gaussian_lsf_power_transfer(target_k):
    n_pixels = 2499
    dv_sim = 1.0004
    sigma_kms = 7.96

    velocity_length = n_pixels * dv_sim

    mode_number = int(np.rint(target_k * velocity_length / (2 * np.pi)))
    k_mode = 2 * np.pi * mode_number / velocity_length

    velocity = np.arange(n_pixels) * dv_sim
    flux = 1.0 + 0.1 * np.cos(k_mode * velocity)

    flux_smoothed, _ = apply_cos_gaussian(flux, dv_sim=dv_sim, sigma_kms=sigma_kms)

    delta_flux = flux - np.mean(flux)
    delta_flux_smoothed = flux_smoothed - np.mean(flux_smoothed)

    fft_original = np.fft.rfft(delta_flux)
    fft_smoothed = np.fft.rfft(delta_flux_smoothed)

    power_original = np.abs(fft_original[mode_number])**2
    power_smoothed = np.abs(fft_smoothed[mode_number])**2

    measured_ratio = power_smoothed / power_original
    expected_ratio = np.exp(-(sigma_kms * k_mode)**2)

    print(
        f"target k={target_k:.3f}, "
        f"actual k={k_mode:.6f}, "
        f"measured ratio={measured_ratio:.6f}, "
        f"expected ratio={expected_ratio:.6f}"
    )

    assert measured_ratio == pytest.approx(expected_ratio, rel=5e-3)

def test_resampling_matches_expected():
    flux  = np.array([0.2, 0.4, 1.0, 0.6, 0.8])

    flux_out, dv_out = resample_flux(flux, dv_in=1.0, n_pixels_out=3)

    #Output bins cover [0, 5/3), [5/3, 10/3), [10/3, 5]
    expected = np.array([0.28, 0.8, 0.72])

    assert flux_out.shape == (3,)
    assert dv_out == pytest.approx(5.0 / 3.0)

    np.testing.assert_allclose(flux_out, expected, rtol=0.0, atol=1e-12)

def test_resampling_preserves_flat_continuum():
    flux = np.full(2499, 0.73, dtype=np.float64)

    flux_out, _ = resample_flux(flux, dv_in=1.0004, n_pixels_out=1249)

    assert flux_out.shape == (1249,)
    np.testing.assert_allclose(flux_out, 0.73, rtol=0.0, atol=1e-12)

def test_resampling_preserves_batch_means_length_and_input():
    rng = np.random.default_rng(seed=42)
    flux = rng.uniform(0.0, 1.0, size=(3, 2499))
    original = flux.copy()
    dv_in = 1.0004

    flux_out, dv_out = resample_flux(flux, dv_in=dv_in, n_pixels_out=1249)

    assert flux_out.shape == (3, 1249)

    length_in = flux.shape[-1] * dv_in
    length_out = flux_out.shape[-1] * dv_out

    assert length_out == pytest.approx(length_in, rel=0.0, abs=1e-10)

    np.testing.assert_allclose(flux_out.mean(axis=1), original.mean(axis=1), rtol=0.0, atol=1e-12)

    np.testing.assert_array_equal(flux, original)  # Ensure input flux is not modified

def test_eight_sigma_kernel_power_transfer_across_full_grid():
    n_pixels = 2499
    dv_sim = 2500.0 / n_pixels
    sigma_kms = 7.96

    impulse = np.zeros(n_pixels, dtype=np.float64)
    impulse[0] = 1.0

    response, dv_out = apply_cos_gaussian(
        impulse,
        dv_sim=dv_sim,
        sigma_kms=sigma_kms,
        truncate=8.0,
    )

    k = 2.0 * np.pi * np.fft.rfftfreq(
        n_pixels,
        d=dv_out,
    )

    measured_transfer = np.abs(np.fft.rfft(response)) ** 2
    expected_transfer = np.exp(-(k * sigma_kms) ** 2)

    np.testing.assert_allclose(
        measured_transfer,
        expected_transfer,
        rtol=1e-6,
        atol=1e-24,
    )
