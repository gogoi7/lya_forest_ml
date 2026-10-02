"""Build instrument-processed features using uvb mean flux calibration and COS LSF."""

import argparse
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np

from .feature_extraction import extract_combined_features, bundle_features
from .instrument import apply_cos_gaussian, resample_flux, apply_cos_lsf, load_cos_lsf, resample_lsf
from .manifests import load_manifest

C_KMS = 299792.458

MODELS = ("EX0", "EX1", "EX2", "EX3")

SIGMA_KMS = 7.96
N_PIXELS_OUT = 1249

TEST_A_ROOT = Path("outputs/uvb_mean_flux")
OUTPUT_ROOT = Path("outputs/cos_lsf_snr")

TRUNCATE_BY_STAGE = {
    "gaussian": 4.0,
    "gaussian_rebinned": 4.0,
    "gaussian_truncate8": 8.0,
    "gaussian_rebinned_truncate8": 8.0,
    "gaussian_rebinned_truncate8_snr10000": 8.0,
}
LP1_STAGES = ("lp1_obs_clean", "lp1_obs_noisy")
STAGES = tuple(TRUNCATE_BY_STAGE) + LP1_STAGES

LP1_LSF_PATH = Path("data/reference/cos_lsf/aa_LSFTable_G130M_1291_LP1_cn.dat")
LP1_WAVELENGTH_A = 1300.0
LP1_DISPERSION_A_PER_PIXEL = 0.00997  # Nominal G130M dispersion
LP1_N_PIXELS_OUT = 362
LP1_NOISE_SIGMA = 0.07

def prepare_lp1_flux(flux, dv_sim, lsf_path=LP1_LSF_PATH):
    """Apply the LP1 LSF and resample to the LP1 pixel grid."""
    lsf_path = Path(lsf_path)
    wavelengths, kernels = load_cos_lsf(lsf_path)

    index = int(np.argmin(np.abs(wavelengths - LP1_WAVELENGTH_A)))
    wavelength = float(wavelengths[index])

    dv_lsf = C_KMS * LP1_DISPERSION_A_PER_PIXEL / wavelength
    kernel = resample_lsf(kernels[:, index], dv_lsf, dv_sim)

    # Convolve each sightline with the LP1 LSF kernel. Convolve on the simulation pixel grid, then resample to the LP1 pixel grid.
    flux, dv_out = apply_cos_lsf(flux, dv_sim, kernel)
    flux, dv_out = resample_flux(flux, dv_out, n_pixels_out=LP1_N_PIXELS_OUT)

    lsf_metadata = {
        "lsf_model": "tabulated_cos",
        "lsf_lifetime_position": 1,
        "lsf_grating": "G130M",
        "lsf_cenwave": 1291,
        "lsf_wavelength_A": wavelength,
        "lsf_dispersion_A_per_pixel": LP1_DISPERSION_A_PER_PIXEL,
        "lsf_configuration_provisional": True,
        "lsf_file": str(lsf_path),
        "lsf_sha256": hashlib.sha256(lsf_path.read_bytes()).hexdigest(),
        "lsf_resampling": "PCHIP cumulative native-pixel weights",
        "lsf_kernel_pixels": int(kernel.size),
        "observation_reference": "pg1048_all.dat",
        "observation_range_A": [1220.0, 1380.0],
        "sampling_convention": "median valid adjacent pixel velocity spacing",
        "pilot_noise_sigma": LP1_NOISE_SIGMA,
        "noise_convention": "median reported error over valid pixels",
    }

    return flux, dv_out, lsf_metadata


def build_cos_features(model, stage):
    """Generate one model's features for one instrument-processing stage."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")

    output_path = (
        OUTPUT_ROOT / "features" / stage / f"z0_b15_{model}.npz"
    )

    if output_path.exists():
        raise FileExistsError(f"Feature file already exists: {output_path}")

    manifest_path = TEST_A_ROOT / "manifests" / "z0_b15_seed42.npz"
    calibration_path = TEST_A_ROOT / "calibration" / "z0_b15_seed42.json"

    manifest = load_manifest(manifest_path)

    with calibration_path.open() as handle:
        calibration = json.load(handle)

    fingerprint = str(manifest["los_sha256"].item())

    if calibration["los_sha256"] != fingerprint:
        raise RuntimeError("Calibration and manifest fingerprints disagree")

    if calibration["redshift"] != 0.0:
        raise ValueError("This experiment currently assumes z = 0")

    tau_scale = float(calibration["results"][model]["tau_scale"])

    if not np.isfinite(tau_scale) or tau_scale <= 0.0:
        raise ValueError("The frozen tau scale must be positive and finite")

    input_path = Path("data/raw") / f"{model}_spectra.hdf5"

    with h5py.File(input_path, "r") as handle:
        tau = np.asarray(
            handle[calibration["tau_key"]][:],
            dtype=np.float64,
        )

    if tau.ndim != 2:
        raise ValueError("Optical depth must have shape (n_los, n_pixels)")

    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise ValueError("Optical depth must be finite and non-negative")

    n_los, n_pixels_in = tau.shape

    if n_los != int(manifest["n_los"].item()):
        raise ValueError("Input sightline count does not match the manifest")

    # At z = 0, the 25 cMpc/h box spans 2500 km/s.
    dv_sim = 2500.0 / n_pixels_in

    # Apply the already-fitted Test A scale to every sightline.
    flux = np.exp(-tau_scale * tau)
    del tau

    mean_flux_before = flux.mean(axis=-1)
    if stage in LP1_STAGES:
        truncate = None
        flux, dv_out, lsf_metadata = prepare_lp1_flux(flux, dv_sim)
    else:
        truncate = TRUNCATE_BY_STAGE[stage]
        lsf_metadata = {"lsf_model": "gaussian"}

        flux, dv_out = apply_cos_gaussian(
            flux,
            dv_sim=dv_sim,
            sigma_kms=SIGMA_KMS,
            truncate=truncate,
        )

        if stage in (
            "gaussian_rebinned",
            "gaussian_rebinned_truncate8",
            "gaussian_rebinned_truncate8_snr10000",
        ):
            flux, dv_out = resample_flux(
                flux,
                dv_in=dv_out,
                n_pixels_out=N_PIXELS_OUT,
            )

    # Check conservation before adding noise.
    if not np.allclose(
        flux.mean(axis=-1),
        mean_flux_before,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Instrument processing changed the mean flux")

    noise_sigma = None
    noise_seed = None

    if stage == "gaussian_rebinned_truncate8_snr10000":
        noise_sigma = 1.0 / 10000.0
        noise_seed = 20260914 + MODELS.index(model)
    elif stage == "lp1_obs_noisy":
        noise_sigma = LP1_NOISE_SIGMA
        noise_seed = 20260921 + MODELS.index(model)

    snr_per_pixel = None if noise_sigma is None else 1.0 / noise_sigma

    if noise_sigma is not None:
        if np.any(flux < 0.0):
            raise RuntimeError("Expected nonnegative clean flux")

        rng = np.random.default_rng(noise_seed)
        flux = flux + rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=flux.shape,
        )
    negative_flux_fraction = float(np.mean(flux < 0.0))

    print(
        f"{model}: negative-flux fraction = "
        f"{negative_flux_fraction:.6e}"
    )

    # Features must use the actual output pixel width.
    los_features = extract_combined_features(flux, dv_out)

    if not np.all(np.isfinite(los_features)):
        raise RuntimeError("Feature extraction produced non-finite values")

    bundled_features = bundle_features(
        los_features,
        manifest["bundles"],
    )

    metadata = {
        "schema_version": 1,
        "model": model,
        "condition": "matched",
        "stage": stage,
        "redshift": 0.0,
        "sigma_kms": None if stage in LP1_STAGES else SIGMA_KMS,
        "truncate": truncate,
        **lsf_metadata,
        "noise_sigma": noise_sigma,
        "noise_added": snr_per_pixel is not None,
        "snr_per_pixel": snr_per_pixel,
        "noise_seed": noise_seed,
        "noise_model": (
            "iid_gaussian_continuum"
            if snr_per_pixel is not None
            else None
        ),
        "negative_flux_fraction": negative_flux_fraction,
        "tau_proxy_log_policy": "log input = max(flux, 0) + 1e-10",
        "tau_scale": tau_scale,
        "dv_sim": dv_sim,
        "dv_out": dv_out,
        "n_pixels_in": n_pixels_in,
        "n_pixels_out": flux.shape[-1],
        "n_los": n_los,
        "manifest": str(manifest_path),
        "calibration": str(calibration_path),
        "los_sha256": fingerprint,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        matched_los=los_features,
        matched_bundles=bundled_features,
        los_sha256=np.array(fingerprint),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )

    print(f"Model: {model}; stage: {stage}")
    print(f"LOS features: {los_features.shape}")
    print(f"Bundled features: {bundled_features.shape}")
    print(f"Pixels: {n_pixels_in} -> {flux.shape[-1]}")
    print(f"dv_out: {dv_out:.10f} km/s")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build matched Gaussian or tabulated COS-LSF feature files.",
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", required=True, choices=STAGES)

    args = parser.parse_args()
    build_cos_features(args.model, args.stage)


if __name__ == "__main__":
    main()