"""Build instrument-processed features using uvb mean flux calibration and COS LSF."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .feature_extraction import extract_combined_features, bundle_features
from .instrument import apply_cos_gaussian, resample_flux
from .manifests import load_manifest


MODELS = ("EX0", "EX1", "EX2", "EX3")
STAGES = ("gaussian", "gaussian_rebinned")

SIGMA_KMS = 7.96
N_PIXELS_OUT = 1249

TEST_A_ROOT = Path("outputs/uvb_mean_flux")
OUTPUT_ROOT = Path("outputs/cos_lsf_snr")


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

    flux, dv_out = apply_cos_gaussian(
        flux,
        dv_sim=dv_sim,
        sigma_kms=SIGMA_KMS,
    )

    if stage == "gaussian_rebinned":
        flux, dv_out = resample_flux(
            flux,
            dv_in=dv_out,
            n_pixels_out=N_PIXELS_OUT,
        )

    if not np.allclose(
        flux.mean(axis=-1),
        mean_flux_before,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Instrument processing changed the mean flux")

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
        "sigma_kms": SIGMA_KMS,
        "noise_added": False,
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
        description="Build matched Gaussian-LSF feature files.",
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", required=True, choices=STAGES)

    args = parser.parse_args()
    build_cos_features(args.model, args.stage)


if __name__ == "__main__":
    main()