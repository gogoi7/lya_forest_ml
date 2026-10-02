"""Configuration and noise generation for SNR sweep experiments."""
import hashlib
import json
from pathlib import Path

import h5py
import scipy
import numpy as np
import inspect
import pywt

from . import feature_extraction as features
from . import build_cos_features as cos
from . import instrument
from .manifests import load_manifest

MODELS = ["EX0", "EX1", "EX2", "EX3"]
SNR_LEVELS = [10, 15, 20, 30, 50, 80]
NOISE_SEEDS = [20260929, 20260930, 20261001]
CLASSIFIER_SEED = 42
BINS_PER_RESEL = 2

def sigma_per_bin(snr_resel, bins_per_resel=BINS_PER_RESEL):
    """Continuum noise sigma for two independent bins per nominal resel."""
    snr_resel = float(snr_resel)

    if not np.isfinite(snr_resel) or snr_resel <= 0.0:
        raise ValueError(f"Invalid SNR per resel: {snr_resel}")

    return np.sqrt(bins_per_resel) / snr_resel

def add_noise(flux, snr_resel, *, model, noise_seed):
    """Add independent Gaussian noise after LP1 smoothing and resampling

    Pass all sightlines for a given model in their original order.
    Do not call seperately from train/test split or restart per batch.

    For a given model, seed, shape and LOS order, different SNR levels will reuse the same standard normal draws, scaled by the appropriate sigma_per_bin.
    The input is preserved and noisy flux is not clipped to [0,1] to avoid biasing the noise distribution.
    """
    flux = np.asarray(flux, dtype=np.float64)

    if flux.ndim != 2 or flux.size == 0 or flux.shape[1] < 2:
        raise ValueError("flux must have shape (n_sightlines, n_bins) with at least 2 bins")

    if not np.all(np.isfinite(flux)):
        raise ValueError("flux contains non-finite values")

    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model}")

    if (
        not isinstance(noise_seed, (int, np.integer))
        or isinstance(noise_seed, (bool, np.bool_))
        or noise_seed < 0
    ):
        raise ValueError(f"noise_seed must be a non-negative integer, got {noise_seed}")

    sigma = sigma_per_bin(snr_resel)

    # Seperate streams for each model to avoid accidental correlation between models
    seed_sequence = np.random.SeedSequence([MODELS.index(model), int(noise_seed)])
    rng = np.random.Generator(np.random.PCG64(seed_sequence))

    return flux + sigma * rng.standard_normal(flux.shape)

def _file_sha256(path):
    """Fingerprint file contents without reading into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def cache_lp1_flux(model, root):
    """Build and cache LP1 flux for a given model, using the manifest"""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model}")

    root = Path(root).resolve()
    test_a = root / "outputs/uvb_mean_flux"

    files = {
        "spectra": root / "data/raw" / f"{model}_spectra.hdf5",
        "manifest": test_a / "manifests/z0_b15_seed42.npz",
        "calibration": test_a / "calibration/z0_b15_seed42.json",
        "lsf": root / cos.LP1_LSF_PATH,
        "forward_model_code": Path(cos.__file__),
        "instrument_code": Path(instrument.__file__),
    }
    cache_path = root / "outputs/cos_lsf_snr/snr_sweep/clean_flux" / f"{model}.npz"

    manifest = load_manifest(files["manifest"])
    calibration = json.loads(files["calibration"].read_text())
    fingerprint = str(manifest["los_sha256"].item())
    frozen_fingerprint = "ddffd1d43106cb33e09b7d0f0ae5d9e1db6b8713639ec6d37275dc5b78a4a4c7"

    if not fingerprint == calibration["los_sha256"] == frozen_fingerprint:
        raise RuntimeError("Manifest and calibration fingerprints disagree")

    if (
        calibration["redshift"] != 0.0
        or int(manifest["n_los"]) != 10_000
        or int(manifest["bundle_size"]) != 15
        or int(manifest["seed"]) != 42
        or manifest["bundles"].shape != (666, 15)
        or manifest["train_bundle_ids"].size != 532
        or manifest["test_bundle_ids"].size != 134
        or cos.LP1_N_PIXELS_OUT != 362
    ):
        raise RuntimeError("This experiment currently assumes z=0, 10k LOS, 15 LOS per bundle, 666 bundles, 532 train bundles, 134 test bundles, and 362 LP1 pixels")

    tau_scale = float(calibration["results"][model]["tau_scale"])
    if not np.isfinite(tau_scale) or tau_scale <= 0.0:
        raise ValueError("The frozen tau scale must be positive and finite")

    signature = {
        "cache_schema": 1,
        "model": model,
        "files_sha256": {name: _file_sha256(path) for name, path in files.items()},
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "velocity_length_kms": 2500.0,
        "lsf_wavelength_A": cos.LP1_WAVELENGTH_A,
        "lsf_dispersion_A_per_pixel": cos.LP1_DISPERSION_A_PER_PIXEL,
        "n_pixels_out": cos.LP1_N_PIXELS_OUT,
    }

    cached = cache_path.exists()

    if cached:
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].item())
            if metadata["signature"] != signature:
                raise RuntimeError(f"Incompatible cache: {cache_path}\nExpected signature: {signature}\nFound signature: {metadata['signature']}")
            flux = saved["flux"].copy()
            mean_before = saved["mean_flux_before"].copy()
        dv_out = float(metadata["dv_out"])
    else:
        with h5py.File(files["spectra"], "r") as handle:
            tau = np.asarray(handle[calibration["tau_key"]][:], dtype=np.float64)

        if tau.shape != (10_000, 2499):
            raise ValueError(f"Expected tau shape (10000, 2499), got {tau.shape}")
        if not np.all(np.isfinite(tau)) or np.any(tau < 0):
            raise ValueError("Optical depth must be finite and non-negative")

        flux = np.exp(-tau_scale * tau)
        del tau
        mean_before = flux.mean(axis=1)

        flux, dv_out, metadata = cos.prepare_lp1_flux(flux, dv_sim=2500.0 / 2499, lsf_path=files["lsf"])

        # Remove noise descriptions inherited from old runs
        metadata.pop("pilot_noise_sigma", None)
        metadata.pop("noise_convention", None)
        metadata.update({
            "signature": signature,
            "model": model,
            "redshift": 0.0,
            "los_sha256": fingerprint,
            "tau_scale": tau_scale,
            "dv_out": float(dv_out),
            "n_los": 10_000,
            "n_pixels_in": 2499,
            "n_pixels_out": 362,
            "noise_added": False,
            "noise_sigma": None,
        })

    # Validate both new and cached flux arrays
    if flux.shape != (10_000, 362) or mean_before.shape != (10_000,):
        raise ValueError(f"Expected flux shape (10000, 362) and mean_before shape (10000,), got {flux.shape} and {mean_before.shape}")
    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(mean_before)):
        raise ValueError("Flux and mean_before must be finite")
    if not np.isclose(dv_out, 2500.0 / 362, rtol=0, atol=1e-12):
        raise ValueError(f"Unexpected LP1 output spacing: {dv_out} km/s, expected {2500.0 / 362} km/s")

    mean_error = float(np.max(np.abs(flux.mean(axis=1) - mean_before)))
    if mean_error > 1e-12:
        raise RuntimeError(f"Mean flux conversion failed: {mean_error:.3e}")

    train_los = manifest["bundles"][manifest["train_bundle_ids"]].ravel()
    train_mean = float(mean_before[train_los].mean())
    expected_mean = calibration["results"][model]["matched_train_mean_flux"]

    if not np.isclose(train_mean, expected_mean, rtol=0, atol=1e-12):
        raise RuntimeError(f"Training mean flux disagrees with Test A setup")

    metadata["max_mean_flux_error"] = mean_error

    if not cached:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")

        # Only give a complete file to the user if everything succeeded
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, flux=flux, mean_flux_before=mean_before, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        temp_path.replace(cache_path)

    print(f"{model}: {'reused' if cached else 'saved'} {cache_path.name}")
    return flux, dv_out, metadata

def _array_sha256(values):
    """Fingerprint an array using a consistent float64 representation."""
    values = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_snr_features(model, snr_resel, noise_seed, clean, root):
    """Build or verify one EX model's features for one noisy condition."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")

    snr_resel = float(snr_resel)
    if snr_resel not in SNR_LEVELS:
        raise ValueError(f"snr_resel must be one of {SNR_LEVELS}")
    if noise_seed not in NOISE_SEEDS:
        raise ValueError(f"noise_seed must be one of {NOISE_SEEDS}")
    noise_seed = int(noise_seed)

    root = Path(root).resolve()
    flux, dv_out, clean_meta = clean
    flux = np.asarray(flux, dtype=np.float64)
    dv_out = float(dv_out)

    if clean_meta["model"] != model or clean_meta["noise_added"]:
        raise ValueError("Expected the corresponding model's clean LP1 flux")
    if flux.shape != (10_000, 362) or not np.all(np.isfinite(flux)):
        raise ValueError("Invalid clean LP1 flux array")
    if not np.isclose(dv_out, 2500.0 / 362, rtol=0, atol=1e-12):
        raise ValueError("Unexpected output velocity spacing")

    manifest_path = (
        root / "outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz"
    )
    manifest = load_manifest(manifest_path)
    fingerprint = str(manifest["los_sha256"].item())

    if (
        _file_sha256(manifest_path)
        != clean_meta["signature"]["files_sha256"]["manifest"]
        or fingerprint != clean_meta["los_sha256"]
    ):
        raise RuntimeError("Manifest changed since clean-flux preparation")

    sigma = sigma_per_bin(snr_resel)
    stage = f"snr{int(snr_resel)}_seed{noise_seed}"
    output_path = (
        root / "outputs/cos_lsf_snr/snr_sweep/features"
        / stage / f"z0_b15_{model}.npz"
    )

    # Hash relevant functions only, so later additions to this module
    # do not invalidate completed feature files.
    sweep_code = "\n\n".join(
        inspect.getsource(function)
        for function in (sigma_per_bin, add_noise, build_snr_features)
    )
    signature = {
        "feature_schema": 1,
        "model": model,
        "snr_resel": snr_resel,
        "noise_seed": noise_seed,
        "seed_entropy": [MODELS.index(model), noise_seed],
        "rng": "PCG64",
        "bins_per_resel": BINS_PER_RESEL,
        "noise_sigma": float(sigma),
        "dv_out": dv_out,
        "clean_signature": clean_meta["signature"],
        "clean_flux_sha256": _array_sha256(flux),
        "feature_code_sha256": _file_sha256(features.__file__),
        "sweep_code_sha256": hashlib.sha256(
            sweep_code.encode("utf-8")
        ).hexdigest(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywavelets": pywt.__version__,
        },
    }

    cached = output_path.exists()

    if cached:
        with np.load(output_path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].item())

            if metadata["signature"] != signature:
                raise RuntimeError(f"Incompatible feature cache: {output_path}")
            if str(saved["los_sha256"].item()) != fingerprint:
                raise RuntimeError("Feature/manifest fingerprints disagree")

            for key in ("bundles", "train_bundle_ids", "test_bundle_ids"):
                np.testing.assert_array_equal(saved[key], manifest[key])

            los_features = saved["matched_los"].copy()
            bundled_features = saved["matched_bundles"].copy()

    else:
        print(f"{stage}/{model}: extracting features", flush=True)

        noisy_flux = add_noise(
            flux,
            snr_resel,
            model=model,
            noise_seed=noise_seed,
        )
        negative_fraction = float(np.mean(noisy_flux < 0.0))

        los_features = features.extract_combined_features(noisy_flux, dv_out)
        del noisy_flux
        bundled_features = features.bundle_features(
            los_features, manifest["bundles"]
        )

        metadata = {
            **clean_meta,
            "signature": signature,
            "schema_version": 1,
            "condition": "matched",
            "stage": stage,
            "bundle_size": 15,
            "n_los_features": 46,
            "n_bundle_features": 92,
            "snr_resel": snr_resel,
            "snr_per_bin": float(1.0 / sigma),
            "bins_per_resel": BINS_PER_RESEL,
            "noise_added": True,
            "noise_sigma": float(sigma),
            "noise_seed": noise_seed,
            "noise_model": "iid_gaussian_continuum",
            "noise_convention": (
                "unit continuum; two independent output bins per nominal resel"
            ),
            "paired_noise_across_snr": True,
            "flux_clipped": False,
            "negative_flux_fraction": negative_fraction,
            "mean_flux_check_stage": "before_noise",
            "tau_proxy_log_policy": "log input = max(flux, 0) + 1e-10",
        }

    # Validate newly computed arrays and completed files on every call.
    for key, array, expected_shape in (
        ("matched_los", los_features, (10_000, 46)),
        ("matched_bundles", bundled_features, (666, 92)),
    ):
        if array.shape != expected_shape or not np.all(np.isfinite(array)):
            raise RuntimeError(f"{model}/{stage}: invalid {key}")

        digest = _array_sha256(array)
        if cached and metadata[f"{key}_sha256"] != digest:
            raise RuntimeError(f"{model}/{stage}: {key} checksum differs")
        metadata[f"{key}_sha256"] = digest

    if not cached:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".tmp")

        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                matched_los=los_features,
                matched_bundles=bundled_features,
                los_sha256=np.array(fingerprint),
                bundles=manifest["bundles"],
                train_bundle_ids=manifest["train_bundle_ids"],
                test_bundle_ids=manifest["test_bundle_ids"],
                metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            )
        temporary_path.replace(output_path)

    print(f"{stage}/{model}: {'reused' if cached else 'saved'}", flush=True)
    return output_path, metadata