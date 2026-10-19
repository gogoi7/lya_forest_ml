"""Compact Sujin export; run in the corrected strong-absorber notebook kernel.

Run the existing bootstrap and plotting cells first. This replaces the old
export cell. It never fits models, extracts features, or changes input caches.
"""

def _sujin_sha256(path):
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sujin_array_hash(array):
    import hashlib
    import numpy as np
    a = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def _sujin_trough_extent(below, los, centers, half_window):
    """Periodic connected-trough sizes and aperture containment at given centers.

    Uses nearest recovery bins in a doubled sightline. A whole-period trough
    has n_bins equal to the sightline length and never fits a shorter window.
    """
    import numpy as np
    below = np.asarray(below, dtype=bool)
    n = below.shape[1]
    if not np.all(below[los, centers]):
        raise ValueError("A retained center is not below the recovery threshold.")
    doubled = np.concatenate([below, below], axis=1)
    index = np.arange(2 * n, dtype=np.int32)[None, :]
    previous = np.maximum.accumulate(np.where(doubled, -n, index), axis=1)
    left = n + centers - previous[los, n + centers]
    del previous
    following = np.minimum.accumulate(
        np.where(doubled, 3 * n, index)[:, ::-1], axis=1
    )[:, ::-1]
    right = following[los, centers] - centers
    whole = np.all(below, axis=1)[los]
    width = np.minimum(left + right - 1, n).astype(np.uint16)
    fits = (left <= half_window + 1) & (right <= half_window + 1) & ~whole
    if np.any(width < 1) or np.any(width > n):
        raise RuntimeError("Invalid periodic trough extent.")
    return width, fits, whole


def _sujin_markdown_table(frame):
    import numpy as np
    import pandas as pd
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def export_sujin_windows():
    import hashlib
    import inspect
    import json
    import shutil
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scipy

    required = (
        "ROOT", "OUT", "catalog", "MODELS", "SNRS", "NOISE_SEEDS", "sweep",
        "manifest_path", "bundles", "train_ids", "test_ids", "los_split", "los_bundle",
        "dv", "offsets", "velocity", "blue_width", "red_width", "source_metadata",
        "FLUX_CUT", "SIGMA_CUT", "RECOVERY_FLUX", "N_TARGET", "N_BOOT", "BOOT_SEED",
        "count", "profile_sum", "clean_profile_sum", "rates", "rate_contrasts",
        "seed_counts", "asymmetry_summary", "stack_rows", "DEPTH_LABELS",
        "smooth_three", "local_minima", "merge_minima", "detect_minima", "get_windows",
        "depth_bin", "window_geometry", "equivalent_widths", "safe_divide",
        "bootstrap_sum", "percentile_bounds", "stack_bootstrap", "display",
    )
    absent = [name for name in required if name not in globals()]
    if absent:
        raise RuntimeError("Run corrected setup/catalog plus bootstrap and plots first. Missing: " + ", ".join(absent))
    if tuple(MODELS) != ("EX0", "EX1", "EX2", "EX3"):
        raise RuntimeError("Expected EX0, EX1, EX2, EX3 in the frozen order.")
    if tuple(NOISE_SEEDS) != (20260929, 20260930, 20261001):
        raise RuntimeError("Unexpected noise seeds/order.")
    if tuple(SNRS) != (10, 15, 20, 30, 50, 80):
        raise RuntimeError("Keep the original six S/N values for the notebook analysis.")
    if (FLUX_CUT, SIGMA_CUT, RECOVERY_FLUX) != (0.85, 3.0, 0.85) or N_TARGET is not None:
        raise RuntimeError("Expected recovery F=0.85, original significance cut, and N_TARGET=None.")
    if OUT.parent != ROOT / "outputs/cos_lsf_snr/strong_absorbers" or not OUT.name.startswith("recovery085_"):
        raise RuntimeError("Use the new recovery085_<timestamp> OUT directory.")
    np.testing.assert_array_equal(offsets, np.arange(-43, 44))
    np.testing.assert_allclose(dv, 2500.0 / 362, rtol=0, atol=1e-12)

    destination = OUT / "sujin_snr20_50"
    if destination.exists():
        raise FileExistsError(f"Completed export already exists; preserving it: {destination}")
    split_path = ROOT / "outputs/cos_lsf_snr/perchunk_tuned/grouped_split.npz"
    signature_path = ROOT / "outputs/cos_lsf_snr/perchunk_tuned/training_signature.json"
    script_path = ROOT / "notebooks/sujin_export_step.py"
    qa_dir = OUT / "window_qa"
    qa_names = (
        "window_examples.csv", "window_examples_snr20.png", "window_examples_snr50.png",
        "qa_settings.json", "native_source_audit.csv", "whole_los_clean_census.csv",
    )
    input_paths = [manifest_path, split_path, signature_path, script_path]
    input_paths += [ROOT / "outputs/cos_lsf_snr/snr_sweep/clean_flux" / f"{m}.npz" for m in MODELS]
    input_paths += [qa_dir / name for name in qa_names]
    missing = [str(p) for p in input_paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Required saved inputs/QA are missing:\n" + "\n".join(missing))
    input_hashes = {str(p.relative_to(ROOT)): _sujin_sha256(p) for p in input_paths}
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if signature.get("manifest_sha256") != _sujin_sha256(manifest_path):
        raise RuntimeError("Per-chunk training signature refers to a different manifest.")
    if signature.get("models") != list(MODELS) or signature.get("seed") != 20261011:
        raise RuntimeError("Unexpected saved per-chunk training configuration.")
    training_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    role_names = ("train", "validation", "calibration", "test")
    split_keys = ("fit_los", "tune_los", "calibration_los", "test_los")
    expected_counts = (6384, 798, 798, 2010)
    with np.load(split_path, allow_pickle=False) as saved:
        needed = set(split_keys) | {"test_bundle_ids", "seed"}
        if not needed.issubset(saved.files):
            raise RuntimeError("grouped_split.npz lacks required arrays.")
        split = {key: saved[key].copy() for key in needed}
    for key, number in zip(split_keys, expected_counts):
        a = split[key]
        if a.shape != (number,) or a.dtype.kind not in "iu" or np.any((a < 0) | (a >= 10000)):
            raise RuntimeError(f"Invalid saved LOS array: {key}")
    if split["seed"].shape != () or int(split["seed"]) != 20261011:
        raise RuntimeError("Saved split seed changed.")
    all_los = np.concatenate([split[key] for key in split_keys])
    if np.unique(all_los).size != all_los.size:
        raise RuntimeError("A parent sightline appears in multiple roles.")
    np.testing.assert_array_equal(np.sort(all_los), np.sort(bundles.ravel()))
    np.testing.assert_array_equal(split["test_bundle_ids"], np.sort(test_ids))
    np.testing.assert_array_equal(split["test_los"], bundles[np.sort(test_ids)].ravel())
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split[key] for key in split_keys[:3]])),
        np.sort(bundles[train_ids].ravel()),
    )
    split_hash = hashlib.sha256(all_los.astype("<i8").tobytes()).hexdigest()
    if split_hash != signature.get("split_sha256"):
        raise RuntimeError("Saved roles disagree with the per-chunk training signature.")
    los_role = np.full(10000, -1, dtype=np.int8)
    for code, key in enumerate(split_keys):
        los_role[split[key]] = code
    np.testing.assert_array_equal(np.flatnonzero(los_role == 1), np.sort(split["tune_los"]))

    n_catalog = len(catalog["labels"])
    bad = {key: len(value) for key, value in catalog.items() if len(value) != n_catalog}
    if bad:
        raise RuntimeError(f"Catalog row lengths disagree: {bad}")
    if np.asarray(catalog["windows"]).shape != (n_catalog, 87):
        raise RuntimeError("Expected 87-bin windows.")
    for name, lower, upper in (("labels", 0, 4), ("los_id", 0, 10000),
                               ("center_pixel", 0, 362), ("n_merged", 1, 363)):
        a = np.asarray(catalog[name])
        if a.dtype.kind not in "iu" or np.any((a < lower) | (a >= upper)):
            raise RuntimeError(f"Invalid catalog values: {name}")
    np.testing.assert_array_equal(catalog["split"], los_split[catalog["los_id"]])
    np.testing.assert_array_equal(catalog["bundle_id"], los_bundle[catalog["los_id"]])
    selected = np.flatnonzero(np.isin(catalog["snr_resel"], (20, 50)))
    if selected.size == 0:
        raise RuntimeError("No S/N 20 or 50 rows.")
    seed_lookup = {int(seed): i for i, seed in enumerate(NOISE_SEEDS)}
    selected_seeds = np.asarray(catalog["noise_seed"])[selected]
    if not set(np.unique(selected_seeds)).issubset(seed_lookup):
        raise RuntimeError("Unexpected noise seed in selected rows.")
    seed_index = np.asarray([seed_lookup[int(s)] for s in selected_seeds], dtype=np.uint64)
    labels = np.asarray(catalog["labels"])[selected].astype(np.uint64)
    snr_values = np.asarray(catalog["snr_resel"])[selected].astype(np.uint64)
    parent = np.asarray(catalog["los_id"])[selected].astype(np.uint64)
    center = np.asarray(catalog["center_pixel"])[selected].astype(np.uint64)
    # ID schema 1: version/model, SNR, seed index, parent LOS, center pixel.
    ids = (((((4 + labels) * 128 + snr_values) * 4 + seed_index) * 16384 + parent) * 512 + center)
    if np.unique(ids).size != selected.size:
        raise RuntimeError("Duplicate window identity (model, SNR, seed, LOS, center).")
    ordering = np.argsort(ids, kind="stable")
    selected, ids = selected[ordering], ids[ordering]
    n = len(selected)
    dtypes = {
        "windows": "<f4", "labels": "u1", "los_id": "<i4", "bundle_id": "<i2",
        "center_pixel": "<i2", "snr_resel": "<i2", "noise_seed": "<i4",
        "sigma_bin": "<f4", "fmin": "<f8", "depth": "<f8", "depth_bin": "u1",
        "n_merged": "<u2", "wraps_boundary": "?",
    }
    rows = {key: np.asarray(catalog[key])[selected].astype(dtype, copy=True) for key, dtype in dtypes.items()}
    rows["window_id"] = ids.astype("<u8")
    rows["group_id"] = rows["los_id"]
    if np.any(los_role[rows["los_id"]] < 0):
        raise RuntimeError("An exported sightline is absent from the saved roles.")
    rows["role"] = los_role[rows["los_id"]].astype(np.uint8)
    rows["trough_n_bins"] = np.zeros(n, dtype=np.uint16)
    rows["trough_fits_window"] = np.zeros(n, dtype=bool)
    rows["whole_los_below_recovery"] = np.zeros(n, dtype=bool)
    for key in ("windows", "fmin", "depth", "sigma_bin"):
        if not np.all(np.isfinite(rows[key])):
            raise RuntimeError(f"Non-finite values in export: {key}")
    np.testing.assert_array_equal(rows["depth"], 1 - rows["fmin"])
    np.testing.assert_array_equal(rows["depth_bin"], depth_bin(rows["fmin"]))
    if np.any(rows["fmin"] >= FLUX_CUT):
        raise RuntimeError("An exported center fails the detection flux cut.")

    # Reject stale notebook statistics before saving anything.
    for si, snr in enumerate(SNRS):
        for mi, model in enumerate(MODELS):
            expected = count[si, :, mi].sum(axis=(1, 2)).mean()
            found = rates[(rates.SNR == snr) & (rates.model == model) & (rates.depth_label == "all")]
            if len(found) != 1 or not np.isclose(float(found.iloc[0].mean_n_test_absorbers), expected, rtol=0, atol=1e-10):
                raise RuntimeError("Bootstrap outputs are stale; rerun the bootstrap and plotting cells.")
    # Recreate noisy arrays in their original LOS order; verify all exported flux.
    verified = np.zeros(n, dtype=bool)
    clean_metadata = {}
    print(f"Preparing {n:,} windows; verifying source flux and adding trough flags...", flush=True)
    for mi, model in enumerate(MODELS):
        clean, cached_dv, meta = sweep.cache_lp1_flux(model, ROOT)
        if clean.shape != (10000, 362) or meta["signature"] != source_metadata[model]["signature"]:
            raise RuntimeError(f"Clean input changed after catalog construction: {model}")
        np.testing.assert_allclose(cached_dv, dv, rtol=0, atol=1e-12)
        clean_metadata[model] = meta
        for snr in (20, 50):
            sigma = float(sweep.sigma_per_bin(snr))
            for seed in NOISE_SEEDS:
                take = np.flatnonzero((rows["labels"] == mi) & (rows["snr_resel"] == snr)
                                      & (rows["noise_seed"] == seed))
                if not take.size:
                    raise RuntimeError(f"Missing export condition: {model}, S/N {snr}, seed {seed}")
                np.testing.assert_array_equal(rows["sigma_bin"][take], np.full(take.size, sigma, dtype=np.float32))
                noisy = sweep.add_noise(clean, snr, model=model, noise_seed=int(seed))
                fs = (np.roll(noisy, 1, axis=1) + noisy + np.roll(noisy, -1, axis=1)) / 3
                ll, cc = rows["los_id"][take].astype(int), rows["center_pixel"][take].astype(int)
                reproduced = noisy[ll[:, None], (cc[:, None] + offsets[None, :]) % 362].astype(np.float32)
                np.testing.assert_array_equal(reproduced, rows["windows"][take])
                np.testing.assert_allclose(fs[ll, cc], rows["fmin"][take], rtol=0, atol=1e-12)
                if np.any((1 - fs[ll, cc]) / (sigma / np.sqrt(3)) < SIGMA_CUT - 1e-12):
                    raise RuntimeError("An exported center fails the significance cut.")
                width, fits, whole = _sujin_trough_extent(fs < RECOVERY_FLUX, ll, cc, half_window=43)
                rows["trough_n_bins"][take] = width
                rows["trough_fits_window"][take] = fits
                rows["whole_los_below_recovery"][take] = whole
                if np.any(rows["n_merged"][take] > width):
                    raise RuntimeError("More merged minima than bins in their trough.")
                verified[take] = True
                del noisy, fs, reproduced
        del clean
        print(f"Verified both S/N levels and all three seeds: {model}", flush=True)
    if not verified.all():
        raise RuntimeError("Some export rows were not checked.")

    frame = pd.DataFrame({
        "snr": rows["snr_resel"], "role": np.asarray(role_names)[rows["role"]],
        "model": np.asarray(MODELS)[rows["labels"]], "noise_seed": rows["noise_seed"],
        "los_id": rows["los_id"], "n_merged": rows["n_merged"],
        "whole_los": rows["whole_los_below_recovery"],
        "trough_outside_window": ~rows["trough_fits_window"],
    })
    if frame.groupby("los_id").role.nunique().max() != 1:
        raise RuntimeError("A sightline occurs in multiple exported roles.")
    def counts_table(keys):
        return frame.groupby(keys, sort=False).agg(
            n_windows=("los_id", "size"), n_los_with_windows=("los_id", "nunique"),
            n_merged_windows=("n_merged", lambda v: int((v > 1).sum())),
            n_whole_los_windows=("whole_los", "sum"),
            n_troughs_outside_window=("trough_outside_window", "sum"),
        ).reset_index()
    counts = counts_table(["snr", "role", "model"])
    seed_counts_export = counts_table(["snr", "role", "model", "noise_seed"])
    role_counts = pd.DataFrame([
        {"role": name, "saved_split_key": key, "assigned_los": len(split[key]),
         "los_with_exported_windows": int(np.unique(rows["los_id"][rows["role"] == code]).size)}
        for code, (name, key) in enumerate(zip(role_names, split_keys))
    ])
    role_counts["los_without_exported_windows"] = role_counts.assigned_los - role_counts.los_with_exported_windows
    order_index = pd.MultiIndex.from_product([(20, 50), role_names], names=["snr", "role"])
    meeting = counts.pivot(index=["snr", "role"], columns="model", values="n_windows")
    meeting = meeting.reindex(index=order_index, columns=list(MODELS), fill_value=0).fillna(0).astype(int)
    meeting["total"] = meeting.sum(axis=1)
    meeting = meeting.reset_index()
    flags = frame.groupby("snr").agg(
        n_windows=("los_id", "size"), n_whole_los_windows=("whole_los", "sum"),
        n_troughs_outside_window=("trough_outside_window", "sum"),
    ).reset_index()
    source_audit = pd.read_csv(qa_dir / "native_source_audit.csv")
    source_census = pd.read_csv(qa_dir / "whole_los_clean_census.csv")
    if len(source_audit) != 2 or np.any(source_audit.rebuild_max_abs_error > 1e-12):
        raise RuntimeError("Native source audit is incomplete or failed.")
    source_totals = source_census.groupby("model", as_index=False).agg(
        n_los=("n_los", "sum"), whole_los_below_085_clean=("whole_los_below_085_clean", "sum")
    )
    source_totals["percent"] = 100 * source_totals.whole_los_below_085_clean / source_totals.n_los

    helpers = (smooth_three, local_minima, merge_minima, detect_minima, get_windows,
               depth_bin, window_geometry, equivalent_widths, safe_divide, bootstrap_sum,
               percentile_bounds, stack_bootstrap)
    helper_source = "\n\n".join(inspect.getsource(f) for f in helpers)
    settings = {
        "schema_version": 2, "detector_version": "recovery085_v1", "id_schema_version": 1,
        "models": list(MODELS), "export_snrs": [20, 50], "analysis_snrs": list(SNRS),
        "noise_seeds": [int(s) for s in NOISE_SEEDS], "dv_kms": float(dv),
        "smoothing": "Arithmetic three-bin mean, periodic on mock sightlines",
        "flux_cut_strict": float(FLUX_CUT), "significance_cut_inclusive": float(SIGMA_CUT),
        "recovery_flux_inclusive": float(RECOVERY_FLUX),
        "merge_rule": "One deepest eligible minimum per connected smoothed-F<0.85 region; join circular ends; ties by smallest pixel index",
        "n_merged": "Number of flux- and significance-qualified minima in that region, including the retained minimum",
        "flux": "Unsmoothed noisy continuum-normalized flux; unclipped; 87 float32 bins",
        "noise": "Independent Gaussian per output bin, sigma=sqrt(2)/SNR; paired standard-normal draws across SNR for each model/seed",
        "window_center_range_kms": [float(velocity[0]), float(velocity[-1])],
        "ew_support_kms": [-300.0, 300.0],
        "role_names": list(role_names), "role_split_keys": dict(zip(role_names, split_keys)),
        "calibration_policy": "Reserved 798 original calibration LOS; excluded from fitting and tuning",
        "grouping": "group_id=los_id, shared across every model, SNR, seed and window",
        "saturated_policy": "Retain all saturated/broad/whole-period troughs; supply observed-noisy-flux flags; no new rejection cut",
        "trough_n_bins": "Count of connected bins below recovery in the periodic smoothed noisy sightline; width=trough_n_bins*dv_kms",
        "trough_fits_window": "All bins of the selected trough are inside the stored 87-bin aperture; other troughs may also appear in that aperture",
        "whole_los_below_recovery": "All 362 smoothed noisy bins are below 0.85; recomputed separately for each SNR/seed",
        "window_id_formula": "(((((4+model_code)*128+snr_resel)*4+noise_seed_index)*16384+los_id)*512+center_pixel)",
        "window_id_scope": "Stable under row reordering/reruns with the same enumeration and inputs; use dataset_id plus window_id across releases; IDs are not model features",
        "bootstrap": {"draws": int(N_BOOT), "seed": int(BOOT_SEED), "n_target": None,
                      "pool": "2010 held-out LOS/model in the original 134 bundles; shared paired bundle resamples; existing seed treatment unchanged"},
    }
    constants = {
        "velocity_kms": np.asarray(velocity, dtype="<f8"),
        "ew_blue_bin_width_kms": np.asarray(blue_width, dtype="<f8"),
        "ew_red_bin_width_kms": np.asarray(red_width, dtype="<f8"),
        "model_names": np.asarray(MODELS), "role_names": np.asarray(role_names),
        "depth_names": np.asarray(DEPTH_LABELS), "snr_levels": np.array([20, 50], dtype="<i2"),
        "noise_seeds": np.asarray(NOISE_SEEDS, dtype="<i4"),
        "los_role": los_role, "los_bundle": np.asarray(los_bundle, dtype="<i2"),
        "manifest_bundles": np.asarray(bundles, dtype="<i4"),
        "test_bundle_ids": np.asarray(test_ids, dtype="<i2"),
    }
    arrays = {**rows, **constants}
    array_hashes = {key: _sujin_array_hash(value) for key, value in arrays.items()}
    helper_hash = hashlib.sha256(helper_source.encode()).hexdigest()
    identity = {"settings": settings, "array_sha256": array_hashes,
                "training_key": training_key, "helper_source_sha256": helper_hash,
                "clean_signatures": {m: clean_metadata[m]["signature"] for m in MODELS}}
    dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    provenance = {
        "dataset_id": dataset_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "n_windows": n, "input_sha256": input_hashes,
        "cached_role_training_key": training_key, "cached_split_sha256": split_hash,
        "clean_metadata": clean_metadata, "helper_source_sha256": helper_hash,
        "array_sha256": array_hashes,
        "schema": {key: {"shape": list(a.shape), "dtype": str(a.dtype), "per_window": key in rows}
                   for key, a in arrays.items()},
        "role_counts": role_counts.to_dict(orient="records"),
        "window_counts": counts.to_dict(orient="records"),
        "flag_counts": flags.to_dict(orient="records"),
        "native_source_audit": source_audit.to_dict(orient="records"),
        "clean_whole_los_census": source_census.to_dict(orient="records"),
        "validation": {"all_exported_flux_reproduced": True, "disjoint_parent_roles": True,
                       "validation_equals_saved_tune_los": True, "no_windows_removed_for_saturation": True},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "analysis_input_hashes": {"test_bundle_counts": _sujin_array_hash(count),
                                  "profile_sum": _sujin_array_hash(profile_sum),
                                  "clean_profile_sum": _sujin_array_hash(clean_profile_sum)},
    }
    metadata_json = json.dumps(provenance, indent=2, allow_nan=False)
    arrays["metadata_json"] = np.array(metadata_json)
    arrays["dataset_id"] = np.array(dataset_id)

    archive_name = "sujin_strong_absorber_windows_snr20_50.npz"
    stage = Path(tempfile.mkdtemp(prefix="sujin_export_tmp_", dir=OUT))
    try:
        archive = stage / archive_name
        print("Writing compact archive...", flush=True)
        np.savez_compressed(archive, **arrays)
        # Read back every saved array and check dtype, shape and content.
        with np.load(archive, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise RuntimeError("Saved archive has missing/unexpected fields.")
            for key, expected_hash in array_hashes.items():
                if _sujin_array_hash(saved[key]) != expected_hash:
                    raise RuntimeError(f"Archive readback mismatch: {key}")
            if str(saved["dataset_id"].item()) != dataset_id or str(saved["metadata_json"].item()) != metadata_json:
                raise RuntimeError("Saved provenance does not match.")
        archive_hash = _sujin_sha256(archive)
        size_mb, size_mib = archive.stat().st_size / 1e6, archive.stat().st_size / 2**20
        counts.to_csv(stage / "window_counts_snr_role_model.csv", index=False)
        seed_counts_export.to_csv(stage / "window_counts_snr_role_model_seed.csv", index=False)
        role_counts.to_csv(stage / "role_sightline_counts.csv", index=False)
        meeting.to_csv(stage / "meeting_window_counts.csv", index=False)
        flags.to_csv(stage / "window_flag_counts.csv", index=False)
        (stage / "provenance.json").write_text(metadata_json + "\n", encoding="utf-8")
        (stage / "analysis_helper_source.py").write_text(helper_source + "\n", encoding="utf-8")
        shutil.copy2(script_path, stage / "export_script.py")
        for name in qa_names:
            shutil.copy2(qa_dir / name, stage / name)
        receipt = {"dataset_id": dataset_id, "archive": archive_name, "sha256": archive_hash,
                   "bytes": archive.stat().st_size, "n_windows": n, "readback_verified": True,
                   "parent_role_overlap_count": 0, "validation_parent_los": 798}
        (stage / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        brief = f'''# Sujin: strong-absorber window dataset

Data: `{archive_name}` — **{n:,} windows**, **{size_mb:.1f} MB ({size_mib:.1f} MiB)**.
S/N per resolution element: **20 and 50**; three noise seeds per condition.
Dataset ID: `{dataset_id}`.

## Task and input

Classify the parent simulation model, EX0–EX3, from one noisy absorber-centered window.
These are z=0 mocks with the frozen mean-flux matching, COS LP1 processing, and
6.906077 km/s bins. Each window has 87 unsmoothed, unclipped float32 flux values,
with centers spanning about −296.96 to +296.96 km/s. Noise sigma per bin is
sqrt(2)/SNR. Clean/noiseless flux windows are excluded from this file.

Detection uses a three-bin mean, Fmin < 0.85 and (1−Fmin)/sigma_smoothed >= 3.
All eligible minima in one connected region below 0.85 are merged; recovery to
0.85 or above separates regions. Periodic endpoints are joined. The deepest
eligible minimum is retained, with pixel index breaking exact ties. `n_merged`
includes the retained minimum. Detection and fixed window extraction are separate:
a window can contain neighboring absorbers, and a large trough can extend beyond it.

## Fixed roles and independence

{_sujin_markdown_table(role_counts)}

Validation is exactly the **798 saved tuning LOS** from the per-chunk notebook.
The 798 calibration LOS remain reserved. Some assigned LOS have no selected windows;
the role assignment still includes them. Every window has a unique `window_id` and
a `role` code. `group_id` equals `los_id`; the same parent ID has one role across
all models, S/N values, noise seeds and overlapping windows. **Zero parent LOS overlap
between roles was verified.** Use the supplied roles; do not randomly split windows.
The three noise realizations and the two S/N conditions reuse parent spectra and
are not independent sightlines. Model labels, IDs, roles and bundle IDs are not input features.

## Window counts for the meeting

Counts below sum all three noise realizations. They count windows, not independent LOS.

{_sujin_markdown_table(meeting)}

Detailed counts and represented parent LOS are in `window_counts_snr_role_model.csv`;
seed-specific counts are in `window_counts_snr_role_model_seed.csv`.

## Saturated and extended troughs

Saturated sightlines are retained. The project interpretation is dense halo gas.
For the inspected EX1/4696 and EX3/5753 cases, native optical depths reproduce the
cached LP1 flux exactly to the recorded tolerance. No additional rejection cut was applied.
`whole_los_below_recovery` flags all 362 **smoothed noisy** bins below 0.85;
`trough_fits_window` records whether the complete selected trough fits the cutout.
These flags vary with S/N and seed. `trough_n_bins` times the bin spacing gives its
threshold-defined bin support. An entirely saturated or flat trough can have a
noise-sensitive selected center. These metadata support a documented sensitivity
comparison; they are not extra features for the depth-only baseline.

{_sujin_markdown_table(flags)}

For reference, the clean-spectrum census counts model–LOS cases (not noise copies):

{_sujin_markdown_table(source_totals)}

See `window_examples_snr20.png`, `window_examples_snr50.png`, `window_examples.csv`
and `native_source_audit.csv` for the completed visual/source checks.

## First baseline and reporting

Fit a depth-only baseline using **only `depth = 1 − fmin`**. Use training rows for fitting
and validation rows for choices; keep calibration reserved and test for the final report.
Report four-class balanced accuracy and per-class recall separately at S/N 20 and 50,
including seed scatter. For uncertainty, resample whole original test bundles with
all their windows/models/seeds together; retain bundles with zero detections in the
resampling universe. Four-class balanced chance accuracy is 25%. Full-window methods
should use the same roles and evaluation rows so comparisons are paired.

```python
import numpy as np
with np.load("{archive_name}", allow_pickle=False) as d:
    X = d["windows"]
    y = d["labels"]
    role_names = d["role_names"].tolist()
    train = d["role"] == role_names.index("train")
    validation = d["role"] == role_names.index("validation")
    test = d["role"] == role_names.index("test")
    depth_only = d["depth"][:, None]
    groups = d["group_id"]
    snr = d["snr_resel"]
    window_id = d["window_id"]
```

`provenance.json` documents fields, input/cache identities, saved split verification,
selection and analysis settings. `export_receipt.json` records archive SHA256 and
successful readback verification. Results describe discrimination among these
mock models under this selection; they do not establish accuracy on real spectra.
'''
        (stage / "SUJIN_BRIEF.md").write_text(brief, encoding="utf-8")
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    # Save the rerun's full six-S/N scientific summaries in the analysis directory.
    rates.to_csv(OUT / "observable_incidence.csv", index=False)
    rate_contrasts.to_csv(OUT / "incidence_differences_vs_EX0.csv", index=False)
    seed_counts.to_csv(OUT / "counts_by_seed_depth.csv", index=False)
    asymmetry_summary.to_csv(OUT / "asymmetry_summary.csv", index=False)
    pd.concat(stack_rows, ignore_index=True).to_csv(OUT / "stack_profiles.csv", index=False)
    (OUT / "provenance.json").write_text(metadata_json + "\n", encoding="utf-8")
    print(f"\nSaved: {destination / archive_name}")
    print(f"Size: {size_mb:.1f} MB ({size_mib:.1f} MiB); windows: {n:,}")
    print("Validation: exactly the saved 798 tuning LOS; parent-role overlaps: 0.")
    print("Readback verification: all saved arrays passed. Saturated windows retained.")
    print(f"Briefing: {destination / 'SUJIN_BRIEF.md'}")
    print("Window counts by S/N, role and model (three seeds combined):")
    display(meeting)
    display(flags)
    return {"directory": destination, "archive": destination / archive_name,
            "dataset_id": dataset_id, "counts": counts, "meeting_counts": meeting,
            "role_counts": role_counts, "flag_counts": flags, "receipt": receipt}


SUJIN_EXPORT = export_sujin_windows()
