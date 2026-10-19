"""Four-class, one-feature baseline for the verified Sujin export.

Run after sujin_export_step.py in the same notebook kernel. Reads the export
without modifying it; writes a separate sibling directory. No source spectra,
window arrays, calibration rows or random window splits are used for fitting.
"""

def _depth_sha256(path):
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _depth_array_hash(array):
    import hashlib
    import numpy as np
    a = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def _depth_metrics(confusion):
    """Returns balanced accuracy, accuracy and recall; last axes are true/pred."""
    import numpy as np
    cm = np.asarray(confusion, dtype=np.float64)
    support = cm.sum(axis=-1)
    if cm.shape[-2:] != (4, 4) or np.any(support <= 0):
        raise ValueError("Each evaluation/resample must contain every true class.")
    correct = np.diagonal(cm, axis1=-2, axis2=-1)
    recall = correct / support
    return recall.mean(axis=-1), correct.sum(axis=-1) / support.sum(axis=-1), recall


def _depth_bundle_confusion(y, predicted, seed_index, bundle_index, n_bundles):
    import numpy as np
    counts = np.zeros((3, n_bundles, 4, 4), dtype=np.int64)
    np.add.at(counts, (seed_index, bundle_index, y, predicted), 1)
    return counts


def _depth_table(frame):
    import numpy as np
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.3f}"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v).replace("|", "\\|")
    result = ["| " + " | ".join(map(str, frame.columns)) + " |",
              "| " + " | ".join("---" for _ in frame.columns) + " |"]
    result += ["| " + " | ".join(cell(v) for v in row) + " |"
               for row in frame.itertuples(index=False, name=None)]
    return "\n".join(result)


def run_sujin_depth_baseline(archive_path, script_path):
    import json
    import shutil
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeClassifier, export_text

    archive = Path(archive_path).resolve()
    script = Path(script_path).resolve()
    destination = archive.parent.parent / "sujin_depth_baseline_v1"
    if destination.exists():
        raise FileExistsError(f"Preserving completed baseline: {destination}")
    for p in (archive, script, archive.parent / "export_receipt.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    receipt = json.loads((archive.parent / "export_receipt.json").read_text())
    archive_hash = _depth_sha256(archive)
    if archive_hash != receipt.get("sha256") or receipt.get("readback_verified") is not True:
        raise RuntimeError("Archive does not match its verified export receipt.")
    fields = ("depth", "fmin", "labels", "role", "los_id", "group_id", "bundle_id",
              "window_id", "snr_resel", "noise_seed")
    constant_keys = ("model_names", "role_names", "noise_seeds", "snr_levels",
                     "los_role", "los_bundle", "manifest_bundles", "test_bundle_ids")
    with np.load(archive, allow_pickle=False) as saved:
        required = set(fields + constant_keys) | {"dataset_id", "metadata_json"}
        if not required.issubset(saved.files):
            raise RuntimeError("This is not the compact Sujin export schema.")
        data = {key: saved[key].copy() for key in fields + constant_keys}
        dataset_id = str(saved["dataset_id"].item())
        provenance = json.loads(saved["metadata_json"].item())
    if dataset_id != receipt.get("dataset_id") or dataset_id != provenance.get("dataset_id"):
        raise RuntimeError("Export dataset identities disagree.")
    for key, a in data.items():
        if _depth_array_hash(a) != provenance["array_sha256"].get(key):
            raise RuntimeError(f"Export field hash mismatch: {key}")
    models = data["model_names"].tolist()
    roles = data["role_names"].tolist()
    seeds = data["noise_seeds"].astype(int)
    snrs = data["snr_levels"].astype(int)
    if models != ["EX0", "EX1", "EX2", "EX3"] or roles != ["train", "validation", "calibration", "test"]:
        raise RuntimeError("Unexpected model/role enumeration.")
    np.testing.assert_array_equal(seeds, [20260929, 20260930, 20261001])
    np.testing.assert_array_equal(snrs, [20, 50])
    n = data["depth"].size
    if n != receipt.get("n_windows") or any(data[k].shape != (n,) for k in fields):
        raise RuntimeError("Export row counts disagree.")
    if not np.isfinite(data["depth"]).all() or np.any(data["depth"] <= 0.15):
        raise RuntimeError("Invalid selected-window depths.")
    np.testing.assert_array_equal(data["depth"], 1 - data["fmin"])
    for key, lo, hi in (("labels", 0, 4), ("role", 0, 4), ("los_id", 0, 10000),
                        ("bundle_id", 0, 666)):
        a = data[key]
        if a.dtype.kind not in "iu" or np.any((a < lo) | (a >= hi)):
            raise RuntimeError(f"Invalid index field: {key}")
    if np.unique(data["window_id"]).size != n:
        raise RuntimeError("Duplicate window IDs.")
    if set(np.unique(data["snr_resel"])) != {20, 50} or set(np.unique(data["noise_seed"])) != set(seeds):
        raise RuntimeError("Unexpected S/N or noise seeds in rows.")
    if data["los_role"].shape != (10000,) or data["los_bundle"].shape != (10000,):
        raise RuntimeError("Invalid parent-role/bundle arrays.")
    np.testing.assert_array_equal(data["group_id"], data["los_id"])
    np.testing.assert_array_equal(data["role"], data["los_role"][data["los_id"]])
    np.testing.assert_array_equal(data["bundle_id"], data["los_bundle"][data["los_id"]])
    np.testing.assert_array_equal([(data["los_role"] == i).sum() for i in range(4)], [6384, 798, 798, 2010])
    bundles = data["manifest_bundles"]
    test_ids = data["test_bundle_ids"].astype(int)
    if bundles.shape != (666, 15) or test_ids.shape != (134,) or np.unique(test_ids).size != 134:
        raise RuntimeError("Expected the original 134 held-out bundles of 15 LOS.")
    np.testing.assert_array_equal(np.sort(bundles[test_ids].ravel()), np.flatnonzero(data["los_role"] == 3))
    if not np.all(data["role"][np.isin(data["bundle_id"], test_ids)] == 3):
        raise RuntimeError("A held-out bundle contains rows assigned to another role.")

    # Protocol fixed before reading validation scores or test performance.
    config = {
        "schema": 1, "task": "four_class", "feature": "depth = 1 - fmin",
        "fit": "One estimator per SNR; pool all three training noise realizations",
        "candidate_max_depth": [0, 1, 2, 3, 4, 5],
        "depth_zero": "Training-majority DummyClassifier control; no depth information",
        "tree": {"criterion": "gini", "class_weight": "balanced", "min_samples_leaf": 0.01,
                 "random_state": 20261019},
        "selection": "Largest mean of the three validation-seed balanced accuracies; exact ties prefer smaller depth",
        "refit_on_validation": False, "calibration_used": False,
        "bootstrap_draws": 5000, "bootstrap_seed": 20261019,
        "bootstrap_unit": "Original 134 test bundles, including any without detections; shared draws across SNR/classes/seeds",
        "interval": "Percentile 95% interval for mean of the three seed metrics; estimator fixed",
        "seed_scatter": "Sample SD across test noise realizations of the same fitted estimator; not three independent datasets",
        "estimand": "Per-window recall within each class; balanced accuracy gives the four classes equal weight",
        "saturation_policy": "All exported windows retained; no flag used as an input feature",
    }
    X = data["depth"][:, None]
    y = data["labels"].astype(int)
    seed_index = np.full(n, -1, dtype=int)
    for i, seed in enumerate(seeds):
        seed_index[data["noise_seed"] == seed] = i
    bundle_lookup = np.full(666, -1, dtype=int)
    bundle_lookup[test_ids] = np.arange(134)
    weights = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        134, np.full(134, 1 / 134), size=config["bootstrap_draws"]
    )
    chosen_models, tuning_rows, summary_rows, seed_rows, recall_rows = {}, [], [], [], []
    per_bundle, bootstrap_metrics, prediction_parts, selection = [], [], [], {}
    for snr in snrs:
        snr = int(snr)
        condition = data["snr_resel"] == snr
        train = condition & (data["role"] == 0)
        validation = condition & (data["role"] == 1)
        test = condition & (data["role"] == 3)
        for name, mask in (("train", train), ("validation", validation), ("test", test)):
            for ri in range(3):
                if not np.array_equal(np.unique(y[mask & (seed_index == ri)]), np.arange(4)):
                    raise RuntimeError(f"Missing class in {name}, S/N {snr}, seed {seeds[ri]}.")
        print(f"S/N {snr}: fitting on {train.sum():,} windows; selecting with {validation.sum():,} validation windows...", flush=True)
        best_score, chosen_depth, chosen = -np.inf, None, None
        for tree_depth in config["candidate_max_depth"]:
            if tree_depth == 0:
                estimator = DummyClassifier(strategy="most_frequent")
            else:
                estimator = DecisionTreeClassifier(max_depth=tree_depth, **config["tree"])
            estimator.fit(X[train], y[train])
            validation_prediction = estimator.predict(X[validation]).astype(int)
            val_cm = np.array([confusion_matrix(
                y[validation][seed_index[validation] == ri],
                validation_prediction[seed_index[validation] == ri], labels=np.arange(4)
            ) for ri in range(3)])
            val_ba, _, _ = _depth_metrics(val_cm)
            score = float(val_ba.mean())
            tuning_rows.append({"snr": snr, "max_depth": tree_depth, "validation_ba_mean": score,
                                **{f"validation_ba_seed{int(seed)}": float(val_ba[ri]) for ri, seed in enumerate(seeds)}})
            if score > best_score + 1e-12:
                best_score, chosen_depth, chosen = score, tree_depth, estimator
        chosen_models[snr] = chosen
        selection[str(snr)] = {"max_depth": chosen_depth, "validation_ba_mean": best_score}
        # Test predictions are computed only after hyperparameter selection has finished.
        index = np.flatnonzero(test)
        predicted = chosen.predict(X[index]).astype(int)
        if chosen.n_features_in_ != 1:
            raise RuntimeError("Baseline unexpectedly uses more than one input feature.")
        cube = _depth_bundle_confusion(y[index], predicted, seed_index[index],
                                       bundle_lookup[data["bundle_id"][index]], 134)
        if int(cube.sum()) != len(index):
            raise RuntimeError("Test bundle counts disagree with test windows.")
        cm = cube.sum(axis=1)
        ba, accuracy, recall = _depth_metrics(cm)
        boot_cm = (weights @ cube.transpose(1, 0, 2, 3).reshape(134, -1)).reshape(-1, 3, 4, 4)
        boot_ba, boot_accuracy, boot_recall = _depth_metrics(boot_cm)
        lo_ba, hi_ba = np.percentile(boot_ba.mean(axis=1), [2.5, 97.5])
        lo_acc, hi_acc = np.percentile(boot_accuracy.mean(axis=1), [2.5, 97.5])
        summary_rows.append({
            "snr": snr, "selected_max_depth": chosen_depth,
            "n_train_windows": int(train.sum()), "n_validation_windows": int(validation.sum()),
            "n_test_windows": len(index), "validation_ba_pct": 100 * best_score,
            "test_ba_pct": 100 * float(ba.mean()), "ba_ci95_low_pct": 100 * float(lo_ba),
            "ba_ci95_high_pct": 100 * float(hi_ba), "ba_seed_sd_pp": 100 * float(ba.std(ddof=1)),
            "test_accuracy_pct": 100 * float(accuracy.mean()),
            "accuracy_ci95_low_pct": 100 * float(lo_acc), "accuracy_ci95_high_pct": 100 * float(hi_acc),
        })
        for ri, seed in enumerate(seeds):
            ba_lo, ba_hi = np.percentile(boot_ba[:, ri], [2.5, 97.5])
            seed_rows.append({"snr": snr, "noise_seed": int(seed), "n_test_windows": int(cm[ri].sum()),
                              "test_ba_pct": 100 * float(ba[ri]), "ba_ci95_low_pct": 100 * float(ba_lo),
                              "ba_ci95_high_pct": 100 * float(ba_hi), "test_accuracy_pct": 100 * float(accuracy[ri]),
                              **{f"recall_{m}_pct": 100 * float(recall[ri, mi]) for mi, m in enumerate(models)}})
        for mi, model in enumerate(models):
            low, high = np.percentile(boot_recall[:, :, mi].mean(axis=1), [2.5, 97.5])
            recall_rows.append({"snr": snr, "true_model": model, "mean_recall_pct": 100 * float(recall[:, mi].mean()),
                                "ci95_low_pct": 100 * float(low), "ci95_high_pct": 100 * float(high)})
        per_bundle.append(cube)
        bootstrap_metrics.append(np.stack([boot_ba, boot_accuracy], axis=-1))
        prediction_parts.append({**{k: data[k][index] for k in ("window_id", "los_id", "bundle_id", "snr_resel", "noise_seed")},
                                 "true_label": y[index].astype(np.uint8), "predicted_label": predicted.astype(np.uint8)})
    summary = pd.DataFrame(summary_rows)
    by_seed = pd.DataFrame(seed_rows)
    recalls = pd.DataFrame(recall_rows)
    tuning = pd.DataFrame(tuning_rows)
    predictions = {key: np.concatenate([p[key] for p in prediction_parts]) for key in prediction_parts[0]}
    order = np.argsort(predictions["window_id"])
    predictions = {key: value[order] for key, value in predictions.items()}
    np.testing.assert_array_equal(predictions["window_id"], np.sort(data["window_id"][data["role"] == 3]))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_dataset_id": dataset_id,
                "archive": str(archive), "archive_sha256": archive_hash,
                "script_sha256": _depth_sha256(script), "config": config, "selection": selection,
                "original_role_counts": dict(zip(roles, [6384, 798, 798, 2010])),
                "input_role_split_sha256": provenance["cached_split_sha256"],
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False)
    report_cols = ["snr", "selected_max_depth", "test_ba_pct", "ba_ci95_low_pct", "ba_ci95_high_pct", "ba_seed_sd_pp", "test_accuracy_pct"]
    report = f'''# Depth-only four-class baseline

Input dataset: `{dataset_id}`. Feature: `depth = 1 - fmin`, the selected three-bin
smoothed minimum's depth. All exported windows, including saturated and extended
troughs, are retained. Accuracy is per selected absorber window, not per sightline
or a combination of windows.

{_depth_table(summary[report_cols])}

Balanced accuracy is the mean of the four class recalls; its chance/control level
is 25%. Ordinary accuracy is also reported because class window counts differ.
Each main result is the mean of three test-noise seed scores. Seed SD is in
percentage points and measures noise sensitivity with the fitted estimator fixed.

## Fitting and selection

One estimator per S/N uses all three training noise copies on the 6,384 fitting
LOS per model. Candidates are class-balanced decision trees of maximum depth 1–5,
with at least 1% of training windows per leaf, plus a training-majority control
(max_depth=0). The 798 validation LOS select the highest mean seed balanced
accuracy; ties within 1e-12 prefer the smaller tree. No refitting on validation.
The 798 calibration LOS remain unused, and the 2,010 test LOS are held out.
Only depth is supplied to the estimator. IDs, counts, roles and trough flags are
not predictors. A shallow tree is a simple reference, not an upper bound on what
could be learned from depth.

## Uncertainty and comparison

The 95% percentile intervals use 5,000 common resamples of the original 134 test
bundles, including bundles with zero detections. All windows, classes and noise
copies from a sampled bundle move together. The same resamples are used at both
S/N values; the three seeds are not treated as independent observed datasets.
Intervals are conditional on the fitted estimators and validation choices; they
do not include retraining uncertainty or variation between independent simulations.
Per-class recall weights windows equally within a class, rather than weighting
parent sightlines equally.

`test_predictions.npz` preserves window IDs for paired comparisons with future
full-window models. `bootstrap_inputs.npz` contains per-bundle confusion counts,
the ordered 134 bundle IDs and the exact shared resampling weights.

## Per-class recall

{_depth_table(recalls)}

Seed results: `metrics_by_seed.csv`. Candidate validation results:
`validation_candidates.csv`. Models and their rules are saved alongside provenance.
These are z=0 mock discrimination results under the stated selection, not measured
accuracy on real observations.
'''
    stage = Path(tempfile.mkdtemp(prefix="depth_baseline_tmp_", dir=destination.parent))
    try:
        summary.to_csv(stage / "metrics_summary.csv", index=False)
        by_seed.to_csv(stage / "metrics_by_seed.csv", index=False)
        recalls.to_csv(stage / "recall_by_class.csv", index=False)
        tuning.to_csv(stage / "validation_candidates.csv", index=False)
        (stage / "provenance.json").write_text(metadata_json + "\n", encoding="utf-8")
        (stage / "DEPTH_BASELINE_RESULTS.md").write_text(report, encoding="utf-8")
        shutil.copy2(script, stage / "baseline_script.py")
        for snr, estimator in chosen_models.items():
            joblib.dump(estimator, stage / f"depth_model_snr{snr}.joblib")
            rules = (export_text(estimator, feature_names=["depth"], max_depth=10, decimals=8)
                     if isinstance(estimator, DecisionTreeClassifier)
                     else f"Training-majority control: always predict {models[int(estimator.predict([[0.5]])[0])]}\n")
            (stage / f"depth_rules_snr{snr}.txt").write_text(rules, encoding="utf-8")
        np.savez_compressed(stage / "test_predictions.npz", **predictions, model_names=np.asarray(models),
                            input_dataset_id=np.array(dataset_id), metadata_json=np.array(metadata_json))
        np.savez_compressed(stage / "bootstrap_inputs.npz", snr_levels=snrs, noise_seeds=seeds,
                            test_bundle_ids=test_ids, confusion_by_bundle=np.stack(per_bundle),
                            resampling_weights=weights.astype(np.uint16),
                            bootstrap_ba_accuracy=np.stack(bootstrap_metrics),
                            metric_names=np.array(["balanced_accuracy", "accuracy"]),
                            input_dataset_id=np.array(dataset_id))
        with np.load(stage / "test_predictions.npz", allow_pickle=False) as saved:
            for key, value in predictions.items():
                np.testing.assert_array_equal(saved[key], value)
        file_hashes = {p.name: _depth_sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage / "completed.json").write_text(json.dumps({"input_dataset_id": dataset_id,
            "n_test_predictions": len(predictions["window_id"]), "files_sha256": file_hashes}, indent=2) + "\n")
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"\nSaved: {destination}")
    print("Four-class depth-only results (%); balanced chance/control = 25%:")
    try:
        from IPython.display import display
        display(summary[report_cols].round(3))
        display(by_seed.round(3))
    except ImportError:
        print(summary[report_cols].round(3).to_string(index=False))
        print(by_seed.round(3).to_string(index=False))
    return {"directory": destination, "summary": summary, "by_seed": by_seed,
            "recalls": recalls, "validation_candidates": tuning}


if "SUJIN_EXPORT" not in globals() or "ROOT" not in globals():
    raise RuntimeError("Run this cell after the verified Sujin export in the same notebook kernel.")
DEPTH_BASELINE = run_sujin_depth_baseline(
    SUJIN_EXPORT["archive"], ROOT / "notebooks/sujin_depth_baseline_step.py"
)
