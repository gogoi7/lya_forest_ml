"""Summarize the S/N sweep with a paired test-bundle bootstrap."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .manifests import load_manifest
from .snr_sweep import SNR_LEVELS, NOISE_SEEDS, _file_sha256
from .snr_training import TASKS, _check_run


def summarize_snr_sweep(root):
    root = Path(root).resolve()
    sweep_root = root / "outputs/cos_lsf_snr/snr_sweep"
    output = sweep_root / "summary"
    manifest_path = (
        root / "outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz"
    )
    index_path = output / "training_runs.csv"

    manifest = load_manifest(manifest_path)
    # Match the sorted bundle order used by the earlier bootstrap.
    order = np.argsort(manifest["test_bundle_ids"])
    bundle_ids = manifest["test_bundle_ids"][order]
    n_bundles = len(bundle_ids)

    if n_bundles != 134 or len(np.unique(bundle_ids)) != 134:
        raise ValueError("Expected 134 distinct held-out bundle IDs.")

    labels = [
        "four-class" if len(models) == 4 else "–".join(models)
        for models in TASKS
    ]

    def stage_name(snr, noise_seed):
        return f"snr{int(snr)}_seed{int(noise_seed)}"

    stages = ["clean"] + [
        stage_name(snr, noise_seed)
        for snr in SNR_LEVELS
        for noise_seed in NOISE_SEEDS
    ]

    index = pd.read_csv(index_path)
    if index.duplicated(["stage", "task"]).any():
        raise ValueError("Duplicate stage/task entries in training_runs.csv.")

    index = index.set_index(["stage", "task"])
    expected = {(stage, task) for stage in stages for task in labels}
    if set(index.index) != expected:
        raise ValueError("The run index does not contain exactly 133 runs.")

    scores = {}
    source_hashes = {}

    for stage in stages:
        for models, task in zip(TASKS, labels):
            row = index.loc[(stage, task), :]
            run_dir = (
                sweep_root / "runs" / stage
                / "z0_b15_s42" / "matched" / "_".join(models)
            )

            if (root / str(row["run_dir"])).resolve() != run_dir.resolve():
                raise ValueError(f"Unexpected run directory: {stage}, {task}")

            # Validates metrics, labels, source models and exact bundle order.
            _check_run(run_dir, models, manifest)

            with np.load(
                run_dir / "predictions.npz", allow_pickle=False
            ) as data:
                correct = data["y_pred"] == data["y_true"]

            # Rows are classes; columns follow the manifest's test IDs.
            # Reorder columns explicitly, then average classes per bundle.
            bundle_accuracy = (
                correct.reshape(len(models), n_bundles)[:, order]
                .mean(axis=0)
            )

            if not np.isclose(
                bundle_accuracy.mean(),
                float(row["accuracy"]),
                rtol=0,
                atol=1e-14,
            ):
                raise ValueError(f"Accuracy mismatch: {stage}, {task}")

            if int(row["n_test"]) != len(models) * n_bundles:
                raise ValueError(f"Test-count mismatch: {stage}, {task}")

            scores[(stage, task)] = bundle_accuracy
            source_hashes[str(run_dir.relative_to(root))] = {
                name: _file_sha256(run_dir / name)
                for name in (
                    "predictions.npz", "metrics.json", "provenance.json"
                )
            }

    # Axes: task, bundle.
    clean = np.stack([scores[("clean", task)] for task in labels])

    # Axes: task, S/N, noise seed, bundle.
    noisy = np.asarray([
        [
            [
                scores[(stage_name(snr, noise_seed), task)]
                for noise_seed in NOISE_SEEDS
            ]
            for snr in SNR_LEVELS
        ]
        for task in labels
    ])

    n_bootstrap = 50_000
    bootstrap_seed = 20260913
    rng = np.random.default_rng(bootstrap_seed)

    # Generate ONCE and reuse for every task, condition and comparison.
    draws = rng.integers(
        0, n_bundles, size=(n_bootstrap, n_bundles)
    )

    def interval_pct(values):
        return 100.0 * np.quantile(values, [0.025, 0.975])

    clean_rows, sweep_rows, seed_rows = [], [], []

    for t, (models, task) in enumerate(zip(TASKS, labels)):
        clean_accuracy = clean[t].mean()
        clean_boot = clean[t][draws].mean(axis=1)
        clean_low, clean_high = interval_pct(clean_boot)

        clean_rows.append({
            "task": task,
            "accuracy_pct": 100.0 * clean_accuracy,
            "ci_low_pct": clean_low,
            "ci_high_pct": clean_high,
            "n_test": len(models) * n_bundles,
            "n_bundles": n_bundles,
        })

        for s, snr in enumerate(SNR_LEVELS):
            replicate_scores = noisy[t, s]

            # Keep all three realizations in every sampled bundle.
            mean_scores = replicate_scores.mean(axis=0)
            accuracy = mean_scores.mean()
            noisy_boot = mean_scores[draws].mean(axis=1)

            low, high = interval_pct(noisy_boot)
            delta_low, delta_high = interval_pct(noisy_boot - clean_boot)

            sweep_rows.append({
                "task": task,
                "snr_resel": int(snr),
                "accuracy_pct": 100.0 * accuracy,
                "ci_low_pct": low,
                "ci_high_pct": high,
                "delta_pp": 100.0 * (accuracy - clean_accuracy),
                "delta_ci_low_pp": delta_low,
                "delta_ci_high_pp": delta_high,
                "n_test": len(models) * n_bundles,
                "n_bundles": n_bundles,
                "n_noise_realizations": len(NOISE_SEEDS),
            })

            for r, noise_seed in enumerate(NOISE_SEEDS):
                seed_rows.append({
                    "task": task,
                    "snr_resel": int(snr),
                    "noise_seed": int(noise_seed),
                    "accuracy_pct": 100.0 * replicate_scores[r].mean(),
                })

    clean_table = pd.DataFrame(clean_rows)
    sweep_table = pd.DataFrame(sweep_rows)
    seed_table = pd.DataFrame(seed_rows)

    tables = {
        "clean_accuracy.csv": clean_table,
        "snr_accuracy.csv": sweep_table,
        "seed_accuracy.csv": seed_table,
    }
    output.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        path = output / name
        temporary = path.with_suffix(".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)

    # Save the actual draws and aligned scores for reproducibility.
    inputs_path = output / "bootstrap_inputs.npz"
    temporary = inputs_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            task_names=np.asarray(labels),
            snr_levels=np.asarray(SNR_LEVELS),
            noise_seeds=np.asarray(NOISE_SEEDS),
            test_bundle_ids=bundle_ids,
            clean_bundle_accuracy=clean,
            noisy_bundle_accuracy=noisy,
            bootstrap_draws=draws.astype(np.int16),
        )
    temporary.replace(inputs_path)

    metadata = {
        "schema_version": 1,
        "method": "paired percentile bootstrap by test-bundle ID",
        "confidence": 0.95,
        "interval_scope": "pointwise",
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "random_generator": "PCG64",
        "n_bundles": n_bundles,
        "noise_seeds": list(NOISE_SEEDS),
        "snr_levels": list(SNR_LEVELS),
        "task_models": {
            task: list(models) for task, models in zip(labels, TASKS)
        },
        "aggregation": "equal mean across classes and noise realizations",
        "paired_change": "noisy minus noiseless, in percentage points",
        "conditioning": (
            "Fixed fitted classifiers, split, simulations and sampled noise "
            "realizations; excludes training and cosmic-variance uncertainty."
        ),
        "array_axes": {
            "clean_bundle_accuracy": ["task", "bundle"],
            "noisy_bundle_accuracy": ["task", "snr", "noise_seed", "bundle"],
            "bootstrap_draws": ["resample", "draw"],
        },
        "manifest_sha256": _file_sha256(manifest_path),
        "run_index_sha256": _file_sha256(index_path),
        "summary_code_sha256": _file_sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "source_hashes": source_hashes,
        "output_sha256": {
            name: _file_sha256(output / name)
            for name in [*tables, "bootstrap_inputs.npz"]
        },
    }

    path = output / "bootstrap_metadata.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

    return clean_table, sweep_table, seed_table