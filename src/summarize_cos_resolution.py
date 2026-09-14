"""Summarize four-class resolution results with paired-bundle bootstrap CIs."""

import json
from pathlib import Path

import numpy as np


def main():
    stages = ("native", "gaussian", "gaussian_rebinned")
    runs_root = Path("outputs/cos_lsf_snr/runs")
    run_suffix = Path("z0_b15_s42/matched/EX0_EX1_EX2_EX3")

    n_bootstrap = 50_000
    seed = 20260913

    predictions = {}

    for stage in stages:
        path = runs_root / stage / run_suffix / "predictions.npz"

        with np.load(path, allow_pickle=False) as data:
            predictions[stage] = {
                key: data[key].copy()
                for key in (
                    "y_true",
                    "y_pred",
                    "bundle_ids",
                    "source_models",
                )
            }

    reference = predictions["native"]

    # Paired comparisons require identical examples in identical order.
    for stage in stages:
        current = predictions[stage]

        for key in ("y_true", "bundle_ids", "source_models"):
            np.testing.assert_array_equal(
                current[key],
                reference[key],
            )

        assert current["y_pred"].shape == reference["y_true"].shape

    bundle_ids, inverse, counts = np.unique(
        reference["bundle_ids"],
        return_inverse=True,
        return_counts=True,
    )

    if not np.all(counts == 4):
        raise ValueError("Each paired bundle must contain four predictions")

    n_bundles = len(bundle_ids)
    bundle_accuracy = np.empty((n_bundles, len(stages)))

    for j, stage in enumerate(stages):
        current = predictions[stage]
        correct = (
            current["y_pred"] == current["y_true"]
        ).astype(np.float64)

        bundle_accuracy[:, j] = (
            np.bincount(inverse, weights=correct) / counts
        )

    accuracy = bundle_accuracy.mean(axis=0)

    # Reuse these same bundle draws for every condition.
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        n_bundles,
        size=(n_bootstrap, n_bundles),
    )

    bootstrap_accuracy = np.column_stack([
        bundle_accuracy[:, j][draws].mean(axis=1)
        for j in range(len(stages))
    ])

    report = {
        "schema_version": 1,
        "models": ["EX0", "EX1", "EX2", "EX3"],
        "n_test_examples": int(reference["y_true"].size),
        "n_paired_bundles": n_bundles,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": 0.95,
        "method": "paired bootstrap by test-bundle ID",
        "accuracy": {},
        "changes": {},
    }

    print(f"\nPaired test bundles: {n_bundles}")
    print(f"{'Condition':<22} {'Accuracy':>10} {'95% CI':>23}")

    for j, stage in enumerate(stages):
        low, high = np.quantile(
            bootstrap_accuracy[:, j],
            [0.025, 0.975],
        )

        report["accuracy"][stage] = {
            "accuracy": float(accuracy[j]),
            "ci_low": float(low),
            "ci_high": float(high),
        }

        print(
            f"{stage:<22} "
            f"{100.0 * accuracy[j]:>9.2f}% "
            f"[{100.0 * low:.2f}, {100.0 * high:.2f}]%"
        )

    comparisons = [
        ("gaussian - native", 1, 0),
        ("gaussian_rebinned - native", 2, 0),
        ("gaussian_rebinned - gaussian", 2, 1),
    ]

    print(f"\n{'Comparison':<32} {'Change':>10} {'95% CI':>24}")

    for name, after, before in comparisons:
        difference_pp = 100.0 * (
            accuracy[after] - accuracy[before]
        )

        bootstrap_difference_pp = 100.0 * (
            bootstrap_accuracy[:, after]
            - bootstrap_accuracy[:, before]
        )

        low, high = np.quantile(
            bootstrap_difference_pp,
            [0.025, 0.975],
        )

        report["changes"][name] = {
            "difference_pp": float(difference_pp),
            "ci_low_pp": float(low),
            "ci_high_pp": float(high),
        }

        print(
            f"{name:<32} "
            f"{difference_pp:>+9.2f} "
            f"[{low:+.2f}, {high:+.2f}] pp"
        )

    output_path = Path(
        "outputs/cos_lsf_snr/summary/"
        "z0_b15_s42_four_class_resolution.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()