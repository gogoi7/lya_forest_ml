"""Training and verified restart support for the LP1 S/N experiment."""

import json
import platform
import shutil
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import sklearn
import xgboost as xgb

from . import experiment, manifests
from . import train_xgboost as training
from .snr_sweep import MODELS, _file_sha256

TASKS = (tuple(MODELS),) + tuple(combinations(MODELS, 2))
PARAMS = {**training.XGB_DEFAULT_PARAMS, "random_state": 42}
ARTIFACTS = ("model.joblib", "metrics.json", "predictions.npz")


def _check_run(run_dir, models, manifest, feature_dir=None):
    """Check a saved run; optionally reproduce its model predictions."""
    run_dir = Path(run_dir)

    for filename in ARTIFACTS:
        if not (run_dir / filename).is_file():
            raise ValueError(f"Incomplete run: missing {filename}")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    n_class = len(models)
    n_test = manifest["test_bundle_ids"].size

    expected = {
        "models": list(models),
        "label_map": {str(i): name for i, name in enumerate(models)},
        "condition": "matched",
        "feature_tag": "z0_b15",
        "los_sha256": str(manifest["los_sha256"].item()),
        "split_seed": 42,
        "model_seed": 42,
        "n_train": int(manifest["train_bundle_ids"].size * n_class),
        "n_test": int(n_test * n_class),
        "n_features": 92,
        "max_pk_bin": 19,
        "feature_columns": list(range(92)),
        "xgboost_version": xgb.__version__,
        "xgboost_params": PARAMS,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"Saved run has incompatible {key}")

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as saved:
        predictions = {
            key: saved[key].copy()
            for key in (
                "y_true", "y_pred", "probabilities",
                "bundle_ids", "source_models",
            )
        }

    expected_arrays = {
        "y_true": np.repeat(np.arange(n_class), n_test),
        "bundle_ids": np.tile(manifest["test_bundle_ids"], n_class),
        "source_models": np.repeat(np.asarray(models), n_test),
    }
    for key, values in expected_arrays.items():
        np.testing.assert_array_equal(
            predictions[key], values, err_msg=f"{key} differs"
        )

    y_pred = predictions["y_pred"]
    probabilities = predictions["probabilities"]

    if (
        y_pred.shape != (n_class * n_test,)
        or not np.all(np.isin(y_pred, np.arange(n_class)))
    ):
        raise ValueError("Invalid predicted labels")

    if (
        probabilities.shape != (n_class * n_test, n_class)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid saved probabilities")

    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6
    )

    scores = training.score_pred(
        predictions["y_true"], y_pred, list(models)
    )
    if (
        not np.isclose(
            metrics["accuracy"], scores["accuracy"], rtol=0, atol=1e-15
        )
        or metrics["confusion_matrix"] != scores["confusion_matrix"]
    ):
        raise ValueError("Metrics disagree with saved predictions")

    # Older runs lack input-file hashes. Before adopting one, check its
    # fitted model against the currently verified clean feature files.
    if feature_dir is not None:
        fitted = joblib.load(run_dir / "model.joblib")
        fitted_params = fitted.get_params()

        if any(fitted_params.get(k) != v for k, v in PARAMS.items()):
            raise ValueError("Saved model parameters disagree")
        if fitted.n_features_in_ != 92:
            raise ValueError("Saved model expects a different feature count")

        np.testing.assert_array_equal(fitted.classes_, np.arange(n_class))
        _, _, X_test, _ = experiment.load_split(
            models=models,
            condition="matched",
            features_dir=feature_dir,
            manifest=manifest,
            tag="z0_b15",
        )
        np.testing.assert_array_equal(fitted.predict(X_test), y_pred)
        np.testing.assert_allclose(
            fitted.predict_proba(X_test),
            probabilities,
            rtol=1e-6,
            atol=1e-7,
        )

    return metrics


def train_or_reuse(
    models,
    feature_dir,
    output_root,
    manifest_path,
    expected_manifest_sha256,
    legacy_run=None,
):
    """Train one task, reuse a verified run, or adopt a compatible legacy run."""
    models = tuple(models)
    print(models)
    if models not in TASKS:
        raise ValueError("Expected models to be one of the following tasks: " + str(TASKS))

    feature_dir = Path(feature_dir).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()

    if _file_sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("The frozen manifest changed")
    manifest = manifests.load_manifest(manifest_path)

    task_name = "_".join(models)
    relative_path = Path("z0_b15_s42") / "matched" / task_name
    run_dir = output_root / relative_path

    signature = {
        "schema_version": 1,
        "models": list(models),
        "manifest_sha256": expected_manifest_sha256,
        "feature_files_sha256": {
            model: _file_sha256(feature_dir / f"z0_b15_{model}.npz")
            for model in models
        },
        "xgboost_params": PARAMS,
        "max_pk_bin": 19,
        "code_sha256": {
            "training": _file_sha256(training.__file__),
            "split": _file_sha256(experiment.__file__),
            "manifest": _file_sha256(manifests.__file__),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "joblib": joblib.__version__,
        },
    }

    if run_dir.exists():
        receipt = json.loads((run_dir / "provenance.json").read_text())
        if receipt["signature"] != signature:
            raise RuntimeError(f"Incompatible completed run: {run_dir}")

        for filename in ARTIFACTS:
            if (
                _file_sha256(run_dir / filename)
                != receipt["artifact_sha256"][filename]
            ):
                raise RuntimeError(f"Changed run artifact: {filename}")

        metrics = _check_run(run_dir, models, manifest)
        print(f"{output_root.name}/{task_name}: reused", flush=True)
        return run_dir, metrics, "reused"

    run_dir.parent.mkdir(parents=True, exist_ok=True)

    # Train in a temporary directory on the same filesystem. The final
    # directory appears only after all artifacts pass their checks.
    with TemporaryDirectory(
        prefix=f".{task_name}_", dir=run_dir.parent
    ) as temporary:
        temporary_root = Path(temporary)
        prepared = temporary_root / relative_path
        prepared.mkdir(parents=True)
        status = None

        if legacy_run is not None and Path(legacy_run).exists():
            legacy_run = Path(legacy_run)
            try:
                _check_run(
                    legacy_run, models, manifest, feature_dir=feature_dir
                )
            except Exception as error:
                print(f"{task_name}: legacy run not reused: {error}")
            else:
                for filename in ARTIFACTS:
                    shutil.copy2(legacy_run / filename, prepared / filename)
                status = "legacy"

        if status is None:
            print(f"{output_root.name}/{task_name}: training", flush=True)

            with (prepared / "training.log").open("w") as log:
                with redirect_stdout(log):
                    training.run_training(
                        models=models,
                        condition="matched",
                        manifest_path=manifest_path,
                        feature_dir=feature_dir,
                        output_dir=temporary_root,
                        tag="z0_b15",
                        seed=42,
                        max_pk_bin=19,
                    )
            status = "trained"

        metrics = _check_run(prepared, models, manifest)
        receipt = {
            "signature": signature,
            "origin": status,
            "legacy_run": str(legacy_run) if status == "legacy" else None,
            "feature_dir": str(feature_dir),
            "manifest": str(manifest_path),
            "wrapper_sha256": _file_sha256(__file__),
            "artifact_sha256": {
                filename: _file_sha256(prepared / filename)
                for filename in ARTIFACTS
            },
        }
        (prepared / "provenance.json").write_text(
            json.dumps(receipt, indent=2) + "\n",
            encoding="utf-8",
        )
        prepared.rename(run_dir)

    print(
        f"{output_root.name}/{task_name}: "
        f"{status}, accuracy={metrics['accuracy']:.2%}",
        flush=True,
    )
    return run_dir, metrics, status