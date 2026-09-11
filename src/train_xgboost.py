"""Configurable XGBoost training script for bundled Ly-alpha forest features."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .experiment import load_split
from .manifests import load_manifest

MODELS = {"EX0", "EX1", "EX2", "EX3"}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "min_child_weight": 1,
    "n_jobs": -1,
}
# Changes added for excluding small scale Pk features
N_LOS_FEATURES = 46
N_BUNDLE_FEATURES = N_LOS_FEATURES * 2
PK_START = 17
PK_STOP = 37
MAX_PK_BIN = 19

def feature_columns(max_pk_bin=MAX_PK_BIN):
    """Return bundled feature column names through a specificed P(k) bin index."""
    if (
        not isinstance(max_pk_bin, (int, np.integer))
        or isinstance(max_pk_bin, bool)
        or not 0 <= max_pk_bin <= MAX_PK_BIN
    ):
        raise ValueError(f"max_pk_bin must be an integer between 0 and {MAX_PK_BIN}")

    first_drop = PK_START + max_pk_bin + 1
    drop_mean = np.arange(first_drop, PK_STOP)
    drop_std = drop_mean + N_LOS_FEATURES

    keep = np.ones(N_BUNDLE_FEATURES, dtype=bool)
    keep[drop_mean] = False
    keep[drop_std] = False

    return np.flatnonzero(keep)

def make_model(seed=42):
    """Create an XGBoost classifier with the specified random seed."""
    return xgb.XGBClassifier(**XGB_DEFAULT_PARAMS, random_state=seed)

def score_pred(y_true, y_pred, models):
    """Compute accuracy and classification report for predictions."""
    labels = np.arange(len(models))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=models, output_dict=True, zero_division=0,
        ),
    }

def run_training(
    models,
    condition,
    manifest_path,
    feature_dir,
    output_dir,
    tag="z0_b15",
    seed=42,
    max_pk_bin=MAX_PK_BIN, # Maximum P(k) bin to include
):
    """Train an XGBoost model on bundled features and save the model, metrics, and predictions."""
    models = list(models)
    manifest_path = Path(manifest_path)
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)

    columns = feature_columns(max_pk_bin=max_pk_bin)

    run_tag = f"{tag}_s{seed}"
    if max_pk_bin < MAX_PK_BIN:
        run_tag += f"_pk{max_pk_bin}"
    run_dir = output_dir / run_tag / condition / "_".join(models)
    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    predict_path = run_dir / "predictions.npz"

    existing = [path for path in [model_path, metrics_path, predict_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Output files already exist: {existing}")

    manifest = load_manifest(manifest_path)
    X_train, y_train, X_test, y_test = load_split(
        models=models,
        condition=condition,
        features_dir=feature_dir,
        manifest=manifest,
        tag=tag,
    )
    X_train = X_train[:, columns]
    X_test = X_test[:, columns]

    print(f"Training {condition} comparison: {' vs '.join(models)}")
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    print(
        f"Using {len(columns)} features (P(k) bins 0-{max_pk_bin}) out of {N_BUNDLE_FEATURES} total features"
    )

    model = make_model(seed=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    scores = score_pred(y_test, y_pred, models)

    params = {**XGB_DEFAULT_PARAMS, "random_state": seed}

    metrics = {
        "models": models,
        "label_map": {
            str(label): model for label, model in enumerate(models)
        },
        "condition": condition,
        "feature_tag": tag,
        "manifest": str(manifest_path),
        "los_sha256": str(manifest["los_sha256"].item()),
        "split_seed": int(manifest["seed"].item()),
        "model_seed": seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "xgboost_version": xgb.__version__,
        "xgboost_params": params,
        "feature_importances": model.feature_importances_.astype(float).tolist(),
        **scores,
        "max_pk_bin": max_pk_bin,
        "feature_columns": columns.tolist(),
    }

    metrics_text = json.dumps(metrics, indent=2)

    n_test_bundle = manifest["test_bundle_ids"].size
    test_bundle_ids = np.tile(manifest["test_bundle_ids"], len(models))
    source_models = np.repeat(np.asarray(models), n_test_bundle)

    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    np.savez_compressed(
        predict_path,
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        bundle_ids=test_bundle_ids,
        source_models=source_models,
    )

    with metrics_path.open("x") as f:
        f.write(metrics_text)
        f.write("\n")

    print(f"Accuracy: {scores['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(np.array(scores["confusion_matrix"]))
    print()
    print(
        classification_report(
            y_test, y_pred, labels=np.arange(len(models)), target_names=models, digits=4, zero_division=0,
        )
    )
    print(f"Saved run directory: {run_dir}")

    return metrics

def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on raw of mean-flux-matched bundled Ly-alpha forest features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=sorted(MODELS),
        help="Models to classify, in label order (e.g., EX0 EX1 EX2). At least two models must be specified.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=("raw", "matched"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz"),
        help="Path to the manifest file describing the bundle scheme and train/test split.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/uvb_mean_flux/features"),
        help="Directory containing the feature bundles saved as .npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/uvb_mean_flux/runs"),
        help="Directory to save the trained model, metrics, and predictions.",
    )
    parser.add_argument(
        "--tag",
        default="z0_b15",
        help="Tag/prefix to identify feature files (default: z0_b15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training (default: 42).",
    )
    parser.add_argument(
        "--max-pk-bin",
        type=int,
        default=MAX_PK_BIN,
        help=f"Maximum P(k) bin index to include in features (default: {MAX_PK_BIN}). Must be between 0 and {MAX_PK_BIN}.",
    )
    return parser

def main(argv=None):
    """Main function to parse arguments and run training."""
    args = build_parser().parse_args(argv)

    run_training(
        models=args.models,
        condition=args.condition,
        manifest_path=args.manifest,
        feature_dir=args.features,
        output_dir=args.output,
        tag=args.tag,
        seed=args.seed,
        max_pk_bin=args.max_pk_bin,
    )

if __name__ == "__main__":
    main()