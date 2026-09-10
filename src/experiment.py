"""Build paired datasets for training and evaluation from saved bundle features."""

from pathlib import Path

import numpy as np

_CONDITIONS = {"raw", "matched"} #raw optical depth vs matched (to targeted mean flux) optical depth

def load_split(
    models,
    condition,
    features_dir,
    manifest,
    tag="z0_b15"
):
    """
    Load the train/test split of features for a given condition and models.

    Parameters:
    models (list of str): List of model names to load features for, e.g., ['EX0', 'EX1', 'EX2'].
    condition (str): Condition to load ('raw' or 'matched').
    features_dir (Path): Directory containing feature bundles saved as .npz files.
    manifest (dict): Manifest containing bundle schemes and train/test splits.
    tag (str): Tag/prefix to identify feature files.

    Returns:
    tuple: (X_train, y_train, X_test, y_test) where X are features and y are labels.
    """
    models = list(models)
    
    if len(models) < 2:
        raise ValueError("At least two models must be specified for a paired dataset")
    
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model names are not allowed")
    
    if condition not in _CONDITIONS:
        raise ValueError(f"Condition must be one of {_CONDITIONS}")
    
    # Load features for each model and concatenate them
    train_ids = manifest["train_bundle_ids"]
    test_ids = manifest["test_bundle_ids"]
    n_bundles = manifest["bundles"].shape[0]
    fingerprint = str(manifest["los_sha256"].item())
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for label, model in enumerate(models):
        path = Path(features_dir) / f"{tag}_{model}.npz"
        key = f"{condition}_bundles"
        
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"Feature file {path} is missing required key '{key}'")
            
            features = data[key].copy()
            saved_fingerprint = str(data["los_sha256"].item())
            
            if saved_fingerprint != fingerprint:
                raise RuntimeError(f"{model}: Feature file {path} fingerprint {saved_fingerprint} does not match manifest fingerprint {fingerprint}")
            
            if features.ndim != 2 or features.shape[0] != n_bundles:
                raise ValueError(f"{model}: Feature array shape {features.shape} does not match expected number of bundles {n_bundles}")
            
            if not np.all(np.isfinite(features)):
                raise ValueError(f"{model}: Feature array contains non-finite values")
            
            X_train.append(features[train_ids])
            X_test.append(features[test_ids])
            y_train.append(np.full(train_ids.size, label, dtype=np.int64))
            y_test.append(np.full(test_ids.size, label, dtype=np.int64))
            
    # Concatenate and return the paired datasets
    return (
        np.concatenate(X_train),
        np.concatenate(y_train),
        np.concatenate(X_test),
        np.concatenate(y_test),
    )