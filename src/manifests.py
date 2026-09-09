"""Reproducible LOS split and bundle generation."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_KEYS = {
    "bundles",
    "train_bundle_ids",
    "test_bundle_ids",
    "unused_los_ids",
    "n_los",
    "bundle_size",
    "test_size",
    "seed",
}

def make_manifest(n_los, bundle_size=15, test_size=0.2, seed=42):
    """
    Create a manifest for splitting LOS into bundles and train/test sets.

    Parameters:
    n_los (int): Total number of LOS.
    bundle_size (int): Number of LOS per bundle.
    test_size (float): Fraction of bundles to use for testing.
    seed (int): Random seed for reproducibility.

    Returns:
    dict: Manifest containing bundle information and train/test splits.
    """
    if (
        not isinstance(n_los, (int, np.integer))
        or isinstance(n_los, bool)
        or n_los <= 0
    ):
        raise ValueError("n_los must be a positive integer")

    if (
        not isinstance(bundle_size, (int, np.integer))
        or isinstance(bundle_size, bool)
        or bundle_size <= 0
    ):
        raise ValueError("bundle_size must be a positive integer")

    if not np.isfinite(test_size) or not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a finite float in the range (0, 1)")
    
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    
    n_bundle = n_los // bundle_size
    if n_bundle < 2:
        raise ValueError("Not enough LOS to create at least two bundles")
    
    n_used = n_bundle * bundle_size
    bundles = np.arange(n_used, dtype=np.int64).reshape(n_bundle, bundle_size)
    
    bundle_ids = np.arange(n_bundle, dtype=np.int64)
    
    train_ids, test_ids = train_test_split(
        bundle_ids, test_size=test_size, random_state=seed,
    )
    
    return {
        "bundles": bundles,
        "train_bundle_ids": train_ids,
        "test_bundle_ids": test_ids,
        "unused_los_ids": np.arange(n_used, n_los, dtype=np.int64),
        "n_los": np.array(n_los, dtype=np.int64),
        "bundle_size": np.array(bundle_size, dtype=np.int64),
        "test_size": np.array(test_size, dtype=np.float64),
        "seed": np.array(seed, dtype=np.int64),
    }
    
def los_ids(manifest, split="train"):
    """
    Get the LOS IDs for a given split (train or test) from the manifest.

    Parameters:
    manifest (dict): Manifest containing bundle information and train/test splits.
    split (str): Split type, either "train" or "test".

    Returns:
    np.ndarray: Array of LOS IDs for the specified split.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    
    bundle_ids = manifest[f"{split}_bundle_ids"]
    
    return manifest["bundles"][bundle_ids].reshape(-1).copy()

def save_manifest(path, manifest):
    """Save a manifest once; refuse to overwrite a frozen assignment"""
    path = Path(path)
    
    if path.suffix != ".npz":
        raise ValueError("Manifest path must have a .npz suffix")
    
    if path.exists():
        raise FileExistsError(f"Manifest file {path} already exists; refusing to overwrite")
    
    missing = _KEYS.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing required keys: {sorted(missing)}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **manifest)
    
def load_manifest(path):
    """Load a manifest from a .npz file"""
    with np.load(path, allow_pickle=False) as data:
        missing = _KEYS.difference(data.files)
        if missing:
            raise ValueError(f"Manifest file {path} is missing required keys: {sorted(missing)}")
        
        return {key: data[key].copy() for key in data.files}