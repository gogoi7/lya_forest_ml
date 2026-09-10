import numpy as np
import pytest

from src.experiment import load_split
from src.manifests import make_manifest

def make_files(tmp_path, manifest, bad_fingerprint=False):
    base = np.arange(12, dtype=np.float64).reshape(6, 2)
    arrays = {}
    
    for i, model in enumerate(["EX0", "EX1"]):
        raw = base + 100 * i
        matched = raw+ 1000
        arrays[model] = {
            "raw": raw,
            "matched": matched,
        }
        
        fingerprint = "wrong" if bad_fingerprint and i == 1 else "test-hash"
        
        np.savez_compressed(
            tmp_path / f"z0_b15_{model}.npz",
            raw_bundles=raw,
            matched_bundles=matched,
            los_sha256=np.array(fingerprint),
        )

    return arrays

@pytest.mark.parametrize("condition", ["raw", "matched"])
def test_load_split_returns_expected_arrays(tmp_path, condition):
    manifest = make_manifest(30, bundle_size=5)
    manifest["los_sha256"] = np.array("test-hash")
    arrays = make_files(tmp_path, manifest)
    
    X_train, y_train, X_test, y_test = load_split(
        ["EX0", "EX1"],
        condition,
        tmp_path,
        manifest,
    )
    
    train_ids = manifest["train_bundle_ids"]
    test_ids = manifest["test_bundle_ids"]
    
    expected_train = np.concatenate([arrays["EX0"][condition][train_ids], arrays["EX1"][condition][train_ids]])
    expected_test = np.concatenate([arrays["EX0"][condition][test_ids], arrays["EX1"][condition][test_ids]])
    
    np.testing.assert_allclose(X_train, expected_train)
    np.testing.assert_allclose(X_test, expected_test)
    np.testing.assert_array_equal(y_train, np.concatenate([np.zeros(train_ids.size, dtype=np.int64), np.ones(train_ids.size, dtype=np.int64)]))
    np.testing.assert_array_equal(y_test, np.concatenate([np.zeros(test_ids.size, dtype=np.int64), np.ones(test_ids.size, dtype=np.int64)]))

@pytest.mark.parametrize(
    ("models", "condition"),
    [
        (["EX0"], "raw"),  # less than 2 models
        (["EX0", "EX0"], "raw"),  # duplicate model names
        (["EX0", "EX1"], "invalid"),  # invalid condition
    ],
)
def test_load_split_invalid_inputs(tmp_path, models, condition):
    manifest = make_manifest(30, bundle_size=5)
    manifest["los_sha256"] = np.array("test-hash")
    
    with pytest.raises(ValueError):
        load_split(models, condition, tmp_path, manifest)

def test_load_split_fingerprint_mismatch(tmp_path):
    manifest = make_manifest(30, bundle_size=5)
    manifest["los_sha256"] = np.array("test-hash")
    
    make_files(tmp_path, manifest, bad_fingerprint=True)
    
    with pytest.raises(RuntimeError):
        load_split(["EX0", "EX1"], "raw", tmp_path, manifest)