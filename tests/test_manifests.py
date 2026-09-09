import numpy as np
import pytest

from src.manifests import make_manifest, los_ids, save_manifest, load_manifest

def test_manifest_matches_current_scheme():
    manifest = make_manifest(10_000)
    
    assert manifest["bundles"].shape == (666, 15)
    np.testing.assert_array_equal(manifest["bundles"][0], np.arange(15))
    assert manifest["train_bundle_ids"].size == 532
    assert manifest["test_bundle_ids"].size == 134
    np.testing.assert_array_equal(manifest["unused_los_ids"], np.arange(9990, 10_000))
    
def test_manifest_is_deterministic_and_partitions_ids():
    first = make_manifest(100, bundle_size=5, seed=7)
    second = make_manifest(100, bundle_size=5, seed=7)
    
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    
    train_ids = los_ids(first, "train")
    test_ids = los_ids(first, "test")
    
    assert np.intersect1d(train_ids, test_ids).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train_ids, test_ids])),
        np.arange(100),
    )
    
def test_manifest_save_and_load_and_overwrite(tmp_path):
    path = tmp_path / "manifest.npz"
    expected = make_manifest(100, bundle_size=5)

    save_manifest(path, expected)
    actual = load_manifest(path)
    
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
        
    with pytest.raises(FileExistsError):
        save_manifest(path, expected)
        
@pytest.mark.parametrize("kwargs", [
    {"n_los": 0},
    {"n_los": 10, "bundle_size": 0},
    {"n_los": 10, "bundle_size": 10},
    {"n_los": 30, "test_size":0},
    {"n_los": 30, "test_size":1},
    {"n_los": 30, "seed": 1.5},
])
def test_manifest_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        make_manifest(**kwargs)
        
def test_los_ids_invalid_split_type():
    with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
        los_ids(make_manifest(30), split="invalid")