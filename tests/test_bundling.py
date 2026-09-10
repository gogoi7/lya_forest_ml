import numpy as np
import pytest

from src.feature_extraction import bundle_features
from src.manifests import make_manifest

def test_bundle_features_matches_expected_calculation():
    features = np.arange(24, dtype=np.float64).reshape(6, 4)
    original = features.copy()
    bundles = np.array([
        [0, 1, 2],
        [3, 4, 5]
    ])
    
    expected_mean = np.array([
        np.mean(features[[0, 1, 2]], axis=0),
        np.mean(features[[3, 4, 5]], axis=0)
    ])
    expected_std = np.array([
        np.std(features[[0, 1, 2]], axis=0),
        np.std(features[[3, 4, 5]], axis=0)
    ])
    
    actual = bundle_features(features, bundles)
    expected = np.concatenate([expected_mean, expected_std], axis=1)
    
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(features, original)  # Ensure input features are not modified
    assert actual.shape == (2, 8)  # 2 bundles, 4 features * 2 (mean + std)
    
def test_bundle_features_accepts_manifest_bundles():
    manifest = make_manifest(30, bundle_size=5)
    features = np.ones((30, 4))  # 30 samples, 4 features
    
    bundled = bundle_features(features, manifest["bundles"])
    assert bundled.shape == (6, 8)  # 6 bundles, 4 features * 2 (mean + std)
    
@pytest.mark.parametrize(
    ("features", "bundles"), [
        (np.ones(6), np.array([[0, 1]])),  # features not 2D
        (np.ones((3, 2)), np.array([0, 1])), # bundles not 2D
        (np.ones((3, 2)), np.array([[0, 3]])), # bundles contain out-of-bounds index
        (np.ones((3, 2)), np.array([[0.0, 1.0]])), # bundles not integer
        (np.ones((3, 2)), np.empty((0, 2), dtype=np.int64)), # bundles empty
    ],
)
def test_bundle_features_invalid_inputs(features, bundles):
    with pytest.raises(ValueError):
        bundle_features(features, bundles)