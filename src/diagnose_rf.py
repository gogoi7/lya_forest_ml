"""
Diagnostic script to understand what Random Forest is learning
that might help improve CNN performance.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_loader import load_data

path0 = 'data/raw/EX0_spectra.hdf5'
path1 = 'data/raw/EX1_spectra.hdf5'

print("=" * 70)
print("RANDOM FOREST DIAGNOSTICS")
print("=" * 70)

# Load data
print("\n[1/4] Loading data...")
X_raw, y_raw = load_data(path0, path1)
print(f"Data shape: {X_raw.shape}")
print(f"Class distribution: EX0={np.sum(y_raw==0)}, EX1={np.sum(y_raw==1)}")

# Structural split
LOS_p_sim = X_raw.shape[0] // 2
uniq_idx = np.arange(LOS_p_sim)
train_idx_base, test_idx_base = train_test_split(
    uniq_idx, test_size=0.2, random_state=42
)

X_train = np.concatenate([X_raw[train_idx_base], X_raw[train_idx_base + LOS_p_sim]])
y_train = np.concatenate([y_raw[train_idx_base], y_raw[train_idx_base + LOS_p_sim]])
X_test = np.concatenate([X_raw[test_idx_base], X_raw[test_idx_base + LOS_p_sim]])
y_test = np.concatenate([y_raw[test_idx_base], y_raw[test_idx_base + LOS_p_sim]])

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# Train RF
print("\n[2/4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

# Feature importance
print("\n[3/4] Analyzing feature importance...")
feature_importance = rf.feature_importances_
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:][::-1]

print(f"\nTop {top_n} most important features (wavelength indices):")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. Index {idx:4d}: {feature_importance[idx]:.6f}")

# Check if features are clustered or spread out
print(f"\nTop features spread:")
print(f"  Min index: {top_indices.min()}")
print(f"  Max index: {top_indices.max()}")
print(f"  Mean index: {top_indices.mean():.1f}")
print(f"  Std of indices: {top_indices.std():.1f}")

# Check feature statistics
print("\n[4/4] Feature statistics for top important regions...")
for idx in top_indices[:5]:
    ex0_values = X_train[y_train == 0, idx]
    ex1_values = X_train[y_train == 1, idx]
    print(f"\nIndex {idx}:")
    print(f"  EX0: mean={ex0_values.mean():.4f}, std={ex0_values.std():.4f}, "
          f"min={ex0_values.min():.4f}, max={ex0_values.max():.4f}")
    print(f"  EX1: mean={ex1_values.mean():.4f}, std={ex1_values.std():.4f}, "
          f"min={ex1_values.min():.4f}, max={ex1_values.max():.4f}")
    print(f"  Difference (EX1-EX0): {ex1_values.mean() - ex0_values.mean():.4f}")

# Check if RF is using local or global patterns
print("\n" + "=" * 70)
print("INSIGHTS:")
print("=" * 70)
if top_indices.std() < 100:
    print("✓ RF uses CLUSTERED features (local patterns)")
    print("  → CNNs should work well with appropriate kernel sizes")
elif top_indices.std() > 500:
    print("✓ RF uses SPREAD OUT features (global patterns)")
    print("  → CNNs might need larger receptive fields or attention")
else:
    print("✓ RF uses MODERATELY SPREAD features")
    print("  → CNNs should capture these with multiple scales")

print(f"\nFeature importance concentration:")
importance_concentration = np.sum(feature_importance[top_indices[:10]]) / np.sum(feature_importance)
print(f"  Top 10 features account for {importance_concentration*100:.1f}% of importance")
if importance_concentration > 0.5:
    print("  → High concentration: few key features matter")
    print("  → Consider focusing CNN on these regions")
else:
    print("  → Low concentration: many features contribute")
    print("  → CNN needs to capture broad patterns")
