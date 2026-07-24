import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

from data_loader import load_data
from feature_extraction import extract_features, extract_combined_features

def calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=None):
    """
    Calculate velocity width per pixel.
    """
    H0 = 100 
    Omega_m = 0.3
    Omega_L = 0.7
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    H_z = H0 * E_z
    v_box = (1 / (1 + z)) * H_z * (L_cMpc_h)
    return v_box / n_pixels

def main():
    print("Loading data (EX1 vs EX3)...")
    path0 = 'data/raw/EX1_spectra.hdf5'
    path1 = 'data/raw/EX3_spectra.hdf5'
    
    X_raw, y_raw = load_data(path0, path1)
    X_flux = np.exp(-X_raw)
    
    n_samples, n_pixels = X_flux.shape
    dv = calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=n_pixels)
    
    # We will use the identical 80/20 train-test split from the baseline to ensure apples-to-apples
    N_sim = len(X_flux) // 2
    indices = np.arange(N_sim)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print("Extracting baseline 46 features...")
    features_all = extract_combined_features(X_flux, dv)
    
    # Feature Names Definitions
    stat_names = [
        "Mean Flux", "Flux Std", "Min Flux", 
        "Skewness", "Kurtosis",
        "P5", "P25", "P50", "P75", "P95",
        "Num Lines",
        "Mean Line Depth", "Max Line Depth",
        "Mean Line Width", "Max Line Width",
        "Total EW",
        "Col Density Proxy"
    ]
    pk_names = [f"Pk_bin_{i}" for i in range(20)]
    bi_names = ["dF_std", "dF_skew", "dF_kurt"]
    wav_names = [f"Wav_E_{i}" for i in range(6)]
    
    all_names = np.array(stat_names + pk_names + bi_names + wav_names)
    
    if len(all_names) != features_all.shape[1]:
        print(f"Warning: Name list length {len(all_names)} != feature count {features_all.shape[1]}")
        all_names = np.array([f"Feat_{i}" for i in range(features_all.shape[1])])
        
    X_train_full = np.concatenate([features_all[train_idx], features_all[train_idx + N_sim]])
    y_train = np.concatenate([y_raw[train_idx], y_raw[train_idx + N_sim]])
    X_test_full = np.concatenate([features_all[test_idx], features_all[test_idx + N_sim]])
    y_test = np.concatenate([y_raw[test_idx], y_raw[test_idx + N_sim]])
    
    # Arrays to track progress
    current_features_mask = np.ones(features_all.shape[1], dtype=bool)
    num_features_history = []
    accuracy_history = []
    feature_sets_history = {}
    
    print("\n--- Starting Recursive Feature Elimination ---")
    
    while np.sum(current_features_mask) > 1:
        n_feats = np.sum(current_features_mask)
        feature_sets_history[n_feats] = list(all_names[current_features_mask])
        
        X_tr = X_train_full[:, current_features_mask]
        X_te = X_test_full[:, current_features_mask]
        current_names = all_names[current_features_mask]
        
        xgb_c = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            n_jobs=-1,
            random_state=42
        )
        
        xgb_c.fit(X_tr, y_train, verbose=False)
        y_pred = xgb_c.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        
        num_features_history.append(n_feats)
        accuracy_history.append(acc)
        
        importances = xgb_c.feature_importances_
        
        least_important_idx_in_active = np.argmin(importances)
        dropped_feature_name = current_names[least_important_idx_in_active]
        
        # print(f"[{n_feats:2d} features] Accuracy: {acc*100:.2f}% | Dropping least important: '{dropped_feature_name}'")
        
        global_indices = np.where(current_features_mask)[0]
        feature_to_drop_global_idx = global_indices[least_important_idx_in_active]
        current_features_mask[feature_to_drop_global_idx] = False

    best_idx = np.argmax(accuracy_history)
    best_n_feats = num_features_history[best_idx]
    print(f"\nBest accuracy {accuracy_history[best_idx]*100:.2f}% achieved with {best_n_feats} features.")
    print(f"\nThe optimal {best_n_feats} features are:")
    for i, f in enumerate(feature_sets_history[best_n_feats], 1):
        print(f"{i}. {f}")

if __name__ == "__main__":
    main()
