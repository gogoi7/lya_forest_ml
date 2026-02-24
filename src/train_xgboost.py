import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd

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
    print(f"Calculated dv: {dv:.4f} km/s per pixel.")
    
    # 1. Experiment 5: XGBoost on Stat Features Only (Baseline for XGB)
    print("\n--- Experiment 5: XGBoost on Statistical Features ---")
    features_stats = extract_features(X_flux, dv)
    
    N_sim = len(X_flux) // 2
    indices = np.arange(N_sim)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train_stats = np.concatenate([features_stats[train_idx], features_stats[train_idx + N_sim]])
    y_train = np.concatenate([y_raw[train_idx], y_raw[train_idx + N_sim]])
    X_test_stats = np.concatenate([features_stats[test_idx], features_stats[test_idx + N_sim]])
    y_test = np.concatenate([y_raw[test_idx], y_raw[test_idx + N_sim]])
    
    xgb_stats = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42)
    xgb_stats.fit(X_train_stats, y_train)
    y_pred_stats = xgb_stats.predict(X_test_stats)
    acc_stats = accuracy_score(y_test, y_pred_stats)
    print(f"XGBoost (Stats) Accuracy: {acc_stats*100:.2f}%")
    
    # 2. Experiment 8: XGBoost on Compact Physical Features (<50)
    print("\n--- Experiment 8: XGBoost on Compact Physical Features (<50) ---")
    print("Extracting compact features... (Stats + Binned P(k) + Bi + Wav)")
    features_compact = extract_combined_features(X_flux, dv)
    print(f"Total feature shape: {features_compact.shape}")
    
    X_train_c = np.concatenate([features_compact[train_idx], features_compact[train_idx + N_sim]])
    X_test_c = np.concatenate([features_compact[test_idx], features_compact[test_idx + N_sim]])
    
    # Tuning for compact feature set
    print("Training XGBoost (Compact)...")
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
    
    xgb_c.fit(X_train_c, y_train, verbose=True)
    y_pred_c = xgb_c.predict(X_test_c)
    acc_c = accuracy_score(y_test, y_pred_c)
    print(f"XGBoost (Compact Features) Accuracy: {acc_c*100:.2f}%")
    
    print("\nConfusion Matrix (Compact):")
    cm = confusion_matrix(y_test, y_pred_c)
    print(cm)
    
    # Feature Importance with Proper Labels
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
    
    all_names = stat_names + pk_names + bi_names + wav_names
    # Adjust for actual length
    if len(all_names) != features_compact.shape[1]:
        print(f"Warning: Name list length {len(all_names)} != feature count {features_compact.shape[1]}")
        all_names = [f"Feat_{i}" for i in range(features_compact.shape[1])]
        
    importances = xgb_c.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop 15 Feature Importances:")
    for i in range(min(15, len(all_names))):
        idx = sorted_idx[i]
        name = all_names[idx] if idx < len(all_names) else f"Feat_{idx}"
        print(f"{i+1}. {name}: {importances[idx]:.4f}")
    
    joblib.dump(xgb_c, "xgb_model_compact.joblib")
    print("Final compact model saved.")


if __name__ == "__main__":
    main()
