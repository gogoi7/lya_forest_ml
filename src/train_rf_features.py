import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib

from data_loader import load_data
from feature_extraction import extract_features

def calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=None):
    """
    Calculate velocity width per pixel.
    
    Parameters:
    L_cMpc_h (float): Box size in comoving Mpc/h.
    z (float): Redshift.
    n_pixels (int): Number of pixels in the spectrum.
    
    Returns:
    float: Velocity width per pixel in km/s.
    """
    # Hubble constant terms
    H0 = 100 # km/s/Mpc * h
    # For a flat LambdaCDM universe (approximate for CAMELS)
    # Omega_m ~ 0.3, Omega_L ~ 0.7
    Omega_m = 0.3
    Omega_L = 0.7
    
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    H_z = H0 * E_z
    
    # Comoving distance relates to velocity via Hubble flow approx in the box
    # The velocity width of the box is H(z) * L_proper / (1+z)? 
    # Wait, H(z) is expansion rate of proper distance. 
    # v = H(z) * d_proper. 
    # d_proper = L_cMpc_h / h / (1+z).
    
    # Let's derive it properly or use the standard approximation.
    # v_box = a * H(z) * L_comoving / h
    # a = 1 / (1+z)
    
    v_box = (1 / (1 + z)) * H_z * (L_cMpc_h) # h factors cancel if H0=100 and L in Mpc/h
    # v_box = H(z) * L_proper
    # H_z is in km/s/Mpc (if h included). L_cMpc_h is in Mpc/h.
    # So L_cMpc_h / h is in Mpc.
    # H_z = 100 * h * E_z.
    # v_box = (100 * h * E_z) * (L_cMpc_h / h) / (1+z)
    #       = 100 * E_z * L_cMpc_h / (1+z)
    
    return v_box / n_pixels

def main():
    # 1. Load Data
    print("Loading data (EX2 vs EX3)...")
    path0 = 'data/raw/EX2_spectra.hdf5'
    path1 = 'data/raw/EX3_spectra.hdf5'
    
    # data_loader.load_data returns X, y
    X_raw, y_raw = load_data(path0, path1)
    
    # 2. Preprocess (Optical Depth -> Flux)
    X_flux = np.exp(-X_raw)
    
    n_samples, n_pixels = X_flux.shape
    print(f"Data Loaded: {n_samples} spectra, {n_pixels} pixels.")
    
    # 3. Calculate dv
    dv = calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=n_pixels)
    print(f"Calculated dv: {dv:.4f} km/s per pixel.")
    
    # 4. Extract Features
    print("Extracting features...")
    features = extract_features(X_flux, dv)
    
    feature_names = [
        "Mean Flux", "Flux Std", "Min Flux", 
        "Skewness", "Kurtosis",
        "P5", "P25", "P50", "P75", "P95",
        "Num Lines",
        "Mean Line Depth", "Max Line Depth",
        "Mean Line Width", "Max Line Width",
        "Total EW",
        "Col Density Proxy"
    ]
    
    print(f"Features extracted: shape {features.shape}")
    
    # 5. Split Data (Preserving LOS structure)
    N_sim = len(X_flux) // 2
    indices = np.arange(N_sim)
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = np.concatenate([features[train_idx], features[train_idx + N_sim]])
    y_train = np.concatenate([y_raw[train_idx], y_raw[train_idx + N_sim]])
    
    X_test = np.concatenate([features[test_idx], features[test_idx + N_sim]])
    y_test = np.concatenate([y_raw[test_idx], y_raw[test_idx + N_sim]])
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 6. Train Random Forest
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 7. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop 10 Feature Importances:")
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[sorted_idx], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.title("Random Forest Feature Importance (EX2 vs EX3)")
    plt.tight_layout()
    plt.savefig("feature_importance_ex2_ex3.png")
    print("Feature importance plot saved to feature_importance_ex2_ex3.png")
    
    # Save Model
    joblib.dump(clf, "rf_model_features_ex2_ex3.joblib")
    print("Model saved to rf_model_features_ex2_ex3.joblib")

if __name__ == "__main__":
    main()
