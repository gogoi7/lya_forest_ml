import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from data_loader import load_all_data
from feature_extraction import extract_combined_features

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
    print("Loading data (EX0, EX1, EX2, EX3)...")
    paths = [
        'data/raw/EX0_spectra.hdf5',
        'data/raw/EX1_spectra.hdf5',
        'data/raw/EX2_spectra.hdf5',
        'data/raw/EX3_spectra.hdf5'
    ]
    X_raw, y_raw = load_all_data(paths)
    
    # Preprocessing
    X_flux = np.exp(-X_raw)
    n_samples, n_pixels = X_flux.shape
    dv = calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=n_pixels)
    
    print("Extracting features...")
    X_features = extract_combined_features(X_flux, dv)
    
    print(f"Total feature shape: {X_features.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
    
    print("Training Multi-Class XGBoost...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Multi-Class Accuracy: {accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["EX0", "EX1", "EX2", "EX3"]))
    
    # Save the model
    joblib.dump(model, 'xgb_model_multiclass.joblib')
    print("Model saved to xgb_model_multiclass.joblib")

if __name__ == "__main__":
    main()
