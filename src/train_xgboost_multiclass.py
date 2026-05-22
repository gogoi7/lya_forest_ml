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

    print("\n--- Multi-Class XGBoost on Batched Input (Batch Size = 10) ---")
    batch_size = 10
    
    def create_batches(features, labels, batch_size):
        unique_classes = np.unique(labels)
        X_batched_list = []
        y_batched_list = []
        
        for c in unique_classes:
            idx = np.where(labels == c)[0]
            features_c = features[idx]
            n_samples_c = len(features_c)
            n_batches = n_samples_c // batch_size
            
            # Truncate to make perfectly divisible
            features_c = features_c[:n_batches * batch_size]
            
            # Reshape to (n_batches, batch_size, n_features)
            features_c = features_c.reshape((n_batches, batch_size, -1))
            
            # Calculate Mean and Std over the batch dimension (axis=1)
            mean_c = np.mean(features_c, axis=1)
            std_c = np.std(features_c, axis=1)
            
            # Concatenate mean and std to form the batched feature vector
            X_batch_c = np.concatenate([mean_c, std_c], axis=1)
            
            X_batched_list.append(X_batch_c)
            y_batched_list.append(np.full(n_batches, c))
            
        return np.concatenate(X_batched_list, axis=0), np.concatenate(y_batched_list, axis=0)
        
    X_batched, y_batched = create_batches(X_features, y_raw, batch_size)
    print(f"Batched feature shape: {X_batched.shape}")
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_batched, y_batched, test_size=0.2, random_state=42, stratify=y_batched
    )
    
    print("Training Multi-Class XGBoost (Batched)...")
    xgb_b = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42,
        tree_method='hist'
    )
    xgb_b.fit(X_train_b, y_train_b)
    
    y_pred_b = xgb_b.predict(X_test_b)
    acc_b = accuracy_score(y_test_b, y_pred_b)
    print(f"XGBoost Multi-Class (Batched) Accuracy: {acc_b * 100:.2f}%")
    
    print("\nConfusion Matrix (Batched):")
    cm_b = confusion_matrix(y_test_b, y_pred_b)
    print(cm_b)
    
    print("\nClassification Report (Batched):")
    print(classification_report(y_test_b, y_pred_b, target_names=["EX0", "EX1", "EX2", "EX3"]))
    
    joblib.dump(xgb_b, 'xgb_model_multiclass_batched.joblib')
    print("Model saved to xgb_model_multiclass_batched.joblib")

if __name__ == "__main__":
    main()
