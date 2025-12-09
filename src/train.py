from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

path0 = 'data/raw/EX0_spectra.hdf5'
path1 = 'data/raw/EX1_spectra.hdf5'
X, y = load_data(path0, path1)

# This fixes the issue of having identical LOS grid for both simulations
LOS_p_sim = 10000
uniq_idx = np.arange(LOS_p_sim)

train_idx, test_idx = train_test_split(uniq_idx, test_size=0.2, random_state=42)

X_train = np.concatenate([X[train_idx], X[train_idx + LOS_p_sim]])
y_train = np.concatenate([y[train_idx], y[train_idx + LOS_p_sim]])

X_test = np.concatenate([X[test_idx], X[test_idx + LOS_p_sim]])
y_test = np.concatenate([y[test_idx], y[test_idx + LOS_p_sim]])

print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape: {X_test.shape}")

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")