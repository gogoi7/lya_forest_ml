import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_data
import h5py

def calculate_dv(L_cMpc_h=25, z=0.1, n_pixels=None):
    H0 = 100 
    Omega_m = 0.3
    Omega_L = 0.7
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    H_z = H0 * E_z
    v_box = (1 / (1 + z)) * H_z * (L_cMpc_h)
    return v_box / n_pixels

def calculate_ensemble_pk(data, dv):
    n_samples, n_pixels = data.shape
    mean_F = np.mean(data, axis=1, keepdims=True)
    delta_F = (data / mean_F) - 1
    
    fft_flux = np.fft.rfft(delta_F, axis=1)
    L = n_pixels * dv
    P_raw = (L / (n_pixels**2)) * np.abs(fft_flux)**2
    
    k_raw = 2 * np.pi * np.fft.rfftfreq(n_pixels, d=dv)
    
    # Calculate ensemble average
    P_ensemble = np.mean(P_raw, axis=0)
    
    return k_raw[1:], P_ensemble[1:] # Drop k=0

def main():
    path0 = 'data/raw/EX1_spectra.hdf5'
    path1 = 'data/raw/EX3_spectra.hdf5'
    
    print("Loading data...")
    with h5py.File(path0, 'r') as f:
        data_ex1_raw = f['/tau/H/1/1215'][:]
    with h5py.File(path1, 'r') as f:
        data_ex3_raw = f['/tau/H/1/1215'][:]
        
    data_ex1 = np.exp(-data_ex1_raw)
    data_ex3 = np.exp(-data_ex3_raw)
    
    n_samples, n_pixels = data_ex1.shape
    dv = calculate_dv(n_pixels=n_pixels)
    
    print("Calculating ensemble P(k)...")
    k, P_ex1 = calculate_ensemble_pk(data_ex1, dv)
    _, P_ex3 = calculate_ensemble_pk(data_ex3, dv)
    
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot absolute P(k)
    ax1.loglog(k, P_ex1, label='EX1', alpha=0.8)
    ax1.loglog(k, P_ex3, label='EX3', alpha=0.8)
    ax1.set_ylabel(r'$P(k)$ [(km/s)$^{-1}$]', fontsize=14)
    ax1.set_title('Ensemble Average 1D Flux Power Spectrum (z=0.1)', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot Percent Difference
    # Ratio = (EX3 - EX1) / EX1
    diff = (P_ex3 - P_ex1) / P_ex1
    diff_percent = diff * 100
    ax2.semilogx(k, diff_percent, color='red')
    ax2.set_xlabel(r'$k$ [s/km]', fontsize=14)
    ax2.set_ylabel('% Difference (EX3 vs EX1)', fontsize=14)
    ax2.axhline(0, color='black', ls='--')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('pk_difference.png', dpi=300)
    print("Saved plot to 'pk_difference.png'.")
    
    # Save Weights
    # We want weights to be positive and proportional to the absolute fractional difference
    # Added 1.0 to avoid zero weights and maintain baseline power
    weights = 1.0 + np.abs(diff)
    
    # Optional: Normalize weights so they average to 1
    weights = weights / np.mean(weights)
    
    np.save('data/processed/pk_weights.npy', weights)
    print("Saved continuous weights to 'data/processed/pk_weights.npy'")
    
if __name__ == '__main__':
    # Ensure processed directory exists
    os.makedirs('data/processed', exist_ok=True)
    main()
