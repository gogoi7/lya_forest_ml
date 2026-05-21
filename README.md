# Lyman-Alpha Forest ML Classification

This repository contains machine learning models for classifying specific 1D Lyman-Alpha forest transmission spectra into their underlying cosmological simulations (EX1 vs EX3).

## Key Results
The champion model is an **XGBoost Classifier** trained on a compact, highly engineered set of physical representations. 

- **Batched Input (Champion)**: By grouping spectra into batches of 10 and computing the mean and standard deviation of their features, the model achieves a test accuracy of **91.00%** on separating EX1 from EX3 at $z=0.1$.
- **Single Spectrum Baseline**: Training on individual spectra achieves an **80.62%** accuracy.

### Architecture: Compact Physical Features
Instead of feeding raw 1D flux (which gradient boosting trees struggle to interpret spatially), we construct 46 custom features:
- **Flux Statistics (17)**: Mean, Variance, Min, Skewness, Kurtosis, Percentiles (P5, P25, P50, P75, P95), and Absorption Line metrics (Count, Width, Depth).
- **1D Power Spectrum (20)**: The unbinned FFT $P(k)$ is log-binned into 20 structural scales, separating large-scale geometry from small-scale noise perfectly.
- **Wavelet Energies (6)**: A 5-level Daubechies 4 wavelet decomposition provides localized frequency representation.
- **Bispectrum Proxy (3)**: Statistics of the flux derivative $\frac{dF}{dx}$.

## Usage
To evaluate the 91.00% batched accuracy and 80.62% baseline benchmarks:
1. Ensure you have the `EX1_spectra.hdf5` and `EX3_spectra.hdf5` data in the `data/raw/` directory.
2. Run the main XGBoost script:
```bash
python src/train_xgboost.py
```

## Explorations
- **Deep Learning**: A PyTorch ResNet, a 1D CNN, and an MLP were evaluated but underperformed the XGBoost model (~60.9% accuracy max). Structured tabular data with explicit, highly-distinct physical meanings remains the domain of Gradient Boosted Trees.
- **Alternative Binning**: Weighted Power Spectrum binning and uniform CDF adaptive binning were explored. Standard Log-Spaced boundaries proved optimal at preserving macro-scale power while sharply suppressing high-k noise in a way decision trees can leverage efficiently.
