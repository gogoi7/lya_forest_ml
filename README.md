# Lyman-Alpha Forest ML Classification

This repository contains machine learning models for classifying specific 1D Lyman-Alpha forest transmission spectra into their underlying cosmological simulations (EX1 vs EX3).

## Current COS analysis

The current experiment compares EX0–EX3 at z = 0 after frozen
mean-flux calibration and COS LP1 processing.

With 15-sightline bundles, four-class accuracy is 51.68% without
added noise and approximately 31–40% across S/N 10–80 per resolution
element. EX2–EX3 is the strongest binary comparison; EX0–EX1 remains
near chance.

Flux statistics and noisy strong-absorber analyses are documented in
[the COS experiment README](experiments/cos_lsf_snr/README.md).

The raw per-chunk aggregation rerun is complete: Gaussian checks pass
in 27/30 task–S/N cases, and N95 is available in 18/30. See the
[raw aggregation results](experiments/cos_lsf_snr/raw_aggregation_results.md)
for results, remaining failure reasons and the cache-only workflow.
Observational S/N-mixture forecasts remain provisional pending
mask/fill validation. These z = 0 mock results are distinct from
the historical benchmark below.

## Historical benchmark: EX1–EX3 at z = 0.1
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
