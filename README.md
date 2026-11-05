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

## Checkpoint: 2026-11-05

### Figures and spectrum viewer

- Compiled the current z≈0 Figures 1–3 into
  `outputs/galaxy_absorption/full/galaxy_figures_1-3_z0.pdf`.
- Selected the comparison examples:
  - Low-sSFR: EX0 halo 58, ray 2811.
  - Star-forming: EX0 halo 77, ray 3738.
- Added `src/galaxy_spectrum_viewer.py`, an offline viewer for comparing
  any two sampled sightlines across all four matched models.
- Each panel independently selects mass range, EX0 star-formation type,
  b/Rvir range, viewing axis, halo, sightline, spectrum mode, and S/N.
  Impact-parameter filters select existing rays; they do not interpolate
  new sightlines.
- Available modes: native noiseless, COS LP1 noiseless, and COS LP1
  with Gaussian noise at arbitrary positive S/N per resolution element.
  Changing S/N rescales a fixed noise realization; the default seed is 42.
- Model labels:
  - EX0: fiducial IllustrisTNG.
  - EX1: extreme AGN feedback.
  - EX2: extreme supernova feedback.
  - EX3: no feedback.
- Spectra use each model's matched-galaxy velocity frame and shared plot
  scales. Annotations include halo/ray IDs, mass, galaxy type, impact
  parameter, axis, redshift, spectrum settings, and equivalent widths.
- Supports PNG export and self-contained interactive HTML for a selected
  pair. The full-sample viewer uses an accompanying indexed `.bin` file.
- Build the viewer from the loaded `full` and `geometry` objects in
  `notebooks/03_experiments.ipynb`. Default output:
  `outputs/galaxy_absorption/full/viewer/`.

### Local cutouts and other snapshots

- Added `src/extract_galaxy_cutouts.py` for chunked local extraction of
  galaxy-centred gas cutouts for the selected comparison halos.
- Updated `scripts/galaxy_full.sh` and `scripts/galaxy_match.slurm` to
  accept a snapshot number while retaining snapshot 90 as the default.
- Matching jobs for the additional redshifts completed successfully.

### Resume later

- Generate and inspect the local cutouts, then build the spatial viewer
  connecting the selected spectra to their surrounding gas.
- Continue the spectra and absorption analysis for snapshots 86 and 80.
- Complete the intrinsic N_HI and random-sightline comparisons from the
  original figure plan.

Large simulation data and generated viewer products remain outside Git.
