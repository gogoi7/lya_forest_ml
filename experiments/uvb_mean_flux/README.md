# UVB mean-flux degeneracy experiment

## Scientific question

Does Ly-alpha forest classification performance persist after removing
differences in the ensemble mean transmission between the CAMELS-EX models?

This tests sensitivity to the UV-background amplitude, not to UVB spectral
shape or thermal-history effects.

## Fixed initial setup

- Redshift: z = 0
- Models: EX0, EX1, EX2, and EX3
- Bundle size: 15 sightlines
- Features: existing 46-feature representation
- Classifier: existing XGBoost configuration
- Sightline IDs are paired across all models
- Raw and matched experiments use identical splits and bundle assignments

## Mean-flux matching

For each model, apply one global optical-depth factor:

tau_matched = A_model * tau_raw

The factor is fitted using training sightlines only, such that:

mean(exp(-A_model * tau_raw)) = target_mean_flux

The initial internal target is the equal-weighted mean of the four
training-set mean fluxes. The same target is used for every pairwise and
four-class experiment.

## Required comparisons

1. Raw optical depths
2. Mean-flux-matched optical depths

All other choices must remain fixed between the two comparisons.

## Prohibited shortcuts

- Do not overwrite the raw HDF5 files
- Do not estimate scaling factors from validation or test sightlines
- Do not normalize individual sightlines or bundles separately
- Do not choose a different target for each model pair
- Do not tune model hyperparameters on the held-out test set

## Generated outputs

Generated files will be written beneath:

outputs/uvb_mean_flux/

Each run must record its configuration, split seeds, target mean flux,
per-model scaling factors, metrics, and predictions.

## Test A completion status

**Status: COMPLETE — PASS (September 11, 2026)**

The prespecified failure criterion was a collapse in classification accuracy
after matching the ensemble mean transmission. No such collapse occurred.
Classification remained strong after matching, so differences in global mean
transmitted flux are not the sole source of model discrimination.

### Calibration

The common training-set target was:

- Target mean flux: `0.9733676942`
- Target effective optical depth: `0.0269933708`
- Fit tolerance: `1e-4`
- Training sightlines per model: `7980`
- Held-out test sightlines per model: `2010`
- Manifest: `outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz`
- Sightline-manifest SHA-256:
  `ddffd1d43106cb33e09b7d0f0ae5d9e1db6b8713639ec6d37275dc5b78a4a4c7`

| Model | Raw train mean flux | Tau scale | Matched train mean flux | Matched test mean flux | Test residual |
|---|---:|---:|---:|---:|---:|
| EX0 | 0.9728469 | 0.9705244 | 0.9733964 | 0.9734334 | +0.0000657 |
| EX1 | 0.9745651 | 1.0675314 | 0.9733925 | 0.9729233 | -0.0004444 |
| EX2 | 0.9708588 | 0.8768067 | 0.9733841 | 0.9737974 | +0.0004297 |
| EX3 | 0.9752000 | 1.1073058 | 0.9734064 | 0.9734106 | +0.0000429 |

All training residuals are below the specified tolerance. Because the scaling
factors were fitted using training sightlines only, the held-out test means
are not forced to equal the target. Nevertheless, the across-model test-set
mean-flux range decreased from `0.0038893` before matching to `0.0008741`
after matching, a reduction of approximately 77.5%.

### Primary classification results

The primary analysis uses all 92 bundled features: the mean and standard
deviation across each 15-sightline bundle for all 46 per-sightline features.
Each model contributes 134 paired held-out bundles.

Accuracy changes are defined as matched minus raw accuracy. Confidence
intervals were calculated with 50,000 bootstrap resamples clustered by paired
bundle ID. P-values were calculated with 100,000 paired permutations and
Holm-corrected across the seven primary comparisons. The inference seed was
`20260911`.

| Comparison | Raw accuracy | Matched accuracy | Change | 95% bootstrap CI | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| EX0–EX1 | 79.48% | 88.81% | +9.33 pp | [+4.85, +13.81] pp | 0.00100 |
| EX0–EX2 | 86.19% | 93.28% | +7.09 pp | [+3.73, +10.82] pp | 0.00100 |
| EX0–EX3 | 83.21% | 88.43% | +5.22 pp | [+1.12, +9.33] pp | 0.07194 |
| EX1–EX2 | 93.28% | 95.52% | +2.24 pp | [-0.37, +4.85] pp | 0.28914 |
| EX1–EX3 | 95.15% | 92.91% | -2.24 pp | [-4.85, 0.00] pp | 0.28914 |
| EX2–EX3 | 74.25% | 95.15% | +20.90 pp | [+16.04, +25.37] pp | 0.00007 |
| Four-class | 65.86% | 84.89% | +19.03 pp | [+14.93, +23.13] pp | 0.00007 |

Matching does not improve every individual comparison significantly. That is
not required for Test A. The decisive result is that every pairwise classifier
retains at least 88.43% matched accuracy and the four-class classifier retains
84.89% accuracy, well above their respective chance levels.

### High-k sensitivity check

As a supplementary check, the mean and standard deviation of power-spectrum
bins 15–19 were removed, reducing the bundled representation from 92 to 82
features.

| Comparison | Raw accuracy | Matched accuracy | Change | 95% bootstrap CI | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| EX2–EX3 | 75.00% | 88.81% | +13.81 pp | [+8.96, +18.66] pp | 0.00002 |
| Four-class | 49.25% | 57.46% | +8.21 pp | [+3.92, +12.50] pp | 0.00037 |

The matched-versus-raw increase therefore persists after removing the five
highest-k power bins. However, the decrease in absolute four-class matched
accuracy from 84.89% to 57.46% shows that the full multiclass classifier is
substantially sensitive to small-scale power. This is a robustness concern to
carry forward; it does not reverse the Test A conclusion.

This high-k check is not the mean-flux-proxy ablation defined as Test C.

### Conclusion and scope

Test A rejects the specific explanation that classification performance is
produced solely by differences in ensemble mean transmitted flux. It does not
yet demonstrate that the remaining signal is caused by galactic feedback.

The following questions remain outside Test A:

- cosmic-realization discrimination, addressed by Test B;
- mean-flux and monotone-proxy feature ablation, addressed by Test C;
- UVB spectral-shape and thermal-history dependence;
- instrumental and observational forward modelling.

At completion, the test suite reports:

```text
51 passed