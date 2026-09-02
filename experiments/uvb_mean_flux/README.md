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