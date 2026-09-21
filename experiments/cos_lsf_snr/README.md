# COS LSF and signal-to-noise experiment

## Current status

**The first tabulated-LP1 noise pilot is complete. The full observing
configuration remains provisional.**

The observed spectrum `pg1048_all.dat` provides representative sampling
and reported-error estimates. Classification results below come from
processed synthetic spectra.

- Branch: `experiment/cos-lsf-snr`
- Foundation: [Test A mean-flux experiment](../uvb_mean_flux/README.md)
- Earlier results: [Historical Gaussian pilot](gaussian_pilot.md)
- Interactive analysis: [COS observation notebook](../../notebooks/cos_obs.ipynb)

## Main result

Both conditions use the same LP1 response, 362-pixel output grid, frozen
bundles, classifier settings, and train/test assignments. A separate
classifier is trained and evaluated under each condition.

| Condition | Four-class accuracy | 95% bootstrap interval |
|---|---:|---:|
| LP1, no added noise | 51.68% | [48.13, 55.22]% |
| LP1, Gaussian noise with sigma = 0.07 | 35.07% | [31.34, 38.81]% |

**Noisy minus clean: -16.60 percentage points,
95% paired interval [-21.83, -11.38] pp.**

Noise substantially reduces performance. Some discrimination survives
among these processed mocks: the noisy accuracy interval remains above
the balanced four-class chance level of 25%.

### Recall by true class

Recall is the fraction of examples from a true class classified correctly.

| Model | Clean recall | Noisy recall |
|---|---:|---:|
| EX0 | 24.6% | 31.3% |
| EX1 | 52.2% | 23.1% |
| EX2 | 56.7% | 42.5% |
| EX3 | 73.1% | 43.3% |

EX2 and EX3 retain the highest noisy recall. The classifiers are trained
separately, so individual class recalls can move in different directions
while overall accuracy decreases.

## Frozen experimental setup

| Setting | Value |
|---|---|
| Simulation redshift | z = 0 |
| Classes | EX0, EX1, EX2, EX3 |
| Sightlines per class | 10,000 |
| Velocity length per sightline | 2500 km/s |
| Native sampling | 2499 pixels; 2500 / 2499 km/s per pixel |
| LP1 pilot sampling | 362 pixels; 6.9060773481 km/s per pixel |
| Bundle size | 15 sightlines |
| Training/test bundles per class | 532 / 134 |
| Four-class training/test examples | 2128 / 536 |
| Features per sightline/bundle | 46 / 92 |
| Power-spectrum features | All 20 bins; max_pk_bin = 19 |
| Split and classifier seed | 42 |

The model-specific optical-depth scales fitted on Test A training
sightlines are reused without refitting.

- Manifest: `outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz`
- Calibration: `outputs/uvb_mean_flux/calibration/z0_b15_seed42.json`

## Observational reference and assumptions

The working forest interval is **1220–1380 Angstrom**.

The known gap near 1300 Angstrom includes zero-error rows at approximately
1301.02–1306.88 Angstrom. Invalid pixels are excluded from sampling/noise
estimates and from the complete strong-trough windows.

The three data columns are currently interpreted as wavelength, normalized
flux, and its 1-sigma error. This interpretation and the continuum
normalization should be confirmed with Taesun.

### Instrument response

- Taesun confirmed lifetime position **LP1**.
- **G130M/1291 and the nominal dispersion remain provisional.**
- Table: `aa_LSFTable_G130M_1291_LP1_cn.dat`
- Representative LSF profile: 1300 Angstrom.
- Assumed native dispersion: 0.00997 Angstrom per pixel.
- Native-bin weights are normalized and resampled through their cumulative
  distribution using PCHIP, retaining the full tabulated support.
- Periodic convolution is applied to complete simulated sightlines.

One representative LSF profile is used throughout this pilot.
Wavelength-dependent instrument modeling remains follow-up work.

### Sampling and noise

The output grid approximates the median velocity spacing between adjacent
valid observed pixels while preserving each mock's 2500 km/s length.

The median positive reported error over valid forest pixels is **0.0700**.
Assuming a unit continuum and 1-sigma errors, this corresponds to
approximately **S/N = 14.3 per pixel**.

For `lp1_obs_noisy`, independent Gaussian noise with constant standard
deviation 0.07 is added after convolution and resampling.

Noise seeds are `20260921 + model_index`, where EX0–EX3 have indices 0–3.
Flux values are retained outside [0, 1]. The existing logarithmic feature
uses `-log(max(flux, 0) + 1e-10)`.

## Processing sequence

1. Load optical depths and apply the frozen Test A scale.
2. Convert to transmitted flux.
3. Apply the tabulated LP1 response on the native simulation grid.
4. Conservatively resample to 362 pixels.
5. Check per-sightline mean-flux conservation.
6. Add noise for the noisy condition.
7. Extract features using the output pixel width and check finiteness.
8. Build the frozen 15-sightline bundles.
9. Train and evaluate one classifier per condition.
10. Compare predictions using a paired bootstrap.

Feature files record the processing settings, LSF checksum, noise seed,
and sightline fingerprint.

## Uncertainty and scope

The bootstrap uses 50,000 resamples with seed 20260913. Its resampling
unit is one of the 134 test-bundle IDs. Each draw keeps the four model-class
examples together and uses the same sampled IDs across conditions.
Intervals are the 2.5th and 97.5th percentiles.

These intervals are conditional on the fitted classifiers, current
simulation volumes, frozen split, and one noise realization. Independent
volume validation and additional noise realizations remain follow-up work.

Other limits of the current pilot:

- Actual errors vary across the observed spectrum; constant independent
  Gaussian noise is a first approximation.
- Continuum uncertainty, line contamination, and coaddition effects require
  further observational characterization.
- Comparisons with the historical Gaussian runs also change sampling.
  Several feature definitions depend on the pixel grid.
- The observed forest currently provides at most **13 complete,
  non-overlapping 2500 km/s chunks**, before ISM/metal masking. The present
  classifier requires **15 sightlines per input**.
- Strong-trough candidates require line identification before being used
  for Ly-alpha absorber or absorber–galaxy analysis.

## Notebook walkthrough

Open `notebooks/cos_obs.ipynb` locally in VS Code or Jupyter for interactive
plots. The notebook was successfully restarted and run from top to bottom.

It covers:

1. Observed flux, reported errors, the known gap, and usable coverage.
2. The tabulated LP1 response and Gaussian comparison.
3. An illustrative EX0 sightline, `MOCK_LOS = 9000`, with instrument
   processing and added noise.
4. Classification accuracies, bootstrap intervals, and confusion matrices.
5. A dropdown viewer for eight strong-trough candidates.

The inspection candidates use flux below 0.85 and complete, non-overlapping
±250 km/s windows. Centers are selected flux minima. Error bars use the
observed file's reported errors; line identifications remain pending.

## Reproduction

Run from the repository root with the project dependencies installed.
The current development environment is Conda `lya_ml`.

Required local inputs:

- `data/raw/EX0_spectra.hdf5` through `EX3_spectra.hdf5`
- The frozen manifest and calibration listed above
- `data/reference/cos_lsf/aa_LSFTable_G130M_1291_LP1_cn.dat`
- `data/raw/observations/pg1048_all.dat` for the notebook

The following commands create a fresh LP1 run. Existing output artifacts
are protected against overwriting.

### Build features and train

```bash
python - <<'PY'
from pathlib import Path
from src.build_cos_features import build_cos_features
from src.train_xgboost import run_training

models = ("EX0", "EX1", "EX2", "EX3")
root = Path("outputs/cos_lsf_snr")

for stage in ("lp1_obs_clean", "lp1_obs_noisy"):
    for model in models:
        build_cos_features(model, stage)

    run_training(
        models=models,
        condition="matched",
        manifest_path=Path(
            "outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz"
        ),
        feature_dir=root / "features" / stage,
        output_dir=root / "runs" / stage,
        tag="z0_b15",
        seed=42,
        max_pk_bin=19,
    )
PY
```

### Summarize

```bash
python -m src.summarize_cos_resolution \
  --stages lp1_obs_clean lp1_obs_noisy \
  --output outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_lp1_obs.json
```

### Artifact locations

- Features: `outputs/cos_lsf_snr/features/<stage>/z0_b15_EX*.npz`
- Models, metrics, and predictions:
  `outputs/cos_lsf_snr/runs/<stage>/z0_b15_s42/matched/EX0_EX1_EX2_EX3/`
- Summary:
  `outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_lp1_obs.json`

## Questions for Taesun

1. What are the exact quasar identity and redshift?
2. Can we confirm the grating, central wavelength settings, binning, and
   coaddition procedure, ideally using the original FITS metadata?
3. How were the continuum and reported errors estimated? Are neighboring
   pixel errors correlated?
4. Which strong troughs are Galactic/ISM, metal, or intrinsic lines, and
   what masks should we use?
5. Are galaxy positions and redshifts available for investigating the
   identified absorbers?
6. Can we obtain the remaining spectra and their observing metadata?

The next scientific priorities are to confirm the observational setup,
establish contamination masks and usable path lengths, and test more
realistic instrument/noise models.