# COS LSF and signal-to-noise experiment

## Status

**Noiseless Gaussian pilot completed. Full COS LSF and S/N experiment in progress.**

- Branch: `experiment/cos-lsf-snr`
- Starting point: `98c0976`, tagged `test-a-uvb-mean-flux`
- Code and results checkpoint: `749a076`
- Checkpoint tag: `cos-gaussian-pilot`
- Observational spectra have not been used.
- The cosmic-variance-floor experiment remains paused.

The completed pilot implements periodic Gaussian convolution and
flux-conserving resampling, validates their numerical behavior, and
measures their effects on four-class classification.

The final deliverable remains a pipeline producing COS-realistic mock
spectra and a thesis figure showing classifier accuracy versus S/N.

## Scientific question

How much classification performance survives instrumental smoothing,
resampling, and eventually observational noise after the ensemble mean
flux has already been matched across simulation models?

This experiment builds on the
[Test A mean-flux experiment](../uvb_mean_flux/README.md).

The Gaussian used here is an initial approximation. The tabulated COS LSF
and noise model have not yet been applied.

## Frozen experimental setup

| Setting | Value |
|---|---|
| Redshift | z = 0 |
| Simulation models | EX0, EX1, EX2, EX3 |
| Sightlines per model | 10,000 |
| Native pixels per sightline | 2499 |
| Velocity length per sightline | 2500 km/s |
| Native pixel width | 2500 / 2499 = 1.0004001601 km/s |
| Bundle size | 15 sightlines |
| Bundles per model | 666 |
| Training bundles per model | 532 |
| Test bundles per model | 134 |
| Four-class training examples | 2128 |
| Four-class test examples | 536 |
| Features per sightline | 46 |
| Features per bundle | 92: feature means and standard deviations |
| Power-spectrum features | All 20 bins retained; `max_pk_bin=19` |
| Split and classifier seed | 42 |
| Noise added | None |
| Four-class chance accuracy | 25% |

All conditions use the same Test A manifest, calibration, feature
extractor, classifier configuration, and train/test assignments.

Manifest:

`outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz`

Calibration:

`outputs/uvb_mean_flux/calibration/z0_b15_seed42.json`

Sightline fingerprint:

`ddffd1d43106cb33e09b7d0f0ae5d9e1db6b8713639ec6d37275dc5b78a4a4c7`

The optical-depth scale for each model was fitted using training
sightlines in Test A. Those scales are reused without refitting.

## Processing sequence

1. Load the original optical depths.
2. Apply the frozen model-specific Test A optical-depth scale.
3. Convert to flux using `flux = exp(-tau_matched)`.
4. Convolve the flux with the Gaussian LSF.
5. Resample the smoothed flux for the rebinned conditions.
6. Extract features using the actual output pixel width.
7. Bundle features using the frozen manifest.
8. Train and evaluate a separate classifier for each condition.

Both training and test spectra receive the same instrument settings.

### Gaussian convolution

Implementation: `apply_cos_gaussian` in
[`src/instrument.py`](../../src/instrument.py).

- Gaussian standard deviation: `sigma_kms = 7.96`.
- Convolution acts along the final array axis.
- Boundary mode: `wrap`, matching the periodic simulation boxes.
- The kernel is normalized.
- Convolution preserves the input grid and mean flux.

Two kernel extents were investigated:

| `truncate` | Approximate extent on each side | Kernel length |
|---|---|---|
| 4.0 | Four standard deviations | 65 pixels |
| 8.0 | Eight standard deviations | 129 pixels |

Changing `truncate` changes the included tails, while the Gaussian
standard deviation remains 7.96 km/s.

The function default remains `truncate=4.0` to preserve the behavior of
earlier calls. The eight-sigma experiment stages explicitly pass
`truncate=8.0`.

### Flux-conserving resampling

Implementation: `resample_flux` in
[`src/instrument.py`](../../src/instrument.py).

The input is treated as piecewise constant within each pixel. The
cumulative flux integral is interpolated to the output pixel edges;
differences of that integral give the output pixel averages.

The full velocity interval is retained:

- Input: 2499 pixels.
- Output: 1249 pixels.
- Output pixel width: `2500 / 1249 = 2.0016012810 km/s`.
- Total velocity length: 2500 km/s.

This replaces the initial proposal to discard one pixel and average
adjacent pairs. No final input pixel is discarded.

The builder checks that each sightline's mean flux is preserved to an
absolute tolerance of `1e-12`. Feature extraction receives `dv_out`.

## Numerical validation

The instrument test module reported **11 passed**:

```bash
python -m pytest tests/test_instrument.py -q
```

The checks cover:

- Preservation of constant flux under convolution.
- Convolution across the periodic boundary and impulse normalization.
- Five Gaussian power-transfer measurements using periodic cosine modes.
- Resampling of a known five-pixel input to three output pixels.
- Preservation of a flat continuum under resampling.
- Preservation of batch means, velocity length, and the original input.
- Eight-sigma kernel power transfer across the full discrete Fourier grid.

A separate reconstruction check also reproduced one cached native EX2
training bundle using the frozen calibration and exact native pixel width.

### Initial five-mode power-transfer check

For a Gaussian with standard deviation sigma, the ideal power transfer is

`P_smoothed(k) / P_original(k) = exp(-(k * sigma)**2)`.

Here `k = 2*pi*f`, with units of s/km.

These measurements used the original four-sigma kernel and convolution
only. Each target was mapped to the nearest allowed periodic Fourier mode.

| Target k | Actual k | Measured ratio | Ideal ratio at actual k |
|---|---|---|---|
| 0.020 | 0.020106 | 0.974729 | 0.974711 |
| 0.050 | 0.050265 | 0.852153 | 0.852067 |
| 0.100 | 0.100531 | 0.527207 | 0.527101 |
| 0.200 | 0.201062 | 0.077181 | 0.077193 |
| 0.300 | 0.299080 | 0.003460 | 0.003456 |

All five passed with relative tolerance `5e-3`.

These initial tests used the rounded pixel width `1.0004 km/s`.
Production feature generation uses the exact value `2500 / 2499`.

### Why the validation was extended

The native power-spectrum features extend to approximately
`3.14 s/km`, beyond the initial five test modes.

Truncating the Gaussian at four standard deviations leaves residual
high-k power above the ideal Gaussian transfer. A separate impulse-response
calculation at `k ≈ 1.000283 s/km` gave approximately:

- Four-sigma kernel power transfer: `1.85e-10`.
- Ideal Gaussian power transfer: `2.93e-28`.

Small residual signals can potentially remain useful in a noiseless
classification experiment.

The eight-sigma impulse-response test checks every mode on the discrete
Fourier grid against the ideal transfer, using:

- Relative tolerance: `1e-6`.
- Absolute tolerance: `1e-24`.

The absolute tolerance governs comparisons where the ideal transfer is
extremely small. Passing this test does not establish relative agreement
below that absolute floor.

## Four-class results

All conditions use mean-flux-matched spectra and the frozen setup above.

| Stage | Kernel extent | Pixels | Accuracy | 95% accuracy CI |
|---|---|---|---|---|
| `native` | No LSF | 2499 | 84.89% | [80.97, 88.62]% |
| `gaussian` | Four sigma | 2499 | 85.07% | [81.34, 88.62]% |
| `gaussian_rebinned` | Four sigma | 1249 | 56.16% | [52.24, 60.07]% |
| `gaussian_truncate8` | Eight sigma | 2499 | 60.63% | Not yet calculated |
| `gaussian_rebinned_truncate8` | Eight sigma | 1249 | 55.97% | Not yet calculated |

The saved eight-sigma summaries contain confidence intervals for paired
accuracy differences, rather than individual accuracy intervals.

### Paired accuracy differences

Changes below are the first condition minus the second, in percentage
points.

| Comparison | Change | 95% paired CI |
|---|---|---|
| `gaussian - native` | +0.19 pp | [-1.68, +2.05] pp |
| `gaussian_rebinned - native` | -28.73 pp | [-33.02, -24.44] pp |
| `gaussian_rebinned - gaussian` | -28.92 pp | [-33.58, -24.25] pp |
| `gaussian_truncate8 - gaussian` | -24.44 pp | [-28.36, -20.52] pp |
| `gaussian_rebinned_truncate8 - gaussian_truncate8` | -4.66 pp | [-8.21, -1.12] pp |

### Bootstrap method

- 50,000 bootstrap resamples.
- Random seed: `20260913`.
- Resampling unit: paired test-bundle ID.
- Number of paired test bundles: 134.
- Each sampled bundle retains its four model-class examples together.
- The same bundle draws are used across conditions.
- Intervals are the 2.5th and 97.5th percentiles.

These are exploratory intervals conditional on the fitted classifiers
and frozen split. They do not include retraining uncertainty or uncertainty
across independent cosmic realizations.

## Findings and interpretation

### 1. The original Gaussian-only accuracy was sensitive to truncation

The initial four-sigma Gaussian gave 85.07%, close to the native 84.89%.

Extending the kernel to eight sigma at the same Gaussian width and native
sampling reduced accuracy to 60.63%. The paired decrease was 24.44 points,
with an interval excluding zero.

The original 85.07% result therefore cannot support a robust claim that
Gaussian smoothing leaves classification performance unchanged.

Residual high-k power is a plausible contributor. The feature families
responsible for the accuracy change have not yet been isolated.

### 2. Resampling produces an additional decrease with the longer kernel

Using the eight-sigma kernel on both grids, resampling changes accuracy
from 60.63% to 55.97%:

`-4.66 pp, 95% CI [-8.21, -1.12] pp`.

This comparison measures resampling together with its effects on the
current feature representation.

### 3. Correct pixel width does not make every feature grid-independent

The current extractor has several dependencies on pixel sampling:

- Power-spectrum bin edges are constructed from each grid's available
  k range, so resampling changes the bin boundaries.
- Fixed wavelet levels correspond to different physical velocity scales
  when pixel width changes.
- The gradient proxy uses `np.gradient(flux)` without velocity spacing.
- Some statistical proxies use unweighted pixel sums.
- The minimum peak-width selection is specified in pixels.

The contribution of these effects to the remaining accuracy difference
has not been measured.

### 4. The current Gaussian-plus-resampling accuracy is 55.97%

This is above the four-class chance level of 25% for this simulated,
noiseless experiment.

It does not yet establish performance on observed COS spectra or show
that the remaining discrimination is uniquely caused by feedback physics.

## Code and saved artifacts

### Implementation

- [`src/instrument.py`](../../src/instrument.py):
  Gaussian convolution and conservative resampling.
- [`src/build_cos_features.py`](../../src/build_cos_features.py):
  Instrument-processed features using the frozen Test A setup.
- [`tests/test_instrument.py`](../../tests/test_instrument.py):
  Numerical validation.
- [`src/summarize_cos_resolution.py`](../../src/summarize_cos_resolution.py):
  Paired bootstrap summaries for the original three conditions.
- [`src/train_xgboost.py`](../../src/train_xgboost.py):
  Existing training implementation used for all conditions.

### Tracked result summaries

- [Original three-condition results](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_resolution.json)
- [Native-grid truncation comparison](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_truncation.json)
- [Eight-sigma resampling comparison](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_resampling_truncate8.json)

### Local inputs and generated outputs

Reproduction requires the local raw spectra, frozen manifest, and native
feature cache in addition to the tracked code and calibration.

Relevant locations, relative to the repository root:

- Raw spectra: `data/raw/EX0_spectra.hdf5` through `EX3_spectra.hdf5`.
- Native features: `outputs/uvb_mean_flux/features/`.
- Instrument-processed features:
  `outputs/cos_lsf_snr/features/<stage>/z0_b15_EX*.npz`.
- Fitted models, metrics, and predictions:
  `outputs/cos_lsf_snr/runs/<stage>/z0_b15_s42/matched/EX0_EX1_EX2_EX3/`.

Generated feature files contain the LOS features, bundled features,
sightline fingerprint, and processing metadata.

The current builder records kernel extent explicitly. The initial
`gaussian` and `gaussian_rebinned` caches predate that metadata field;
both used `truncate=4.0`.

### Entry points

Example command to build one feature file:

```bash
python -m src.build_cos_features \
    --model EX0 \
    --stage gaussian_rebinned_truncate8
```

Four-class training uses `src.train_xgboost.run_training` with:

- Models: `EX0`, `EX1`, `EX2`, `EX3`.
- Condition: `matched`.
- Frozen Test A manifest.
- Stage-specific feature and run directories.
- Tag: `z0_b15`.
- Seed: `42`.
- `max_pk_bin=19`.

The original three-condition summary can be generated with:

```bash
python -m src.summarize_cos_resolution
```

Feature builders, training runs, and summary writers protect existing
outputs from overwriting.

The two eight-sigma comparison summaries were generated using terminal
Python blocks. Their inputs and bootstrap settings are recorded in the
JSON files and this README. A reusable comparison entry point covering
all stages remains to be added.

## Remaining work

1. Review feature definitions that change with pixel sampling.
2. Consolidate comparison and bootstrap code into a reusable entry point,
   including individual accuracy intervals for the eight-sigma conditions.
3. Incorporate the appropriate tabulated COS LSF once the observing setup
   is established.
4. Define the noise model and S/N convention, including whether S/N is
   specified per pixel or per resolution element.
5. Run the four-class and six pairwise instrument/noise experiments.
6. Produce the accuracy-versus-S/N thesis figure with uncertainty estimates.
7. Validate the observational processing once collaborator data are available.

The six pairwise instrument-processed comparisons and the S/N sweep have
not yet been run.# COS LSF and signal-to-noise experiment

## Status

**Noiseless Gaussian pilot completed. Full COS LSF and S/N experiment in progress.**

- Branch: `experiment/cos-lsf-snr`
- Starting point: `98c0976`, tagged `test-a-uvb-mean-flux`
- Code and results checkpoint: `749a076`
- Checkpoint tag: `cos-gaussian-pilot`
- Observational spectra have not been used.
- The cosmic-variance-floor experiment remains paused.

The completed pilot implements periodic Gaussian convolution and
flux-conserving resampling, validates their numerical behavior, and
measures their effects on four-class classification.

The final deliverable remains a pipeline producing COS-realistic mock
spectra and a thesis figure showing classifier accuracy versus S/N.

## Scientific question

How much classification performance survives instrumental smoothing,
resampling, and eventually observational noise after the ensemble mean
flux has already been matched across simulation models?

This experiment builds on the
[Test A mean-flux experiment](../uvb_mean_flux/README.md).

The Gaussian used here is an initial approximation. The tabulated COS LSF
and noise model have not yet been applied.

## Frozen experimental setup

| Setting | Value |
|---|---|
| Redshift | z = 0 |
| Simulation models | EX0, EX1, EX2, EX3 |
| Sightlines per model | 10,000 |
| Native pixels per sightline | 2499 |
| Velocity length per sightline | 2500 km/s |
| Native pixel width | 2500 / 2499 = 1.0004001601 km/s |
| Bundle size | 15 sightlines |
| Bundles per model | 666 |
| Training bundles per model | 532 |
| Test bundles per model | 134 |
| Four-class training examples | 2128 |
| Four-class test examples | 536 |
| Features per sightline | 46 |
| Features per bundle | 92: feature means and standard deviations |
| Power-spectrum features | All 20 bins retained; `max_pk_bin=19` |
| Split and classifier seed | 42 |
| Noise added | None |
| Four-class chance accuracy | 25% |

All conditions use the same Test A manifest, calibration, feature
extractor, classifier configuration, and train/test assignments.

Manifest:

`outputs/uvb_mean_flux/manifests/z0_b15_seed42.npz`

Calibration:

`outputs/uvb_mean_flux/calibration/z0_b15_seed42.json`

Sightline fingerprint:

`ddffd1d43106cb33e09b7d0f0ae5d9e1db6b8713639ec6d37275dc5b78a4a4c7`

The optical-depth scale for each model was fitted using training
sightlines in Test A. Those scales are reused without refitting.

## Processing sequence

1. Load the original optical depths.
2. Apply the frozen model-specific Test A optical-depth scale.
3. Convert to flux using `flux = exp(-tau_matched)`.
4. Convolve the flux with the Gaussian LSF.
5. Resample the smoothed flux for the rebinned conditions.
6. Extract features using the actual output pixel width.
7. Bundle features using the frozen manifest.
8. Train and evaluate a separate classifier for each condition.

Both training and test spectra receive the same instrument settings.

### Gaussian convolution

Implementation: `apply_cos_gaussian` in
[`src/instrument.py`](../../src/instrument.py).

- Gaussian standard deviation: `sigma_kms = 7.96`.
- Convolution acts along the final array axis.
- Boundary mode: `wrap`, matching the periodic simulation boxes.
- The kernel is normalized.
- Convolution preserves the input grid and mean flux.

Two kernel extents were investigated:

| `truncate` | Approximate extent on each side | Kernel length |
|---|---|---|
| 4.0 | Four standard deviations | 65 pixels |
| 8.0 | Eight standard deviations | 129 pixels |

Changing `truncate` changes the included tails, while the Gaussian
standard deviation remains 7.96 km/s.

The function default remains `truncate=4.0` to preserve the behavior of
earlier calls. The eight-sigma experiment stages explicitly pass
`truncate=8.0`.

### Flux-conserving resampling

Implementation: `resample_flux` in
[`src/instrument.py`](../../src/instrument.py).

The input is treated as piecewise constant within each pixel. The
cumulative flux integral is interpolated to the output pixel edges;
differences of that integral give the output pixel averages.

The full velocity interval is retained:

- Input: 2499 pixels.
- Output: 1249 pixels.
- Output pixel width: `2500 / 1249 = 2.0016012810 km/s`.
- Total velocity length: 2500 km/s.

This replaces the initial proposal to discard one pixel and average
adjacent pairs. No final input pixel is discarded.

The builder checks that each sightline's mean flux is preserved to an
absolute tolerance of `1e-12`. Feature extraction receives `dv_out`.

## Numerical validation

The instrument test module reported **11 passed**:

```bash
python -m pytest tests/test_instrument.py -q
```

The checks cover:

- Preservation of constant flux under convolution.
- Convolution across the periodic boundary and impulse normalization.
- Five Gaussian power-transfer measurements using periodic cosine modes.
- Resampling of a known five-pixel input to three output pixels.
- Preservation of a flat continuum under resampling.
- Preservation of batch means, velocity length, and the original input.
- Eight-sigma kernel power transfer across the full discrete Fourier grid.

A separate reconstruction check also reproduced one cached native EX2
training bundle using the frozen calibration and exact native pixel width.

### Initial five-mode power-transfer check

For a Gaussian with standard deviation sigma, the ideal power transfer is

`P_smoothed(k) / P_original(k) = exp(-(k * sigma)**2)`.

Here `k = 2*pi*f`, with units of s/km.

These measurements used the original four-sigma kernel and convolution
only. Each target was mapped to the nearest allowed periodic Fourier mode.

| Target k | Actual k | Measured ratio | Ideal ratio at actual k |
|---|---|---|---|
| 0.020 | 0.020106 | 0.974729 | 0.974711 |
| 0.050 | 0.050265 | 0.852153 | 0.852067 |
| 0.100 | 0.100531 | 0.527207 | 0.527101 |
| 0.200 | 0.201062 | 0.077181 | 0.077193 |
| 0.300 | 0.299080 | 0.003460 | 0.003456 |

All five passed with relative tolerance `5e-3`.

These initial tests used the rounded pixel width `1.0004 km/s`.
Production feature generation uses the exact value `2500 / 2499`.

### Why the validation was extended

The native power-spectrum features extend to approximately
`3.14 s/km`, beyond the initial five test modes.

Truncating the Gaussian at four standard deviations leaves residual
high-k power above the ideal Gaussian transfer. A separate impulse-response
calculation at `k ≈ 1.000283 s/km` gave approximately:

- Four-sigma kernel power transfer: `1.85e-10`.
- Ideal Gaussian power transfer: `2.93e-28`.

Small residual signals can potentially remain useful in a noiseless
classification experiment.

The eight-sigma impulse-response test checks every mode on the discrete
Fourier grid against the ideal transfer, using:

- Relative tolerance: `1e-6`.
- Absolute tolerance: `1e-24`.

The absolute tolerance governs comparisons where the ideal transfer is
extremely small. Passing this test does not establish relative agreement
below that absolute floor.

## Four-class results

All conditions use mean-flux-matched spectra and the frozen setup above.

| Stage | Kernel extent | Pixels | Accuracy | 95% accuracy CI |
|---|---|---|---|---|
| `native` | No LSF | 2499 | 84.89% | [80.97, 88.62]% |
| `gaussian` | Four sigma | 2499 | 85.07% | [81.34, 88.62]% |
| `gaussian_rebinned` | Four sigma | 1249 | 56.16% | [52.24, 60.07]% |
| `gaussian_truncate8` | Eight sigma | 2499 | 60.63% | Not yet calculated |
| `gaussian_rebinned_truncate8` | Eight sigma | 1249 | 55.97% | Not yet calculated |

The saved eight-sigma summaries contain confidence intervals for paired
accuracy differences, rather than individual accuracy intervals.

### Paired accuracy differences

Changes below are the first condition minus the second, in percentage
points.

| Comparison | Change | 95% paired CI |
|---|---|---|
| `gaussian - native` | +0.19 pp | [-1.68, +2.05] pp |
| `gaussian_rebinned - native` | -28.73 pp | [-33.02, -24.44] pp |
| `gaussian_rebinned - gaussian` | -28.92 pp | [-33.58, -24.25] pp |
| `gaussian_truncate8 - gaussian` | -24.44 pp | [-28.36, -20.52] pp |
| `gaussian_rebinned_truncate8 - gaussian_truncate8` | -4.66 pp | [-8.21, -1.12] pp |

### Bootstrap method

- 50,000 bootstrap resamples.
- Random seed: `20260913`.
- Resampling unit: paired test-bundle ID.
- Number of paired test bundles: 134.
- Each sampled bundle retains its four model-class examples together.
- The same bundle draws are used across conditions.
- Intervals are the 2.5th and 97.5th percentiles.

These are exploratory intervals conditional on the fitted classifiers
and frozen split. They do not include retraining uncertainty or uncertainty
across independent cosmic realizations.

## Findings and interpretation

### 1. The original Gaussian-only accuracy was sensitive to truncation

The initial four-sigma Gaussian gave 85.07%, close to the native 84.89%.

Extending the kernel to eight sigma at the same Gaussian width and native
sampling reduced accuracy to 60.63%. The paired decrease was 24.44 points,
with an interval excluding zero.

The original 85.07% result therefore cannot support a robust claim that
Gaussian smoothing leaves classification performance unchanged.

Residual high-k power is a plausible contributor. The feature families
responsible for the accuracy change have not yet been isolated.

### 2. Resampling produces an additional decrease with the longer kernel

Using the eight-sigma kernel on both grids, resampling changes accuracy
from 60.63% to 55.97%:

`-4.66 pp, 95% CI [-8.21, -1.12] pp`.

This comparison measures resampling together with its effects on the
current feature representation.

### 3. Correct pixel width does not make every feature grid-independent

The current extractor has several dependencies on pixel sampling:

- Power-spectrum bin edges are constructed from each grid's available
  k range, so resampling changes the bin boundaries.
- Fixed wavelet levels correspond to different physical velocity scales
  when pixel width changes.
- The gradient proxy uses `np.gradient(flux)` without velocity spacing.
- Some statistical proxies use unweighted pixel sums.
- The minimum peak-width selection is specified in pixels.

The contribution of these effects to the remaining accuracy difference
has not been measured.

### 4. The current Gaussian-plus-resampling accuracy is 55.97%

This is above the four-class chance level of 25% for this simulated,
noiseless experiment.

It does not yet establish performance on observed COS spectra or show
that the remaining discrimination is uniquely caused by feedback physics.

## Code and saved artifacts

### Implementation

- [`src/instrument.py`](../../src/instrument.py):
  Gaussian convolution and conservative resampling.
- [`src/build_cos_features.py`](../../src/build_cos_features.py):
  Instrument-processed features using the frozen Test A setup.
- [`tests/test_instrument.py`](../../tests/test_instrument.py):
  Numerical validation.
- [`src/summarize_cos_resolution.py`](../../src/summarize_cos_resolution.py):
  Paired bootstrap summaries for the original three conditions.
- [`src/train_xgboost.py`](../../src/train_xgboost.py):
  Existing training implementation used for all conditions.

### Tracked result summaries

- [Original three-condition results](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_resolution.json)
- [Native-grid truncation comparison](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_truncation.json)
- [Eight-sigma resampling comparison](../../outputs/cos_lsf_snr/summary/z0_b15_s42_four_class_resampling_truncate8.json)

### Local inputs and generated outputs

Reproduction requires the local raw spectra, frozen manifest, and native
feature cache in addition to the tracked code and calibration.

Relevant locations, relative to the repository root:

- Raw spectra: `data/raw/EX0_spectra.hdf5` through `EX3_spectra.hdf5`.
- Native features: `outputs/uvb_mean_flux/features/`.
- Instrument-processed features:
  `outputs/cos_lsf_snr/features/<stage>/z0_b15_EX*.npz`.
- Fitted models, metrics, and predictions:
  `outputs/cos_lsf_snr/runs/<stage>/z0_b15_s42/matched/EX0_EX1_EX2_EX3/`.

Generated feature files contain the LOS features, bundled features,
sightline fingerprint, and processing metadata.

The current builder records kernel extent explicitly. The initial
`gaussian` and `gaussian_rebinned` caches predate that metadata field;
both used `truncate=4.0`.

### Entry points

Example command to build one feature file:

```bash
python -m src.build_cos_features \
    --model EX0 \
    --stage gaussian_rebinned_truncate8
```

Four-class training uses `src.train_xgboost.run_training` with:

- Models: `EX0`, `EX1`, `EX2`, `EX3`.
- Condition: `matched`.
- Frozen Test A manifest.
- Stage-specific feature and run directories.
- Tag: `z0_b15`.
- Seed: `42`.
- `max_pk_bin=19`.

The original three-condition summary can be generated with:

```bash
python -m src.summarize_cos_resolution
```

Feature builders, training runs, and summary writers protect existing
outputs from overwriting.

The two eight-sigma comparison summaries were generated using terminal
Python blocks. Their inputs and bootstrap settings are recorded in the
JSON files and this README. A reusable comparison entry point covering
all stages remains to be added.

## Remaining work

1. Review feature definitions that change with pixel sampling.
2. Consolidate comparison and bootstrap code into a reusable entry point,
   including individual accuracy intervals for the eight-sigma conditions.
3. Incorporate the appropriate tabulated COS LSF once the observing setup
   is established.
4. Define the noise model and S/N convention, including whether S/N is
   specified per pixel or per resolution element.
5. Run the four-class and six pairwise instrument/noise experiments.
6. Produce the accuracy-versus-S/N thesis figure with uncertainty estimates.
7. Validate the observational processing once collaborator data are available.

The six pairwise instrument-processed comparisons and the S/N sweep have
not yet been run.