# Strong-absorber export and depth-only baseline: 19 October 2026

The corrected recovery-threshold detector produced **433,195 noisy windows at
S/N 20 and 50**, exported in a **106.9 MB** compressed NPZ file. Saturated and
extended troughs are retained with flags. The depth-only four-class baseline
achieves **25.704%** balanced accuracy at S/N 20 and **26.343%** at S/N 50.
These are results for selected absorber windows in the frozen z=0 mocks.

This checkpoint supersedes the earlier 50 km/s merging procedure for the Sujin
export. The earlier README incidence and profile summaries describe that older
selection; newly generated incidence/stack tables live in the corrected run's
output directory and must be consulted before quoting revised contrasts.

## Detection and visual checks

- Preserve the fixed mean-flux matching, COS LP1 processing, 362 bins per
  2,500 km/s sightline and periodic mock boundaries.
- Detect local minima of the three-bin mean of the noisy flux. Require
  F_min < 0.85 and (1 - F_min)/(sigma_bin/sqrt(3)) >= 3, with
  sigma_bin = sqrt(2)/SNR.
- Eligible minima belong to one absorber while their connected smoothed-flux
  region stays below 0.85. Recovery to 0.85 or above separates regions.
  Join connected regions across the periodic endpoint.
- Keep the deepest eligible minimum; exact ties use the smallest pixel index.
  `n_merged` counts all eligible minima in the region, including the retained one.
- Extract 87 unsmoothed, unclipped noisy flux bins centered on the detection.
  Bin spacing is 6.906077 km/s, and bin centers span about ±296.96 km/s.
  The fixed cutout can contain a neighboring absorber or truncate a broad trough.

All six modeled S/N levels were rerun for the notebook analysis; the compact
export contains only S/N 20 and 50. The six-level catalog contains 1,357,131
windows, including three noise realizations per condition. Detection counts
need not vary monotonically with S/N because the significance cut changes the
effective flux threshold at low S/N.

Visual QA inspected single, typical merged and largest merged examples at both
exported S/N levels. Merging is determined by recovery, so a two-minimum absorber
need not be wider than every single-minimum absorber. The largest examples have
100 qualifying minima in a connected trough covering the entire sightline.

For EX1/LOS 4696 and EX3/LOS 5753, native optical depths reproduce cached LP1 flux
with maximum absolute error 0.0 in the recorded check. Their native maximum
fluxes are approximately 0.53238 and 0.00999. The project interpretation is
saturated sightlines through dense halo gas; the reconstruction verifies that
the saturation is already present in the native spectra.

The clean-spectrum census has 12, 8, 7 and 17 whole-period cases for EX0, EX1,
EX2 and EX3, respectively: 44 model–LOS cases among 39,960 used model–LOS cases.
This does not count distinct parent IDs across models or noise realizations.
No additional saturation or width rejection cut was introduced.

## Compact export and fixed roles

Archive: `sujin_strong_absorber_windows_snr20_50.npz`.
Only noisy float32 flux windows are exported; noiseless windows are omitted.
Noise seeds: 20260929, 20260930 and 20261001.

| Role | Saved per-chunk split | Assigned LOS per model |
|---|---|---:|
| train | fit_los | 6,384 |
| validation | tune_los | 798 |
| calibration | calibration_los | 798 |
| test | test_los | 2,010 |

Validation uses exactly the saved tuning sightlines. Calibration remains reserved.
Every window has a stable `window_id`, a role and `group_id = los_id`.
Parent IDs have one role across models, S/N values, seeds and overlapping windows;
zero parent-role overlap was verified. Some assigned sightlines may yield no
selected windows, and remain assigned in the exported parent-role array.

Window counts below sum the three noise realizations. They are not independent
sightline counts.

| S/N | Role | EX0 | EX1 | EX2 | EX3 | Total |
|---:|---|---:|---:|---:|---:|---:|
| 20 | train | 40,171 | 39,608 | 39,711 | 41,921 | 161,411 |
| 20 | validation | 5,049 | 4,984 | 4,930 | 5,126 | 20,089 |
| 20 | calibration | 5,247 | 5,337 | 5,412 | 5,452 | 21,448 |
| 20 | test | 12,487 | 12,412 | 12,503 | 13,071 | 50,473 |
| 50 | train | 28,225 | 28,038 | 28,212 | 29,663 | 114,138 |
| 50 | validation | 3,638 | 3,555 | 3,519 | 3,718 | 14,430 |
| 50 | calibration | 3,766 | 3,821 | 3,846 | 3,944 | 15,377 |
| 50 | test | 8,844 | 8,781 | 8,870 | 9,334 | 35,829 |

| S/N | Windows | Whole-period trough windows | Trough extends beyond cutout |
|---:|---:|---:|---:|
| 20 | 253,421 | 99 | 660 |
| 50 | 179,774 | 121 | 742 |
| Total | 433,195 | 220 | 1,402 |

Whole-period windows are a subset of the windows extending beyond the cutout.
The latter comprise 0.324% of the export. These are window records, not distinct
parent sightlines. Flags are measured separately on each smoothed noisy spectrum:
`whole_los_below_recovery`, `trough_fits_window` and `trough_n_bins`.

The exporter verified the saved split and training signature, reproduced every
exported noisy window, and checked all arrays after archive readback. Input hashes,
field descriptions and the archive checksum are recorded with the data.

## Depth-only four-class baseline

The only predictor is `depth = 1 - fmin`, where fmin is the selected three-bin
smoothed minimum. One model is fitted per S/N using all three training noise
copies. Candidates are class-balanced decision trees with maximum depth 1–5,
a minimum leaf size of 1% of training windows, and a training-majority control.
The 798 validation LOS select the largest mean balanced accuracy across noise
seeds; ties within 1e-12 prefer the smaller tree. Training is not repeated on
validation data, and calibration data remain unused. No window is removed using
its saturation or width flags.

Balanced accuracy is the mean of the four class recalls, with a 25% chance/control
reference. It weights windows equally within each class. Each reported score is
the mean of the three test-seed scores, rather than a score for aggregated windows.

| S/N | Selected tree depth | Balanced accuracy (%) | 95% bundle interval (%) | Seed SD (pp) | Ordinary accuracy (%) |
|---:|---:|---:|---|---:|---:|
| 20 | 4 | 25.704 | 25.359–26.064 | 0.054 | 25.976 |
| 50 | 2 | 26.343 | 25.916–26.777 | 0.202 | 26.928 |

| S/N | Noise seed | Test windows | Balanced accuracy (%) | EX0 recall (%) | EX1 recall (%) | EX2 recall (%) | EX3 recall (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20260929 | 16,832 | 25.642 | 27.242 | 10.382 | 17.213 | 47.731 |
| 20 | 20260930 | 16,822 | 25.737 | 26.675 | 10.722 | 17.611 | 47.942 |
| 20 | 20261001 | 16,819 | 25.733 | 26.567 | 11.355 | 17.894 | 47.118 |
| 50 | 20260929 | 11,956 | 26.170 | 0.000 | 21.044 | 14.732 | 68.905 |
| 50 | 20260930 | 11,913 | 26.566 | 0.000 | 21.436 | 15.652 | 69.176 |
| 50 | 20261001 | 11,960 | 26.294 | 0.000 | 20.933 | 15.147 | 69.096 |

Intervals use 5,000 shared bootstrap draws of the original 134 test bundles,
including any bundles without detections. All windows, classes and noise copies
from a sampled bundle move together. The same bundle draws are used at both S/N
levels. Intervals are conditional on the fitted models and validation choices;
they exclude retraining uncertainty and variation between independent simulations.
Seed scatter describes test-noise sensitivity of the same fitted estimator.

The gains over balanced chance are 0.704 and 1.343 percentage points. Both
conditional intervals exclude 25%, but practical discrimination is weak and
class performance is uneven. At S/N 50, EX0 recall is zero in every seed, while
EX3 recall is about 69%. This result establishes a simple reference; it does not
measure the maximum information obtainable from depth or establish how well
full-window classifiers will perform.

Future full-window comparisons should use the same roles and test-window IDs,
and estimate paired differences using the same test bundles. These per-window
scores are not directly comparable to classifiers operating on full sightlines,
15-sightline bundles or sums across multiple chunks. No observational accuracy
or posterior model probability is inferred here.

## Files and reproducibility

Notebook: [cos_strong_absorbers.ipynb](../../notebooks/cos_strong_absorbers.ipynb).
Helpers, run in the notebook kernel:

- [sujin_window_qa_step.py](../../notebooks/sujin_window_qa_step.py)
- [sujin_export_step.py](../../notebooks/sujin_export_step.py)
- [sujin_depth_baseline_step.py](../../notebooks/sujin_depth_baseline_step.py)

Each corrected analysis is stored under
`outputs/cos_lsf_snr/strong_absorbers/recovery085_<UTC>/`:

- `window_qa/`: inspected examples, native-source audit and clean-spectrum census.
- Analysis root: corrected six-S/N incidence, contrasts, stacks, asymmetry,
  figures and provenance.
- `sujin_snr20_50/`: compact archive, `SUJIN_BRIEF.md`, meeting counts,
  role/model/seed counts, flags, QA copies, provenance and export receipt.
- `sujin_depth_baseline_v1/`: fitted models, decision rules, validation candidates,
  summary/seed/class metrics, `DEPTH_BASELINE_RESULTS.md`, test predictions keyed
  by window ID, per-bundle confusion counts, shared bootstrap weights and provenance.

The numerical tables and file size above are the user's verified local outputs
reported on 19 October 2026. Run-specific dataset IDs, input identities, software
versions and checksums are retained in the generated provenance/receipt files.
