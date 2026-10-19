# Raw per-chunk aggregation: 18 October 2026 checkpoint

The cache-only rerun sums the **raw, uncalibrated XGBoost log-probabilities**
and selects the largest class score with equal priors. It reuses the saved
tuned models, held-out predictions and grouped split. No models were fitted
and no features were rebuilt for this rerun.

| Result across 5 tasks × 6 S/N levels | Raw aggregation | Isotonic reference |
|---|---:|---:|
| All Gaussian checks pass, including all three noise seeds | 27/30 | 4/30 |
| N95 available from the mean of the three seed curves | 18/30 | 6/30 |

The isotonic reference was recomputed with the same draws and settings.
All 864 old Gaussian-check rows matched in direct accuracy, Gaussian
accuracy and acceptance, and all 120 old N95 rows matched in threshold,
method and sustained-crossing flag. Numeric comparisons used absolute
tolerance 1e-10. This establishes continuity with the previous run.

## Workflow and statistical procedure

- Safe entry point: [cos_per_chunk_raw_analysis.ipynb](../../notebooks/cos_per_chunk_raw_analysis.ipynb).
- Analysis implementation: [raw_combination_analysis_step.py](../../notebooks/raw_combination_analysis_step.py).
- Original training and executed isotonic notebook:
  [cos_per_chunk_tuned.ipynb](../../notebooks/cos_per_chunk_tuned.ipynb).
- Cache inputs: `outputs/cos_lsf_snr/perchunk_tuned/`.
- Corrected results: `outputs/cos_lsf_snr/perchunk_tuned/raw_combination/`.

Run the validation cell in the safe notebook with the `lya_ml` kernel,
then its analysis step. The original training notebook can load features
and fall back to fitting, so it is not the cache-only rerun entry point.
The validation cell reads 129 required cache inputs and checks training
signatures, receipts, class/seed/LOS ordering and original bundle IDs.
Saved calibrators reproduce the cached isotonic predictions. Both sources
remain available for reliability and diagnostic comparisons.

The setup retains z = 0 EX0–EX3 mocks, frozen mean-flux matching, COS LP1
processing and 46 features per sightline. Per class, the grouped split has
6,384 fitting, 798 tuning, 798 calibration and 2,010 test sightlines.
Three noisy copies give 19,152 fitting rows per class. Hyperparameters
were tuned at S/N 20 and reused across S/N 10, 15, 20, 30, 50 and 80.
Noise seeds are 20260929, 20260930 and 20261001, with per-bin noise
sigma = sqrt(2)/SNR under the mock convention.

Direct evaluation uses 10,000 draws without replacement for every integer
N from 1 to 200. Draw indices are shared across sources, tasks, S/N levels
and noise seeds. Gaussian checks at N = 50, 100, 150 and 200 use
N(1 - N/2010) times the sample covariance (ddof = 1). Acceptance remains
an absolute class-accuracy difference no larger than max(0.02, 3 MCSE).
Four-class evaluation retains the full covariance of its three competing
margins. Extrapolation to independent new chunks uses N times the mean
and N times the covariance, without finite-population correction.

## Why the aggregation source changed

Diagnostics of the earlier held-out test results motivated this change.
It is a post-diagnostic analysis, not a choice made before seeing those
test results. The finding concerns this run, not isotonic calibration
in general.

The saved raw arrays show no floor or ceiling signatures. They had already
passed the original normalization/clipping helper; EPS = 1e-6 and that
helper are unchanged. Pre-helper raw outputs were not saved.

The earlier printed isotonic metric counted probability entries at either
boundary before the helper. The audit reconstructs that metric and counts
lower-bound entries, upper-bound entries and affected chunks separately.
For a saturated binary prediction there is one floor and one ceiling:
the either-bound entry fraction equals the affected-chunk fraction,
whereas the lower-entry fraction is half that fraction.

| Task | Isotonic lower entries (%) | Chunk records with a floor (%) |
|---|---:|---:|
| EX0–EX3 | 0.03662 | 0.07324 |
| EX2–EX3 | 0.18173 | 0.36346 |
| EX0–EX2 | 0.03386 | 0.06772 |
| EX0–EX1 | 0.03870 | 0.07739 |
| Four-class | 0.14044 | 0.44776 |

These percentages pool six S/N levels and three noise realizations:
72,360 class-conditioned chunk records per binary task and 144,720 for
four-class. They are not counts of independent parent sightlines.
Four-class renormalization shifts the upper endpoint below 1 - EPS:
there are zero literal upper-bound entries but 41 chunk records at the
renormalized ceiling.

For EX0–EX1 at S/N 10, seed 20260929, true EX1, just 7/2,010 chunks
(0.348%) contribute 55.653% of the absolute calibrated log-margin mass
and 43.066% of the harmful (negative-margin) mass. Their maximum absolute
margin is 13.815510, versus 0.138209 for raw predictions in that subset.
The calibrated median absolute margin is 0.029237. Rates depend on the
condition: EX2–EX3 at S/N 10 has 18–19 floor chunks out of 2,010 true-EX2
records in the displayed seeds. Extreme margins can favor either class;
absolute evidence concentration alone does not determine batch accuracy.

## N95 from the mean seed curve

| Task | S/N 10 | 15 | 20 | 30 | 50 | 80 |
|---|---:|---:|---:|---:|---:|---:|
| EX0–EX3 | 3745 | 473 | 440 | 198 | 164 | 136 |
| EX2–EX3 | 179 | 132 | 116 | 88 | 69 | 60 |
| EX0–EX2 | G | G | 803 | 427 | 325 | 189 |
| EX0–EX1 | D | D | >10000 | D | 2685 | 814 |
| Four-class | D | G + D | D | D | D | >10000 |

- **G:** at least one required Gaussian check failed.
- **D:** at least one estimated true-versus-competitor mean margin is nonpositive.
- **>10000:** the otherwise accepted Gaussian extrapolation does not reach
  95% by the existing maximum N = 10,000; N95 is stored as missing.

Counts through 200 are direct first crossings under without-replacement
sampling from the finite held-out pool; larger counts use the checked
independent-chunk Gaussian extrapolation. All reported direct crossings
in this table remain at or above 95% through N = 200. The mean-curve
crossing is not the mean of individual-seed N95 values. Individual-seed
scatter and missing reasons are saved separately.

The three failed task/S/N checks are EX0–EX2 at S/N 10 and 15, and
four-class at S/N 15. The other missing N95 values survive the switch
to raw probabilities because of their estimated margin drift or the
maximum-N limit. They do not establish a physical cosmic-variance floor.

## Exact N = 15 comparison

The comparison uses the original 134 held-out bundles, aligned by bundle
ID. The same 5,000 bundle bootstrap draws estimate paired differences
after averaging the three noise seeds within each bundle. Intervals are
pointwise 95% percentile intervals, conditional on the fitted pipelines
and this held-out pool.

Four-class results:

| S/N | Per-chunk sum (%) | Bundled classifier (%) | Difference [95% interval], pp |
|---|---:|---:|---:|
| 10 | 34.83 | 31.59 | +3.23 [0.56, 5.91] |
| 15 | 37.00 | 32.71 | +4.29 [1.43, 7.03] |
| 20 | 38.74 | 30.91 | +7.84 [5.16, 10.57] |
| 30 | 42.97 | 36.26 | +6.72 [4.10, 9.39] |
| 50 | 44.84 | 39.55 | +5.29 [2.11, 8.40] |
| 80 | 47.45 | 39.93 | +7.52 [4.48, 10.57] |

For example, EX2–EX3 at S/N 20 improves by 5.35 pp [1.74, 8.83],
and EX0–EX1 at S/N 80 by 10.82 pp [6.34, 15.30]. Full results for all
four pairs are in `n15_vs_bundled.csv`. This compares complete pipelines:
the per-chunk model also changes the training sample construction,
feature representation and tuning. The differences cannot be attributed
solely to summing predictions.

## Observational QA and provisional forecasts

The independently inventoried archive contains 52 spectra and 1,139
coverage candidates. The earlier assumed total of 1,113 is not imposed.
Snapshot: `outputs/cos_observational_qa/qa_20261018T223903588165Z/`.

The S/N mapping uses sqrt(2/mean propagated output-bin variance) and
assigns each in-range chunk to the nearest modeled noise variance
2/SNR^2; ties select the lower S/N. Values below 10 or above 80 remain
unassigned and are excluded from the scenario mixtures. Propagated
neighboring-bin covariance is recorded separately; this assignment
does not reproduce that covariance in the cached mocks.

The exported scenarios require a companion fit file, coverage inside
the supplied limits and an assigned S/N bin. `any_observed_bins` requires
at least one unmasked bin and gives 1,025 chunks. Requiring at least half
of the 362 bins to remain observed gives 995, 938 and 864 chunks for
mask half-widths of 75, 150 and 225 km/s, respectively. These are
sensitivity scenarios, not adopted retention cuts or validated filling.
Unresolved line identifications and the supplied-window interpretation
remain part of the recorded observational QA.

For each true class and noise seed, mixture forecasts use
sum_s n_s mu_s and sum_s n_s Sigma_s, then compute batch classification
accuracy. They do not average per-chunk accuracies. Every occupied S/N
bin must pass its Gaussian checks; seed-mean forecasts require all
three seeds to pass.

For the provisional 1,025-chunk scenario:

| Pair | Expected accuracy (%) | Range across three seeds (%) |
|---|---:|---:|
| EX0–EX3 | 99.9201 | 99.8898–99.9378 |
| EX2–EX3 | Approximately 100; rounded | Approximately 100; rounded |
| EX0–EX2 | Withheld | Gaussian checks fail in occupied S/N bins 10 and 15 |
| EX0–EX1 | 81.2158 | 69.7580–88.1727 |

The seed range is not a confidence interval. Identical forecasts for
`any_observed_bins` at different mask widths follow from identical
mixture counts; the cached moments contain no mask-width dependence.
No final mask width, retention rule or filling method has been adopted.

These are expected discrimination accuracies under **unfilled z = 0
mocks and independent chunks**. They are not measured accuracy on the
unlabeled observations or posterior probabilities for the models.
Masking/filling, continuum errors, observed noise covariance, redshift
differences and correlations among chunks from the same QSO are not
validated by this forecast. A final observational result requires the
adopted processing to be tested on development mocks before relying on
the held-out evaluation.

## Saved outputs and provenance

The validation diagnostics are in
`raw_combination/cache_diagnostics_20261018T225437143629Z/`.
Each analysis run has a separate `raw_combination/analysis_<UTC>/`
directory with:

- Accuracy curves, class-by-seed Gaussian checks, N95 and missing reasons.
- PNG/PDF curves for S/N 10, 20, 30 and 50; N95 across all six S/N values
  with individual-seed scatter; Gaussian checks and the N = 15 comparison.
- Full class-conditional means and covariances for both sources.
- Shared direct draws, joint-normal draws, bootstrap weights and bundle IDs.
- Paired N = 15 scores and raw/isotonic comparison tables.
- QA mixture counts, mixed moments and provisional forecasts.
- Input identities, code hash, analysis settings, package versions and
  output checksums in `provenance.json`.

The result summary used for this note has SHA256
`a30e7737ac771b65c0d441e4dd0e92e622ac3ac11609000e094e4be787313ea2`
(`analysis_summary.txt`, supplied on 18 October 2026). The complete
training-cache identity and QA mapping/counts are retained in each
run's provenance. Original training artifacts and isotonic results
remain preserved.

The corrected raw sum is ready for subsequent mock work, including
Test B and the z approximately 0.1 rerun. Observational mask/fill
validation remains a separate unfinished part of the analysis.
