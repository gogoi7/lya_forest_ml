"""Run in the validated cache-only notebook namespace; no training or features."""

from datetime import datetime, timezone
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from IPython.display import display

if not globals().get("CACHE_VALIDATED", False):
    raise RuntimeError("Run the cache-validation cell successfully first.")
if AGGREGATION_SOURCE != "raw":
    raise ValueError("This corrected analysis requires AGGREGATION_SOURCE = 'raw'.")
if "ANALYSIS_STEP_SOURCE" not in globals():
    raise RuntimeError("Use the supplied notebook cell to load this step and record its source hash.")
if not np.array_equal(test_los, bundles[test_bundle_ids].ravel()):
    raise ValueError("Test LOS no longer follow the original sorted bundle IDs.")
if len(test_bundle_ids) != 134 or bundle_size != 15 or len(test_los) != 2010:
    raise ValueError("Expected the original 134 bundles of 15 held-out LOS.")

ANALYSIS_COMPLETE = False
SOURCES = (AGGREGATION_SOURCE, "calibrated")
SOURCE_LABELS = {"raw": "Raw", "calibrated": "Isotonic reference"}
DRAW_BLOCK = 1000                      # Memory control only; same draws and sums.
created = datetime.now(timezone.utc)
RUN_DIR = OUT / created.strftime("analysis_%Y%m%dT%H%M%S%fZ")
start_time = time.monotonic()


def write_json_result(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_npz_result(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


draw_rng = np.random.default_rng(SEED + 1)
draw_indices = np.stack([draw_rng.choice(len(test_los), DIRECT_MAX_N, replace=False)
                         for _ in range(DIRECT_DRAWS)])
if not np.all(np.diff(np.sort(draw_indices, axis=1), axis=1) > 0):
    raise RuntimeError("A direct batch contains duplicate LOS.")
direct_ns = np.arange(1, DIRECT_MAX_N + 1)
long_ns = np.unique(np.r_[DIRECT_MAX_N + 1,
    np.rint(np.geomspace(DIRECT_MAX_N + 1, GAUSSIAN_MAX_N, 100)).astype(int)])
sobol = qmc.Sobol(d=3, scramble=True, seed=SEED + 2)
normal_draws = ndtri(np.clip(sobol.random_base2(GAUSSIAN_SOBOL_POWER), 1e-12, 1-1e-12))
bootstrap_weights = np.random.default_rng(SEED + 3).multinomial(
    len(test_bundle_ids), np.full(len(test_bundle_ids), 1/len(test_bundle_ids)),
    size=BASELINE_BOOTSTRAPS)


def margin_moments(probabilities):
    """Full class-conditional mean/covariance of competing log margins."""
    k = probabilities.shape[0]
    moments = []
    for c in range(k):
        logp = np.log(probabilities[c])
        others = [j for j in range(k) if j != c]
        margins = logp[:, [c]] - logp[:, others]
        mu = margins.mean(axis=0)
        covariance = np.atleast_2d(np.cov(margins, rowvar=False, ddof=1))
        eigval, eigvec = np.linalg.eigh(covariance)
        root = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 0)))
        z = normal_draws[:, :k-1] @ root.T
        moments.append(dict(mu=mu, cov=covariance, z=z, other_classes=others))
    return moments


def orthant_accuracy(mean, covariance, z=None):
    """Original binary analytic / joint-normal multiclass calculation and tie rule."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.atleast_2d(covariance)
    if len(mean) == 1:
        variance = max(float(covariance[0, 0]), 0)
        if variance == 0:
            return float(mean[0] > 0) + 0.5*float(mean[0] == 0)
        return float(ndtr(mean[0]/np.sqrt(variance)))
    if z is None:
        eigval, eigvec = np.linalg.eigh(covariance)
        root = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 0)))
        z = normal_draws[:, :len(mean)] @ root.T
    margins = mean + z
    scores = np.column_stack([np.zeros(len(margins)), -margins])
    return float(tie_credit(scores, 0).mean())


def gaussian_accuracy(moment, n, finite_pool=None):
    factor = float(n)
    if finite_pool is not None:
        if n > finite_pool:
            raise ValueError("Without-replacement N exceeds the test pool.")
        factor *= 1-float(n)/finite_pool
    return orthant_accuracy(n*moment["mu"], factor*moment["cov"],
                            np.sqrt(factor)*moment["z"] if len(moment["mu"]) > 1 else None)


def direct_accuracies(probabilities):
    k = probabilities.shape[0]
    credits = np.empty((k, DIRECT_DRAWS, DIRECT_MAX_N), dtype=float)
    for c in range(k):
        logp = np.log(probabilities[c])
        for first in range(0, DIRECT_DRAWS, DRAW_BLOCK):
            last = min(first + DRAW_BLOCK, DIRECT_DRAWS)
            scores = logp[draw_indices[first:last]].cumsum(axis=1)
            credits[c, first:last] = tie_credit(scores, c)
    by_class = credits.mean(axis=1)
    class_mcse = credits.std(axis=1, ddof=1)/np.sqrt(DIRECT_DRAWS)
    # Shared LOS draws across true classes: average classes BEFORE calculating MCSE.
    balanced_draws = credits.mean(axis=0)
    balanced = balanced_draws.mean(axis=0)
    balanced_mcse = balanced_draws.std(axis=0, ddof=1)/np.sqrt(DIRECT_DRAWS)
    return by_class, class_mcse, balanced, balanced_mcse


RUN_PROVENANCE = {
    "created_utc": created.isoformat(), "status": "running", "aggregation_source": AGGREGATION_SOURCE,
    "reference_source": "calibrated", "analysis_code_sha256": hashlib.sha256(ANALYSIS_STEP_SOURCE.encode()).hexdigest(),
    "training_key": TRAINING_KEY, "cached_input_identity": INPUT_IDENTITY, "input_sha256": INPUT_HASHES,
    "cache_validation_provenance": str((DIAGNOSTIC_DIR/"provenance.json").relative_to(ROOT)),
    "cache_validation_provenance_sha256": file_hash(DIAGNOSTIC_DIR/"provenance.json"),
    "settings": dict(ANALYSIS_SETTINGS, computation_draw_block=DRAW_BLOCK),
    "comparison": "Both saved sources recomputed with identical draw indices, Gaussian checks and bootstrap weights",
    "n95_rule": {
        "direct": "First integer crossing through N=200; later dips reported by sustained flag",
        "extrapolation": "Only after all required Gaussian checks pass and every mean margin is positive",
        "maximum_n": GAUSSIAN_MAX_N,
        "missing_reasons": "Gaussian check failure and nonpositive margins recorded separately; max-N failure only for otherwise valid extrapolation",
    },
    "qa_snapshot": str(QA_SNAPSHOT.relative_to(ROOT)), "qa_input_identity": QA_INPUT_IDENTITY,
    "qa_input_sha256": QA_INPUT_HASHES, "qa_provenance": QA_PROVENANCE,
    "qa_scenario_counts": QA_SCENARIO_SUMMARY.to_dict(orient="records"), "adopted_mixture": None,
    "scenario_forecast_scope": (
        "Provisional S/N-only forecasts for each exported retention/mask-width scenario. "
        "Moments come from unfilled z=0 mocks, not mask/fill-matched mocks. "
        "S/N mapping does not reproduce observational noise covariance. "
        "Independent chunks and equal priors; chunks in one QSO may be correlated. "
        "Not measured observational accuracy or a posterior probability for a true model."
    ),
    "n15_scope": "Complete-pipeline comparison; per-chunk fitting sample construction and tuning also differ",
    "motivation": "Raw aggregation selected following diagnostics of earlier isotonic held-out test results",
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "scipy": scipy.__version__},
}
json.dumps(RUN_PROVENANCE, allow_nan=False)
RUN_DIR.mkdir(parents=True, exist_ok=False)
write_json_result(RUN_DIR/"provenance.json", RUN_PROVENANCE)
save_npz_result(RUN_DIR/"sampling_inputs.npz", draw_indices=draw_indices, test_los=test_los,
                normal_draws=normal_draws, bootstrap_weights=bootstrap_weights, test_bundle_ids=test_bundle_ids)
QA_SNR_MIXTURES.to_csv(RUN_DIR/"qa_snr_mixture_scenarios_used.csv", index=False)

ALL_MOMENTS, ALL_DIRECT, ALL_GAUSSIAN_OK = {}, {}, {}
curve_rows, check_rows, moment_arrays = [], [], {}
for snr in SNRS:
    for task, models in TASKS.items():
        status = {}
        for source in SOURCES:
            status[source] = []
            for r, seed in enumerate(NOISE_SEEDS):
                p = PREDICTIONS[(task, snr)][source][r]
                by_class, class_mcse, balanced, balanced_mcse = direct_accuracies(p)
                moments = margin_moments(p)
                key = (source, task, snr, seed)
                ALL_MOMENTS[key], ALL_DIRECT[key] = moments, balanced
                stem = f"{source}_{task}_snr{snr}_seed{seed}"
                moment_arrays[stem+"_mean"] = np.stack([m["mu"] for m in moments])
                moment_arrays[stem+"_cov"] = np.stack([m["cov"] for m in moments])
                passed_checks = []
                for n in CHECK_NS:
                    for c, model in enumerate(models):
                        approx = gaussian_accuracy(moments[c], n, finite_pool=len(test_los))
                        tolerance = max(CHECK_ABS_TOL, 3*float(class_mcse[c, n-1]))
                        difference = approx-float(by_class[c, n-1])
                        passed = bool(abs(difference) <= tolerance)
                        passed_checks.append(passed)
                        check_rows.append(dict(aggregation_source=source, task=task, snr=snr, noise_seed=seed,
                            n=n, true_model=model, direct_accuracy=float(by_class[c, n-1]),
                            gaussian_accuracy=approx, difference=difference, direct_mcse=float(class_mcse[c, n-1]),
                            tolerance=tolerance, passed=passed, finite_pool_size=len(test_los)))
                passed = all(passed_checks)
                ALL_GAUSSIAN_OK[key] = passed
                status[source].append("PASS" if passed else "FAIL")
                positive_drift = all(np.all(m["mu"] > 0) for m in moments)
                common = dict(aggregation_source=source, task=task, snr=snr, noise_seed=seed,
                              gaussian_check_passed=passed, all_margins_positive=positive_drift)
                for ix, n in enumerate(direct_ns):
                    row = dict(common, n=int(n), method="direct_without_replacement",
                               accuracy=float(balanced[ix]), mcse=float(balanced_mcse[ix]), reportable=True)
                    row.update({f"accuracy_{model}": float(by_class[c, ix]) for c, model in enumerate(models)})
                    curve_rows.append(row)
                # Failed Gaussian values stay in the CSV explicitly marked unreportable.
                for n in long_ns:
                    values = [gaussian_accuracy(m, n) for m in moments]
                    row = dict(common, n=int(n), method="gaussian_iid", accuracy=float(np.mean(values)),
                               mcse=np.nan, reportable=passed)
                    row.update({f"accuracy_{model}": float(values[c]) for c, model in enumerate(models)})
                    curve_rows.append(row)
        print(f"{TASK_LABELS[task]}, S/N {snr}: raw {','.join(status['raw'])}; "
              f"isotonic {','.join(status['calibrated'])}", flush=True)
    # Save completed S/N levels as a partial checkpoint if execution is interrupted.
    pd.DataFrame(curve_rows).to_csv(RUN_DIR/"accuracy_vs_n_by_seed.csv", index=False)
    pd.DataFrame(check_rows).to_csv(RUN_DIR/"gaussian_validation.csv", index=False)

curves = pd.DataFrame(curve_rows)
checks = pd.DataFrame(check_rows)
summary = curves.groupby(["aggregation_source", "task", "snr", "n", "method"], as_index=False).agg(
    accuracy=("accuracy", "mean"), seed_sd=("accuracy", "std"), seed_min=("accuracy", "min"),
    seed_max=("accuracy", "max"), gaussian_check_passed=("gaussian_check_passed", "all"),
    all_margins_positive=("all_margins_positive", "all"), reportable=("reportable", "all"))
summary.to_csv(RUN_DIR/"accuracy_vs_n.csv", index=False)
save_npz_result(RUN_DIR/"log_ratio_moments.npz", **moment_arrays,
    metadata_json=np.array(json.dumps(dict(model_order=TASKS, sources=SOURCES, covariance_ddof=1,
                                          training_key=TRAINING_KEY, cached_input_identity=INPUT_IDENTITY))))

# Primary-source globals for subsequent analysis, including observational mixtures.
MOMENTS = {key[1:]: value for key, value in ALL_MOMENTS.items() if key[0] == AGGREGATION_SOURCE}
DIRECT = {key[1:]: value for key, value in ALL_DIRECT.items() if key[0] == AGGREGATION_SOURCE}
GAUSSIAN_OK = {key[1:]: value for key, value in ALL_GAUSSIAN_OK.items() if key[0] == AGGREGATION_SOURCE}


def n95_estimate(source, task, snr, seeds):
    direct = np.mean([ALL_DIRECT[(source, task, snr, seed)] for seed in seeds], axis=0)
    passed = all(ALL_GAUSSIAN_OK[(source, task, snr, seed)] for seed in seeds)
    moments = [m for seed in seeds for m in ALL_MOMENTS[(source, task, snr, seed)]]
    positive = all(np.all(m["mu"] > 0) for m in moments)
    common = dict(gaussian_check_passed=passed, all_margins_positive=positive)
    crossing = np.flatnonzero(direct >= .95)
    if len(crossing):
        n = int(crossing[0])+1
        return dict(common, n95=n, method="direct_first_crossing", missing_reasons="",
                    sustained_through_200_or_monotone=bool(np.all(direct[n-1:] >= .95)))
    reasons = []
    if not passed:
        reasons.append("gaussian_check_failed")
    if not positive:
        reasons.append("nonpositive_margin_drift")
    if reasons:
        return dict(common, n95=np.nan, method=reasons[0], missing_reasons=";".join(reasons),
                    sustained_through_200_or_monotone=False)
    def accuracy(n):
        return float(np.mean([gaussian_accuracy(m, n) for m in moments]))
    if accuracy(GAUSSIAN_MAX_N) < .95:
        return dict(common, n95=np.nan, method=f"above_{GAUSSIAN_MAX_N}",
                    missing_reasons=f"above_{GAUSSIAN_MAX_N}", sustained_through_200_or_monotone=False)
    lo, hi = DIRECT_MAX_N+1, GAUSSIAN_MAX_N
    while lo < hi:
        mid = (lo+hi)//2
        if accuracy(mid) >= .95:
            hi = mid
        else:
            lo = mid+1
    return dict(common, n95=lo, method="gaussian_iid", missing_reasons="",
                sustained_through_200_or_monotone=True)


n95_rows = []
for source in SOURCES:
    for task in TASKS:
        for snr in SNRS:
            for label, seeds in [(str(seed), (seed,)) for seed in NOISE_SEEDS]+[("mean", NOISE_SEEDS)]:
                n95_rows.append(dict(aggregation_source=source, task=task, snr=snr, noise_seed=label,
                                     **n95_estimate(source, task, snr, seeds)))
n95 = pd.DataFrame(n95_rows)
n95.to_csv(RUN_DIR/"n95_by_snr.csv", index=False)
n95.loc[n95.n95.isna()].to_csv(RUN_DIR/"n95_missing_reasons.csv", index=False)

# Exact N=15, original test bundles; all comparisons use the same paired weights.
comparison_rows, bundle_score_arrays = [], {}
for source in SOURCES:
    for task, models in TASKS.items():
        for snr in SNRS:
            per_chunk, bundled = [], []
            for r, seed in enumerate(NOISE_SEEDS):
                p = PREDICTIONS[(task, snr)][source][r]
                logp = np.log(p).reshape(len(models), len(test_bundle_ids), bundle_size, len(models)).sum(axis=2)
                chunk_score = np.mean([tie_credit(logp[c], c) for c in range(len(models))], axis=0)
                baseline = BASELINE_ALIGNED[(task, snr, seed)]
                per_chunk.append(chunk_score)
                bundled.append(baseline)
                stem = f"{source}_{task}_snr{snr}_seed{seed}"
                bundle_score_arrays[stem+"_per_chunk"] = chunk_score
                bundle_score_arrays[stem+"_baseline"] = baseline
                comparison_rows.append(dict(aggregation_source=source, task=task, snr=snr, noise_seed=str(seed),
                    per_chunk_accuracy=float(chunk_score.mean()), bundled_accuracy=float(baseline.mean()),
                    difference=float((chunk_score-baseline).mean()), low=np.nan, high=np.nan))
            pc, bb = np.mean(per_chunk, axis=0), np.mean(bundled, axis=0)
            delta = bootstrap_weights @ (pc-bb)/len(test_bundle_ids)
            lo, hi = np.quantile(delta, [.025, .975])
            comparison_rows.append(dict(aggregation_source=source, task=task, snr=snr, noise_seed="mean",
                per_chunk_accuracy=float(pc.mean()), bundled_accuracy=float(bb.mean()),
                difference=float((pc-bb).mean()), low=float(lo), high=float(hi)))
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(RUN_DIR/"n15_vs_bundled.csv", index=False)
save_npz_result(RUN_DIR/"n15_paired_bundle_scores.npz", **bundle_score_arrays, test_bundle_ids=test_bundle_ids)


def mixture_seed_forecast(source, task, seed, mix):
    """Combine evidence moments across occupied bins, then classify the batch."""
    failed_bins = [int(s) for s in mix.snr_resel
                   if not ALL_GAUSSIAN_OK[(source, task, int(s), seed)]]
    passed = not failed_bins
    rows, values = [], []
    k = len(TASKS[task])
    for c, true_model in enumerate(TASKS[task]):
        mixed_mean = np.zeros(k-1)
        mixed_covariance = np.zeros((k-1, k-1))
        for snr, count in mix[["snr_resel", "n_chunks"]].itertuples(index=False, name=None):
            moment = ALL_MOMENTS[(source, task, int(snr), seed)][c]
            mixed_mean += int(count)*moment["mu"]
            mixed_covariance += int(count)*moment["cov"]
        values.append(orthant_accuracy(mixed_mean, mixed_covariance) if passed else np.nan)
        rows.append(dict(true_model=true_model, mixed_mean_json=json.dumps(mixed_mean.tolist()),
                         mixed_covariance_json=json.dumps(mixed_covariance.tolist()),
                         expected_class_accuracy=values[-1]))
    return values, rows, failed_bins


forecast_rows, mixed_moment_rows = [], []
for (width, scenario), mix in QA_SNR_MIXTURES.groupby(["half_width_kms", "scenario"]):
    mix = mix.loc[mix.n_chunks.gt(0), ["snr_resel", "n_chunks"]].sort_values("snr_resel")
    count = int(mix.n_chunks.sum())
    if count <= 0:
        raise ValueError(f"Empty QA mixture: {width}, {scenario}")
    scenario_info = QA_SCENARIO_SUMMARY.loc[
        QA_SCENARIO_SUMMARY.half_width_kms.eq(width) & QA_SCENARIO_SUMMARY.scenario.eq(scenario)].iloc[0]
    for source in SOURCES:
        for task, models in TASKS.items():
            if len(models) != 2:
                continue
            common = dict(aggregation_source=source, task=task, half_width_kms=int(width), scenario=scenario,
                          n_chunks=count, n_qsos=int(scenario_info["QSOs"]), scope="provisional_snr_only")
            seed_values, all_failed = [], set()
            for seed in NOISE_SEEDS:
                values, moment_rows, failed = mixture_seed_forecast(source, task, seed, mix)
                value = float(np.mean(values)) if not failed else np.nan
                seed_values.append(value)
                all_failed.update(failed)
                forecast_rows.append(dict(common, noise_seed=str(seed), expected_accuracy=value,
                    gaussian_check_passed=not failed, failed_snr_bins=json.dumps(failed),
                    missing_reason="occupied_bin_gaussian_check_failed" if failed else "",
                    accuracy_class0=values[0], accuracy_class1=values[1], seed_min=np.nan, seed_max=np.nan))
                mixed_moment_rows.extend(dict(common, noise_seed=seed, **row) for row in moment_rows)
            good = bool(np.isfinite(seed_values).all())
            forecast_rows.append(dict(common, noise_seed="mean",
                expected_accuracy=float(np.mean(seed_values)) if good else np.nan,
                gaussian_check_passed=good, failed_snr_bins=json.dumps(sorted(all_failed)),
                missing_reason="occupied_bin_gaussian_check_failed" if not good else "",
                accuracy_class0=np.nan, accuracy_class1=np.nan,
                seed_min=float(np.min(seed_values)) if good else np.nan,
                seed_max=float(np.max(seed_values)) if good else np.nan))
qa_forecasts = pd.DataFrame(forecast_rows)
qa_forecasts.to_csv(RUN_DIR/"qa_scenario_forecasts.csv", index=False)
pd.DataFrame(mixed_moment_rows).to_csv(RUN_DIR/"qa_scenario_mixed_moments.csv", index=False)

# Summarize the change using matched task/SNR/seed calculations.
comparison_summary_rows = []
for task in TASKS:
    for snr in SNRS:
        row = dict(task=task, snr=snr)
        for source in SOURCES:
            seeds = [ALL_GAUSSIAN_OK[(source, task, snr, seed)] for seed in NOISE_SEEDS]
            nrow = n95.loc[(n95.aggregation_source == source) & (n95.task == task)
                          & (n95.snr == snr) & (n95.noise_seed == "mean")].iloc[0]
            n15row = comparison.loc[(comparison.aggregation_source == source) & (comparison.task == task)
                                   & (comparison.snr == snr) & (comparison.noise_seed == "mean")].iloc[0]
            b = BOUNDARY_DIAGNOSTICS.loc[(BOUNDARY_DIAGNOSTICS.task == task) & (BOUNDARY_DIAGNOSTICS.snr == snr)
                & (BOUNDARY_DIAGNOSTICS.stage == ("raw_saved" if source == "raw" else "isotonic_saved"))]
            e = EVIDENCE_DIAGNOSTICS.loc[(EVIDENCE_DIAGNOSTICS.task == task) & (EVIDENCE_DIAGNOSTICS.snr == snr)
                                        & (EVIDENCE_DIAGNOSTICS.source == source)]
            row.update({f"{source}_gaussian_all_seeds": all(seeds), f"{source}_gaussian_seeds_passed": sum(seeds),
                f"{source}_n95": nrow["n95"], f"{source}_n95_method": nrow["method"],
                f"{source}_n95_missing_reasons": nrow["missing_reasons"],
                f"{source}_direct_n200": float(np.mean([ALL_DIRECT[(source, task, snr, s)][-1] for s in NOISE_SEEDS])),
                f"{source}_exact_n15": n15row["per_chunk_accuracy"],
                f"{source}_floor_chunk_fraction": float(b.n_chunks_any_lower.sum()/b.n_chunks.sum()),
                f"{source}_max_abs_margin": float(e.max_abs_margin.max()),
                f"{source}_max_floor_abs_evidence_share": float(e.floor_abs_evidence_share.max())})
        row["raw_minus_isotonic_n200"] = row["raw_direct_n200"]-row["calibrated_direct_n200"]
        row["raw_minus_isotonic_n15"] = row["raw_exact_n15"]-row["calibrated_exact_n15"]
        comparison_summary_rows.append(row)
aggregation_comparison = pd.DataFrame(comparison_summary_rows)
aggregation_comparison.to_csv(RUN_DIR/"aggregation_comparison.csv", index=False)

# Compare against old on-disk outputs when present; never alter them.
legacy_rows, legacy_hashes = [], {}
for filename in ("gaussian_validation.csv", "n95_by_snr.csv"):
    path = CACHE_ROOT/filename
    if not path.is_file():
        legacy_rows.append(dict(file=filename, status="not_present"))
        continue
    legacy_hashes[str(path.relative_to(ROOT))] = file_hash(path)
    old = pd.read_csv(path, dtype={"noise_seed": str})
    new = (checks if filename == "gaussian_validation.csv" else n95)
    new = new.loc[new.aggregation_source.eq("calibrated")].copy()
    new["noise_seed"] = new.noise_seed.astype(str)
    keys = ["task", "snr", "noise_seed"]
    if filename == "gaussian_validation.csv":
        keys += ["n", "true_model"]
        fields = ["direct_accuracy", "gaussian_accuracy", "passed"]
    else:
        fields = ["n95", "method", "sustained_through_200_or_monotone"]
    if not set(keys+fields).issubset(old.columns) or old.duplicated(keys).any():
        legacy_rows.append(dict(file=filename, status="schema_differs"))
        continue
    merged = old[keys+fields].merge(new[keys+fields], on=keys, suffixes=("_old", "_recomputed"),
                                    how="outer", indicator=True, validate="one_to_one")
    legacy_rows.append(dict(file=filename, field="row_alignment", status="compared",
                            matches=int(merged._merge.eq("both").sum()), rows=len(merged)))
    for field in fields:
        a, b = merged[field+"_old"], merged[field+"_recomputed"]
        if field in ("direct_accuracy", "gaussian_accuracy", "n95"):
            same = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True)
        else:
            same = a.eq(b) | (a.isna() & b.isna())
        same = np.asarray(same) & merged._merge.eq("both").to_numpy()
        legacy_rows.append(dict(file=filename, field=field, status="compared", matches=int(same.sum()), rows=len(merged)))
pd.DataFrame(legacy_rows).to_csv(RUN_DIR/"legacy_isotonic_reproduction.csv", index=False)


def save_figure(fig, stem):
    fig.savefig(RUN_DIR/(stem+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(RUN_DIR/(stem+".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


colors = {10: "#b23a48", 20: "#d48c1f", 30: "#087e8b", 50: "#6055a4"}
raw_summary = summary.loc[summary.aggregation_source.eq(AGGREGATION_SOURCE)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for ax, (task, models) in zip(axes.ravel(), TASKS.items()):
    for snr in PLOT_SNRS:
        sub = raw_summary.loc[raw_summary.task.eq(task) & raw_summary.snr.eq(snr)]
        for method, style in (("direct_without_replacement", "-"), ("gaussian_iid", "--")):
            data = sub.loc[sub.method.eq(method)].sort_values("n")
            if not len(data) or (method == "gaussian_iid" and not data.reportable.all()):
                continue
            ax.plot(data.n, data.accuracy, color=colors[snr], ls=style,
                    label=f"S/N {snr}" if method == "direct_without_replacement" else None)
            ax.fill_between(data.n.to_numpy(), data.seed_min.to_numpy(), data.seed_max.to_numpy(),
                            color=colors[snr], alpha=.13)
    ax.axhline(.95, color=".3", lw=.8, ls=":")
    ax.axhline(1/len(models), color=".7", lw=.6)
    ax.axvline(DIRECT_MAX_N, color=".6", lw=.7, ls=":")
    ax.set(xscale="log", xlim=(1, GAUSSIAN_MAX_N), ylim=(0, 1.015), title=TASK_LABELS[task],
           xlabel="Chunks in a batch, N", ylabel="Balanced accuracy")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="lower right")
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(.03, .94,
    "Raw log-probability sums · equal class priors\n\n"
    "Solid: direct, without replacement, N ≤ 200\n"
    "Dashed: checked Gaussian, independent chunks\n"
    "Shading: range of the three noise seeds\n"
    "Failed Gaussian curves are omitted.\n\n"
    "z = 0 mock discrimination accuracy\nSeed scatter is not a confidence interval.", va="top", fontsize=10, linespacing=1.6)
fig.tight_layout()
save_figure(fig, "accuracy_vs_n_raw")

raw_n95 = n95.loc[n95.aggregation_source.eq(AGGREGATION_SOURCE)]
fig, ax = plt.subplots(figsize=(10, 5.5))
for task in TASKS:
    rows = raw_n95.loc[raw_n95.task.eq(task) & raw_n95.noise_seed.eq("mean")].sort_values("snr")
    line = ax.plot(rows.snr, rows.n95, "o-", label=TASK_LABELS[task])[0]
    for r, seed in enumerate(NOISE_SEEDS):
        points = raw_n95.loc[raw_n95.task.eq(task) & raw_n95.noise_seed.eq(str(seed)) & raw_n95.n95.notna()]
        ax.scatter(points.snr.to_numpy()*(1+.012*(r-1)), points.n95, s=18, marker="x",
                   color=line.get_color(), alpha=.65)
ax.set(yscale="log", ylim=(.9, GAUSSIAN_MAX_N*1.1), xlabel="S/N per resolution element",
       ylabel="First N reaching 95% balanced accuracy", title="Raw aggregation: N₉₅ with individual noise seeds")
ax.set_xticks(SNRS)
ax.legend(ncol=2, fontsize=9)
ax.grid(alpha=.2)
fig.text(.12, .015, "Circles: mean-curve crossing. Crosses: each available seed (slightly offset). Missing reasons are saved separately.", fontsize=9)
fig.tight_layout(rect=(0, .04, 1, 1))
save_figure(fig, "n95_vs_snr_raw")

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
for ax, task in zip(axes.ravel(), TASKS):
    for source, color, marker in (("calibrated", "#aaaaaa", "x"), ("raw", "#087e8b", "o")):
        rows = checks.loc[checks.task.eq(task) & checks.aggregation_source.eq(source)]
        ax.scatter(rows.direct_accuracy, rows.gaussian_accuracy, s=11, alpha=.5, color=color,
                   marker=marker, label=SOURCE_LABELS[source])
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.01), title=TASK_LABELS[task],
           xlabel="Direct class accuracy", ylabel="Gaussian class accuracy (finite pool)")
    ax.grid(alpha=.15)
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(.05, .9, "Each point: class × S/N × seed × check N\n"
    "Check N = 50, 100, 150, 200\n"
    "Covariance includes finite-population correction.\n\n"
    "Acceptance remains |difference| ≤ max(0.02, 3 MCSE).\n"
    "All required checks must pass for extrapolation.", va="top", fontsize=10, linespacing=1.6)
axes.ravel()[0].legend(fontsize=8)
fig.tight_layout()
save_figure(fig, "direct_vs_gaussian")

raw_n15 = comparison.loc[comparison.aggregation_source.eq(AGGREGATION_SOURCE) & comparison.noise_seed.eq("mean")]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for ax, task in zip(axes.ravel(), TASKS):
    rows = raw_n15.loc[raw_n15.task.eq(task)].sort_values("snr")
    ax.plot(rows.snr, 100*rows.difference, "o-", color="#087e8b")
    ax.fill_between(rows.snr.to_numpy(), 100*rows.low.to_numpy(), 100*rows.high.to_numpy(), color="#087e8b", alpha=.18)
    ax.axhline(0, color=".5", lw=.8)
    ax.set(title=TASK_LABELS[task], xlabel="S/N per resolution element", ylabel="Per-chunk − bundled (percentage points)")
    ax.set_xticks(SNRS)
    ax.grid(alpha=.15)
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(.03, .94, "Exact original 134 held-out bundles\n"
    "N = 15 LOS per bundle; aligned by bundle ID\n"
    "Shading: paired 95% bootstrap intervals\n"
    "Seed-mean differences, resampling bundles\n\n"
    "Complete-pipeline comparison:\n"
    "training construction and tuning also differ.", va="top", fontsize=10, linespacing=1.6)
fig.tight_layout()
save_figure(fig, "n15_vs_bundled_raw")

# A concise report remains usable without copying notebook output.
raw_mean_n95 = raw_n95.loc[raw_n95.noise_seed.eq("mean")]
raw_mean_forecasts = qa_forecasts.loc[qa_forecasts.aggregation_source.eq(AGGREGATION_SOURCE)
                                     & qa_forecasts.noise_seed.eq("mean")]
source_counts = []
for source in SOURCES:
    q = n95.loc[n95.aggregation_source.eq(source) & n95.noise_seed.eq("mean")]
    successful = sum(all(ALL_GAUSSIAN_OK[(source, task, snr, seed)] for seed in NOISE_SEEDS)
                     for task in TASKS for snr in SNRS)
    source_counts.append(dict(source=source, gaussian_task_snr_passed=successful,
                              gaussian_task_snr_total=len(TASKS)*len(SNRS), mean_curve_n95_available=int(q.n95.notna().sum())))
source_counts = pd.DataFrame(source_counts)
source_counts.to_csv(RUN_DIR/"analysis_summary.csv", index=False)
lines = ["Raw-probability aggregation analysis", "", source_counts.to_string(index=False), "",
    "N95 from the mean of three seed curves:",
    raw_mean_n95[["task", "snr", "n95", "method", "missing_reasons", "sustained_through_200_or_monotone"]].to_string(index=False),
    "", "Exact N=15 paired comparison (accuracy differences are fractions):",
    raw_n15[["task", "snr", "per_chunk_accuracy", "bundled_accuracy", "difference", "low", "high"]].to_string(index=False),
    "", "Provisional S/N-only scenarios (not a final mask/fill-matched forecast):",
    raw_mean_forecasts[["half_width_kms", "scenario", "task", "n_chunks", "expected_accuracy",
                        "seed_min", "seed_max", "failed_snr_bins"]].to_string(index=False),
    "", "Reproduction of old on-disk isotonic results, if available:",
    pd.DataFrame(legacy_rows).to_string(index=False),
    "", "Boundary and tail diagnostics describe this run, not a general failure of isotonic calibration.",
    "Test-result diagnostics motivated the aggregation change; this is a post-diagnostic analysis.",
    "Seed scatter is not a confidence interval. Monte Carlo errors condition on the fixed test pool.",
    "N15 compares complete pipelines, including different training construction and tuning.",
    "QA forecasts are provisional S/N-only scenarios using unfilled z=0 mocks and independent chunks.",
    "They do not validate masking/filling or estimate measured observational accuracy or model posterior probabilities.",
    "Within-QSO correlations and observational noise covariance remain outside the cached forecast."]
(RUN_DIR/"analysis_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
RUN_PROVENANCE.update(status="complete", elapsed_seconds=time.monotonic()-start_time,
                      legacy_input_sha256=legacy_hashes, summary=source_counts.to_dict(orient="records"))
RUN_PROVENANCE["output_sha256"] = {p.name: file_hash(p) for p in sorted(RUN_DIR.iterdir())
                                   if p.is_file() and p.name != "provenance.json"}
write_json_result(RUN_DIR/"provenance.json", RUN_PROVENANCE)
ANALYSIS_COMPLETE = True

print("\nRaw versus isotonic: all classes, check sizes and three seeds required per task/SNR:")
display(source_counts)
print("Raw N95 (mean curve); all seed results and missing reasons are in the CSV files:")
display(raw_mean_n95[["task", "snr", "n95", "method", "missing_reasons", "sustained_through_200_or_monotone"]])
print("Exact N=15: complete pipelines, paired by the original bundle IDs:")
display(raw_n15[["task", "snr", "per_chunk_accuracy", "bundled_accuracy", "difference", "low", "high"]].round(5))
print("Provisional raw S/N-only forecasts; final mask, retention and filling choices remain open:")
display(qa_forecasts.loc[qa_forecasts.aggregation_source.eq("raw") & qa_forecasts.noise_seed.eq("mean"),
    ["half_width_kms", "scenario", "task", "n_chunks", "expected_accuracy", "seed_min", "seed_max", "failed_snr_bins"]].round(6))
print(f"Completed in {(time.monotonic()-start_time)/60:.1f} minutes. Saved analysis: {RUN_DIR}")
