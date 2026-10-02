"""Absorber census on noise-free, periodic COS LSF convolved spectra."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def _minima(flux):
    """One midpoint per minimum, including flat and boundary-crossing minima."""
    n = len(flux)
    peaks, _ = find_peaks(-np.tile(flux, 3))
    return peaks[(peaks >= n) & (peaks < 2 * n)] - n


def _distances(a, b, n_bins, dv):
    separation = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    return np.minimum(separation, n_bins - separation) * dv


def _strong_environment(flux, pixels, dv, strong_cut):
    """Distances to edges of contiguous strong-bin regions."""
    strong = flux < strong_cut
    inside = strong[pixels]
    missing = np.full(len(pixels), np.nan)
    if not strong.any():
        return missing.copy(), missing.copy(), inside, 0

    starts = strong & ~np.roll(strong, 1)
    if strong.all():
        labels = np.zeros(len(flux), dtype=int)
    else:
        labels = np.cumsum(starts) - 1
        if strong[0] and strong[-1]:
            labels[labels < 0] = labels[-1]
        labels[~strong] = -1

    strong_pixels = np.flatnonzero(strong)
    distance = np.maximum(
        _distances(pixels, strong_pixels, len(flux), dv) - dv / 2,
        0.0,
    )
    nearest = distance.min(axis=1)

    # Exclude the whole containing complex, not just the minimum itself.
    own = labels[pixels]
    same_complex = (
        (own[:, None] >= 0)
        & (own[:, None] == labels[strong_pixels][None, :])
    )
    other = np.where(same_complex, np.inf, distance).min(axis=1)
    other[~np.isfinite(other)] = np.nan
    return nearest, other, inside, int(labels.max() + 1)


def _match(a, b, n_bins, dv, match_kms):
    """Maximize match count, then minimize total periodic velocity separation."""
    distance = _distances(a, b, n_bins, dv)
    allowed = distance <= match_kms
    if not len(a) or not len(b):
        empty = np.array([], dtype=int)
        return empty, empty, allowed

    # One forbidden assignment costs more than every allowed edge combined.
    penalty = (min(len(a), len(b)) + 1) * (match_kms + 1.0)
    rows, columns = linear_sum_assignment(
        np.where(allowed, distance, penalty)
    )
    keep = allowed[rows, columns]
    return rows[keep], columns[keep], allowed


def build_absorber_census(
    data, *, flux_cuts=(0.85, 0.95), match_kms=30.0, strong_cut=0.5
):
    """Detect minima and classify pairwise/all-model catalog uniqueness.

    Threshold catalogs are nested. Matching is one-to-one within each pair
    and threshold. A unique_all detection is unmatched against every other
    model. An isolated_all detection additionally has no eligible neighbor
    in any other model, so its uniqueness is not a matching competition.

    Depth is 1 - F_min. Strong complexes use the same model's F < strong_cut
    bins. Absence of a strong complex is retained as NaN, not zero distance.
    """
    flux = np.asarray(data.lp1, dtype=np.float64)
    models = tuple(data.models)
    los_ids = np.asarray(data.los_ids)
    bundle_ids = np.asarray(data.bundle_ids)
    dv = float(data.dv_lp1)
    cuts = tuple(sorted(set(float(c) for c in flux_cuts)))

    if (
        flux.ndim != 3
        or flux.shape[:2] != (len(models), len(los_ids))
        or len(models) < 2
        or len(set(models)) != len(models)
        or len(los_ids) == 0
        or len(np.unique(los_ids)) != len(los_ids)
        or bundle_ids.shape != los_ids.shape
        or flux.shape[2] < 3
        or not np.all(np.isfinite(flux))
        or not np.isfinite(dv)
        or dv <= 0
    ):
        raise ValueError("Invalid paired flux arrays, IDs or pixel spacing")
    if (
        not cuts
        or not all(strong_cut < c < 1.0 for c in cuts)
        or not 0.0 < strong_cut < 1.0
        or not np.isfinite(match_kms)
        or not 0.0 < match_kms < flux.shape[2] * dv / 2
    ):
        raise ValueError("Invalid flux thresholds or matching window")

    n_bins = flux.shape[2]
    length = n_bins * dv
    records, groups = [], {}

    for position, los_id in enumerate(los_ids):
        for m, model in enumerate(models):
            profile = flux[m, position]
            pixels = _minima(profile)
            pixels = pixels[profile[pixels] < cuts[-1]]
            near, other, inside, n_complexes = _strong_environment(
                profile, pixels, dv, strong_cut
            )
            first = len(records)
            for j, pixel in enumerate(pixels):
                records.append((
                    model, int(los_id), int(bundle_ids[position]),
                    int(pixel), (pixel + 0.5) * dv,
                    profile[pixel], 1.0 - profile[pixel],
                    n_complexes, bool(inside[j]), near[j], other[j],
                ))
            groups[m, position] = np.arange(
                first, len(records), dtype=int
            )

    catalog = pd.DataFrame(records, columns=[
        "model", "los_id", "bundle_id", "pixel", "velocity_kms",
        "f_min", "depth", "n_strong_complexes", "inside_strong",
        "distance_to_strong_kms", "distance_to_other_complex_kms",
    ])
    catalog.insert(0, "absorber_id", np.arange(len(catalog)))
    catalog = catalog.astype({
        "pixel": int, "f_min": float, "depth": float,
        "inside_strong": bool, "n_strong_complexes": int,
        "distance_to_strong_kms": float,
        "distance_to_other_complex_kms": float,
    })
    f_min = catalog["f_min"].to_numpy()
    pixels = catalog["pixel"].to_numpy()
    pair_rows = []

    for cut in cuts:
        for position, los_id in enumerate(los_ids):
            for a, b in combinations(range(len(models)), 2):
                broad_a = groups[a, position]
                broad_b = groups[b, position]
                ids_a = broad_a[f_min[broad_a] < cut]
                ids_b = broad_b[f_min[broad_b] < cut]

                ia, ib, allowed = _match(
                    pixels[ids_a], pixels[ids_b],
                    n_bins, dv, match_kms,
                )
                partner_a = np.full(len(ids_a), -1, dtype=int)
                partner_b = np.full(len(ids_b), -1, dtype=int)
                partner_a[ia], partner_b[ib] = ids_b[ib], ids_a[ia]

                for m, other_m, ids, partners, neighbors, broad_other in (
                    (a, b, ids_a, partner_a, allowed.sum(axis=1), broad_b),
                    (b, a, ids_b, partner_b, allowed.sum(axis=0), broad_a),
                ):
                    broad_near = (
                        _distances(
                            pixels[ids], pixels[broad_other], n_bins, dv
                        ) <= match_kms
                    ).any(axis=1)

                    for j, absorber_id in enumerate(ids):
                        partner = int(partners[j])
                        offset = (
                            ((pixels[partner] - pixels[absorber_id]) * dv
                             + length / 2) % length - length / 2
                            if partner >= 0 else np.nan
                        )
                        pair_rows.append((
                            cut, models[m], models[other_m], int(los_id),
                            int(absorber_id), partner, partner >= 0,
                            int(neighbors[j]), bool(neighbors[j] == 0),
                            bool(broad_near[j]), offset,
                        ))

    pair_status = pd.DataFrame(pair_rows, columns=[
        "flux_cut", "model", "other_model", "los_id", "absorber_id",
        "partner_id", "matched", "n_candidates", "no_neighbor",
        "has_broader_neighbor", "offset_kms",
    ]).astype({
        "flux_cut": float, "absorber_id": int, "matched": bool,
        "n_candidates": int, "no_neighbor": bool,
        "has_broader_neighbor": bool,
    })
    pair_status["unmatched"] = ~pair_status["matched"]

    flags = pair_status.groupby(["flux_cut", "absorber_id"]).agg(
        n_compared=("other_model", "size"),
        n_matched_models=("matched", "sum"),
        n_no_neighbor_models=("no_neighbor", "sum"),
        n_broader_neighbor_models=("has_broader_neighbor", "sum"),
    ).reset_index()
    if not flags["n_compared"].eq(len(models) - 1).all():
        raise RuntimeError("Incomplete model comparisons")

    detections = flags.merge(
        catalog, on="absorber_id", how="left", validate="many_to_one"
    )
    detections["unique_all"] = detections["n_matched_models"].eq(0)
    detections["isolated_all"] = (
        detections["n_no_neighbor_models"].eq(len(models) - 1)
    )
    detections["isolated_at_broadest_cut"] = (
        detections["n_broader_neighbor_models"].eq(0)
    )
    detections["unique_with_competition"] = (
        detections["unique_all"] & ~detections["isolated_all"]
    )
    detections["isolated_with_weaker_neighbor"] = (
        detections["isolated_all"]
        & ~detections["isolated_at_broadest_cut"]
    )

    # Include sightlines with zero detections in the path denominator.
    index = pd.MultiIndex.from_product(
        [cuts, models, los_ids],
        names=["flux_cut", "model", "los_id"],
    )
    los_counts = detections.groupby(
        ["flux_cut", "model", "los_id"]
    ).agg(
        n_detected=("absorber_id", "size"),
        n_unique_all=("unique_all", "sum"),
        n_isolated_all=("isolated_all", "sum"),
        n_competition=("unique_with_competition", "sum"),
        n_weaker_neighbor=("isolated_with_weaker_neighbor", "sum"),
    ).reindex(index, fill_value=0).reset_index()

    count_columns = [c for c in los_counts if c.startswith("n_")]
    los_counts[count_columns] = los_counts[count_columns].astype(int)
    bundle_lookup = dict(zip(los_ids, bundle_ids))
    los_counts["bundle_id"] = los_counts["los_id"].map(bundle_lookup)
    los_counts["path_kms"] = length

    summary = los_counts.groupby(
        ["flux_cut", "model"], sort=False
    ).agg(
        **{c: (c, "sum") for c in count_columns},
        path_kms=("path_kms", "sum"),
    )
    for stem in ("detected", "unique_all", "isolated_all"):
        summary[f"{stem}_per_1e4_kms"] = (
            1e4 * summary[f"n_{stem}"] / summary["path_kms"]
        )
    summary["unique_fraction"] = (
        summary["n_unique_all"]
        / summary["n_detected"].replace(0, np.nan)
    )

    unique = detections.loc[detections["unique_all"]].copy()
    unique["no_strong_complex"] = unique["n_strong_complexes"].eq(0)
    unique["outside_distance_kms"] = (
        unique["distance_to_strong_kms"].where(~unique["inside_strong"])
    )
    properties = unique.groupby(["flux_cut", "model"]).agg(
        unique_median_depth=("depth", "median"),
        unique_inside_strong_fraction=("inside_strong", "mean"),
        unique_no_strong_fraction=("no_strong_complex", "mean"),
        unique_outside_median_distance_kms=("outside_distance_kms", "median"),
    )
    summary = summary.join(properties).reset_index()

    pair_index = pd.MultiIndex.from_tuples(
        [
            (cut, a, b)
            for cut in cuts
            for a in models
            for b in models
            if a != b
        ],
        names=["flux_cut", "model", "other_model"],
    )
    pair_summary = pair_status.groupby(
        ["flux_cut", "model", "other_model"]
    ).agg(
        n_detected=("absorber_id", "size"),
        n_matched=("matched", "sum"),
        n_unmatched=("unmatched", "sum"),
        n_no_neighbor=("no_neighbor", "sum"),
    ).reindex(pair_index, fill_value=0).reset_index()
    pair_summary["unmatched_per_1e4_kms"] = (
        1e4 * pair_summary["n_unmatched"] / (len(los_ids) * length)
    )

    return {
        "catalog": catalog,
        "detections": detections,
        "pair_status": pair_status,
        "los_counts": los_counts,
        "summary": summary,
        "pair_summary": pair_summary,
        "settings": {
            "flux_cuts": list(cuts),
            "match_kms": float(match_kms),
            "strong_cut": float(strong_cut),
            "periodic": True,
            "distance_reference": "Edges of same-model strong-bin regions",
            "path_per_los_kms": float(length),
            "n_los": len(los_ids),
            "matching": "Maximum cardinality, then minimum total separation",
        },
    }

def bootstrap_absorber_rates(
    census, *, n_draws=50_000, seed=20261003, batch_size=1000
):
    """Pointwise percentile intervals using shared draws of whole LOS bundles."""
    for name, value, minimum in (
        ("n_draws", n_draws, 2),
        ("seed", seed, 0),
        ("batch_size", batch_size, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    counts = census["los_counts"].copy()
    counts["n_isolated_broad"] = (
        counts["n_isolated_all"] - counts["n_weaker_neighbor"]
    )
    cuts = sorted(counts["flux_cut"].unique())
    models = list(census["summary"]["model"].unique())
    bundles = np.sort(counts["bundle_id"].unique())
    columns = [
        "n_detected", "n_unique_all",
        "n_isolated_all", "n_isolated_broad",
    ]

    index = pd.MultiIndex.from_product(
        [cuts, models, bundles],
        names=["flux_cut", "model", "bundle_id"],
    )
    grouped = counts.groupby(
        ["flux_cut", "model", "bundle_id"]
    )[columns + ["path_kms"]].sum().reindex(index)

    if grouped.isna().any().any():
        raise ValueError("Missing model/threshold/bundle combinations")

    n_cut, n_model, n_bundle = len(cuts), len(models), len(bundles)
    if n_bundle < 2:
        raise ValueError("At least two bundles are needed")

    values = grouped[columns].to_numpy(dtype=float).reshape(
        n_cut, n_model, n_bundle, 4
    ).transpose(2, 0, 1, 3)
    paths = grouped["path_kms"].to_numpy(dtype=float).reshape(
        n_cut, n_model, n_bundle
    ).transpose(2, 0, 1)

    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.all(np.isfinite(paths))
        or np.any(paths <= 0)
        or not np.allclose(paths, paths[:, :1, :1])
    ):
        raise ValueError("Invalid counts or unequal paired path coverage")

    bundle_path = paths[:, 0, 0]

    def measures(total, path):
        rates = 1e4 * total / np.asarray(path)[..., None]
        percent = np.divide(
            100.0 * total[..., 1], total[..., 0],
            out=np.full(total.shape[:-1], np.nan),
            where=total[..., 0] > 0,
        )
        return np.concatenate([rates, percent[..., None]], axis=-1)

    estimate = measures(values.sum(axis=0), bundle_path.sum())
    draws = np.empty((n_draws, n_cut, n_model, 5))
    rng = np.random.Generator(np.random.PCG64(seed))
    probability = np.full(n_bundle, 1.0 / n_bundle)

    for start in range(0, n_draws, batch_size):
        size = min(batch_size, n_draws - start)
        weights = rng.multinomial(n_bundle, probability, size=size)
        total = (weights @ values.reshape(n_bundle, -1)).reshape(
            size, n_cut, n_model, 4
        )
        path = (weights @ bundle_path)[:, None, None]
        draws[start:start + size] = measures(total, path)

    def interval(sample):
        finite = sample[np.isfinite(sample)]
        if finite.size:
            low, high = np.quantile(finite, [0.025, 0.975])
            return float(low), float(high), len(finite)
        return np.nan, np.nan, 0

    metrics = (
        "detected_rate", "unique_rate", "isolated_rate",
        "isolated_broad_rate", "unique_percent",
    )
    levels, contrasts = [], []

    for c, cut in enumerate(cuts):
        for k, metric in enumerate(metrics):
            for m, model in enumerate(models):
                low, high, valid = interval(draws[:, c, m, k])
                levels.append({
                    "flux_cut": cut,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate[c, m, k],
                    "low": low,
                    "high": high,
                    "valid_draws": valid,
                    "unit": "%" if k == 4 else "per 10000 km/s",
                })

            for a, b in combinations(range(n_model), 2):
                difference = draws[:, c, b, k] - draws[:, c, a, k]
                low, high, valid = interval(difference)
                contrasts.append({
                    "flux_cut": cut,
                    "metric": metric,
                    "model_a": models[a],
                    "model_b": models[b],
                    "comparison": f"{models[b]} - {models[a]}",
                    "difference": estimate[c, b, k] - estimate[c, a, k],
                    "low": low,
                    "high": high,
                    "valid_draws": valid,
                    "unit": "pp" if k == 4 else "per 10000 km/s",
                })

    return {
        "rates": pd.DataFrame(levels),
        "contrasts": pd.DataFrame(contrasts),
        "bundle_counts": grouped.reset_index(),
        "settings": {
            "n_draws": n_draws,
            "seed": seed,
            "batch_size": batch_size,
            "n_bundles": n_bundle,
            "rng": "PCG64",
            "interval": "Pointwise 2.5th-97.5th percentile",
            "resampling": "Shared whole-bundle draws across models and cuts",
            "scope": (
                "Conditional on the frozen calibration and simulation volumes"
            ),
        },
    }


def plot_absorber_patterns(census, inference):
    """Incidence intervals and descriptive depth/distance ECDFs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    detections = census["detections"]
    rates = inference["rates"].set_index(
        ["flux_cut", "model", "metric"]
    )
    models = list(census["summary"]["model"].unique())
    cuts = sorted(census["settings"]["flux_cuts"])
    if len(models) != 4:
        raise ValueError("Expected four models")

    colors = ("#222222", "#0072B2", "#D55E00", "#009E73")
    broad_cut = max(cuts)
    max_distance = census["settings"]["path_per_los_kms"] / 2

    fig, axes = plt.subplots(
        len(cuts), 3, figsize=(15, 3.4 * len(cuts) + 2),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.17,
        top=0.85, hspace=0.35, wspace=0.30,
    )
    selected_rates = inference["rates"].loc[
        inference["rates"]["metric"].isin(
            ["unique_rate", "isolated_broad_rate"]
        )
    ]
    rate_limit = max(0.01, 1.15 * selected_rates["high"].max())

    def ecdf(ax, values, color, linestyle, maximum, alpha=1.0):
        values = np.asarray(values, dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if not len(values):
            return
        x = np.r_[0.0, values, maximum]
        y = np.r_[
            0.0,
            np.arange(1, len(values) + 1) / len(values),
            1.0,
        ]
        ax.step(
            x, y, where="post", color=color, ls=linestyle,
            lw=1.8 if linestyle == "-" else 1.2,
            alpha=alpha,
        )

    for row, cut in enumerate(cuts):
        count_ax, depth_ax, distance_ax = axes[row]

        for m, (model, color) in enumerate(zip(models, colors)):
            for offset, metric, filled in (
                (-0.10, "unique_rate", True),
                (+0.10, "isolated_broad_rate", False),
            ):
                result = rates.loc[(cut, model, metric)]
                x = m + offset
                count_ax.vlines(
                    x, result["low"], result["high"], color=color
                )
                count_ax.hlines(
                    [result["low"], result["high"]],
                    x - 0.035, x + 0.035, color=color,
                )
                count_ax.plot(
                    x, result["estimate"], "o", color=color, ms=6,
                    markerfacecolor=color if filled else "white",
                )

            all_lines = detections.loc[
                (detections["flux_cut"] == cut)
                & (detections["model"] == model)
            ]
            for subset, linestyle, alpha in (
                (all_lines, "--", 0.45),
                (all_lines.loc[all_lines["unique_all"]], "-", 1.0),
            ):
                ecdf(
                    depth_ax, subset["depth"],
                    color, linestyle, 1.0, alpha,
                )
                outside = subset.loc[~subset["inside_strong"]]
                ecdf(
                    distance_ax, outside["distance_to_strong_kms"],
                    color, linestyle, max_distance, alpha,
                )

        count_ax.set(
            title=f"F < {cut:g} | Incidence",
            ylabel="Detections per 10,000 km/s",
            ylim=(0, rate_limit),
            xlim=(-0.45, len(models) - 0.55),
        )
        count_ax.set_xticks(range(len(models)), labels=models)
        depth_ax.set(
            title=f"F < {cut:g} | Depth",
            xlabel=r"Depth, $1-F_{\min}$",
            ylabel="Cumulative fraction",
            xlim=(0, 1), ylim=(0, 1.02),
        )
        distance_ax.set(
            title=f"F < {cut:g} | Strong-complex proximity",
            xlabel="Distance to nearest complex edge [km/s]",
            ylabel="Cumulative fraction",
            xlim=(0, max_distance), ylim=(0, 1.02),
        )

        for ax in axes[row]:
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    count_handles = [
        Line2D(
            [], [], marker="o", ls="none", color="0.25",
            label="Unmatched against all models",
        ),
        Line2D(
            [], [], marker="o", ls="none", color="0.25",
            markerfacecolor="white",
            label=f"No neighbor in other models at F < {broad_cut:g}",
        ),
    ]
    axes[0, 0].legend(
        handles=count_handles, fontsize=8,
        frameon=False, loc="upper left",
    )

    handles = [
        Line2D([], [], color=color, lw=2, label=model)
        for model, color in zip(models, colors)
    ] + [
        Line2D([], [], color="0.4", lw=2, label="CDF: unique"),
        Line2D(
            [], [], color="0.4", ls="--",
            label="CDF: all detections",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.94), ncol=6, frameon=False,
    )
    fig.suptitle(
        "Model-unique absorber patterns", fontsize=17, y=0.985
    )
    fig.text(
        0.5, 0.085,
        "Incidence bars: pointwise 95% paired bundle bootstrap intervals. "
        "CDFs are descriptive and normalized within each subset.",
        ha="center", fontsize=9,
    )
    fig.text(
        0.5, 0.045,
        "Distance CDFs include only detections outside strong complexes "
        "on sightlines containing a same-model F < 0.5 complex.",
        ha="center", fontsize=9, color="0.35",
    )
    return fig