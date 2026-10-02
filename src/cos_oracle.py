"""Known-field Gaussian oracle for paired, noiseless LP1 spectra."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .snr_sweep import SNR_LEVELS, sigma_per_bin

CHUNK_COUNTS = (1, 2, 5, 10, 13, 15, 30, 50, 100, 200)


def pair_diagnostics(data, flux_cut=0.85):
    """Return squared separations, per-LOS scores, and pooled flux splits."""
    flux = np.asarray(data.lp1, dtype=np.float64)
    if (
        flux.ndim != 3
        or flux.shape[:2] != (len(data.models), len(data.los_ids))
        or flux.shape[2] == 0
        or not np.all(np.isfinite(flux))
    ):
        raise ValueError("Expected finite flux with shape (model, LOS, bin)")
    if not 0.0 < flux_cut < 1.0:
        raise ValueError("flux_cut must lie between zero and one")

    pairs = list(combinations(range(len(data.models)), 2))
    sse = np.empty((flux.shape[1], len(pairs)), dtype=np.float64)
    sightlines, splits = [], []

    for column, (a, b) in enumerate(pairs):
        label = f"{data.models[a]}-{data.models[b]}"
        squared = (flux[a] - flux[b]) ** 2
        low_flux = np.minimum(flux[a], flux[b]) < flux_cut
        total = squared.sum(axis=1)
        low = np.where(low_flux, squared, 0.0).sum(axis=1)
        sse[:, column] = total

        fraction = np.divide(
            low, total,
            out=np.full(total.shape, np.nan),
            where=total > 0.0,
        )
        table = pd.DataFrame({
            "model_pair": label,
            "los_id": data.los_ids,
            "bundle_id": data.bundle_ids,
            "squared_difference": total,
            "low_flux_squared_difference": low,
            "low_flux_bin_fraction": low_flux.mean(axis=1),
            "low_flux_chi2_fraction": fraction,
        })
        for snr in SNR_LEVELS:
            table[f"delta_chi2_snr{snr}"] = (
                total / sigma_per_bin(snr) ** 2
            )
        sightlines.append(table)

        pooled_total = total.sum()
        splits.append({
            "model_pair": label,
            "flux_cut": flux_cut,
            "low_flux_bin_fraction": low_flux.mean(),
            "low_flux_chi2_fraction": (
                low.sum() / pooled_total if pooled_total > 0.0 else np.nan
            ),
        })

    return (
        sse,
        pd.concat(sightlines, ignore_index=True),
        pd.DataFrame(splits),
    )


def chunk_oracle(
    squared_difference,
    pair_labels,
    *,
    n_draws=50_000,
    seed=20261002,
    max_chunks=200,
    batch_size=512,
):
    """Average balanced, known-template Gaussian accuracy over iid LOS draws.

    Chunk identities are sampled with replacement. Each draw uses one class
    consistently across all its chunks. The same LOS sequences are reused
    for every pair, S/N level, and prefix length N.

    mc_se measures Monte Carlo integration error only.
    """
    sse = np.asarray(squared_difference, dtype=np.float64)
    pair_labels = tuple(pair_labels)
    if (
        sse.ndim != 2
        or sse.shape[0] == 0
        or sse.shape[1] != len(pair_labels)
        or len(pair_labels) == 0
        or not np.all(np.isfinite(sse))
        or np.any(sse < 0.0)
    ):
        raise ValueError("Expected finite non-negative squared separations")

    for name, value, minimum in (
        ("n_draws", n_draws, 2),
        ("seed", seed, 0),
        ("max_chunks", max_chunks, 1),
        ("batch_size", batch_size, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    snrs = tuple(SNR_LEVELS)
    shape = (len(snrs), max_chunks, len(pair_labels))
    error_sum = np.zeros(shape)
    error_square_sum = np.zeros(shape)
    rng = np.random.Generator(np.random.PCG64(seed))

    for start in range(0, n_draws, batch_size):
        count = min(batch_size, n_draws - start)
        indices = rng.integers(
            0, sse.shape[0], size=(count, max_chunks)
        )
        distance = np.cumsum(sse[indices], axis=1)
        np.sqrt(distance, out=distance)

        for i, snr in enumerate(snrs):
            # Work with error probabilities to retain precision near 100%.
            error = ndtr(-distance / (2.0 * sigma_per_bin(snr)))
            error_sum[i] += error.sum(axis=0)
            error_square_sum[i] += (error * error).sum(axis=0)

    mean_error = error_sum / n_draws
    accuracy = 1.0 - mean_error
    variance = np.maximum(
        (error_square_sum - n_draws * mean_error**2) / (n_draws - 1),
        0.0,
    )
    mc_se = np.sqrt(variance / n_draws)

    rows, crossings = [], []
    for p, label in enumerate(pair_labels):
        for i, snr in enumerate(snrs):
            values = accuracy[i, :, p]
            errors = mc_se[i, :, p]
            rows.append(pd.DataFrame({
                "model_pair": label,
                "snr_resel": snr,
                "n_chunks": np.arange(1, max_chunks + 1),
                "oracle_accuracy": values,
                "mc_se": errors,
            }))

            hits = np.flatnonzero(values >= 0.95)
            index = int(hits[0]) if hits.size else None
            crossings.append({
                "model_pair": label,
                "snr_resel": snr,
                "n95": index + 1 if index is not None else pd.NA,
                "n95_display": (
                    str(index + 1) if index is not None
                    else f">{max_chunks}"
                ),
                "accuracy_at_n95": (
                    values[index] if index is not None else np.nan
                ),
                "mc_se_at_n95": (
                    errors[index] if index is not None else np.nan
                ),
                "accuracy_before_n95": (
                    values[index - 1] if index is not None and index > 0
                    else np.nan
                ),
                "accuracy_at_max_chunks": values[-1],
            })

    curves = pd.concat(rows, ignore_index=True)
    n95 = pd.DataFrame(crossings)
    n95["n95"] = n95["n95"].astype("Int64")
    return curves, n95

def plot_oracle_results(curves, flux_split):
    """Return full-range, N=1-5 detail, and pooled flux-split figures."""
    import matplotlib.pyplot as plt

    pairs = list(flux_split["model_pair"])
    snrs = sorted(curves["snr_resel"].unique())
    if len(pairs) != 6 or len(snrs) != 6:
        raise ValueError("Expected six model pairs and six S/N levels")

    colors = (
        "#0072B2", "#56B4E9", "#009E73",
        "#E69F00", "#D55E00", "#CC79A7",
    )
    figures = {}
    y_min = 5.0 * np.floor(
        100.0 * curves["oracle_accuracy"].min() / 5.0
    )
    max_n = int(curves["n_chunks"].max())

    for name, limit, logarithmic in (
        ("oracle_accuracy", max_n, True),
        ("oracle_accuracy_detail", min(5, max_n), False),
    ):
        fig, axes = plt.subplots(
            2, 3, figsize=(13, 7.4), sharex=True, sharey=True
        )
        fig.subplots_adjust(
            left=0.075, right=0.985, bottom=0.15,
            top=0.79, hspace=0.30, wspace=0.17,
        )

        for ax, pair in zip(axes.flat, pairs):
            for snr, color in zip(snrs, colors):
                selected = curves.loc[
                    (curves["model_pair"] == pair)
                    & (curves["snr_resel"] == snr)
                    & (curves["n_chunks"] <= limit)
                ].sort_values("n_chunks")

                x = selected["n_chunks"].to_numpy()
                y = 100.0 * selected["oracle_accuracy"].to_numpy()
                marks = [
                    i for i, n in enumerate(x)
                    if n <= 5 or n in CHUNK_COUNTS
                ]
                ax.plot(
                    x, y, color=color, lw=1.8,
                    marker="o", markersize=3.5, markevery=marks,
                    label=f"{snr:g}",
                )

            ax.axhline(95, color="0.4", ls="--", lw=1, zorder=0)
            ax.set_title(pair, fontsize=12)
            ax.set_ylim(y_min, 100.6)
            ax.grid(axis="both", which="major", alpha=0.20)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            if logarithmic:
                ax.set_xscale("log")
                ticks = [
                    n for n in (1, 2, 5, 10, 30, 100, 200)
                    if n <= limit
                ]
                ax.set_xticks(ticks, labels=[str(n) for n in ticks])
                ax.set_xlim(0.95, limit * 1.07)
                ax.minorticks_off()
            else:
                ax.set_xticks(range(1, limit + 1))
                ax.set_xlim(0.9, limit + 0.1)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.925), ncol=6, frameon=False,
            title="S/N per resel; dashed line = 95%",
        )
        detail = " | N = 1–5" if not logarithmic else ""
        fig.suptitle(
            "Known-field Gaussian oracle" + detail,
            fontsize=17, y=0.98,
        )
        fig.supxlabel("Number of independent chunks, N", y=0.065)
        fig.supylabel("Mean oracle accuracy (%)", x=0.018)
        fig.text(
            0.5, 0.02,
            "Equal priors; known paired fields; independent noise and chunks."
            " Ceiling conditional on these assumptions.",
            ha="center", fontsize=9, color="0.35",
        )
        figures[name] = fig

    split = flux_split.set_index("model_pair").loc[pairs]
    cuts = split["flux_cut"].unique()
    if len(cuts) != 1:
        raise ValueError("Expected one flux threshold")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(
        left=0.14, right=0.96, bottom=0.13, top=0.77
    )
    positions = np.arange(len(pairs))

    for offset, column, color, label in (
        (-0.18, "low_flux_bin_fraction", "#9AA3AB",
         "Fraction of bins"),
        (+0.18, "low_flux_chi2_fraction", "#0072B2",
         r"Fraction of $\Delta\chi^2$"),
    ):
        values = 100.0 * split[column].to_numpy()
        bars = ax.barh(
            positions + offset, values, height=0.30,
            color=color, label=label,
        )
        ax.bar_label(
            bars, labels=[f"{v:.2f}%" for v in values],
            padding=5, fontsize=10,
        )

    ax.set_yticks(positions, labels=pairs)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Share of the pair's total (%)")
    ax.grid(axis="x", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Flux split: min(F_A, F_B) < {cuts[0]:g}",
        fontsize=16, y=0.98,
    )
    fig.text(
        0.5, 0.915,
        "Noiseless LP1 spectra; pooled across held-out paired sightlines",
        ha="center", fontsize=10,
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875), ncol=2, frameon=False,
    )
    figures["oracle_flux_split"] = fig
    return figures