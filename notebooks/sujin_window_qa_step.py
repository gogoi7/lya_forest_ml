"""Run after the corrected catalog cell in cos_strong_absorbers.ipynb.

Uses the notebook's live catalog and helpers. Recreates the original noise
streams on all LOS in their original order. Saves examples, never an export.
"""

def run_sujin_window_qa():
    import hashlib
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    required = (
        "catalog", "ROOT", "OUT", "MODELS", "sweep", "dv", "n_los",
        "offsets", "detect_minima", "local_minima", "get_windows",
        "FLUX_CUT", "SIGMA_CUT", "RECOVERY_FLUX", "display",
    )
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError("Run the corrected setup/helpers/catalog first: " + ", ".join(missing))

    expected_parent = ROOT / "outputs/cos_lsf_snr/strong_absorbers"
    if OUT.parent != expected_parent or not OUT.name.startswith("recovery085_"):
        raise RuntimeError("OUT must be the new recovery085_<timestamp> directory; check for a duplicate OUT assignment.")
    n_rows = len(catalog["labels"])
    for key, value in catalog.items():
        if len(value) != n_rows:
            raise RuntimeError(f"Catalog row mismatch: {key} has {len(value):,}, expected {n_rows:,}.")
    if np.any(catalog["n_merged"] < 1):
        raise RuntimeError("Every retained detection must have n_merged >= 1.")

    qa_seed = 20261019
    qa_rng = np.random.default_rng(qa_seed)
    chosen = []
    for snr in (20, 50):
        # Original training pool only; no held-out test windows are selected.
        pool = np.flatnonzero((catalog["snr_resel"] == snr) & (catalog["split"] == 0))
        singles = pool[catalog["n_merged"][pool] == 1]
        merged = pool[catalog["n_merged"][pool] > 1]
        if singles.size == 0 or merged.size == 0:
            raise RuntimeError(f"Need both single and merged examples at S/N {snr}.")
        multiplicity = catalog["n_merged"][merged]
        distance = np.abs(multiplicity - np.median(multiplicity))
        typical = merged[distance == distance.min()]
        choices = (
            ("single", int(qa_rng.choice(singles))),
            ("typical_merged", int(qa_rng.choice(typical))),
            ("largest_merged", int(merged[np.argmax(multiplicity)])),
        )
        for case, row in choices:
            chosen.append({
                "snr": snr, "case": case, "catalog_row": row,
                "model": MODELS[int(catalog["labels"][row])],
                "los_id": int(catalog["los_id"][row]),
                "noise_seed": int(catalog["noise_seed"][row]),
                "center_pixel": int(catalog["center_pixel"][row]),
                "n_merged": int(catalog["n_merged"][row]),
            })
    selections = pd.DataFrame(chosen)

    # Cache only six small sightline vectors, not whole noisy arrays.
    snapshots = {}
    for model, model_rows in selections.groupby("model", sort=False):
        cache_path = ROOT / "outputs/cos_lsf_snr/snr_sweep/clean_flux" / f"{model}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        clean, cached_dv, _ = sweep.cache_lp1_flux(model, ROOT)
        expected_pixels = int(round(2500.0 / dv))
        if clean.shape != (n_los, expected_pixels):
            raise RuntimeError(f"Unexpected clean-cache shape for {model}: {clean.shape}")
        np.testing.assert_allclose(cached_dv, dv, rtol=0, atol=1e-12)
        for (snr, seed), group in model_rows.groupby(["snr", "noise_seed"], sort=False):
            noisy = sweep.add_noise(clean, int(snr), model=model, noise_seed=int(seed))
            for q in group.itertuples(index=False):
                snapshots[int(q.catalog_row)] = noisy[int(q.los_id)].copy()
            del noisy
        del clean

    qa_dir = OUT / "window_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for snr in (20, 50):
        examples = selections[selections.snr == snr]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharey="row", squeeze=False)
        for panel, q in enumerate(examples.itertuples(index=False)):
            row, center = int(q.catalog_row), int(q.center_pixel)
            flux = snapshots[row]
            sigma = float(sweep.sigma_per_bin(snr))
            centers, fs, es, multiplicities = detect_minima(flux, sigma, dv)
            position = np.flatnonzero(centers == center)
            if position.size != 1 or int(multiplicities[position[0]]) != int(q.n_merged):
                raise RuntimeError(f"Detection/multiplicity did not reproduce for catalog row {row}.")
            rebuilt = get_windows(flux, np.array([center]), offsets)[0].astype(np.float32)
            np.testing.assert_array_equal(rebuilt, catalog["windows"][row])
            np.testing.assert_allclose(fs[center], catalog["fmin"][row], rtol=0, atol=1e-12)

            # Independently walk the trough in both directions to audit membership.
            below = np.isfinite(fs) & (fs < RECOVERY_FLUX)
            component = np.zeros(len(fs), dtype=bool)
            component[center] = True
            for direction in (-1, 1):
                for distance in range(1, len(fs)):
                    pixel = (center + direction * distance) % len(fs)
                    if not below[pixel] or component[pixel]:
                        break
                    component[pixel] = True
            candidates = local_minima(fs, np.ones(len(fs), dtype=bool), periodic=True)
            eligible = candidates[
                (fs[candidates] < FLUX_CUT)
                & ((1.0 - fs[candidates]) / es[candidates] >= SIGMA_CUT)
            ]
            members = eligible[component[eligible]]
            if len(members) != int(q.n_merged):
                raise RuntimeError(f"Independent trough audit failed for catalog row {row}.")
            winner = members[np.lexsort((members, fs[members]))[0]]
            if winner != center:
                raise RuntimeError(f"Retained center is not the deepest qualifying minimum: row {row}.")

            aperture = np.zeros(len(fs), dtype=bool)
            aperture[(center + offsets) % len(fs)] = True
            fits = not np.any(component & ~aperture)
            width = int(component.sum()) * dv  # Bin support, not interpolated crossing width.
            audit_rows.append({
                **q._asdict(), "fmin": float(fs[center]),
                "trough_bins": int(component.sum()), "trough_bin_width_kms": float(width),
                "trough_fits_window": bool(fits), "whole_los_below_recovery": bool(component.all()),
                "trough_wraps_native_boundary": bool(component[0] and component[-1]),
                "window_wraps_native_boundary": bool(catalog["wraps_boundary"][row]),
            })

            # Display one period centered on the retained detection, plus its window.
            relative_pixels = (np.arange(len(fs)) - center + len(fs)//2) % len(fs) - len(fs)//2
            x = relative_pixels * dv
            order = np.argsort(x)
            for column, ax in enumerate(axes[panel]):
                ax.plot(x[order], flux[order], color="0.65", lw=0.7, label="Noisy flux")
                ax.plot(x[order], fs[order], color="C0", lw=1.15, label="3-bin smoothed")
                ax.fill_between(x[order], 0, 1, where=component[order], step="mid",
                                transform=ax.get_xaxis_transform(), color="C0", alpha=0.09)
                ax.scatter(x[members], fs[members], s=18, color="C1", zorder=4,
                           label="Qualifying minima in this trough")
                ax.scatter([0], [fs[center]], marker="*", s=110, color="C3", zorder=5,
                           label="Retained center")
                ax.axhline(RECOVERY_FLUX, color="black", ls="--", lw=0.8)
                ax.axhline(1.0, color="0.8", lw=0.7)
                ax.set_xlim((-len(fs)*dv/2, len(fs)*dv/2) if column == 0 else (-300, 300))
                if column == 0:
                    ax.axvline(-300, color="0.4", ls=":", lw=0.8)
                    ax.axvline(300, color="0.4", ls=":", lw=0.8)
                ax.grid(alpha=0.15)
                ax.set_xlabel("Velocity from retained center [km/s]")
            axes[panel, 0].set_ylabel("Normalized flux")
            axes[panel, 0].set_title(
                f"{q.case} | {q.model}, LOS {q.los_id}, seed {q.noise_seed}\n"
                f"{q.n_merged} minima; trough bin support {width:.0f} km/s", fontsize=9)
            axes[panel, 1].set_title(
                f"87-bin window | complete trough fits: {'yes' if fits else 'no'}", fontsize=9)
        axes[0, 0].legend(fontsize=6, loc="best")
        fig.suptitle(f"Absorber detection audit | S/N {snr} | recovery F = {RECOVERY_FLUX}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figure_path = qa_dir / f"window_examples_snr{snr}.png"
        fig.savefig(figure_path, dpi=170, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {figure_path}")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(qa_dir / "window_examples.csv", index=False)
    script_path = ROOT / "notebooks/sujin_window_qa_step.py"
    metadata = {
        "selection_seed": qa_seed,
        "selection_pool": "Original manifest training LOS only (catalog split == 0)",
        "cases_per_snr": ["random single", "random nearest-median merged multiplicity", "largest merge; first catalog row breaks ties"],
        "snrs": [20, 50],
        "detection": {"flux_cut": float(FLUX_CUT), "sigma_cut": float(SIGMA_CUT),
                      "recovery_flux": float(RECOVERY_FLUX), "dv_kms": float(dv)},
        "checks": ["Exact float32 window reproduction", "Center and multiplicity reproduction",
                   "Independent periodic trough walk", "Deepest eligible center and tie rule"],
        "trough_width_definition": "Number of connected sub-threshold smoothed bins times dv; no crossing interpolation",
        "catalog_row_is_permanent_window_id": False,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest() if script_path.is_file() else None,
    }
    (qa_dir / "qa_settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("All six example checks passed. These examples do not measure classifier accuracy.")
    display(audit[["snr", "case", "model", "los_id", "n_merged", "trough_bin_width_kms",
                   "trough_fits_window", "whole_los_below_recovery"]].round(2))
    return audit


QA_EXAMPLES = run_sujin_window_qa()
