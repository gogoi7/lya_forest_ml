from pathlib import Path
from collections import Counter

import argparse
import json
import sys
import h5py
import numpy as np
import csv
import hashlib

def periodic_distance(x, y, box):
    """Compute the periodic distance between two points in a simulation box."""
    delta = (x - y + box / 2) % box - box / 2
    return np.linalg.norm(delta, axis=1)

def check_dm_ids(root, snapnum):
    rng = np.random.default_rng(42)
    results = []
    header_keys = ("BoxSize", "Time", "HubbleParam", "Omega0", "OmegaLambda")

    for i in range(4):
        model = f"EX_{i}"
        path = root / model / f"snapshot_{snapnum:03d}.hdf5"
        print(f"reading {path}", flush=True)

        with h5py.File(path, "r") as f:
            ids = f["PartType1/ParticleIDs"][:]
            xyz = f["PartType1/Coordinates"][:]
            attrs = f["Header"].attrs
            header = {key: float(attrs[key]) for key in header_keys}
            expected_count = int(attrs["NumPart_Total"][1])

        if len(ids) != expected_count or xyz.shape != (len(ids), 3):
            raise ValueError(f"{model}: incomplete or inconsistent DM arrays")

        order = np.argsort(ids)
        ids_sorted = ids[order]
        duplicates = int(np.count_nonzero(ids_sorted[1:] == ids_sorted[:-1]))

        if duplicates:
            raise ValueError(f"{model}: found {duplicates} duplicate DM IDs")

        if i == 0:
            reference_ids = ids_sorted
            reference_header = header
            sample = rng.choice(len(ids), size=min(4096, len(ids)), replace=False)
            reference_xyz = xyz[order[sample]].astype(np.float64)
            shuffled_xyz = reference_xyz[rng.permutation(len(sample))]

        same_ids = np.array_equal(ids_sorted, reference_ids)
        same_header = all(np.isclose(header[key], reference_header[key], rtol=1e-8, atol=1e-8) for key in header_keys)
        row = {
            "model": model,
            "snapshot": str(path),
            "header": header,
            "n_dm": len(ids),
            "duplicate_ids": duplicates,
            "same_ids_as_EX0": bool(same_ids),
            "same_header_as_EX0": bool(same_header),
        }

        print(
            f"{model}: N_DM={len(ids):,}, duplicates={duplicates}, "
            f"IDs_match_EX0={same_ids}, header_match_EX0={same_header}",
            flush=True,
        )

        if i > 0 and same_ids and same_header:
            sample_xyz = xyz[order[sample]].astype(np.float64)

            if not np.isfinite(sample_xyz).all() or not np.isfinite(reference_xyz).all():
                raise ValueError(f"{model}: non-finite coordinates in sample")

            matched = periodic_distance(sample_xyz, reference_xyz, header["BoxSize"]) 
            shuffled = periodic_distance(sample_xyz, shuffled_xyz, header["BoxSize"])
            percentiles = np.percentile(matched, [50, 95, 99])

            row["sample_size"] = len(sample)
            row["distance_ckpch_p50_p95_p99"] = percentiles.tolist()
            row["shuffled_median_ckpch"] = float(np.median(shuffled))

            print(" Same-ID separation p50/p95/p99 [ckpch]:", np.round(percentiles, 3), flush=True)
            print(f" Shuffled median separation [ckpch]: {row['shuffled_median_ckpch']:.3f}", flush=True)

        results.append(row)
        del ids, xyz, order, ids_sorted

    return {"snapnum": snapnum, "seed": 42, "models": results}

def load_dm_groups(root, model, snapnum):
    catalog_path = root / model / f"groups_{snapnum:03d}.hdf5"
    snapshot_path = root / model / f"snapshot_{snapnum:03d}.hdf5"

    with h5py.File(catalog_path, "r") as handle:
        ndm = handle["Group/GroupLenType"][:, 1].astype(np.int64)
        first = handle["Group/GroupFirstSub"][:].astype(np.int64)
        subpos = handle["Subhalo/SubhaloPos"][:]
        rvir = handle["Group/Group_R_TopHat200"][:]
        box = float(handle["Header"].attrs["BoxSize"])

    valid = (first >= 0) & (first < len(subpos))
    centers = np.full((len(ndm), 3), np.nan)
    centers[valid] = subpos[first[valid]]
    first[~valid] = -1

    with h5py.File(snapshot_path, "r") as handle:
        ids = handle["PartType1/ParticleIDs"][:]

    offsets = np.concatenate(([0], np.cumsum(ndm)))
    if offsets[-1] > len(ids):
        raise ValueError(f"{model}: catalog DM counts exceed snapshot count")

    owner = np.full(len(ids), -1, dtype=np.int32)
    owner[:offsets[-1]] = np.repeat(np.arange(len(ndm), dtype=np.int32), ndm)
    order = np.argsort(ids)

    return {
        "ids": ids[order], "owner": owner[order], "order": order,
        "offsets": offsets, "ndm": ndm, "first": first,
        "centers": centers, "rvir": rvir, "box": box,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    }


def strongest_group(labels):
    counts = np.bincount(labels[labels >= 0])
    if counts.size == 0:
        return -1, 0, False, 0

    group = int(np.argmax(counts))
    shared = int(counts[group])
    tied = bool(np.count_nonzero(counts == shared) > 1)
    counts[group] = 0
    second = int(counts.max(initial=0))
    return group, shared, tied, second


def match_halos(root, snapnum, sample_path, output_path):
    with sample_path.open(newline="") as handle:
        sample = list(csv.DictReader(handle))

    reference = load_dm_groups(root, "EX_0", snapnum)
    print("EX0 catalog SHA256:", reference["catalog_sha256"], flush=True)

    common = {int(row["group_id"]) for row in sample}
    pilot = {
        int(row["group_id"]) for row in sample
        if row["pilot"].lower() == "true"
    }
    hashes = {"EX_0": reference["catalog_sha256"]}
    summary = {}
    results = []

    for source in sample:
        group = int(source["group_id"])
        if (int(source["subhalo_id"]) != reference["first"][group]
                or int(source["Ndm_fof"]) != reference["ndm"][group]):
            raise ValueError(f"EX0 sample and TACC catalog disagree for group {group}")

    for number in (1, 2, 3):
        model = f"EX_{number}"
        target = load_dm_groups(root, model, snapnum)

        if not np.array_equal(reference["ids"], target["ids"]):
            raise ValueError(f"{model}: DM ID arrays differ from EX0")
        if not np.isclose(reference["box"], target["box"]):
            raise ValueError(f"{model}: box sizes differ")

        hashes[model] = target["catalog_sha256"]

        # Translate membership between the two snapshot row orders.
        target_at_reference = np.empty(len(reference["ids"]), dtype=np.int32)
        target_at_reference[reference["order"]] = target["owner"]

        reference_at_target = np.empty(len(target["ids"]), dtype=np.int32)
        reference_at_target[target["order"]] = reference["owner"]

        accepted = set()
        statuses = Counter()
        center_shifts = []

        for source in sample:
            group0 = int(source["group_id"])
            start, stop = reference["offsets"][group0:group0 + 2]
            group1, shared, tied, second = strongest_group(
                target_at_reference[start:stop]
            )

            # Keep candidate information even when the match is rejected.
            row = dict(source)
            row.update({
                "model": model,
                "matched_group_id": group1,
                "matched_subhalo_id": -1,
                "n_shared": shared,
                "fraction_EX0": shared / reference["ndm"][group0],
                "fraction_model": 0.0,
                "second_fraction_EX0": second / reference["ndm"][group0],
                "reciprocal": False,
                "status": "unmatched",
                "center_sep_ckpch": float("nan"),
                "center_sep_Rvir_EX0": float("nan"),
            })

            if group1 >= 0:
                start, stop = target["offsets"][group1:group1 + 2]
                reverse, _, reverse_tied, _ = strongest_group(
                    reference_at_target[start:stop]
                )
                reciprocal = reverse == group0 and not tied and not reverse_tied

                row["reciprocal"] = bool(reciprocal)
                row["fraction_model"] = shared / target["ndm"][group1]
                row["matched_subhalo_id"] = int(target["first"][group1])
                valid_center = np.isfinite(target["centers"][group1]).all()

                if valid_center:
                    distance = periodic_distance(
                        reference["centers"][group0:group0 + 1],
                        target["centers"][group1:group1 + 1],
                        reference["box"],
                    )[0]
                    row["center_sep_ckpch"] = float(distance)
                    row["center_sep_Rvir_EX0"] = float(
                        distance / float(source["Rvir_ckpch"])
                    )

                if tied or reverse_tied:
                    row["status"] = "ambiguous_tie"
                elif not reciprocal:
                    row["status"] = "nonreciprocal"
                elif min(row["fraction_EX0"], row["fraction_model"]) <= 0.5:
                    row["status"] = "weak_overlap"
                elif not valid_center:
                    row["status"] = "no_central"
                else:
                    row["status"] = "accepted"
                    accepted.add(group0)
                    center_shifts.append(row["center_sep_Rvir_EX0"])

            statuses[row["status"]] += 1
            results.append(row)

        common.intersection_update(accepted)
        summary[model] = dict(statuses)
        print(f"{model}: {dict(statuses)}", flush=True)

        if center_shifts:
            print(
                "  Accepted center shift / EX0 Rvir [p50, p95, max]:",
                np.round(np.percentile(center_shifts, [50, 95, 100]), 3),
                flush=True,
            )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Accepted in all four models: {len(common)}/{len(sample)}", flush=True)
    print(f"Original pilot retained: {len(common & pilot)}/{len(pilot)}", flush=True)
    print(f"Saved {output_path}", flush=True)

    return {
        "snapnum": snapnum,
        "reference_model": "EX_0",
        "method": "Unique reciprocal maximum shared FoF DM count; both fractions > 0.5",
        "center": "Central subhalo of each matched FoF halo",
        "catalog_sha256": hashes,
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "per_model_status": summary,
        "common_group_ids_EX0": sorted(common),
        "pilot_group_ids_retained_EX0": sorted(common & pilot),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("check-ids", "match"), default="check-ids")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapnum", type=int, default=90)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Python: {sys.executable}; NumPy: {np.__version__}; h5py: {h5py.__version__}")

    if args.task == "check-ids":
        report = check_dm_ids(args.root.resolve(), args.snapnum)
        report_path = args.output
    else:
        if args.sample is None:
            parser.error("--sample is required for --task match")

        report = match_halos(
            args.root.resolve(), args.snapnum, args.sample, args.output
        )
        report_path = args.output.with_suffix(".json")

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {report_path}", flush=True)