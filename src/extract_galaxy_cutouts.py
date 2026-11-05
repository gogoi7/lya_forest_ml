"""Extract gas cubes and ray metadata for the two-halo viewer."""

import argparse
import csv
import io
from pathlib import Path

import h5py
import numpy as np


def append(dataset, values):
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=[f"EX_{i}" for i in range(4)], required=True
    )
    parser.add_argument("--rays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapnum", type=int, default=90)
    parser.add_argument("--halos", type=int, nargs="+", default=[58, 77])
    parser.add_argument("--half-width", type=float, default=2.0)
    args = parser.parse_args()

    if args.half_width <= 0:
        parser.error("--half-width must be positive")

    with args.rays.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [
            r for r in reader
            if r["model"] == args.model
            and int(r["group_id_EX0"]) in args.halos
        ]

    snapshot = args.root / args.model / f"snapshot_{args.snapnum:03d}.hdf5"
    catalog = args.root / args.model / f"groups_{args.snapnum:03d}.hdf5"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(snapshot, "r") as src, h5py.File(catalog, "r") as cat:
        gas = src["PartType0"]
        box = float(src["Header"].attrs["BoxSize"])

        required = [
            "Coordinates", "Masses", "Density", "InternalEnergy",
            "ElectronAbundance", "StarFormationRate",
        ]
        missing = [name for name in required if name not in gas]
        if missing:
            raise ValueError(f"{args.model}: missing gas fields {missing}")

        optional = [
            "ParticleIDs", "Velocities", "NeutralHydrogenAbundance",
            "GFM_Metals", "GFM_Metallicity",
        ]
        fields = required + [name for name in optional if name in gas]
        print(f"{args.model}: saving {', '.join(fields)}", flush=True)

        # 'x' prevents overwriting an existing cutout.
        with h5py.File(args.output, "x") as out:
            out.attrs.update({
                "complete": False,
                "model": args.model,
                "snapnum": args.snapnum,
                "half_width_Rvir_EX0": args.half_width,
                "selection": "Gas-cell centres inside a periodic cube",
                "position_unit": "ckpc/h",
                "gas_fields": "Raw snapshot units",
                "source_snapshot": str(snapshot.resolve()),
                "source_catalog": str(catalog.resolve()),
                "source_rays": str(args.rays.resolve()),
            })
            out.create_group("source_header").attrs.update(src["Header"].attrs)
            targets = {}

            for halo in args.halos:
                selected = [
                    r for r in rows if int(r["group_id_EX0"]) == halo
                ]
                if (
                    len(selected) != 48
                    or len({int(r["ray_id"]) for r in selected}) != 48
                ):
                    raise ValueError(f"Halo {halo}: expected 48 distinct rays")

                identities = {
                    (
                        int(r["snapnum"]),
                        int(r["group_id_model"]),
                        int(r["subhalo_id_model"]),
                    )
                    for r in selected
                }
                if (
                    len(identities) != 1
                    or next(iter(identities))[0] != args.snapnum
                ):
                    raise ValueError(f"Halo {halo}: inconsistent ray metadata")

                row = selected[0]
                group_id = int(row["group_id_model"])
                sub_id = int(row["subhalo_id_model"])
                center = cat["Subhalo/SubhaloPos"][sub_id].astype(np.float64)
                rvir0 = float(row["Rvir_EX0_ckpch"])
                half_width = args.half_width * rvir0

                if not 0 < half_width < box / 2:
                    raise ValueError(f"Halo {halo}: invalid cutout size")

                # Verify that this centre reproduces the stored ray geometry.
                cofm = np.array([
                    [float(r[f"cofm_{c}_ckpch"]) for c in "xyz"]
                    for r in selected
                ])
                axes = np.array([int(r["axis"]) - 1 for r in selected])
                offsets = (cofm - center + box / 2) % box - box / 2

                np.testing.assert_allclose(
                    offsets[np.arange(48), axes], 0, atol=0.01
                )
                np.testing.assert_allclose(
                    np.linalg.norm(offsets, axis=1),
                    [float(r["b_ckpch"]) for r in selected],
                    rtol=0, atol=0.01,
                )

                target = out.create_group(f"halo_{halo}")
                target.attrs.update({
                    "group_id_EX0": halo,
                    "group_id_model": group_id,
                    "subhalo_id_model": sub_id,
                    "type_EX0": row["type_EX0"],
                    "M200c_EX0_Msun": float(row["M200c_EX0_Msun"]),
                    "Rvir_EX0_ckpch": rvir0,
                    "half_width_ckpch": half_width,
                    "Rvir_model_ckpch": float(
                        cat["Group/Group_R_TopHat200"][group_id]
                    ),
                })
                target.create_dataset("center_ckpch", data=center)
                target.create_dataset(
                    "center_velocity_kms",
                    data=cat["Subhalo/SubhaloVel"][sub_id],
                )

                text = io.StringIO()
                writer = csv.DictWriter(text, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(selected)
                target.create_dataset(
                    "manifest_csv", data=text.getvalue(),
                    dtype=h5py.string_dtype("utf-8"),
                )

                dest = target.create_group("gas")
                for name in fields:
                    tail = gas[name].shape[1:]
                    dataset = dest.create_dataset(
                        name, shape=(0,) + tail, maxshape=(None,) + tail,
                        dtype=gas[name].dtype, chunks=True,
                        compression="gzip", compression_opts=1, shuffle=True,
                    )
                    dataset.attrs.update(gas[name].attrs)

                dest.create_dataset(
                    "RelativeCoordinates_ckpch",
                    shape=(0, 3), maxshape=(None, 3), dtype="f8",
                    chunks=True, compression="gzip",
                    compression_opts=1, shuffle=True,
                )
                targets[halo] = (center, half_width, dest)

            # Read each snapshot once, keeping memory bounded.
            chunk_size = 262144
            for start in range(0, len(gas["Coordinates"]), chunk_size):
                stop = min(start + chunk_size, len(gas["Coordinates"]))
                xyz = gas["Coordinates"][start:stop].astype(np.float64)
                masks = {}

                for halo, (center, half_width, dest) in targets.items():
                    delta = (xyz - center + box / 2) % box - box / 2
                    keep = np.all(np.abs(delta) <= half_width, axis=1)
                    if keep.any():
                        masks[halo] = keep
                        append(dest["RelativeCoordinates_ckpch"], delta[keep])

                if not masks:
                    continue

                for name in fields:
                    values = (
                        xyz if name == "Coordinates"
                        else gas[name][start:stop]
                    )
                    for halo, keep in masks.items():
                        append(targets[halo][2][name], values[keep])

            for halo, (_, _, dest) in targets.items():
                print(
                    f"{args.model}, halo {halo}: "
                    f"{len(dest['Coordinates']):,} gas cells; 48 rays",
                    flush=True,
                )
            out.attrs["complete"] = True

    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()