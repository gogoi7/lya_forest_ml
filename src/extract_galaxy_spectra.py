"""Extract galaxy sightlines using the existing fake_spectra defaults."""

import argparse
import csv
import io
from pathlib import Path

import h5py
import numpy as np
from fake_spectra import spectra

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--model", choices=["EX_0", "EX_1", "EX_2", "EX_3"], required=True)
parser.add_argument("--snapnum", type=int, default=90)
parser.add_argument("--rays", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

with args.rays.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if row["model"] == args.model]

if not rows:
    raise ValueError(f"No rays for {args.model}")
if any(int(row["snapnum"]) != args.snapnum for row in rows):
    raise ValueError("Ray manifest and requested snapshot disagree")

ray_ids = np.array([int(row["ray_id"]) for row in rows], dtype=np.int64)
axis = np.array([int(row["axis"]) for row in rows], dtype=np.int32)
position_columns = ["cofm_x_ckpch", "cofm_y_ckpch", "cofm_z_ckpch"]
cofm = np.array(
    [[float(row[column]) for column in position_columns] for row in rows],
    dtype=np.float64,
)

assert len(np.unique(ray_ids)) == len(rows), "Duplicate ray IDs"
assert np.isfinite(cofm).all() and np.isin(axis, [1, 2, 3]).all()

output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(f"Output already exists: {output}")

n_halos = len({row["group_id_EX0"] for row in rows})
print(f"{args.model}: {n_halos} halos, {len(rows)} rays", flush=True)

spec = spectra.Spectra(
    args.snapnum,
    str((args.root / args.model).resolve()),
    cofm,
    axis,
    reload_file=True,
    savedir=str(output.parent),
    savefile=output.name,
)

tau = spec.get_tau("H", 1, 1215)
assert tau.shape == (len(rows), spec.nbins)
assert np.isfinite(tau).all() and np.all(tau >= 0), "Invalid optical depths"

spec.get_col_density("H", 1)
spec.save_file()

# Preserve the input rows in exactly the same order as the spectra.
manifest_text = io.StringIO()
writer = csv.DictWriter(manifest_text, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(rows)

with h5py.File(output, "a") as handle:
    metadata = handle.create_group("galaxy_metadata")
    metadata.create_dataset("ray_id", data=ray_ids)
    metadata.create_dataset(
        "manifest_csv",
        data=manifest_text.getvalue(),
        dtype=h5py.string_dtype("utf-8"),
    )
    metadata.attrs["model"] = args.model
    metadata.attrs["snapnum"] = args.snapnum
    metadata.attrs["dvbin_kms"] = spec.dvbin
    metadata.attrs["vmax_kms"] = spec.vmax
    metadata.attrs["velfac_kms_per_ckpch"] = spec.velfac
    metadata.attrs["kernel_int"] = spec.kernel_int

print(f"tau shape: {tau.shape}; dv = {spec.dvbin:.9f} km/s", flush=True)
print(f"Saved {output}", flush=True)