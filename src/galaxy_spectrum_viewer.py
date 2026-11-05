"""Build a local, offline, two-sightline CAMELS/IllustrisTNG spectrum viewer.

Use from notebooks/03_experiments.ipynb after loading ``full`` and ``geometry``::

    from src.galaxy_spectrum_viewer import build_spectrum_viewer
    viewer = build_spectrum_viewer(full, geometry, FULL_OUT_ROOT / "viewer")

Open the returned HTML in a browser and select the accompanying .bin file.
The browser reads only selected records. No server, upload, or kernel is needed.
"Export this pair as HTML" saves a self-contained, interactive two-ray version.

Inputs are the existing ``full[EX_i]`` dictionaries: ``rays`` (ray_id index),
``flux``, ``edges``, ``lp1_flux``, and ``lp1_edges``. Arrays must follow the row
order of that model's rays table. Models are aligned by ray_id, never by position.
Dependencies: numpy, pandas (already used by the notebook), and plotly.

Noise is Gaussian, with sigma = sqrt(2) / SNR for the existing COS grid's two
output bins per resolution element. The unit-noise stream replays
default_rng([noise_seed, model_number]) in the ORIGINAL model row order.
Changing SNR rescales the same realization; identical ray/model/seed selections
give identical spectra in both panels. Native spectra are noiseless.

The binary is a float32 display cache, not a replacement for science products.
Equivalent widths are recomputed with exact partial-pixel periodic overlaps;
rounding to float32 can introduce small differences from float64 source values.
No clipping, interpolation of sightlines, or mean-flux rescaling is performed.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np


MODEL_INFO = (
    ("EX_0", "EX0", "Fiducial IllustrisTNG", "#111111", "A_SN1=1, A_AGN1=1, A_SN2=1, A_AGN2=1"),
    ("EX_1", "EX1", "Extreme AGN feedback", "#0072B2", "A_SN1=1, A_AGN1=100, A_SN2=1, A_AGN2=1"),
    ("EX_2", "EX2", "Extreme supernova feedback", "#E69F00", "A_SN1=100, A_AGN1=1, A_SN2=1, A_AGN2=1"),
    ("EX_3", "EX3", "No feedback", "#D55E00", "A_SN1=0, A_AGN1=0, A_SN2=0, A_AGN2=0"),
)
MODEL_SOURCE = (
    "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/EX/"
    "CosmoAstroSeed_IllustrisTNG_L25n256_EX.txt"
)
MAGIC = b"LYACGM01"


def build_spectrum_viewer(
    full,
    geometry,
    output_dir,
    *,
    noise_seed=42,
    preferred_rays=(2811, 3738),
    retained_halo_ids=None,
    chunk_size=128,
):
    """Export the full matched ray sample as an HTML viewer plus indexed data.

    ``retained_halo_ids`` optionally enables an additional offset-cut filter.
    SNR is per resolution element, with two existing COS output bins per element.
    All mass, type, and impact-parameter filters use the EX0 reference definition.
    Returns a dict with ``html``, ``data``, ``n_rays``, and ``size_mb``.
    """
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ImportError("Install plotly in this notebook kernel: %pip install plotly") from exc

    if int(noise_seed) != noise_seed or noise_seed < 0:
        raise ValueError("noise_seed must be a nonnegative integer.")
    noise_seed = int(noise_seed)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    a, h = float(geometry["scale_factor"]), float(geometry["HubbleParam"])
    if not (math.isfinite(a) and math.isfinite(h) and a > 0 and h > 0):
        raise ValueError("geometry needs positive finite scale_factor and HubbleParam.")

    reference = full["EX_0"]["rays"]
    if reference.empty or not reference.index.is_unique:
        raise ValueError("EX0 rays must have a nonempty, unique ray_id index.")
    ids = reference.index.to_numpy(dtype=np.int64)
    if not np.array_equal(reference.index.to_numpy(), ids):
        raise ValueError("ray_id must be an integer index.")
    required = {
        "group_id_EX0", "group_id_model", "subhalo_id_model", "axis",
        "b_over_Rvir_EX0", "b_over_Rvir_model", "b_ckpch", "type_EX0",
        "M200c_EX0_Msun", "Rvir_EX0_ckpch", "galaxy_velocity_kms", "snapnum",
    }
    n = len(reference)
    aligned, models = [], []
    width = 0
    for key, short, label, color, parameters in MODEL_INFO:
        d, table = full[key], full[key]["rays"]
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{key} rays missing columns: {sorted(missing)}")
        if len(table) != n or not table.index.is_unique:
            raise ValueError(f"{key} must contain the same unique ray IDs as EX0.")
        positions = table.index.get_indexer(reference.index)
        if (positions < 0).any():
            raise ValueError(f"{key} is missing one or more EX0 ray IDs.")
        paired = table.iloc[positions]
        for col in ("group_id_EX0", "axis", "type_EX0"):
            if not np.array_equal(paired[col].to_numpy(), reference[col].to_numpy()):
                raise ValueError(f"{key} has inconsistent paired metadata: {col}")
        if not np.allclose(paired["b_over_Rvir_EX0"], reference["b_over_Rvir_EX0"], rtol=1e-7, atol=1e-9):
            raise ValueError(f"{key} has inconsistent reference impact parameters.")
        aligned.append(paired)
        model = dict(key=key, short=short, label=label, color=color, parameters=parameters)
        for kind, array_key, edge_key in (("native", "flux", "edges"), ("cos", "lp1_flux", "lp1_edges")):
            edges = np.asarray(d[edge_key], dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
                raise ValueError(f"{key}/{edge_key} must be finite increasing pixel edges.")
            count = len(edges) - 1
            if tuple(d[array_key].shape) != (n, count):
                raise ValueError(f"{key}/{array_key} shape does not match rays and edges.")
            model[kind] = dict(offset=width, count=count, edges=edges.tolist(), period=float(edges[-1] - edges[0]))
            width += count
        model["noise_offset"] = width
        width += model["cos"]["count"]
        models.append(model)

    retained = None if retained_halo_ids is None else {int(x) for x in retained_halo_ids}
    halo_rows, ray_rows = {}, []
    for i, ray_id in enumerate(ids):
        row = reference.iloc[i]
        halo_id = int(row["group_id_EX0"])
        halo_key = str(halo_id)
        mass = float(row["M200c_EX0_Msun"])
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid EX0 halo mass for ray {ray_id}.")
        if halo_key not in halo_rows:
            halo_rows[halo_key] = dict(
                logm=math.log10(mass), type=str(row["type_EX0"]),
                rvir_kpc=float(row["Rvir_EX0_ckpch"]) * a / h,
                groups=[int(t.iloc[i]["group_id_model"]) for t in aligned],
                subhalos=[int(t.iloc[i]["subhalo_id_model"]) for t in aligned],
                retained=None if retained is None else halo_id in retained,
            )
        ray_rows.append(dict(
            id=int(ray_id), halo=halo_id, axis=int(row["axis"]),
            b=float(row["b_over_Rvir_EX0"]), b_kpc=float(row["b_ckpch"]) * a / h,
            bm=[float(t.iloc[i]["b_over_Rvir_model"]) for t in aligned],
            v=[float(t.iloc[i]["galaxy_velocity_kms"]) for t in aligned],
        ))
    snapshots = reference["snapnum"].unique()
    if len(snapshots) != 1:
        raise ValueError("Build one viewer per snapshot.")
    header = dict(
        format="lya-galaxy-viewer", version=1, n_rows=n, record_floats=width,
        dtype="float32-le", models=models, rays=ray_rows, halos=halo_rows,
        snapnum=int(snapshots[0]), z=0.0 if abs(1.0 / a - 1.0) < 1e-12 else 1.0 / a - 1.0,
        scale_factor=a, hubble_param=h, noise_seed=noise_seed, cos_bins_per_resel=2,
        preferred_rays=[int(x) for x in preferred_rays], model_source=MODEL_SOURCE,
        has_retained_filter=retained is not None,
    )
    header_bytes = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"galaxy_spectrum_viewer_snap{header['snapnum']:03d}"
    data_path, html_path = output_dir / f"{stem}.bin", output_dir / f"{stem}.html"
    temporary = data_path.with_suffix(".bin.tmp")
    data_start = 16 + len(header_bytes)
    expected_size = data_start + n * width * 4
    print(f"Exporting {n:,} paired sightlines ({expected_size / 1e6:.1f} MB)...", flush=True)
    try:
        with temporary.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.truncate(expected_size)
        cache = np.memmap(temporary, dtype="<f4", mode="r+", offset=data_start, shape=(n, width))
        try:
            for model_number, model in enumerate(models):
                d = full[model["key"]]
                # Iterate the ORIGINAL model row order to reproduce its noise stream.
                to_reference = reference.index.get_indexer(d["rays"].index)
                rng = np.random.default_rng([noise_seed, model_number])
                for start in range(0, n, chunk_size):
                    stop = min(start + chunk_size, n)
                    rows = to_reference[start:stop]
                    for kind, array_key in (("native", "flux"), ("cos", "lp1_flux")):
                        spec = model[kind]
                        block = np.asarray(d[array_key][start:stop], dtype="<f4")
                        if not np.isfinite(block).all():
                            raise ValueError(f"Nonfinite flux in {model['key']}/{array_key}.")
                        cache[rows, spec["offset"]:spec["offset"] + spec["count"]] = block
                    noise = rng.standard_normal((stop - start, model["cos"]["count"]))
                    off = model["noise_offset"]
                    cache[rows, off:off + noise.shape[1]] = noise.astype("<f4")
                cache.flush()
                print(f"  {model['short']}: {model['label']}", flush=True)
        finally:
            del cache
        temporary.replace(data_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    page = _HTML.replace("__PLOTLY_BUNDLE__", get_plotlyjs())
    html_tmp = html_path.with_suffix(".html.tmp")
    html_tmp.write_text(page, encoding="utf-8")
    html_tmp.replace(html_path)
    print(f"Ready: {html_path}\nOpen the HTML, then select {data_path.name}", flush=True)
    return dict(html=html_path, data=data_path, n_rays=n, size_mb=expected_size / 1e6)


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galaxy sightline comparison · CAMELS IllustrisTNG</title>
<style>
:root{color-scheme:light;--ink:#182b3a;--muted:#586774;--line:#d9e1e7;--blue:#17648b;--paper:#fff;--bg:#f3f6f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1540px;margin:auto;padding:26px 30px 40px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:.13em;color:var(--blue);text-transform:uppercase}
h1{font-size:29px;line-height:1.2;margin:6px 0 10px;letter-spacing:-.025em}h2{font-size:17px;margin:0 0 12px}p{margin:8px 0}.muted{color:var(--muted)}
.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px 20px}.load{margin:20px 0 14px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}.load p{margin:0;flex:1;min-width:260px}
button,.file-button{font:inherit;border-radius:7px;border:1px solid #bacbd7;background:#fff;color:var(--ink);padding:8px 13px;cursor:pointer;font-weight:650;display:inline-block}
button:hover,.file-button:hover{background:#eef5fa}button:disabled{cursor:default;opacity:.45}.primary,.file-button{background:var(--blue);border-color:var(--blue);color:white}.primary:hover,.file-button:hover{background:#104f70}
input,select{font:inherit;color:var(--ink);border:1px solid #b7c6d2;background:#fff;border-radius:6px;padding:7px 8px;min-width:0;width:100%;height:36px}input:disabled{background:#eef1f4;color:#75838d}
input:focus,select:focus,button:focus-visible,.file-button:focus-within{outline:2px solid #3599c8;outline-offset:2px}label{display:block;font-size:12px;font-weight:650;color:#435765}label input,label select{display:block;margin-top:4px}
.screen-reader{position:absolute;left:-9999px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.fields{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.wide{grid-column:span 2}.full{grid-column:1/-1}.pair-inputs{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.panel-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px}.badge{font-size:12px;color:var(--muted);font-weight:500}.panel-letter{display:inline-block;color:var(--blue);margin-right:7px}.panel-error,#error{color:#a02927;font-size:13px;margin-top:8px;min-height:0}.panel-error:empty,#error:empty{display:none}
.toolbar{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:14px 0}.toolbar label{width:150px}.toolbar .cohort{width:190px}.toolbar .spacer{flex:1}.plot-card{padding:4px 8px 0;overflow:hidden}#plot{width:100%;min-height:620px}
.details{margin-top:14px}.details .card{padding:16px 20px}.meta{font-size:13px;margin:0 0 12px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;border-bottom:1px solid #e7ecf0;padding:7px 6px}th{font-weight:650;color:var(--muted)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.swatch{display:inline-block;width:17px;height:3px;vertical-align:middle;margin-right:5px}details.card{margin-top:16px;font-size:13px}summary{cursor:pointer;font-weight:650}details ul{padding-left:20px}.model-info{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px;margin:14px 0}a{color:var(--blue)}footer{margin-top:12px;color:var(--muted);font-size:12px}.empty{padding:100px 25px;text-align:center;color:var(--muted)}[hidden]{display:none!important}
@media(max-width:850px){main{padding:18px 12px}.fields{grid-template-columns:repeat(2,minmax(0,1fr))}.two{gap:10px}.card{padding:14px}.panel-top{display:block}.badge{display:block;margin-bottom:10px}.details{grid-template-columns:1fr}.toolbar .spacer{display:none}h1{font-size:25px}.model-info{grid-template-columns:1fr}}
@media(max-width:540px){.controls{grid-template-columns:1fr}.plot-card{overflow-x:auto}#plot{min-width:680px}.load{gap:10px}.toolbar label{width:calc(50% - 10px)}}
</style>
<script>__PLOTLY_BUNDLE__</script>
</head>
<body><main>
<div class="eyebrow">CAMELS · IllustrisTNG · Galaxy-centred Lyα</div>
<h1>Compare two sightlines</h1>
<p class="muted">Choose each sightline independently. Every panel overlays all four matched feedback models.</p>
<section class="card load">
  <label class="file-button" for="data-file">Open spectra file<input class="screen-reader" id="data-file" type="file" accept=".bin"></label>
  <p id="file-status" role="status" aria-live="polite">Select the .bin file created alongside this HTML. Everything runs locally in your browser.</p>
</section>
<div id="error" role="alert"></div>
<div id="workspace" hidden>
<div class="two controls"><section class="card" id="left-controls"></section><section class="card" id="right-controls"></section></div>
<div class="toolbar">
  <label>Velocity ± (km/s)<input id="velocity-window" type="number" min="1" step="any" value="300"></label>
  <label>EW window ± (km/s)<input id="ew-window" type="number" min="1" step="any" value="300"></label>
  <label class="cohort" id="cohort-label" hidden>Halo sample<select id="cohort"><option value="all">All matched halos</option><option value="retained">Pass centroid-offset cut</option></select></label>
  <button id="reset-pair">Reset example pair</button><div class="spacer"></div>
  <button id="save-png">Save plot as PNG</button><button id="export-pair" class="primary">Export this pair as HTML</button>
</div>
<section class="card plot-card"><div id="plot"><div class="empty">Select a spectra file to begin.</div></div></section>
<div class="two details"><section class="card" id="left-details"></section><section class="card" id="right-details"></section></div>
<details class="card"><summary>Model definitions and measurement conventions</summary>
  <div class="model-info" id="model-info"></div>
  <p><a id="model-source" target="_blank" rel="noopener">CAMELS EX parameter table</a>. These factors are the simulation input parameters.</p>
  <ul>
    <li>Mass, star-formation type, and selection in b/Rvir use the <b>EX0 reference halo</b>. Low-sSFR means sSFR &lt; 10⁻¹¹ yr⁻¹; star-forming means sSFR ≥ 10⁻¹¹ yr⁻¹.</li>
    <li>Impact-parameter limits select existing sampled rays (both limits inclusive). Mass limits use lower ≤ log₁₀(M₂₀₀c/M☉) &lt; upper. No new sightlines are interpolated. Distances in kpc are physical, converted from ckpc/h using a/h.</li>
    <li>Each model's velocity is measured relative to its own matched galaxy, with periodic wrapping. Both panels share the displayed velocity and flux scales. A matched ray has the same transverse offset about each model's central.</li>
    <li>COS uses the precomputed tabulated LP1 line-spread-function spectra. S/N is per resolution element, with two output bins per element: σ<sub>F</sub> = √2 / (S/N). The fixed seeded Gaussian realization is rescaled when S/N changes. Native mode is noiseless.</li>
    <li>W<sub>r</sub> = (1215.67 Å / 299792.458 km s⁻¹) ∫(1 − F) dv, evaluated in the stated EW window with exact partial-pixel overlaps and periodic boundaries. No flux clipping or mean-flux rescaling is applied.</li>
    <li>The original full-sample viewer uses its companion data file. An exported pair embeds those two rays, with all four models, both resolutions, and interactive S/N and velocity controls.</li>
  </ul>
</details>
<footer id="footer"></footer>
</div>
</main>
<script id="embedded-data" type="application/json">null</script>
<script>
(() => {
'use strict';
const $ = id => document.getElementById(id);
const AXES = {1:'x', 2:'y', 3:'z'};
const SIDES = ['left', 'right'];
const TYPE = t => t === 'low_sSFR' ? 'Low sSFR' : t === 'star-forming' ? 'Star-forming' : t;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mod = (x,p) => ((x % p) + p) % p;
let data = null, rawReader = null, recordCache = new Map(), drawToken = 0, validPair = false, embedded = false, busy = false;
const options = (items, selected) => items.map(([value,label]) => `<option value="${esc(value)}"${String(value) === String(selected) ? ' selected' : ''}>${esc(label)}</option>`).join('');
const field = (side,name) => $(`${side}-${name}`);

function renderControls(side) {
  $(side+'-controls').innerHTML = `
  <div class="panel-top"><h2><span class="panel-letter">${side === 'left' ? 'A' : 'B'}</span>${side === 'left' ? 'Left' : 'Right'} sightline</h2><span class="badge" id="${side}-count"></span></div>
  <div class="fields">
    <label class="wide">Mass bin · EX0<select id="${side}-mass-bin">${options([['all','All masses'],['11:11.5','11 ≤ log M < 11.5'],['11.5:12','11.5 ≤ log M < 12'],['12:12.5','12 ≤ log M < 12.5'],['12.5:','log M ≥ 12.5'],['custom','Custom range']], 'all')}</select></label>
    <label>Min log M<input id="${side}-mmin" type="number" step="any" placeholder="No limit"></label>
    <label>Max log M (exclusive)<input id="${side}-mmax" type="number" step="any" placeholder="No limit"></label>
    <label class="wide">Star-formation type · EX0<select id="${side}-type"></select></label>
    <label>Min b/Rvir · EX0<input id="${side}-bmin" type="number" step="any" min="0" value="0"></label>
    <label>Max b/Rvir · EX0<input id="${side}-bmax" type="number" step="any" min="0" value="1"></label>
    <label>Line-of-sight axis<select id="${side}-axis">${options([['all','All axes'],['1','x'],['2','y'],['3','z']], 'all')}</select></label>
    <label>EX0 halo<select id="${side}-halo"></select></label>
    <label class="wide">Sightline<select id="${side}-ray"></select></label>
    <label class="wide">Spectrum<select id="${side}-mode">${options([['native','Native · noiseless'],['clean','COS LP1 · noiseless'],['noisy','COS LP1 + noise']], 'noisy')}</select></label>
    <label class="wide">S/N per resolution element<input id="${side}-snr" type="number" min="0.000001" step="any" value="20"></label>
  </div><div class="panel-error" id="${side}-error" role="alert"></div>`;
  const types = [...new Set(Object.values(data.halos).map(h => h.type))].sort();
  field(side,'type').innerHTML = options([['all','All types'], ...types.map(t => [t, TYPE(t)])], 'all');
  field(side,'mass-bin').addEventListener('change', () => {
    const v = field(side,'mass-bin').value;
    if (v !== 'custom') {
      const [lo, hi] = v === 'all' ? ['', ''] : v.split(':');
      field(side,'mmin').value = lo; field(side,'mmax').value = hi;
    }
    updateChoices(side); draw();
  });
  for (const name of ['mmin','mmax','bmin','bmax','type','axis']) field(side,name).addEventListener('change', () => {
    if (name === 'mmin' || name === 'mmax') field(side,'mass-bin').value = 'custom';
    updateChoices(side); draw();
  });
  field(side,'halo').addEventListener('change', () => { updateChoices(side); draw(); });
  for (const name of ['ray','mode','snr']) field(side,name).addEventListener('change', () => {
    field(side,'snr').disabled = field(side,'mode').value !== 'noisy'; draw();
  });
}

function bounds(side) {
  const num = (name, fallback) => field(side,name).value.trim() === '' ? fallback : Number(field(side,name).value);
  const mmin = num('mmin', -Infinity), mmax = num('mmax', Infinity), bmin = num('bmin', 0), bmax = num('bmax', Infinity);
  if ([mmin,mmax,bmin,bmax].some(Number.isNaN) || mmin >= mmax || bmin < 0 || bmin > bmax) throw Error('Enter increasing mass limits and nonnegative b/Rvir limits.');
  return {mmin,mmax,bmin,bmax};
}

function updateChoices(side, preferredID = null) {
  field(side,'error').textContent = '';
  let matches = [];
  try {
    const {mmin,mmax,bmin,bmax} = bounds(side), type = field(side,'type').value, axis = field(side,'axis').value;
    matches = data.rays.map((r,i) => ({r,i})).filter(({r}) => {
      const h = data.halos[r.halo];
      return h.logm >= mmin && h.logm < mmax && r.b >= bmin && r.b <= bmax &&
        (type === 'all' || h.type === type) && (axis === 'all' || r.axis === Number(axis)) &&
        ($('cohort').value !== 'retained' || h.retained === true);
    });
  } catch (err) { field(side,'error').textContent = err.message; }
  const halos = [...new Set(matches.map(({r}) => r.halo))].sort((a,b)=>a-b);
  const oldRayID = preferredID === null ? Number(field(side,'ray').selectedOptions[0]?.dataset.rayId) : Number(preferredID);
  const preferredRay = preferredID === null ? null : matches.find(({r}) => r.id === Number(preferredID));
  let halo = preferredRay?.r.halo ?? Number(field(side,'halo').value);
  if (!halos.includes(halo)) halo = halos[0];
  field(side,'halo').innerHTML = options(halos.map(id => [id, String(id)]), halo);
  const available = matches.filter(({r}) => r.halo === halo).sort((a,b)=>a.r.id-b.r.id);
  const selected = available.find(({r}) => r.id === oldRayID)?.i ?? available[0]?.i;
  field(side,'ray').innerHTML = available.map(({r,i}) => `<option value="${i}" data-ray-id="${r.id}"${i === selected ? ' selected' : ''}>${r.id} · ${AXES[r.axis]} · b/Rvir=${r.b.toFixed(4)}</option>`).join('');
  field(side,'halo').disabled = !halos.length; field(side,'ray').disabled = !available.length;
  field(side,'count').textContent = `${matches.length.toLocaleString()} rays · ${halos.length.toLocaleString()} halos`;
  if (!available.length && !field(side,'error').textContent) field(side,'error').textContent = 'No sampled sightlines match these filters. Widen a range or choose another type.';
}

function resetExamples() {
  $('cohort').value = 'all';
  for (let j=0;j<SIDES.length;j++) {
    const side = SIDES[j], ray = data.rays.find(r => r.id === data.preferred_rays[j]) || data.rays[Math.min(j,data.rays.length-1)];
    const halo = data.halos[ray.halo];
    const bin = halo.logm < 11 ? 'all' : halo.logm >= 12.5 ? '12.5:' : `${Math.floor(halo.logm*2)/2}:${Math.floor(halo.logm*2)/2+.5}`;
    const [lo,hi] = bin === 'all' ? ['',''] : bin.split(':');
    field(side,'mass-bin').value=bin; field(side,'mmin').value=lo; field(side,'mmax').value=hi;
    field(side,'type').value=halo.type; field(side,'axis').value='all';
    field(side,'bmin').value=Math.floor(ray.b*4)/4; field(side,'bmax').value=Math.floor(ray.b*4)/4+.25;
    field(side,'mode').value='noisy'; field(side,'snr').value=20; field(side,'snr').disabled=false;
    updateChoices(side,ray.id);
  }
  draw();
}

function currentIndex(side) { return field(side,'ray').value === '' ? null : Number(field(side,'ray').value); }
async function readRecord(index) {
  if (!recordCache.has(index)) {
    const promise = rawReader(index).then(buffer => {
      if (buffer.byteLength !== data.record_floats*4) throw Error('The data file is truncated. Rebuild it from the notebook.');
      const view = new DataView(buffer), values = new Float32Array(data.record_floats);
      for (let i=0;i<values.length;i++) values[i]=view.getFloat32(i*4,true);
      return values;
    });
    recordCache.set(index,promise);
    if (recordCache.size>16) recordCache.delete(recordCache.keys().next().value);
  }
  return recordCache.get(index);
}

function ew(flux, edges, galaxyVelocity, halfWindow) {
  const period = edges.at(-1)-edges[0], c = mod(galaxyVelocity-edges[0],period)+edges[0];
  let sum=0;
  for (let i=0;i<flux.length;i++) {
    let weight=0;
    for (const shift of [-period,0,period]) weight+=Math.max(0, Math.min(edges[i+1]+shift,c+halfWindow)-Math.max(edges[i]+shift,c-halfWindow));
    sum+=(1-flux[i])*weight;
  }
  return 1215.67/299792.458*sum;
}

function modeLabel(side) {
  const mode=field(side,'mode').value;
  return mode==='native' ? 'Native · noiseless' : mode==='clean' ? 'COS LP1 · noiseless' : `COS LP1 · S/N ${Number(field(side,'snr').value)}/resel`;
}

function chosenSpectrum(side, record, model) {
  const mode=field(side,'mode').value, spec=model[mode==='native'?'native':'cos'];
  const snr=Number(field(side,'snr').value), sigma=mode==='noisy'?Math.sqrt(data.cos_bins_per_resel)/snr:0;
  if (mode==='noisy' && (!(snr>0) || !Number.isFinite(snr) || !Number.isFinite(sigma))) throw Error('S/N must be a positive, finite number.');
  const flux=new Float64Array(spec.count);
  for(let i=0;i<flux.length;i++) flux[i]=record[spec.offset+i]+(sigma?record[model.noise_offset+i]*sigma:0);
  return {flux,spec};
}

async function draw() {
  if(!data) return;
  const token=++drawToken; validPair=false; toggleExport(); $('error').textContent='';
  const half=Number($('velocity-window').value), ewHalf=Number($('ew-window').value);
  const maxHalf=Math.min(...data.models.flatMap(m=>[m.native.period/2,m.cos.period/2]));
  if (!(half>0 && half<=maxHalf && ewHalf>0 && ewHalf<=maxHalf)) {
    $('error').textContent=`Both half-windows must be greater than zero and at most ${maxHalf.toFixed(2)} km/s (half the periodic box).`;
    $('plot').style.opacity='.35'; return;
  }
  $('plot').style.opacity='1';
  const traces=[], annotations=[], details={}, valid=[];
  let ymin=0, ymax=1;
  try {
    const indices=SIDES.map(currentIndex);
    const records=await Promise.all(indices.map(i=>i===null?null:readRecord(i)));
    if(token!==drawToken) return;
    for(let j=0;j<SIDES.length;j++) {
      const side=SIDES[j], index=indices[j], xDomain=j===0?[0,.465]:[.535,1];
      if(index===null) {
        annotations.push({xref:'paper',yref:'paper',x:(xDomain[0]+xDomain[1])/2,y:.5,text:'No matching sightline',showarrow:false,font:{size:15,color:'#647583'}});
        details[side]='<h2>No sightline selected</h2><p class="muted">Change the filters above to select a ray.</p>'; continue;
      }
      const ray=data.rays[index], halo=data.halos[ray.halo], rows=[];
      field(side,'error').textContent='';
      try {
        for(let m=0;m<data.models.length;m++) {
          const model=data.models[m], {flux,spec}=chosenSpectrum(side,records[j],model), pairs=[];
          for(let i=0;i<flux.length;i++) {
            const v=mod((spec.edges[i]+spec.edges[i+1])/2-ray.v[m]+spec.period/2,spec.period)-spec.period/2;
            pairs.push([v,flux[i]]);
            if(Math.abs(v)<=half) {ymin=Math.min(ymin,flux[i]);ymax=Math.max(ymax,flux[i]);}
          }
          pairs.sort((a,b)=>a[0]-b[0]);
          const equivalentWidth=ew(flux,spec.edges,ray.v[m],ewHalf);
          traces.push({type:'scatter',mode:'lines',x:pairs.map(p=>p[0]),y:pairs.map(p=>p[1]),
            xaxis:j?'x2':'x',yaxis:j?'y2':'y',name:`${model.short} — ${model.label}`,legendgroup:model.key,
            showlegend:j===0,line:{color:model.color,width:1.7},
            hovertemplate:`${model.short} · ${esc(model.label)}<br>v = %{x:.1f} km/s<br>F = %{y:.4f}<extra></extra>`});
          rows.push(`<tr><td><span class="swatch" style="background:${model.color}"></span>${model.short}</td><td class="num">${halo.groups[m]} / ${halo.subhalos[m]}</td><td class="num">${ray.bm[m].toFixed(4)}</td><td class="num" data-ew="${model.key}" data-value="${equivalentWidth}">${equivalentWidth.toFixed(4)}</td></tr>`);
        }
        valid.push(side);
        const title=`<b>${j?'B':'A'} · Ray ${ray.id} | EX0 halo ${ray.halo}</b><br>${esc(TYPE(halo.type))} · log₁₀(M₂₀₀c/M☉) = ${halo.logm.toFixed(3)}<br>b/Rvir = ${ray.b.toFixed(4)} · ${ray.b_kpc.toFixed(1)} kpc · ${AXES[ray.axis]} axis<br>${modeLabel(side)} · z = ${data.z.toFixed(3)}`;
        annotations.push({xref:'paper',yref:'paper',x:xDomain[0],y:1.035,xanchor:'left',yanchor:'bottom',align:'left',text:title,showarrow:false,font:{size:12,color:'#182b3a'}});
        details[side]=`<h2>${j?'B':'A'} · Ray ${ray.id} / EX0 halo ${ray.halo}</h2>
          <p class="meta">${esc(TYPE(halo.type))} · M₂₀₀c = ${Math.pow(10,halo.logm).toExponential(3)} M☉ · Rvir,EX0 = ${halo.rvir_kpc.toFixed(1)} physical kpc<br>b = ${ray.b_kpc.toFixed(2)} physical kpc · b/Rvir,EX0 = ${ray.b.toFixed(6)} · axis ${AXES[ray.axis]}<br>${modeLabel(side)}${field(side,'mode').value==='noisy'?` · noise seed ${data.noise_seed}`:''}</p>
          <table><thead><tr><th>Model</th><th class="num">Group / subhalo</th><th class="num">b/Rvir,model</th><th class="num">Wᵣ (Å)</th></tr></thead><tbody>${rows.join('')}</tbody></table>
          <p class="meta" style="margin-top:9px;margin-bottom:0">Wᵣ integrated over ±${ewHalf} km/s about each matched galaxy.</p>`;
      } catch(err) {
        field(side,'error').textContent=err.message;
        details[side]='<h2>Invalid selection</h2><p class="muted">Correct the highlighted input above.</p>';
      }
    }
    if(token!==drawToken) return;
    // If the left selection is empty, still show the shared model legend.
    if(!valid.includes('left')) for(const t of traces) t.showlegend=true;
    const margin=Math.max(.08,.08*(ymax-ymin)), yrange=[ymin-margin,ymax+margin];
    const xBase={range:[-half,half],title:{text:'Velocity relative to matched galaxy (km/s)',font:{size:12}},zeroline:true,zerolinecolor:'#9baab5',gridcolor:'#edf1f4',showline:true,linecolor:'#bac6d0',ticks:'outside',tickfont:{size:11}};
    const yBase={range:yrange,gridcolor:'#edf1f4',zeroline:false,showline:true,linecolor:'#bac6d0',ticks:'outside',tickfont:{size:11}};
    const shapes=[0,1].flatMap(j=>[0,1].map(y=>({type:'line',xref:j?'x2':'x',yref:j?'y2':'y',x0:-half,x1:half,y0:y,y1:y,line:{color:'#bec8cf',width:1,dash:'dot'},layer:'below'})));
    const layout={height:640,margin:{l:68,r:24,t:125,b:150},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Arial, sans-serif',color:'#213747'},
      xaxis:{...xBase,domain:[0,.465],anchor:'y'},xaxis2:{...xBase,domain:[.535,1],anchor:'y2',matches:'x'},
      yaxis:{...yBase,domain:[0,1],anchor:'x',title:{text:'Normalized flux'}},yaxis2:{...yBase,domain:[0,1],anchor:'x2',matches:'y'},
      annotations,shapes,hovermode:'closest',legend:{orientation:'h',x:0,y:-.24,xanchor:'left',yanchor:'top',font:{size:12},itemclick:false,itemdoubleclick:false},
      title:{text:`Lyα · snapshot ${data.snapnum} · EW window ±${ewHalf} km/s${valid.some(s=>field(s,'mode').value==='noisy')?` · noise seed ${data.noise_seed}`:''}`,x:.5,xanchor:'center',y:.995,font:{size:12,color:'#637582'}}};
    await Plotly.react($('plot'),traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['toggleSpikelines','select2d','lasso2d'],toImageButtonOptions:{format:'png',filename:'sightline_comparison',scale:2}});
    if(token!==drawToken) return;
    for(const side of SIDES) $(side+'-details').innerHTML=details[side];
    validPair=valid.length===2; toggleExport();
  } catch(err) {if(token===drawToken) $('error').textContent=err.message;}
}

function toggleExport() {
  $('export-pair').disabled=!validPair||busy; $('save-png').disabled=!validPair||busy;
}
function captureState() {
  const state={velocity:$('velocity-window').value,ew:$('ew-window').value,cohort:$('cohort').value,sides:{}};
  for(const side of SIDES) {
    state.sides[side]=Object.fromEntries(['mass-bin','mmin','mmax','type','bmin','bmax','axis','halo','mode','snr'].map(k=>[k,field(side,k).value]));
    state.sides[side].rayID=data.rays[currentIndex(side)].id;
  }
  return state;
}
function restoreState(state) {
  $('velocity-window').value=state.velocity; $('ew-window').value=state.ew; $('cohort').value=state.cohort;
  for(const side of SIDES) {
    for(const [k,v] of Object.entries(state.sides[side])) if(k!=='rayID') field(side,k).value=v;
    field(side,'snr').disabled=field(side,'mode').value!=='noisy'; updateChoices(side,state.sides[side].rayID);
  }
  draw();
}
function setup(header, reader, isEmbedded=false, state=null) {
  if(header.format!=='lya-galaxy-viewer'||header.version!==1||header.models?.length!==4||!header.rays?.length) throw Error('This is not a supported galaxy spectrum dataset.');
  ++drawToken; data=header; rawReader=reader; recordCache=new Map(); embedded=isEmbedded;
  $('plot').querySelector('.empty')?.remove();
  $('workspace').hidden=false; $('cohort-label').hidden=!data.has_retained_filter;
  $('file-status').textContent=`${embedded?'Embedded comparison':'Full sample'} · ${data.rays.length.toLocaleString()} paired sightlines · ${Object.keys(data.halos).length.toLocaleString()} halos · z = ${data.z.toFixed(3)} · noise seed ${data.noise_seed}`;
  $('footer').textContent=`Snapshot ${data.snapnum} · a = ${data.scale_factor.toFixed(6)} · h = ${data.hubble_param} · All selection labels reference EX0. ${embedded?'This file contains the selected pair.':'Spectra stay on this computer; only selected records are read.'}`;
  $('model-info').innerHTML=data.models.map(m=>`<div><span class="swatch" style="background:${m.color}"></span><b>${m.short} — ${esc(m.label)}</b><br><span class="muted">${esc(m.parameters)}</span></div>`).join('');
  $('model-source').href=data.model_source;
  const maxHalf=Math.min(...data.models.flatMap(m=>[m.native.period/2,m.cos.period/2]));
  for(const id of ['velocity-window','ew-window']) {$(id).max=maxHalf;$(id).value=Math.min(300,maxHalf);}
  $('cohort').value='all';
  for(const side of SIDES) renderControls(side);
  if(state) restoreState(state); else resetExamples();
}

$('data-file').addEventListener('change',async event=>{
  const file=event.target.files[0]; if(!file) return;
  $('error').textContent='';
  try {
    const prefix=await file.slice(0,16).arrayBuffer();
    if(prefix.byteLength!==16||new TextDecoder().decode(prefix.slice(0,8))!=='LYACGM01') throw Error('Choose the .bin file exported by build_spectrum_viewer.');
    const headerLength=Number(new DataView(prefix).getBigUint64(8,true));
    if(!Number.isSafeInteger(headerLength)||headerLength<2||headerLength>128*1024*1024||headerLength+16>file.size) throw Error('Invalid data-file header.');
    const header=JSON.parse(await file.slice(16,16+headerLength).text()), start=16+headerLength, bytes=header.record_floats*4;
    if(!Number.isSafeInteger(bytes)||bytes<=0||start+header.n_rows*bytes!==file.size||header.n_rows!==header.rays.length) throw Error('Data-file size does not match its index. Rebuild the viewer data.');
    setup(header,index=>file.slice(start+index*bytes,start+(index+1)*bytes).arrayBuffer());
  } catch(err) {$('error').textContent=err.message;}
});
for(const id of ['velocity-window','ew-window']) $(id).addEventListener('change',draw);
$('cohort').addEventListener('change',()=>{for(const side of SIDES) updateChoices(side);draw();});
$('reset-pair').addEventListener('click',resetExamples);
$('save-png').addEventListener('click',()=>Plotly.downloadImage($('plot'),{format:'png',filename:`lya_rays_${SIDES.map(s=>data.rays[currentIndex(s)].id).join('_')}`,width:1500,height:640,scale:2}));

function toBase64(bytes) {
  let text=''; for(let i=0;i<bytes.length;i+=32768) text+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return btoa(text);
}
$('export-pair').addEventListener('click',async()=>{
  if(!validPair||busy) return;
  busy=true; toggleExport();
  try {
    const state=captureState(), indices=[...new Set(SIDES.map(currentIndex))], bytes=data.record_floats*4;
    const raw=new Uint8Array(indices.length*bytes);
    for(let j=0;j<indices.length;j++) raw.set(new Uint8Array(await rawReader(indices[j])),j*bytes);
    const subset={...data,n_rows:indices.length,rays:indices.map(i=>data.rays[i]),preferred_rays:SIDES.map(s=>state.sides[s].rayID)};
    subset.halos=Object.fromEntries([...new Set(subset.rays.map(r=>String(r.halo)))].map(id=>[id,data.halos[id]]));
    const payload={header:subset,records:toBase64(raw),state};
    const clone=document.documentElement.cloneNode(true);
    clone.querySelector('#embedded-data').textContent=JSON.stringify(payload).replace(/</g,'\\u003c');
    clone.querySelector('#plot').innerHTML=''; clone.querySelector('#error').textContent='';
    clone.querySelector('#data-file').value='';
    const blob=new Blob(['<!doctype html>\n',clone.outerHTML],{type:'text/html;charset=utf-8'}), url=URL.createObjectURL(blob);
    const link=document.createElement('a');link.href=url;link.download=`lya_compare_${state.sides.left.rayID}_${state.sides.right.rayID}.html`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),30000);
  } catch(err) {$('error').textContent=err.message;} finally {busy=false;toggleExport();}
});
try {
  const payload=JSON.parse($('embedded-data').textContent);
  if(payload) {
    const bytes=Uint8Array.from(atob(payload.records),c=>c.charCodeAt(0)), stride=payload.header.record_floats*4;
    if(bytes.byteLength!==stride*payload.header.n_rows) throw Error('Embedded comparison data are incomplete.');
    setup(payload.header,index=>Promise.resolve(bytes.buffer.slice(index*stride,(index+1)*stride)),true,payload.state);
  }
} catch(err) {$('error').textContent=err.message;}
})();
</script>
</body></html>
'''
