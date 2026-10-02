"""Paired test-sighting overlay for the COS instrument."""

import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .manifests import load_manifest, los_ids
from .snr_sweep import MODELS, cache_lp1_flux, sigma_per_bin

OVERLAY_SNRS = (10, 20, 50, 80)

class OverlayData:
    """Keep track of the data (spectra, LOS IDs, pairwise scores) for a given sightline pair."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.models = tuple(MODELS)
        self.pairs = tuple(combinations(self.models, 2))

        test_a = self.root / "outputs/uvb_mean_flux"
        manifest = load_manifest(test_a / "manifests/z0_b15_seed42.npz")
        self.calibration = json.loads((test_a / "calibration/z0_b15_seed42.json").read_text())
        self.los_ids = np.sort(los_ids(manifest, "test"))

        if (
            self.los_ids.size != 2010
            or np.unique(self.los_ids).size != 2010
            or np.intersect1d(self.los_ids, los_ids(manifest, "train")).size
        ):
            raise ValueError("Expected 2010 unique test LOS IDs")

        self._positions = {int(los): i for i, los in enumerate(self.los_ids)}
        bundle_lookup = {
            int(los): int(bundle_id)
            for bundle_id in manifest["test_bundle_ids"]
            for los in manifest["bundles"][bundle_id]
        }
        self.bundle_ids = np.array([bundle_lookup[int(los)] for los in self.los_ids])

        # Reuse the snr_sweep cache validation to avoid recomputing the LP1 fluxes for each pair of models.
        lp1_arrays = []
        self.metadata = {}

        for model in self.models:
            flux, dv, metadata = cache_lp1_flux(model, self.root)

            if metadata["tau_scale"] != self.calibration["results"][model]["tau_scale"]:
                raise ValueError(f"{model} tau_scale mismatch between calibration and LP1 flux cache")

            if not np.isclose(dv, 2500.0 / 362, rtol=0.0, atol=1e-12):
                raise ValueError(f"{model} dv mismatch between calibration and LP1 flux cache")

            lp1_arrays.append(flux[self.los_ids].copy())
            self.metadata[model] = metadata
            del flux

        # Axes: (model, los, velocity)
        self.lp1 = np.stack(lp1_arrays)
        self.lp1.setflags(write=False)

        self.dv_native = 2500.0 / 2499
        self.dv_lp1 = 2500.0 / 362

        # Pixel centers on the same [0, 2500] km/s grid as the LP1 fluxes, but with the native COS resolution.
        self.v_native = (np.arange(2499) + 0.5) * self.dv_native
        self.v_lp1 = (np.arange(362) + 0.5) * self.dv_lp1

        # Compute unweighted separations (need to do it only once since different SNRs only rescales the separations).
        separations = []
        for a, b in combinations(range(len(self.models)), 2):
            difference = self.lp1[a] - self.lp1[b]
            separations.append(np.sum(difference**2, axis=1, dtype=np.float64))

        # Axes: test_los, model_pair
        self.squared_diff = np.column_stack(separations)
        self.squared_diff.setflags(write=False)

    def position(self, los_id):
        """Get the index of a given LOS ID in the test set."""
        if not isinstance(los_id, (int, np.integer)) or isinstance(los_id, (bool, np.bool_)):
            raise ValueError("los_id must be an integer")

        if int(los_id) not in self._positions:
            raise ValueError(f"los_id {los_id} not found in test set")

        return self._positions[int(los_id)]

    @lru_cache(maxsize=32)
    def native_flux(self, los_id):
        """Get the native-resolution flux for a given LOS ID."""
        pos = self.position(los_id)
        spectra = []

        for model in self.models:
            path = self.root / "data/raw" / f"{model}_spectra.hdf5"

            with h5py.File(path, "r") as f:
                tau = np.asarray(f[self.calibration["tau_key"]][int(los_id), :], dtype=np.float64)

            if tau.shape != (2499,) or not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
                raise ValueError(f"Invalid optical depth for {model} LOS {los_id}")

            scale = self.calibration["results"][model]["tau_scale"]
            spectra.append(np.exp(-scale * tau))

        flux = np.stack(spectra)

        # Connect the native-resolution flux to their LP1-resolution counterparts
        np.testing.assert_allclose(
            flux.mean(axis=1), self.lp1[:, pos, :].mean(axis=1), rtol=0.0, atol=1e-12, err_msg=f"Native and LP1 fluxes do not match for LOS {los_id}"
        )

        flux.setflags(write=False)
        return flux

    def rank_table(self, pair=("EX0", "EX3"), snr_resel=20, descending=True):
        """Get a DataFrame of LOS pairs ranked by their pairwise score for a given SNR."""
        pair = tuple(pair)
        if pair not in self.pairs:
            raise ValueError(f"Invalid model pair {pair}, must be one of {self.pairs}")

        column = self.pairs.index(pair)
        separation = self.squared_diff[:, column]
        scores = separation / sigma_per_bin(snr_resel)**2

        # Break ties by LOS ID to ensure deterministic ranking. Sort before SNR scaling.
        order = np.lexsort((self.los_ids, -separation if descending else separation))

        return pd.DataFrame(
            {
                "rank": np.arange(1, len(order) + 1),
                "los_id": self.los_ids[order],
                "bundle_id": self.bundle_ids[order],
                "delta_chi2": scores[order],
            }
        )

    def chi2_table(self, los_id):
        """Get a DataFrame of pairwise chi-squared values for a given LOS ID for the full SNR range."""
        pos = self.position(los_id)
        variances = np.array([sigma_per_bin(snr)**2 for snr in OVERLAY_SNRS])

        values = self.squared_diff[pos, :, None] / variances[None, :]
        return pd.DataFrame(
            values,
            index=pd.Index(["-".join(pair) for pair in self.pairs], name="model_pair"),
            columns=[f"S/N {snr}" for snr in OVERLAY_SNRS],
        )

def make_overlay_viewer(data):
    """Interactive overlays with selectable display noise and S/N."""
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = ("#222222", "#0072B2", "#D55E00", "#009E73")
    style = {"description_width": "initial"}

    pair = widgets.Dropdown(
        options=[("–".join(p), p) for p in data.pairs],
        value=("EX0", "EX3"),
        description="Ranking pair:",
        style=style,
        layout=widgets.Layout(width="250px"),
    )
    order = widgets.Dropdown(
        options=[
            ("Most separated first", "most"),
            ("Least separated first", "least"),
        ],
        value="most",
        description="Order:",
        style=style,
        layout=widgets.Layout(width="260px"),
    )
    snr = widgets.Dropdown(
        options=OVERLAY_SNRS,
        value=20,
        description="S/N per resel:",
        style=style,
        layout=widgets.Layout(width="190px"),
    )
    noisy = widgets.Checkbox(
        value=True,
        description="Add noise",
        indent=False,
        layout=widgets.Layout(width="140px"),
    )
    draw = widgets.BoundedIntText(
        value=0,
        min=0,
        max=1_000_000,
        description="Noise draw:",
        continuous_update=False,
        style=style,
        layout=widgets.Layout(width="185px"),
    )
    new_noise = widgets.Button(
        description="New noise",
        layout=widgets.Layout(width="110px"),
    )
    autoscale = widgets.Checkbox(
        value=False,
        description="Autoscale ΔF",
        indent=False,
        layout=widgets.Layout(width="160px"),
    )
    sightline = widgets.Dropdown(
        options=[],
        description="Test LOS:",
        style=style,
        layout=widgets.Layout(width="440px"),
    )

    previous = widgets.Button(
        description="Previous", layout=widgets.Layout(width="85px")
    )
    following = widgets.Button(
        description="Next", layout=widgets.Layout(width="70px")
    )
    most = widgets.Button(
        description="Most", layout=widgets.Layout(width="70px")
    )
    least = widgets.Button(
        description="Least", layout=widgets.Layout(width="70px")
    )
    reset = widgets.Button(
        description="Full view", layout=widgets.Layout(width="95px")
    )

    status = widgets.HTML()
    note = widgets.HTML()
    score_panel = widgets.HTML()
    errors = widgets.Output()

    figure = go.FigureWidget(make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.085,
        subplot_titles=[
            f"Native, noiseless ({data.dv_native:.4f} km/s per bin)",
            "LP1 + rebin",
            "LP1 differences: ΔF = F(EXi) − F(EX0)",
        ],
    ))

    trace_ids = {"native": [], "lp1": [], "delta": []}

    for stage, row, velocity in (
        ("native", 1, data.v_native),
        ("lp1", 2, data.v_lp1),
    ):
        for i, model in enumerate(data.models):
            trace_ids[stage].append(len(figure.data))
            figure.add_trace(
                go.Scattergl(
                    x=velocity,
                    y=np.zeros_like(velocity),
                    mode="lines",
                    name=model,
                    legendgroup=model,
                    legendrank=i,
                    showlegend=(stage == "native"),
                    line=dict(
                        color=colors[i],
                        width=1.3 if stage == "native" else 1.7,
                    ),
                    hovertemplate=(
                        "v: %{x:.2f} km/s"
                        "<br>Flux: %{y:.5f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=row,
                col=1,
            )

    # One band, updated whenever the selected S/N or noise mode changes.
    band_id = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[0, 2500, 2500, 0, 0],
            y=[0, 0, 0, 0, 0],
            mode="lines",
            fill="toself",
            fillcolor="rgba(100,110,120,0.18)",
            line=dict(color="rgba(100,110,120,0.65)", width=0.8),
            name="Noise band",
            legendgroup="noise_band",
            legendrank=10,
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )

    for i, model in enumerate(data.models):
        trace_ids["delta"].append(len(figure.data))
        figure.add_trace(
            go.Scatter(
                x=data.v_lp1,
                y=np.zeros_like(data.v_lp1),
                mode="lines",
                name=f"{model} − EX0",
                legendgroup=model,
                showlegend=False,
                line=dict(
                    color=colors[i],
                    width=1.0 if i == 0 else 1.6,
                    dash="dot" if i == 0 else "solid",
                ),
                hovertemplate=(
                    "v: %{x:.2f} km/s"
                    "<br>ΔF: %{y:+.5f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=3,
            col=1,
        )

    for row in (1, 2):
        figure.add_hline(
            y=1.0, line_color="#999999", line_dash="dot",
            line_width=0.8, row=row, col=1,
        )
        figure.update_yaxes(title_text="Flux", row=row, col=1)

    figure.update_xaxes(range=[0, 2500], gridcolor="#E5E7EB")
    figure.update_xaxes(
        title_text="Velocity along sightline [km/s]", row=3, col=1
    )
    figure.update_yaxes(gridcolor="#E5E7EB", zeroline=False)
    figure.update_yaxes(title_text="ΔF", row=3, col=1)
    figure.update_annotations(font_size=12)
    figure.update_layout(
        template="plotly_white",
        width=1050,
        height=830,
        margin=dict(l=75, r=30, t=110, b=120),
        hovermode="x unified",
        dragmode="zoom",
        uirevision="cos-overlay",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
            yanchor="top",
            groupclick="togglegroup",
        ),
    )

    clean_delta_max = float(
        np.max(np.abs(data.lp1[1:] - data.lp1[0]))
    )
    state = {
        "busy": False,
        "ranked": None,
        "last_los": None,
        "shown": None,
        "difference": None,
        "band_sigma": None,
    }

    def display_flux(los_id, clean):
        if not noisy.value:
            return clean

        # Display-only realizations, independent between models.
        # Reusing these seeds keeps standard-normal draws fixed across S/N.
        standard_noise = np.stack([
            np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([
                        20261001, int(draw.value), i, int(los_id)
                    ])
                )
            ).standard_normal(clean.shape[1])
            for i in range(len(data.models))
        ])

        # Preserve all flux values, including those outside [0, 1].
        return clean + sigma_per_bin(snr.value) * standard_noise

    def update_y_ranges():
        shown = state["shown"]
        difference = state["difference"]

        figure.update_yaxes(
            range=[-0.025, 1.025],
            autorange=False,
            row=1,
            col=1,
        )
        figure.update_yaxes(
            range=[
                min(-0.025, float(shown.min()) - 0.025),
                max(1.025, float(shown.max()) + 0.025),
            ],
            autorange=False,
            row=2,
            col=1,
        )

        extent = max(
            state["band_sigma"],
            float(np.max(np.abs(difference))),
        )
        if not autoscale.value:
            extent = max(extent, clean_delta_max)

        limit = 1.05 * extent
        figure.update_yaxes(
            range=[-limit, limit],
            autorange=False,
            row=3,
            col=1,
        )

    @errors.capture(clear_output=True)
    def render(change=None):
        if state["busy"] or sightline.value is None:
            return

        los_id = int(sightline.value)
        position = data.position(los_id)
        new_los = los_id != state["last_los"]
        record = state["ranked"].set_index("los_id").loc[los_id]

        sigma_bin = float(sigma_per_bin(snr.value))
        clean = data.lp1[:, position, :]
        shown = display_flux(los_id, clean)
        difference = shown - shown[0]

        # In noisy mode both members of each nontrivial difference
        # contain independent noise. EX0 minus itself remains exactly zero.
        band_sigma = (
            np.sqrt(2.0) * sigma_bin if noisy.value else sigma_bin
        )
        band_label = "σ(ΔF)" if noisy.value else "σ/bin"
        mode = "noisy" if noisy.value else "noiseless"

        state["shown"] = shown
        state["difference"] = difference
        state["band_sigma"] = band_sigma

        with figure.batch_update():
            if new_los:
                native = data.native_flux(los_id)
                for i in range(len(data.models)):
                    figure.data[trace_ids["native"][i]].y = native[i]
                figure.update_xaxes(
                    range=[0, 2500], autorange=False
                )

            # Update these on EVERY render, including S/N and draw changes.
            for i in range(len(data.models)):
                figure.data[trace_ids["lp1"][i]].y = shown[i]
                figure.data[trace_ids["delta"][i]].y = difference[i]

            figure.data[band_id].y = [
                -band_sigma, -band_sigma,
                band_sigma, band_sigma, -band_sigma,
            ]
            figure.data[band_id].name = (
                f"±1{band_label}: S/N {snr.value}"
            )

            figure.layout.annotations[1].text = (
                f"LP1 + rebin, {mode}: S/N {snr.value} per resel "
                f"({data.dv_lp1:.4f} km/s per bin)"
            )
            figure.layout.annotations[2].text = (
                f"ΔF = F(EXi) − F(EX0), from the {mode} LP1 spectra"
            )
            figure.update_layout(
                title=dict(
                    text=(
                        f"Paired held-out sightline {los_id}"
                        "<br><sup>z = 0; frozen mean-flux matching; "
                        "LP1, assumed G130M/1291 at 1300 Å</sup>"
                    ),
                    x=0.07,
                    font=dict(size=16),
                ),
            )
            update_y_ranges()

        state["last_los"] = los_id
        direction = (
            "most → least" if order.value == "most" else "least → most"
        )
        status.value = (
            f"<b>LOS {los_id}</b> · test bundle {int(record['bundle_id'])}"
            f" · rank {int(record['rank'])}/{len(data.los_ids)}"
            f" ({direction})"
            f" · {'–'.join(pair.value)}:"
            f" template Δχ² = <b>{record['delta_chi2']:,.3f}</b>"
            f" at S/N {snr.value}"
        )

        # Only the selected S/N column.
        table = data.chi2_table(los_id)[[f"S/N {snr.value}"]]
        score_panel.value = (
            "<style>"
            ".cos-overlay-scores th,.cos-overlay-scores td"
            "{padding:5px 14px;text-align:right;}"
            "</style>"
            f"<b>Template Δχ² at S/N {snr.value} per resel</b>"
            + table.to_html(
                border=0,
                classes="cos-overlay-scores",
                float_format=lambda value: f"{value:,.3f}",
            )
        )

        band_description = (
            "Band: ±σ(ΔF) = ±√2 σ/bin, including noise in both spectra."
            if noisy.value else
            "Band: ±σ/bin, the single-spectrum observational noise reference."
        )
        note.value = (
            f"<small>LP1 σ/bin = {sigma_bin:.5f}. "
            f"{band_description} "
            "Scores use noiseless LP1 templates and σ/bin over all 362 bins, "
            "including when zoomed. "
            "Noise draws are independent between models and fixed across "
            "S/N changes. Click legend labels to hide curves or the band."
            "</small>"
        )

        draw.disabled = not noisy.value
        new_noise.disabled = not noisy.value
        previous.disabled = sightline.index == 0
        following.disabled = sightline.index == len(sightline.options) - 1

    @errors.capture(clear_output=True)
    def rerank(change=None):
        old_los = sightline.value
        keep_los = change is not None and change["owner"] is snr

        ranked = data.rank_table(
            pair=pair.value,
            snr_resel=snr.value,
            descending=(order.value == "most"),
        )
        state["ranked"] = ranked
        state["busy"] = True

        try:
            sightline.options = [
                (
                    f"{int(row.rank):4d} | LOS {int(row.los_id)}"
                    f" | Δχ² = {row.delta_chi2:.5g}",
                    int(row.los_id),
                )
                for row in ranked.itertuples(index=False)
            ]
            sightline.value = (
                old_los if keep_los and old_los is not None
                else int(ranked.iloc[0]["los_id"])
            )
        finally:
            state["busy"] = False

        render()

    def step(amount):
        sightline.index = max(
            0,
            min(
                len(sightline.options) - 1,
                sightline.index + amount,
            ),
        )

    def jump(direction):
        if order.value == direction:
            sightline.index = 0
        else:
            order.value = direction

    @errors.capture(clear_output=True)
    def reset_view(_):
        with figure.batch_update():
            figure.update_xaxes(range=[0, 2500], autorange=False)
            update_y_ranges()

    sightline.observe(render, names="value")
    for control in (noisy, draw, autoscale):
        control.observe(render, names="value")
    for control in (pair, order, snr):
        control.observe(rerank, names="value")

    previous.on_click(lambda _: step(-1))
    following.on_click(lambda _: step(1))
    most.on_click(lambda _: jump("most"))
    least.on_click(lambda _: jump("least"))
    reset.on_click(reset_view)
    new_noise.on_click(
        lambda _: setattr(draw, "value", (draw.value + 1) % (draw.max + 1))
    )

    rerank()

    def control_row(children):
        return widgets.HBox(
            children,
            layout=widgets.Layout(
                flex_flow="row wrap", align_items="center"
            ),
        )

    viewer = widgets.VBox([
        control_row([pair, order, snr, noisy]),
        control_row([sightline, previous, following, most, least, reset]),
        control_row([draw, new_noise, autoscale]),
        status,
        figure,
        note,
        score_panel,
        errors,
    ])

    return viewer, figure