"""Plot the saved COS S/N sweep summaries."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_snr_sweep(root):
    sweep_root = Path(root) / "outputs/cos_lsf_snr/snr_sweep"
    summary = sweep_root / "summary"
    output = sweep_root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (summary / "bootstrap_metadata.json").read_text()
    )

    def load_table(name):
        path = summary / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["output_sha256"][name]:
            raise ValueError(f"{name} differs from the bootstrap output.")
        return pd.read_csv(path)

    clean = load_table("clean_accuracy.csv").set_index("task")
    results = load_table("snr_accuracy.csv")
    seeds = load_table("seed_accuracy.csv")

    tasks = list(metadata["task_models"])
    snr = np.asarray(metadata["snr_levels"], dtype=float)
    noise_seeds = metadata["noise_seeds"]

    positions = [(1, 1)] + [
        (row, col) for row in range(2, 5) for col in (1, 2)
    ]
    titles = [
        "Four-class" if task == "four-class" else task
        for task in tasks
    ]

    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#6B7280"
    xlim = np.array([8.0, 100.0])
    offsets = np.exp(np.linspace(-0.035, 0.035, len(noise_seeds)))
    plotly_symbols = ("circle-open", "square-open", "diamond-open")
    mpl_symbols = ("o", "s", "D")

    interactive = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=titles,
        vertical_spacing=0.075,
        horizontal_spacing=0.09,
        x_title="Continuum S/N per nominal resolution element",
        y_title="Held-out accuracy (%)",
    )

    static = plt.figure(figsize=(11, 12.8), facecolor="white")
    grid = static.add_gridspec(4, 2)
    axes = [
        static.add_subplot(
            grid[row - 1, :] if i == 0 else grid[row - 1, col - 1]
        )
        for i, (row, col) in enumerate(positions)
    ]

    def add_band(x, low, high, color, group, row, col):
        x = np.asarray(x)
        low = np.broadcast_to(low, x.shape)
        high = np.broadcast_to(high, x.shape)

        interactive.add_trace(
            go.Scatter(
                x=np.r_[x, x[::-1]],
                y=np.r_[low, high[::-1]],
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=color,
                legendgroup=group,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    for i, (task, (row, col), ax) in enumerate(
        zip(tasks, positions, axes)
    ):
        data = (
            results.loc[results["task"].eq(task)]
            .set_index("snr_resel")
            .loc[snr]
        )
        reference = clean.loc[task]
        chance = 100.0 / len(metadata["task_models"][task])
        show_legend = i == 0

        # Task-specific noiseless reference and its confidence interval.
        add_band(
            xlim,
            reference["ci_low_pct"],
            reference["ci_high_pct"],
            "rgba(213,94,0,0.13)",
            "clean",
            row,
            col,
        )
        interactive.add_trace(
            go.Scatter(
                x=xlim,
                y=[reference["accuracy_pct"]] * 2,
                mode="lines",
                name="Noiseless and 95% CI",
                legendgroup="clean",
                legendrank=20,
                showlegend=show_legend,
                line=dict(color=orange, dash="dash", width=2),
                customdata=np.tile(
                    reference[["ci_low_pct", "ci_high_pct"]].to_numpy(),
                    (2, 1),
                ),
                hovertemplate=(
                    "Noiseless: %{y:.2f}%"
                    "<br>95% CI: [%{customdata[0]:.2f}, "
                    "%{customdata[1]:.2f}]%<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        interactive.add_trace(
            go.Scatter(
                x=xlim,
                y=[chance] * 2,
                mode="lines",
                name="Chance",
                legendgroup="chance",
                legendrank=30,
                showlegend=show_legend,
                line=dict(color=gray, dash="dot", width=1.5),
                hovertemplate="Chance: %{y:.0f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Mean accuracy across the three noise realizations.
        add_band(
            snr,
            data["ci_low_pct"].to_numpy(),
            data["ci_high_pct"].to_numpy(),
            "rgba(0,114,178,0.17)",
            "mean",
            row,
            col,
        )
        interactive.add_trace(
            go.Scatter(
                x=snr,
                y=data["accuracy_pct"],
                mode="lines+markers",
                name="Mean and 95% CI",
                legendgroup="mean",
                legendrank=10,
                showlegend=show_legend,
                line=dict(color=blue, width=2),
                marker=dict(color=blue, size=7),
                customdata=data[
                    [
                        "ci_low_pct",
                        "ci_high_pct",
                        "delta_pp",
                        "delta_ci_low_pp",
                        "delta_ci_high_pp",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "S/N/resel: %{x:.0f}"
                    "<br>Accuracy: %{y:.2f}%"
                    "<br>95% CI: [%{customdata[0]:.2f}, "
                    "%{customdata[1]:.2f}]%"
                    "<br>Change from noiseless: %{customdata[2]:+.2f} pp"
                    "<br>95% paired CI: [%{customdata[3]:+.2f}, "
                    "%{customdata[4]:+.2f}] pp<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Draw the same quantities in the static figure.
        ax.axvspan(10, 80, color="black", alpha=0.035, zorder=0)
        ax.axhspan(
            reference["ci_low_pct"],
            reference["ci_high_pct"],
            color=orange,
            alpha=0.13,
            zorder=1,
        )
        ax.axhline(
            reference["accuracy_pct"],
            color=orange,
            linestyle="--",
            linewidth=1.5,
            label="Noiseless and 95% CI",
        )
        ax.axhline(
            chance,
            color=gray,
            linestyle=":",
            linewidth=1.3,
            label="Chance",
        )
        ax.fill_between(
            snr,
            data["ci_low_pct"].to_numpy(),
            data["ci_high_pct"].to_numpy(),
            color=blue,
            alpha=0.17,
            zorder=2,
        )
        ax.plot(
            snr,
            data["accuracy_pct"],
            "-o",
            color=blue,
            linewidth=1.6,
            markersize=4,
            label="Mean and 95% CI",
            zorder=5,
        )

        for j, noise_seed in enumerate(noise_seeds):
            replicate = (
                seeds.loc[
                    seeds["task"].eq(task)
                    & seeds["noise_seed"].eq(noise_seed)
                ]
                .set_index("snr_resel")
                .loc[snr, "accuracy_pct"]
            )
            label = f"Noise seed {noise_seed}"

            interactive.add_trace(
                go.Scatter(
                    x=snr * offsets[j],
                    y=replicate,
                    mode="markers",
                    name=label,
                    legendgroup=f"seed_{noise_seed}",
                    legendrank=40 + j,
                    showlegend=show_legend,
                    marker=dict(
                        symbol=plotly_symbols[j],
                        color=gray,
                        size=6,
                        line=dict(width=1),
                    ),
                    customdata=snr,
                    hovertemplate=(
                        "S/N/resel: %{customdata:.0f}"
                        "<br>Accuracy: %{y:.2f}%"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            ax.scatter(
                snr * offsets[j],
                replicate,
                marker=mpl_symbols[j],
                facecolors="none",
                edgecolors=gray,
                s=22,
                linewidths=0.8,
                label=label,
                zorder=4,
            )

        interactive.add_vrect(
            x0=10,
            x1=80,
            fillcolor="gray",
            opacity=0.045,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )

        ax.set_title(titles[i], loc="left", fontsize=11)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(0, 100)
        ax.set_xticks(snr)
        ax.set_xticklabels([f"{value:g}" for value in snr])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.minorticks_off()
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)

    setup = (
        "z = 0 · 15 LOS/bundle · LP1; assumed G130M/1291 at 1300 Å"
    )
    uncertainty = (
        f"Three noise seeds · Pointwise 95% CIs from "
        f"{metadata['n_bootstrap']:,} resamples of "
        f"{metadata['n_bundles']} bundle IDs"
    )
    note = (
        "Shading: S/N 10–80. Seed markers offset horizontally. "
        "Nominal resolution element: two output bins."
    )

    interactive.update_annotations(font_size=12)
    interactive.update_xaxes(
        type="log",
        range=np.log10(xlim).tolist(),
        tickmode="array",
        tickvals=snr,
        ticktext=[f"{value:g}" for value in snr],
        gridcolor="#E5E7EB",
    )
    interactive.update_yaxes(
        range=[0, 100],
        dtick=25,
        gridcolor="#E5E7EB",
        zeroline=False,
    )
    interactive.update_layout(
        title=dict(
            text=(
                "COS LP1: accuracy versus S/N"
                f"<br><sup>{setup}<br>{uncertainty}</sup>"
            ),
            x=0.07,
            font=dict(size=18),
        ),
        template="plotly_white",
        width=1050,
        height=1320,
        margin=dict(l=85, r=35, t=145, b=215),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.10,
            yanchor="top",
            font=dict(size=11),
            groupclick="togglegroup",
        ),
    )
    interactive.add_annotation(
        x=0.5,
        y=-0.20,
        xref="paper",
        yref="paper",
        text=note,
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=10),
    )

    static.subplots_adjust(
        left=0.085,
        right=0.98,
        top=0.90,
        bottom=0.17,
        hspace=0.48,
        wspace=0.18,
    )
    static.suptitle("COS LP1: accuracy versus S/N", y=0.99, fontsize=16)
    static.text(0.5, 0.965, setup, ha="center", fontsize=10)
    static.text(0.5, 0.945, uncertainty, ha="center", fontsize=9)
    static.text(
        0.5, 0.12,
        "Continuum S/N per nominal resolution element",
        ha="center", fontsize=11,
    )
    static.text(
        0.015, 0.54, "Held-out accuracy (%)",
        va="center", rotation="vertical", fontsize=11,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [2, 0, 1, 3, 4, 5]
    static.legend(
        [handles[i] for i in legend_order],
        [labels[i] for i in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    static.text(0.5, 0.022, note, ha="center", fontsize=8)

    interactive.write_html(
        str(output / "accuracy_vs_snr.html"),
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    with plt.rc_context({"pdf.fonttype": 42}):
        static.savefig(output / "accuracy_vs_snr.pdf", bbox_inches="tight")
        static.savefig(
            output / "accuracy_vs_snr.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(static)
    print(f"Saved HTML, PDF and PNG in {output}")
    return interactive, static