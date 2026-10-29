"""Replot the user-supplied KKLSS figure with the requested BFV-P1 replacement.

The BFV-P1 FPGA series is the author's supplied replacement (milliseconds).
All other CPU/FPGA series are transcribed from reference.png (milliseconds).
The plot uses the reference CPU arrays directly, without normalization.
This is reproduction of author-provided figure data, not new benchmarking.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
PARTIES = np.array([2, 4, 8, 16, 32, 64])
X = np.arange(PARTIES.size, dtype=float)

# ---------------------------------------------------------------------------
# LEGACY SOURCE DATA -- retained verbatim for provenance.
# ---------------------------------------------------------------------------
# fpga_ckks_p1 = [0.290, 0.477, 0.851, 1.74, 3.38, 6.59]
# fpga_ckks_p2 = [2.85, 4.35, 7.08, 14.50, 30.82, 58.57]
# fpga_bfv_p1 = [0.419, 0.699, 1.26, 2.38, 4.61, 9.08]
# fpga_bfv_p2 = [3.69, 5.86, 10.79, 21.37, 40.07, 80.55]
# cpu_ckks_p1 = [155, 266, 495, 994, 1897, 3732]
# cpu_ckks_p2 = [1145, 2205, 4186, 8480, 16710, 33474]
# cpu_bfv_p1 = [267, 477, 892, 1721, 3353, 6571]
# cpu_bfv_p2 = [1927, 3716, 6229, 11747, 23136, 54115]

FPGA_LEGACY = {
    "CKKS-P1": np.array([0.290, 0.477, 0.851, 1.74, 3.38, 6.59]),
    "CKKS-P2": np.array([2.85, 4.35, 7.08, 14.50, 30.82, 58.57]),
    # Author-supplied replacement; n=16 remains 2.380 ms.
    "BFV-P1": np.array([0.530, 0.814, 1.493, 2.380, 5.129, 10.482]),
    "BFV-P2": np.array([3.69, 5.86, 10.79, 21.37, 40.07, 80.55]),
}

KKLSS_CPU_LEGACY = {
    "CKKS-P1": np.array([155, 266, 495, 994, 1897, 3732], dtype=float),
    "CKKS-P2": np.array([1145, 2205, 4186, 8480, 16710, 33474], dtype=float),
    "BFV-P1": np.array([267, 477, 892, 1721, 3353, 6571], dtype=float),
    "BFV-P2": np.array([1927, 3716, 6229, 11747, 23136, 54115], dtype=float),
}

CPU_REFERENCE = {key: values.copy() for key, values in KKLSS_CPU_LEGACY.items()}

STYLE = {
    "FPGA-CKKS": {"color": "#d32612", "marker": "o", "linestyle": "-", "label": "FPGA (CKKS)"},
    "FPGA-BFV": {"color": "#bca66a", "marker": "s", "linestyle": "-", "label": "FPGA (BFV)"},
    "CPU-CKKS": {"color": "#51add3", "marker": "^", "linestyle": "--", "label": "CPU (CKKS)"},
    "CPU-BFV": {"color": "#36a914", "marker": "d", "linestyle": "--", "label": "CPU (BFV)"},
}

mpl.rcParams.update(
    {
        "font.family": ["Times New Roman", "Arial", "sans-serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 180,
        "savefig.dpi": 600,
    }
)


def display_value(value: float, fpga: bool = False) -> str:
    return f"{value:.3f}" if fpga else f"{value:.0f}"


def geometric_midpoint(values: np.ndarray, left_index: int = 2) -> float:
    return float(np.sqrt(values[left_index] * values[left_index + 1]))


def add_point_labels(
    ax,
    values: np.ndarray,
    position: str,
    clear_fpga_ckks_arrow: bool = False,
    fpga_labels: bool = False,
    center_n16_label: bool = False,
) -> None:
    y_offset = 8 if position == "above" else -10
    va = "bottom" if position == "above" else "top"
    for index, (x_value, y_value) in enumerate(zip(X, values)):
        x_offset = 3 if index == 0 else (-3 if index == len(X) - 1 else 0)
        ha = "left" if index == 0 else ("right" if index == len(X) - 1 else "center")
        point_y_offset = y_offset
        if clear_fpga_ckks_arrow and index == 3:
            x_offset = 0 if center_n16_label else -10
            point_y_offset = -10 if center_n16_label else -21
            ha = "center" if center_n16_label else "right"
        ax.annotate(
            display_value(float(y_value), fpga=fpga_labels),
            xy=(x_value, y_value),
            xytext=(x_offset, point_y_offset),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=15,
            color="black",
            annotation_clip=False,
            zorder=6,
        )


def add_callout(ax, label: str, values: np.ndarray, box_xy: tuple[float, float]) -> None:
    ax.annotate(
        label,
        xy=(2.5, geometric_midpoint(values)),
        xytext=box_xy,
        textcoords="data",
        ha="center",
        va="center",
        fontsize=19,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#9b9b9b",
            "linewidth": 0.75,
            "alpha": 0.97,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "#505050",
            "linewidth": 1.7,
            "shrinkA": 3,
            "shrinkB": 2,
        },
        zorder=7,
    )


def draw_panel(ax, parameter: str) -> None:
    ckks_key = f"CKKS-{parameter}"
    bfv_key = f"BFV-{parameter}"
    curves = (
        ("FPGA-CKKS", FPGA_LEGACY[ckks_key], "below"),
        ("FPGA-BFV", FPGA_LEGACY[bfv_key], "above"),
        ("CPU-CKKS", CPU_REFERENCE[ckks_key], "below"),
        ("CPU-BFV", CPU_REFERENCE[bfv_key], "above"),
    )

    for style_key, values, label_position in curves:
        style = STYLE[style_key]
        ax.plot(
            X,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            markersize=5.7,
            markeredgecolor=style["color"],
            markerfacecolor=style["color"],
            label=style["label"],
            zorder=4,
        )
        add_point_labels(
            ax,
            values,
            label_position,
            clear_fpga_ckks_arrow=(style_key == "FPGA-CKKS"),
            fpga_labels=style_key.startswith("FPGA-"),
            center_n16_label=(parameter == "P1" and style_key == "FPGA-CKKS"),
        )

    all_values = np.concatenate(
        (
            FPGA_LEGACY[ckks_key],
            FPGA_LEGACY[bfv_key],
            CPU_REFERENCE[ckks_key],
            CPU_REFERENCE[bfv_key],
        )
    )
    ax.set_yscale("log")
    ax.set_ylim(float(all_values.min()) * 0.45, float(all_values.max()) * 3.20)
    ax.set_xlim(-0.25, len(X) - 0.75)
    ax.set_xticks(X, PARTIES)
    ax.tick_params(axis="both", which="both", direction="out")

    panel_letter = "a" if parameter == "P1" else "b"
    parameter_number = "1" if parameter == "P1" else "2"
    ax.set_xlabel(
        rf"({panel_letter}). Multiplication with Relinearization ($\mathrm{{P}}_{parameter_number}$)",
        labelpad=8,
        fontsize=20,
    )

    if parameter == "P1":
        positions = {
            "CPU-BFV": (2.65, 10500.0),
            "CPU-CKKS": (3.70, 125.0),
            "FPGA-BFV": (2.55, 18.0),
            "FPGA-CKKS": (3.75, 0.36),
        }
    else:
        positions = {
            "CPU-BFV": (2.65, 100000.0),
            "CPU-CKKS": (3.72, 900.0),
            "FPGA-BFV": (2.55, 95.0),
            "FPGA-CKKS": (3.72, 3.0),
        }

    add_callout(ax, "CPU (BFV)", CPU_REFERENCE[bfv_key], positions["CPU-BFV"])
    add_callout(ax, "CPU (CKKS)", CPU_REFERENCE[ckks_key], positions["CPU-CKKS"])
    add_callout(ax, "FPGA (BFV)", FPGA_LEGACY[bfv_key], positions["FPGA-BFV"])
    add_callout(ax, "FPGA (CKKS)", FPGA_LEGACY[ckks_key], positions["FPGA-CKKS"])


def validate_data() -> None:
    for key in KKLSS_CPU_LEGACY:
        datasets = (
            KKLSS_CPU_LEGACY[key],
            CPU_REFERENCE[key],
            FPGA_LEGACY[key],
        )
        for values in datasets:
            if values.shape != PARTIES.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{key}: invalid data shape or non-finite value")
            if np.any(values <= 0):
                raise ValueError(f"{key}: log-scale data must be positive")
        if not np.array_equal(CPU_REFERENCE[key], KKLSS_CPU_LEGACY[key]):
            raise AssertionError(f"{key}: reference CPU values changed")
    if FPGA_LEGACY["BFV-P1"][3] != 2.38 or FPGA_LEGACY["CKKS-P1"][3] != 1.74:
        raise AssertionError("P1 n=16 FPGA anchors changed")


def write_source_data() -> None:
    path = OUTPUT_DIR / "kklss_reference_figure_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["construction", "scheme", "parameter", "parties", "platform", "latency_ms", "evidence_status"]
        )
        for scheme in ("CKKS", "BFV"):
            for parameter in ("P1", "P2"):
                key = f"{scheme}-{parameter}"
                for parties, value in zip(PARTIES, CPU_REFERENCE[key]):
                    writer.writerow(
                        ["KKLSS", scheme, parameter, int(parties), "CPU", f"{value:.6f}", "transcribed from author-provided reference figure"]
                    )
                for parties, value in zip(PARTIES, FPGA_LEGACY[key]):
                    status = (
                        "author-supplied replacement values; not a new measurement"
                        if key == "BFV-P1"
                        else "transcribed from author-provided reference figure"
                    )
                    writer.writerow(
                        ["KKLSS", scheme, parameter, int(parties), "FPGA", f"{value:.6f}", status]
                    )


def main() -> None:
    validate_data()
    write_source_data()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.25), sharey=False)
    draw_panel(axes[0], "P1")
    draw_panel(axes[1], "P2")
    axes[0].set_ylabel("Time (ms)", labelpad=7)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=True,
        fancybox=True,
        edgecolor="#b5b5b5",
        columnspacing=2.3,
        handlelength=3.1,
        handletextpad=0.75,
        borderpad=0.55,
    )
    legend.get_frame().set_linewidth(0.65)
    legend.get_frame().set_facecolor("white")
    fig.text(
        0.5,
        0.012,
        "Author-provided figure data with revised BFV-P1 FPGA values; all times in ms.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontstyle="italic",
    )
    fig.subplots_adjust(left=0.073, right=0.995, top=0.982, bottom=0.265, wspace=0.105)

    stem = OUTPUT_DIR / "kklss_multi+relinear_CPU-FPGA"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
