"""Reproduce the original two-panel MKF latency figure with revised data.

The visual grammar intentionally follows the manuscript's legacy figure:
two log-scale panels, point labels, four arrow callouts, and one bottom
legend.  Only the active data have changed.

Evidence status
---------------
CPU curves:
    Calibrated from the active scaled FPGA baseline using the
    manuscript-specified FPGA/CPU speedups: CKKS-P1 848x, BFV-P1 828x,
    CKKS-P2 837x, and BFV-P2 823x.  These values are derived, not measured.
FPGA curves:
    Author-requested 85% scaling of the confirmed 70%-normalized FPGA
    baseline supplied for this revision, with the author's 2026-10-28
    correction of BFV-P1 at n=16 to 1.290 ms. Existing derived CPU values
    remain fixed to the pre-correction baseline. No new measurements.
"""

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
# LEGACY SOURCE DATA — retained for provenance; no longer plotted.
# ---------------------------------------------------------------------------
# mult_mkf_ckks_p1 = [0.136, 0.233, 0.421, 0.794, 1.54, 3.04]
# mult_mkf_ckks_p2 = [1.12, 1.87, 3.36, 6.36, 12.36, 24.34]
# mult_mkf_bfv_p1 = [0.223, 0.371, 0.669, 1.26, 2.46, 4.83]
# mult_mkf_bfv_p2 = [2.01, 3.36, 5.62, 10.18, 19.26, 37.42]
#
# The later author-provided list changed only the BFV-P2 n=2 legacy point
# from 2.01 to 2.22.  The active projection below follows that later list.
#
# cpu_ckks_p1 = [110, 199, 365, 708, 1298, 2585]
# cpu_ckks_p2 = [743, 1526, 2825, 5554, 11024, 22279]
# cpu_bfv_p1 = [208.5, 366.00, 670.5, 1141.5, 2085.0, 3898.9]
# cpu_bfv_p2 = [1298.2, 2691.0, 4677.7, 9124.5, 17486.2, 33053.5]

# ORIGINAL ISOLATED RELINEARIZATION DATA -- retained verbatim for provenance.
# relin_mkf_ckks_p1 = [0.108, 0.179, 0.323, 0.609, 1.18, 2.35]
# relin_mkf_ckks_p2 = [0.898, 1.49, 2.68, 5.06, 9.84, 19.06]
# relin_mkf_bfv_p1 = [0.161, 0.269, 0.484, 0.914, 1.77, 3.49]
# relin_mkf_bfv_p2 = [1.57, 2.29, 4.26, 8.24, 15.99, 29.97]

# PREVIOUS FPGA OPERATOR-SUM DATA -- retained for provenance; no longer plotted.
# FPGA_EVALUATED = {
#     "CKKS-P1": np.array([0.244, 0.412, 0.744, 1.403, 2.72, 5.39]),
#     "CKKS-P2": np.array([2.018, 3.36, 6.04, 11.42, 22.20, 43.40]),
#     "BFV-P1": np.array([0.384, 0.640, 1.153, 2.174, 4.23, 8.32]),
#     "BFV-P2": np.array([3.79, 5.65, 9.88, 18.42, 35.25, 67.39]),
# }

# AUTHOR-CONFIRMED FPGA BASELINE -- retained for provenance before 85% scaling.
# FPGA_BASELINE = {
#     "CKKS-P1": np.array([0.1708, 0.2884, 0.5208, 0.9821, 1.904, 3.773]),
#     "CKKS-P2": np.array([1.4126, 2.352, 4.228, 7.994, 15.54, 30.38]),
#     "BFV-P1": np.array([0.2688, 0.448, 0.8071, 1.5218, 2.961, 5.824]),
#     "BFV-P2": np.array([2.653, 3.955, 6.916, 12.894, 24.675, 47.173]),
# }

# SUPERSEDED LOW-LATENCY BASELINE INTERPRETATION -- retained for provenance.
# FPGA_BASELINE = {
#     "CKKS-P1": np.array([0.0488, 0.0824, 0.1488, 0.2806, 0.544, 1.078]),
#     "CKKS-P2": np.array([0.4036, 0.672, 1.208, 2.284, 4.44, 8.68]),
#     "BFV-P1": np.array([0.0768, 0.128, 0.2306, 0.4348, 0.846, 1.664]),
#     "BFV-P2": np.array([0.758, 1.13, 1.976, 3.684, 7.05, 13.478]),
# }

# ACTIVE FPGA DATA: 85% of the author-confirmed baseline (milliseconds).
FPGA_EVALUATED = {
    "CKKS-P1": np.array([0.14518, 0.24514, 0.44268, 0.834785, 1.6184, 3.20705]),
    "CKKS-P2": np.array([1.20071, 1.9992, 3.5938, 6.7949, 13.209, 25.823]),
    "BFV-P1": np.array([0.22848, 0.3808, 0.686035, 1.29353, 2.51685, 4.9504]),
    "BFV-P2": np.array([2.25505, 3.36175, 5.8786, 10.9599, 20.97375, 40.09705]),
}

# ---------------------------------------------------------------------------
# PREVIOUS CPU DATA -- measured means in milliseconds; retained for provenance
# and intentionally not plotted after speedup calibration.
# ---------------------------------------------------------------------------
# CPU_MEASURED = {
#     "CKKS-P1": np.array(
#         [67.609175, 120.343600, 220.864088, 427.987403, 837.712935, 1657.420458]
#     ),
#     "CKKS-P2": np.array(
#         [500.352623, 878.607651, 1639.422614, 3159.823539, 6220.808010, 12307.562016]
#     ),
#     "BFV-P1": np.array(
#         [95.976758, 147.803380, 253.384452, 457.026845, 873.617227, 1700.749769]
#     ),
#     "BFV-P2": np.array(
#         [677.159665, 1055.495843, 1835.195392, 3361.442927, 6426.922406, 12661.828298]
#     ),
# }
#
# CPU_SAMPLE_SD = {
#     "CKKS-P1": np.array(
#         [1.516946, 3.107455, 0.759463, 9.503471, 10.951074, 14.937417]
#     ),
#     "CKKS-P2": np.array(
#         [2.837658, 3.100259, 15.529361, 19.400741, 30.513733, 79.370592]
#     ),
#     "BFV-P1": np.array(
#         [0.629291, 1.004514, 6.102013, 1.466075, 12.847963, 9.551374]
#     ),
#     "BFV-P2": np.array(
#         [13.689056, 0.964387, 22.935381, 13.961664, 22.232985, 84.264330]
#     ),
# }

# ACTIVE CPU DATA: pointwise calibration against the unchanged FPGA baseline.
# CPU latency = FPGA latency x target speedup (milliseconds).
TARGET_SPEEDUP = {
    "CKKS-P1": 848.0,
    "CKKS-P2": 837.0,
    "BFV-P1": 828.0,
    "BFV-P2": 823.0,
}
CPU_CALIBRATED = {
    key: FPGA_EVALUATED[key] * speedup
    for key, speedup in TARGET_SPEEDUP.items()
}

# Plot-only author correction. Keep the historical baseline above unchanged
# so the existing CPU curves do not change when this FPGA point is corrected.
FPGA_PLOTTED = {key: values.copy() for key, values in FPGA_EVALUATED.items()}
FPGA_PLOTTED["BFV-P1"][3] = 1.290

# ---------------------------------------------------------------------------
STYLE = {
    "FPGA-CKKS": {
        "color": "#d32612",
        "marker": "o",
        "linestyle": "-",
        "label": "FPGA (CKKS)",
    },
    "FPGA-BFV": {
        "color": "#bca66a",
        "marker": "s",
        "linestyle": "-",
        "label": "FPGA (BFV)",
    },
    "CPU-CKKS": {
        "color": "#51add3",
        "marker": "^",
        "linestyle": "--",
        "label": "CPU (CKKS)",
    },
    "CPU-BFV": {
        "color": "#36a914",
        "marker": "d",
        "linestyle": "--",
        "label": "CPU (BFV)",
    },
}

mpl.rcParams.update(
    {
        # Times preserves the original manuscript style; Arial/sans-serif are
        # explicit publication-safe fallbacks if Times is unavailable.
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


def display_value(value: float) -> str:
    """Compact labels matching the legacy figure's readable precision."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value >= 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def geometric_midpoint(values: np.ndarray, left_index: int = 2) -> float:
    """Midpoint that is visually centered on a logarithmic y-axis."""
    return float(np.sqrt(values[left_index] * values[left_index + 1]))


def add_point_labels(
    ax,
    values: np.ndarray,
    position: str,
    clear_fpga_ckks_arrow: bool = False,
    fixed_three_decimals: bool = False,
) -> None:
    y_offset = 8 if position == "above" else -10
    va = "bottom" if position == "above" else "top"

    for index, (x_value, y_value) in enumerate(zip(X, values)):
        x_offset = 3 if index == 0 else (-3 if index == len(X) - 1 else 0)
        ha = "left" if index == 0 else ("right" if index == len(X) - 1 else "center")
        point_y_offset = y_offset
        if clear_fpga_ckks_arrow and index == 3:
            # Keep the n=16 value next to its marker, shifted slightly right.
            # The FPGA-CKKS callout arrow targets x=2.5, so left-shifting this
            # label would place the value on the arrow rather than the point.
            x_offset = 7
            point_y_offset = -10
            ha = "left"
        label_text = (
            f"{float(y_value):.3f}"
            if fixed_three_decimals
            else display_value(float(y_value))
        )
        ax.annotate(
            label_text,
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


def add_callout(
    ax,
    label: str,
    values: np.ndarray,
    box_xy: tuple[float, float],
    target_x: float = 2.5,
) -> None:
    ax.annotate(
        label,
        xy=(target_x, geometric_midpoint(values)),
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
        ("FPGA-CKKS", FPGA_PLOTTED[ckks_key], "below"),
        ("FPGA-BFV", FPGA_PLOTTED[bfv_key], "above"),
        ("CPU-CKKS", CPU_CALIBRATED[ckks_key], "below"),
        ("CPU-BFV", CPU_CALIBRATED[bfv_key], "above"),
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
            fixed_three_decimals=style_key.startswith("FPGA-"),
        )

    all_values = np.concatenate(
        (
            FPGA_PLOTTED[ckks_key],
            FPGA_PLOTTED[bfv_key],
            CPU_CALIBRATED[ckks_key],
            CPU_CALIBRATED[bfv_key],
        )
    )
    ax.set_yscale("log")
    # Match the legacy figure's generous log-scale headroom so the callout
    # boxes and endpoint labels remain fully inside the plotting frame.
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
        callout_positions = {
            "CPU-BFV": (2.65, 6000.0),
            "CPU-CKKS": (3.70, 135.0),
            "FPGA-BFV": (2.60, 10.0),
            "FPGA-CKKS": (3.75, 0.17),
        }
    else:
        callout_positions = {
            "CPU-BFV": (2.65, 70000.0),
            "CPU-CKKS": (3.72, 920.0),
            "FPGA-BFV": (2.55, 90.0),
            "FPGA-CKKS": (3.72, 1.38),
        }

    add_callout(ax, "CPU (BFV)", CPU_CALIBRATED[bfv_key], callout_positions["CPU-BFV"])
    add_callout(ax, "CPU (CKKS)", CPU_CALIBRATED[ckks_key], callout_positions["CPU-CKKS"])
    add_callout(ax, "FPGA (BFV)", FPGA_PLOTTED[bfv_key], callout_positions["FPGA-BFV"])
    add_callout(ax, "FPGA (CKKS)", FPGA_PLOTTED[ckks_key], callout_positions["FPGA-CKKS"])


def validate_data() -> None:
    datasets = {
        **CPU_CALIBRATED,
        **{f"FPGA-{key}": values for key, values in FPGA_EVALUATED.items()},
        **{f"Plotted-FPGA-{key}": values for key, values in FPGA_PLOTTED.items()},
    }
    for name, values in datasets.items():
        if values.shape != PARTIES.shape:
            raise ValueError(f"{name}: expected {PARTIES.size} points, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: all points must be finite")
        if np.any(values <= 0):
            raise ValueError(f"{name}: log-scale data must be strictly positive")

    for key, target in TARGET_SPEEDUP.items():
        ratios = CPU_CALIBRATED[key] / FPGA_EVALUATED[key]
        if not np.allclose(ratios, target, rtol=0.0, atol=1e-12):
            raise ValueError(f"{key}: calibrated speedup does not equal {target:g}x")

    changes = [
        (key, index)
        for key in FPGA_EVALUATED
        for index in range(PARTIES.size)
        if FPGA_EVALUATED[key][index] != FPGA_PLOTTED[key][index]
    ]
    if changes != [("BFV-P1", 3)] or FPGA_PLOTTED["BFV-P1"][3] != 1.290:
        raise ValueError("Expected only the BFV-P1 n=16 author correction to 1.290 ms")

def write_source_data() -> None:
    path = OUTPUT_DIR / "mkf_revised_data_20260826.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["construction", "scheme", "parameter", "parties", "platform", "latency_ms", "evidence_status"]
        )
        for scheme in ("CKKS", "BFV"):
            for parameter in ("P1", "P2"):
                key = f"{scheme}-{parameter}"
                for parties, value in zip(PARTIES, CPU_CALIBRATED[key]):
                    writer.writerow(
                        [
                            "MKF",
                            scheme,
                            parameter,
                            int(parties),
                            "CPU",
                            f"{value:.6f}",
                            f"calibrated from FPGA baseline to {TARGET_SPEEDUP[key]:.0f}x; not measured",
                        ]
                    )
                for parties, value in zip(PARTIES, FPGA_PLOTTED[key]):
                    provenance = "author-requested 85% scaling of confirmed 70%-normalized FPGA baseline"
                    if key == "BFV-P1" and int(parties) == 16:
                        provenance += "; author correction 2026-10-28 to 1.290 ms; not a new board measurement"
                    writer.writerow(
                        [
                            "MKF",
                            scheme,
                            parameter,
                            int(parties),
                            "FPGA",
                            f"{value:.6f}",
                            provenance,
                        ]
                    )


def main() -> None:
    validate_data()
    write_source_data()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.25), sharey=False)
    draw_panel(axes[0], "P1")
    draw_panel(axes[1], "P2")
    axes[0].set_ylabel("Time (ms)", labelpad=7)

    handles, labels = axes[0].get_legend_handles_labels()
    figure_legend = fig.legend(
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
    figure_legend.get_frame().set_linewidth(0.65)
    figure_legend.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.073, right=0.995, top=0.982, bottom=0.24, wspace=0.105)

    stem = OUTPUT_DIR / "mkf_multi+relinear_CPU-FPGA"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
