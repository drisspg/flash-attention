#!/usr/bin/env python3
"""Generate the diagrams and measured policy plots embedded in the FA4 stack PRs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent

NAVY = "#14213d"
BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#16a34a"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
RED = "#dc2626"
GRAY = "#64748b"
LIGHT = "#f8fafc"
BORDER = "#cbd5e1"
TEXT = "#0f172a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titleweight": "bold",
        "axes.titlesize": 22,
        "axes.labelsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, title, detail, *, color=BLUE, fontsize=14):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=2,
        edgecolor=color,
        facecolor=LIGHT,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=fontsize,
        fontweight="bold",
        family="DejaVu Sans Mono" if "(" in title or "Config" in title else None,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=fontsize - 2,
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, text="", text_xy=None, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    if text:
        x, y = text_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, text, ha="center", va="center", fontsize=11, color=color)


def setup_diagram(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, title, ha="left", va="top", fontsize=25, color=TEXT, fontweight="bold")
    ax.text(0.02, 0.895, subtitle, ha="left", va="top", fontsize=13, color=GRAY)
    return fig, ax


def typed_config_diagram() -> None:
    fig, ax = setup_diagram(
        "One typed boundary for FA4 forward launch choices",
        "Policy inputs, resolved launch configuration, and code-generation keys stay separate.",
    )
    box(
        ax,
        (0.03, 0.49),
        0.20,
        0.22,
        "FwdHeuristicInputs",
        "host-visible metadata\nshape values are okay here",
        color=CYAN,
    )
    box(
        ax,
        (0.30, 0.49),
        0.20,
        0.22,
        "default_fwd_config(...) ",
        "cheap pure selector\nconfig=None only",
        color=PURPLE,
        fontsize=13,
    )
    box(
        ax,
        (0.57, 0.49),
        0.18,
        0.22,
        "FwdConfig",
        "immutable and resolved\nexplicit means exact",
        color=BLUE,
    )
    box(
        ax,
        (0.81, 0.63),
        0.16,
        0.17,
        "FwdMainKernelSpec",
        "main codegen only",
        color=GREEN,
        fontsize=12,
    )
    box(
        ax,
        (0.81, 0.36),
        0.16,
        0.17,
        "FwdCombineKernelSpec",
        "combine codegen only",
        color=ORANGE,
        fontsize=11,
    )
    arrow(ax, (0.23, 0.60), (0.30, 0.60), color=CYAN)
    arrow(ax, (0.50, 0.60), (0.57, 0.60), color=PURPLE)
    arrow(ax, (0.75, 0.62), (0.81, 0.71), color=GREEN)
    arrow(ax, (0.75, 0.57), (0.81, 0.45), color=ORANGE)

    box(
        ax,
        (0.31, 0.18),
        0.18,
        0.15,
        "explicit FwdConfig",
        "benchmark / caller",
        color=BLUE,
        fontsize=13,
    )
    arrow(
        ax,
        (0.49, 0.255),
        (0.62, 0.49),
        color=BLUE,
        text="validate, never rewrite",
        text_xy=(0.58, 0.29),
        connectionstyle="arc3,rad=-0.12",
    )

    ax.text(
        0.03,
        0.12,
        "Runtime shapes choose a config; only the typed specs key compilation.\nSame spec = cache reuse. Crossing a codegen boundary = one new specialization.",
        ha="left",
        va="top",
        fontsize=14,
        color=TEXT,
        fontweight="bold",
        linespacing=1.45,
    )
    save(fig, "typed-config-boundary.png")


def runtime_diagram() -> None:
    fig, ax = setup_diagram(
        "_flash_attn_fwd is the single selection and execution boundary",
        "Public APIs stay unchanged. Direct callers can force an exact config; normal calls use config=None.",
    )
    box(
        ax,
        (0.03, 0.50),
        0.18,
        0.20,
        "_flash_attn_fwd(...) ",
        "config=None or\nexplicit FwdConfig",
        color=CYAN,
        fontsize=13,
    )
    box(
        ax,
        (0.27, 0.50),
        0.17,
        0.20,
        "select + validate",
        "one path\nexact semantics",
        color=PURPLE,
        fontsize=13,
    )
    box(
        ax,
        (0.50, 0.50),
        0.17,
        0.20,
        "main projection",
        "compile-cache key",
        color=GREEN,
        fontsize=13,
    )
    box(
        ax,
        (0.73, 0.63),
        0.23,
        0.16,
        "main attention kernel",
        "always",
        color=GREEN,
        fontsize=13,
    )
    box(
        ax,
        (0.73, 0.34),
        0.23,
        0.16,
        "SplitKV combine kernel",
        "only when num_splits > 1",
        color=ORANGE,
        fontsize=13,
    )
    arrow(ax, (0.21, 0.60), (0.27, 0.60), color=CYAN)
    arrow(ax, (0.44, 0.60), (0.50, 0.60), color=PURPLE)
    arrow(ax, (0.67, 0.61), (0.73, 0.70), color=GREEN)
    arrow(ax, (0.61, 0.50), (0.73, 0.42), color=ORANGE, connectionstyle="arc3,rad=-0.1")

    box(
        ax,
        (0.27, 0.17),
        0.24,
        0.15,
        "out_partial + lse_partial",
        "optional reusable FP32 workspaces",
        color=ORANGE,
        fontsize=12,
    )
    arrow(
        ax,
        (0.51, 0.245),
        (0.73, 0.39),
        color=ORANGE,
        text="allocation-free replay",
        text_xy=(0.64, 0.23),
        connectionstyle="arc3,rad=-0.12",
    )

    ax.text(
        0.03,
        0.11,
        "Main and combine kernels have independent typed cache keys. Split counts can reuse the main specialization\nwhile compiling only the combine variants they actually need.",
        ha="left",
        va="top",
        fontsize=14,
        color=TEXT,
        fontweight="bold",
        linespacing=1.4,
    )
    save(fig, "forward-runtime-flow.png")


def benchmark_diagram() -> None:
    fig, ax = setup_diagram(
        "A new tuning campaign is a YAML grid, not a new benchmark harness",
        "One runner owns compilation, correctness, timing, checkpointing, and review-ready summaries.",
    )
    xs = [0.025, 0.22, 0.415, 0.61, 0.805]
    titles = [
        "campaign YAML",
        "compile barrier",
        "correctness gate",
        "paired timing",
        "results.json",
    ]
    details = [
        "shape grid +\nbaseline/candidate",
        "isolated fake-tensor\nsubprocesses",
        "independent FP32\nreference",
        "7 rounds × 31\nCUDA-graph replays",
        "raw samples +\npolicy summary",
    ]
    colors = [CYAN, PURPLE, GREEN, ORANGE, BLUE]
    for x, title, detail, color in zip(xs, titles, details, colors):
        box(ax, (x, 0.49), 0.16, 0.22, title, detail, color=color, fontsize=13)
    for left, right, color in zip(xs[:-1], xs[1:], colors[1:]):
        arrow(ax, (left + 0.16, 0.60), (right, 0.60), color=color)

    ax.text(
        0.04,
        0.32,
        "Timing contract",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
    )
    contract = [
        (0.04, 0.265, "fixed pointers"),
        (0.48, 0.265, "preallocated outputs/workspaces"),
        (0.04, 0.215, "cyclic arm order"),
        (0.48, 0.215, "20 untimed BF16 GEMMs before each arm"),
    ]
    for x, y, item in contract:
        ax.text(x, y, f"✓  {item}", fontsize=12.5, color=TEXT, ha="left")

    ax.text(
        0.04,
        0.115,
        "Output: n, geomean + paired-round 95% CI, time-weighted speedup, min–max, and W/N/L.\nCandidate generation stays out of production code.",
        ha="left",
        va="top",
        fontsize=14,
        color=TEXT,
        fontweight="bold",
        linespacing=1.45,
    )
    save(fig, "benchmark-contract.png")


def policy_plot(
    *,
    title: str,
    subtitle: str,
    rows: list[dict],
    source: str,
    timing: str,
    name: str,
    xlim: tuple[float, float],
    confirmation: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.31, right=0.96, top=0.78, bottom=0.24)
    colors = [BLUE, PURPLE, GREEN, ORANGE, CYAN, RED]
    positions = list(range(len(rows) - 1, -1, -1))

    ax.axvline(1.0, color=TEXT, linewidth=1.5, linestyle="--", alpha=0.7, zorder=0)
    ax.text(1.002, len(rows) - 0.45, "old policy", color=TEXT, fontsize=11, va="top")

    for y, row, color in zip(positions, rows, colors):
        minimum = row["minimum"]
        maximum = row["maximum"]
        ci_low, ci_high = row["ci95"]
        geomean = row["geomean"]
        weighted = row["weighted"]
        is_control = row.get("control", False)
        if is_control:
            color = RED
        ax.hlines(y, minimum, maximum, color=color, alpha=0.32, linewidth=4, zorder=1)
        ax.plot([minimum, maximum], [y, y], "|", color=color, markersize=13, markeredgewidth=2)
        ax.hlines(y, ci_low, ci_high, color=color, linewidth=12, alpha=0.92, zorder=3)
        ax.scatter(geomean, y, s=150, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.scatter(weighted, y, s=105, color=color, marker="D", edgecolor="white", linewidth=1.2, zorder=4)
        label_x = min(maximum + (xlim[1] - xlim[0]) * 0.012, xlim[1] - 0.005)
        align = "left" if label_x < xlim[1] - 0.02 else "right"
        ax.text(
            label_x,
            y + 0.11,
            f"max {maximum:.3f}×",
            color=color,
            fontsize=10.5,
            va="bottom",
            ha=align,
            fontweight="bold",
        )
        ax.text(
            geomean,
            y - 0.19,
            f"geo {geomean:.3f}×  •  weighted {weighted:.3f}×  •  n={row['cells']}",
            color=TEXT,
            fontsize=11.5,
            va="top",
            ha="center",
            fontweight="bold",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=13, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65, len(rows) - 0.25)
    ax.set_xlabel("speedup (old config latency / new config latency)", labelpad=12)
    ax.grid(axis="x", color=BORDER, linewidth=0.8, alpha=0.6)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", colors=GRAY)

    fig.text(0.035, 0.94, title, fontsize=25, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.035, 0.885, subtitle, fontsize=13.5, color=GRAY, ha="left", va="top")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=11, label="geomean"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=BLUE, markersize=9, label="time-weighted"),
        Line2D([0], [0], color=BLUE, linewidth=9, label="paired-round 95% CI"),
        Line2D([0], [0], color=BLUE, alpha=0.35, linewidth=4, marker="|", markersize=10, label="cell median min–max"),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.825),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        handlelength=2.4,
        columnspacing=1.6,
    )

    fig.text(0.035, 0.145, timing, fontsize=11.5, color=TEXT, ha="left", va="top", fontweight="bold")
    fig.text(0.035, 0.098, confirmation, fontsize=11.5, color=GREEN, ha="left", va="top", fontweight="bold")
    fig.text(0.035, 0.052, f"Source: {source}", fontsize=10.8, color=GRAY, ha="left", va="top")
    save(fig, name)


def gb300_plot() -> None:
    data = json.loads((ROOT / "gb300_summary.json").read_text())
    labels = {
        "long_k_d64_nonpersistent": "Dense noncausal D64\npersistent → single",
        "balanced_varlen_mha_clc": "Balanced packed-varlen MHA\nnon-CLC → CLC",
        "high_head_varlen_clc": "Packed-varlen H≥24\nnon-CLC → CLC",
        "dense_short_k_clc": "Dense causal B≥32, short-K\nnon-CLC → CLC",
    }
    rows = []
    for policy in data["policies"]:
        rows.append(
            {
                "label": labels[policy["experiment"]],
                "cells": policy["cells"],
                "geomean": policy["geomean_speedup"],
                "weighted": policy["time_weighted_speedup"],
                "minimum": policy["minimum_speedup"],
                "maximum": policy["maximum_speedup"],
                "ci95": policy["geomean_ci95"],
            }
        )
    policy_plot(
        title="GB300 / SM103: four narrow selector wins",
        subtitle="BF16 output-only policies promoted from 297 frozen boundary + model-family holdout cells.",
        rows=rows,
        source="benchmarks/configs/fwd_config_sm103.yaml → benchmarks/fwd_config_bench.py",
        timing="7 alternating rounds × 31 fixed-pointer CUDA-graph replays per arm; preallocated outputs/workspaces; 20 untimed BF16 GEMMs before each arm.",
        confirmation="No retained holdout regression and no paired interval wholly below 1.0; config=None selector checks and independent correctness passed.",
        name="gb300-policy-gains.png",
        xlim=(0.985, 1.455),
    )


def b200_plot() -> None:
    data = json.loads((ROOT / "b200_summary.json").read_text())
    summaries = data["phase_summaries"]
    control = data["excluded_controls"]["timed_excluded_strata"]["d64_causal"]["summary"]
    rows = [
        {
            "label": "Dense noncausal D64\nTMA O → direct O",
            "cells": summaries["d64_direct_output"]["all"]["cells"],
            "geomean": summaries["d64_direct_output"]["all"]["geomean_speedup"],
            "weighted": summaries["d64_direct_output"]["all"]["time_weighted_speedup"],
            "minimum": summaries["d64_direct_output"]["all"]["minimum_speedup"],
            "maximum": summaries["d64_direct_output"]["all"]["maximum_speedup"],
            "ci95": summaries["d64_direct_output"]["all"]["geomean_ci95"],
        },
        {
            "label": "Dense noncausal D128\n2CTA → 1CTA",
            "cells": summaries["d128_1cta"]["all"]["cells"],
            "geomean": summaries["d128_1cta"]["all"]["geomean_speedup"],
            "weighted": summaries["d128_1cta"]["all"]["time_weighted_speedup"],
            "minimum": summaries["d128_1cta"]["all"]["minimum_speedup"],
            "maximum": summaries["d128_1cta"]["all"]["maximum_speedup"],
            "ci95": summaries["d128_1cta"]["all"]["geomean_ci95"],
        },
        {
            "label": "D64 causal control\nrejected",
            "cells": control["cells"],
            "geomean": control["geomean_speedup"],
            "weighted": control["time_weighted_speedup"],
            "minimum": control["minimum_speedup"],
            "maximum": control["maximum_speedup"],
            "ci95": control["geomean_ci95"],
            "control": True,
        },
    ]
    policy_plot(
        title="B200 / SM100: direct output for D64, 1CTA for D128",
        subtitle="BF16 output-only policies promoted from 132 retained discovery + boundary + independent holdout cells; causal was measured and rejected.",
        rows=rows,
        source="benchmarks/configs/fwd_config_b200.yaml → benchmarks/fwd_config_bench.py",
        timing="7 alternating rounds × 31 fixed-pointer CUDA-graph replays per arm; preallocated outputs/workspaces; 20 untimed BF16 GEMMs before each arm.",
        confirmation="Fresh old-baseline vs config=None confirmation: D64 1.208× and D128 1.211× geomean across 12 cells each; 24/24 selector + correctness checks passed.",
        name="b200-policy-gains.png",
        xlim=(0.975, 1.595),
    )


def main() -> None:
    typed_config_diagram()
    runtime_diagram()
    benchmark_diagram()
    gb300_plot()
    b200_plot()


if __name__ == "__main__":
    main()
