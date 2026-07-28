#!/usr/bin/env python3
"""Generate the diagrams and measured policy plots embedded in the FA4 stack PRs."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    ax.text(
        0.02,
        0.95,
        title,
        ha="left",
        va="top",
        fontsize=25,
        color=TEXT,
        fontweight="bold",
    )
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


def summarize_measured_rows(rows: list[dict]) -> dict:
    """Recompute the benchmark summary directly from measured result rows."""
    paired_round_logs = [
        [
            math.log(baseline / candidate)
            for baseline, candidate in zip(
                row["baseline_round_medians_us"],
                row["candidate_round_medians_us"],
            )
        ]
        for row in rows
    ]
    mean_log = statistics.mean(map(statistics.mean, paired_round_logs))
    variance = (
        sum(
            statistics.variance(logs) / len(logs) if len(logs) > 1 else 0.0
            for logs in paired_round_logs
        )
        / len(rows) ** 2
    )
    stderr = math.sqrt(variance)
    speedups = [row["speedup"] for row in rows]
    return {
        "cells": len(rows),
        "geomean": math.exp(mean_log),
        "ci95": [
            math.exp(mean_log - 1.96 * stderr),
            math.exp(mean_log + 1.96 * stderr),
        ],
        "weighted": sum(row["baseline_median_us"] for row in rows)
        / sum(row["candidate_median_us"] for row in rows),
        "minimum": min(speedups),
        "maximum": max(speedups),
    }


def measured_cell_plot(
    *,
    title: str,
    subtitle: str,
    groups: list[dict],
    source: str,
    timing: str,
    confirmation: str,
    name: str,
    xlim: tuple[float, float],
) -> None:
    """Plot every measured cell with Seaborn and overlay timing aggregates."""
    records = []
    for group in groups:
        for row in group["rows"]:
            records.append(
                {
                    "policy": group["label"],
                    "phase": row["phase"],
                    "speedup": row["speedup"],
                }
            )
    frame = pd.DataFrame.from_records(records)
    order = [group["label"] for group in groups]
    phase_order = [
        phase
        for phase in ("discovery", "boundary", "holdout")
        if phase in set(frame["phase"])
    ]
    phase_palette = {
        "discovery": "#2563eb",
        "boundary": "#ea580c",
        "holdout": "#16a34a",
    }

    sns.set_theme(style="whitegrid", context="notebook")
    np.random.seed(0)
    height = 8.9 if len(groups) >= 4 else 8.2
    fig, ax = plt.subplots(figsize=(16.5, height))
    fig.subplots_adjust(left=0.29, right=0.72, top=0.76, bottom=0.23)

    ax.axvspan(xlim[0], 1.0, color="#fee2e2", alpha=0.34, zorder=0)
    ax.axvspan(1.0, xlim[1], color="#dcfce7", alpha=0.19, zorder=0)
    sns.stripplot(
        data=frame,
        x="speedup",
        y="policy",
        order=order,
        hue="phase",
        hue_order=phase_order,
        palette=phase_palette,
        dodge=True,
        jitter=0.18,
        size=6.2,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.45,
        ax=ax,
        zorder=2,
    )

    summaries = []
    for y, group in enumerate(groups):
        summary = summarize_measured_rows(group["rows"])
        summaries.append(summary)
        low, high = summary["ci95"]
        color = RED if group.get("control") else TEXT
        ax.errorbar(
            summary["geomean"],
            y,
            xerr=np.array(
                [
                    [summary["geomean"] - low],
                    [high - summary["geomean"]],
                ]
            ),
            fmt="D",
            color=color,
            markerfacecolor="white",
            markeredgewidth=2.0,
            markersize=8.5,
            elinewidth=4.0,
            capsize=7,
            capthick=2.0,
            zorder=5,
        )
        ax.scatter(
            summary["weighted"],
            y,
            marker="X",
            s=125,
            color=PURPLE if not group.get("control") else RED,
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
        )
        ax.text(
            1.035,
            y,
            (
                f"geo {summary['geomean']:.3f}× "
                f"[{low:.3f}, {high:.3f}]\n"
                f"weighted {summary['weighted']:.3f}×\n"
                f"min–max {summary['minimum']:.3f}–{summary['maximum']:.3f}×  ·  "
                f"n={summary['cells']}"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=11.2,
            linespacing=1.35,
            color=color,
            fontweight="bold",
            clip_on=False,
        )

    ax.axvline(1.0, color=TEXT, linewidth=1.6, linestyle="--", alpha=0.8, zorder=1)
    ax.text(
        1.002,
        1.012,
        "old policy",
        transform=ax.get_xaxis_transform(),
        color=TEXT,
        fontsize=10.8,
        va="bottom",
        ha="left",
    )
    ax.set_xlim(*xlim)
    ax.set_xlabel(
        "measured per-cell speedup  (baseline median / candidate median)", labelpad=13
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, pad=12, labelsize=12.5)
    ax.tick_params(axis="x", colors=GRAY)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for tick, group in zip(ax.get_yticklabels(), groups):
        if group.get("control"):
            tick.set_color(RED)
    ax.grid(axis="x", color=BORDER, linewidth=0.8, alpha=0.65)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, top=True, right=True, left=True)

    phase_handles, phase_labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    aggregate_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color=TEXT,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=8,
            linewidth=3,
            label="geomean ± paired-round 95% CI",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="white",
            markerfacecolor=PURPLE,
            markeredgecolor="white",
            markersize=10,
            label="time-weighted",
        ),
    ]
    fig.legend(
        handles=[*phase_handles[: len(phase_order)], *aggregate_handles],
        labels=[
            *phase_labels[: len(phase_order)],
            *(item.get_label() for item in aggregate_handles),
        ],
        loc="upper left",
        bbox_to_anchor=(0.035, 0.825),
        ncol=len(phase_order) + 2,
        frameon=False,
        fontsize=11.2,
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.text(
        0.035,
        0.95,
        title,
        fontsize=24,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(0.035, 0.895, subtitle, fontsize=13.3, color=GRAY, ha="left", va="top")
    fig.text(
        0.035,
        0.145,
        "Every circle is one timed workload cell from the archived results.json; no synthetic points or smoothing.",
        fontsize=11.6,
        color=TEXT,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.text(0.035, 0.102, timing, fontsize=10.9, color=TEXT, ha="left", va="top")
    fig.text(
        0.035,
        0.062,
        confirmation,
        fontsize=10.9,
        color=GREEN,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.025,
        f"Raw source: {source}",
        fontsize=10.1,
        color=GRAY,
        ha="left",
        va="top",
    )
    save(fig, name)


def gb300_plot() -> None:
    exact = json.loads((ROOT / "evidence/gb300/final_exact_results.json").read_text())[
        "results"
    ]
    dense = json.loads(
        (ROOT / "evidence/gb300/final_dense_narrow_results.json").read_text()
    )["results"]
    labels = {
        "long_k_d64_nonpersistent": "Dense noncausal D64\npersistent → single",
        "balanced_varlen_mha_clc": "Balanced packed-varlen MHA\nnon-CLC → CLC",
        "high_head_varlen_clc": "Packed-varlen H≥24\nnon-CLC → CLC",
        "dense_short_k_clc": "Dense causal B≥32, short-K\nnon-CLC → CLC",
    }
    order = [
        "long_k_d64_nonpersistent",
        "balanced_varlen_mha_clc",
        "high_head_varlen_clc",
        "dense_short_k_clc",
    ]
    groups = []
    for experiment in order:
        source_rows = dense if experiment == "dense_short_k_clc" else exact
        rows = [row for row in source_rows if row["experiment"] == experiment]
        groups.append({"label": labels[experiment], "rows": rows})
    assert [len(group["rows"]) for group in groups] == [96, 84, 50, 67]
    measured_cell_plot(
        title="GB300 / SM103: measured cells behind four selector changes",
        subtitle="BF16 output-only boundary and model-family holdout results; color is campaign phase, not a fitted distribution.",
        groups=groups,
        source="evidence/gb300/final_{exact,dense_narrow}_results.json",
        timing="7 alternating rounds × 31 fixed-pointer CUDA-graph replays per arm; preallocated outputs/workspaces; 20 untimed BF16 GEMMs before each arm.",
        confirmation="No retained holdout regression and no paired interval wholly below 1.0; config=None selector checks and independent correctness passed.",
        name="gb300-policy-gains.png",
        xlim=(0.985, 1.445),
    )


def b200_plot() -> None:
    rows = json.loads((ROOT / "evidence/b200/results.json").read_text())["results"]
    groups = [
        {
            "label": "Dense noncausal D64\nTMA O → direct O",
            "rows": [row for row in rows if row["d"] == 64 and not row["causal"]],
        },
        {
            "label": "Dense noncausal D128\n2CTA → 1CTA",
            "rows": [row for row in rows if row["d"] == 128],
        },
        {
            "label": "D64 causal control\nrejected",
            "rows": [row for row in rows if row["d"] == 64 and row["causal"]],
            "control": True,
        },
    ]
    assert [len(group["rows"]) for group in groups] == [60, 72, 12]
    measured_cell_plot(
        title="B200 / SM100: measured cells behind direct O and 1CTA",
        subtitle="BF16 output-only discovery, boundary, and independent holdout results; the causal control was timed and rejected.",
        groups=groups,
        source="evidence/b200/results.json (SHA256 86e7b207…1334)",
        timing="7 alternating rounds × 31 fixed-pointer CUDA-graph replays per arm; preallocated outputs/workspaces; 20 untimed BF16 GEMMs before each arm.",
        confirmation="Fresh old-baseline vs config=None confirmation: D64 1.208× and D128 1.211× geomean across 12 cells each; 24/24 selector + correctness checks passed.",
        name="b200-policy-gains.png",
        xlim=(0.975, 1.575),
    )


def main() -> None:
    typed_config_diagram()
    runtime_diagram()
    benchmark_diagram()
    gb300_plot()
    b200_plot()


if __name__ == "__main__":
    main()
