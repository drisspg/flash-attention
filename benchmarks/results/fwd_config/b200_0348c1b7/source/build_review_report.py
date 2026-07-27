#!/usr/bin/env python3
"""Build review-ready campaign phase/control summaries from frozen results."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "agent_space/b200_fwd_config/full/results.json"
OUT = ROOT / "agent_space/b200_fwd_config/review_artifacts"


def paired(row):
    logs = [
        math.log(b / c)
        for b, c in zip(
            row["baseline_round_medians_us"], row["candidate_round_medians_us"]
        )
    ]
    mean = statistics.mean(logs)
    se = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    return math.exp(mean), math.exp(mean - 1.96 * se), math.exp(mean + 1.96 * se)


def summary(rows):
    cell_logs = []
    for row in rows:
        cell_logs.append(
            [
                math.log(b / c)
                for b, c in zip(
                    row["baseline_round_medians_us"],
                    row["candidate_round_medians_us"],
                )
            ]
        )
    mean = statistics.mean(map(statistics.mean, cell_logs))
    variance = sum(
        statistics.variance(logs) / len(logs) if len(logs) > 1 else 0.0
        for logs in cell_logs
    ) / len(rows) ** 2
    se = math.sqrt(variance)
    ratios = [row["speedup"] for row in rows]
    return {
        "cells": len(rows),
        "geomean_speedup": math.exp(mean),
        "geomean_ci95": [math.exp(mean - 1.96 * se), math.exp(mean + 1.96 * se)],
        "time_weighted_speedup": sum(row["baseline_median_us"] for row in rows)
        / sum(row["candidate_median_us"] for row in rows),
        "minimum_speedup": min(ratios),
        "maximum_speedup": max(ratios),
        "wins_neutral_losses": [
            sum(x > 1.01 for x in ratios),
            sum(0.99 <= x <= 1.01 for x in ratios),
            sum(x < 0.99 for x in ratios),
        ],
        "paired_intervals_wholly_below_one": sum(paired(row)[2] < 1 for row in rows),
    }


def compact_cell(row):
    pg, low, high = paired(row)
    return {
        "name": row["name"],
        "phase": row["phase"],
        "batch": row["batch"],
        "q_heads": row["q_heads"],
        "kv_heads": row["kv_heads"],
        "seqlen_q": row["seqlen_q"],
        "seqlen_k": row["seqlen_k"],
        "d": row["d"],
        "causal": row["causal"],
        "speedup": row["speedup"],
        "paired_round_speedup": pg,
        "paired_round_ci95": [low, high],
        "median_below_one": row["speedup"] < 1.0,
        "paired_interval_wholly_below_one": high < 1.0,
        "exclusion_reason": "causal stratum failed promotion and is outside the output-only rule",
    }


def fmt(s):
    w, n, l = s["wins_neutral_losses"]
    lo, hi = s["geomean_ci95"]
    return (
        f"{s['cells']} cells | geomean {s['geomean_speedup']:.4f}x "
        f"[{lo:.4f}, {hi:.4f}] | weighted {s['time_weighted_speedup']:.4f}x | "
        f"min {s['minimum_speedup']:.4f}x | max {s['maximum_speedup']:.4f}x | "
        f"W/N/L {w}/{n}/{l} | paired-CI-below-1 "
        f"{s['paired_intervals_wholly_below_one']}"
    )


def main():
    payload = json.loads(RESULTS.read_text())
    rows = payload["results"]
    retained = {
        "d64_direct_output": [row for row in rows if row["d"] == 64 and not row["causal"]],
        "d128_1cta": [row for row in rows if row["d"] == 128],
    }
    phase_summaries = {
        rule: {
            "all": summary(group),
            "discovery": summary([row for row in group if row["phase"] == "discovery"]),
            "boundary": summary([row for row in group if row["phase"] == "boundary"]),
            "holdout": summary([row for row in group if row["phase"] == "holdout"]),
        }
        for rule, group in retained.items()
    }
    causal = [row for row in rows if row["d"] == 64 and row["causal"]]
    causal_cells = [compact_cell(row) for row in causal]
    host_exclusions = {
        "timed_excluded_strata": {
            "d64_causal": {
                "summary": summary(causal),
                "cells": causal_cells,
                "losing_cells": [cell for cell in causal_cells if cell["median_below_one"]],
            }
        },
        "host_selector_exclusions_not_claimed_as_timed_cells": [
            "FP16 (separate dtype stratum)",
            "device_arch != 100, including SM103/GB300",
            "Q packed length <= 256",
            "max K length < 2048",
            "causal or local/windowed attention",
            "varlen paths",
            "SplitKV / num_splits != 1",
            "paged KV",
            "block-sparse attention",
            "QV/MLA",
            "gather KV",
            "score modifiers",
            "mask modifiers",
            "learnable sinks",
            "LSE-producing, preallocated-LSE, Flex/FLASH, and training/autograd paths",
            "requested tile_m or tile_n other than 128",
            "D != DV or D outside {64, 128}",
            "GQA ratio 2",
            "unpacked GQA; retained GQA ratios 4/8/16 require pack_gqa",
        ],
        "note": (
            "Only the 12 D64 causal controls were performance-timed and excluded. "
            "The remaining exclusions are conservative host-visible scope guards, "
            "covered by selector tests but not represented as measured performance cells."
        ),
    }
    rationale = {
        "independence": (
            "The holdout tables were frozen before promotion and were not used to choose "
            "the candidate arm or selector boundaries. They use unseen batch/head-count "
            "combinations and mostly odd/non-power-of-two Q/K lengths, rather than copies "
            "of the discovery or +/-1 boundary triplets."
        ),
        "distribution": (
            "The campaign is a stratified shape envelope, not a frequency-weighted claim: "
            "batch 1-7, heads 8-48 (and up to 64 for D128 controls), symmetric/asymmetric "
            "Q/K, short-prefill/wave-boundary and long-prefill contexts, and legal packed "
            "MHA/GQA/MQA ratios 1/4/8/16. Odd lengths model real request/padding residue and "
            "prevent power-of-two-only tuning."
        ),
        "model_family_proxies": (
            "D64 MHA cells proxy dense-head encoder/GPT-style families and D64 serving "
            "variants; D128 MHA and ratios 4/8/16 proxy the scheduling geometry common in "
            "Llama/Mistral/Qwen-like decoder families and MQA-style deployments. These are "
            "geometry proxies only; no production model mix or traffic frequency is inferred."
        ),
        "holdout_composition": {
            "d64": {
                "cells": 20,
                "mha_long": 12,
                "mha_short": 4,
                "packed_gqa_mqa_long": 4,
            },
            "d128": {
                "cells": 24,
                "mha_long": 12,
                "mha_short": 6,
                "packed_gqa_mqa_long": 6,
            },
        },
    }
    report = {
        "source_results": str(RESULTS.relative_to(ROOT)),
        "phase_summaries": phase_summaries,
        "excluded_controls": host_exclusions,
        "holdout_rationale": rationale,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "campaign_review_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# B200 campaign review summary",
        "",
        "## Frozen campaign phase summaries",
        "",
    ]
    for rule, phases in phase_summaries.items():
        md += [f"### {rule}", ""]
        for phase in ("discovery", "boundary", "holdout", "all"):
            md.append(f"- **{phase}:** {fmt(phases[phase])}")
        md.append("")
    md += [
        "## Timed excluded controls: all 12 D64 causal cells",
        "",
        f"Aggregate: {fmt(host_exclusions['timed_excluded_strata']['d64_causal']['summary'])}",
        "",
    ]
    for cell in causal_cells:
        lo, hi = cell["paired_round_ci95"]
        md.append(
            f"- {cell['phase']} B{cell['batch']} H{cell['q_heads']}/KV{cell['kv_heads']} "
            f"Q{cell['seqlen_q']} K{cell['seqlen_k']}: median "
            f"{cell['speedup']:.4f}x; paired {cell['paired_round_speedup']:.4f}x "
            f"[{lo:.4f}, {hi:.4f}]"
        )
    md += ["", "Eight of these have median speedup below 1.0; all 12 are excluded because the causal stratum failed promotion.", ""]
    md += ["## Conservative host-only exclusions (not timed performance cells)", ""]
    md += [f"- {item}" for item in host_exclusions["host_selector_exclusions_not_claimed_as_timed_cells"]]
    md += ["", "## Independent-holdout rationale", ""]
    md += [
        f"- **Independence:** {rationale['independence']}",
        f"- **Distribution:** {rationale['distribution']}",
        f"- **Model-family proxies:** {rationale['model_family_proxies']}",
        "- **D64 holdout:** 20 cells = 12 long MHA + 4 short MHA + 4 long packed GQA/MQA.",
        "- **D128 holdout:** 24 cells = 12 long MHA + 6 short MHA + 6 long packed GQA/MQA.",
        "",
    ]
    (OUT / "campaign_review_summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
