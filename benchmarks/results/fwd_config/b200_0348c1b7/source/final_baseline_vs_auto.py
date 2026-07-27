#!/usr/bin/env python3
"""Final B200 old-baseline versus config=None confirmation.

This is a review artifact, not production source. It freezes 12 representative
retained cells per B200 rule (four discovery, four boundary, four independent
holdout), verifies that config=None selects the originally measured candidate,
compiles both exact specializations behind an isolated fake-tensor barrier,
checks direct output agreement, and times seven alternating paired rounds with
31 fixed-pointer CUDA-graph samples per arm and round.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmarks" / "fwd_config_bench.py"
CAMPAIGN = REPO / "benchmarks" / "configs" / "fwd_config_b200.yaml"
SOURCE_RESULTS = REPO / "agent_space" / "b200_fwd_config" / "full" / "results.json"
OUTPUT = Path(__file__).resolve().parent / "review_artifacts" / "final_7x31_baseline_vs_auto.json"
SPEC = importlib.util.spec_from_file_location("fwd_config_bench_final_confirmation", BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)

# Four discovery, four boundary, and four holdout cells per rule. The retained
# set spans MHA, GQA/MQA ratios 4/8/16, batch/head variation, long and short
# contexts, odd lengths, K=2048, Q=257, and 152-SM wave-boundary controls.
SELECTED_NAMES = (
    # D64 discovery
    "d64_direct_long_noncausal_mha__q32768_k16384_b1_c0_s1__h16kv16d64v64",
    "d64_direct_long_noncausal_mha__q4096_k24576_b4_c0_s1__h32kv32d64v64",
    "d64_direct_scope_controls__q16384_k8192_b2_c0_s1__h16kv4d64v64",
    "d64_direct_scope_controls__q8193_k16385_b3_c0_s1__h24kv3d64v64",
    # D64 boundary
    "d64_direct_long_noncausal_mha__q32767_k9216_b1_c0_s1__h16kv16d64v64",
    "d64_direct_long_noncausal_mha__q24577_k8193_b2_c0_s1__h16kv16d64v64",
    "d64_direct_scope_controls__q511_k8192_b1_c0_s1__h32kv32d64v64",
    "d64_direct_scope_controls__q257_k2049_b4_c0_s1__h8kv8d64v64",
    # D64 holdout
    "d64_direct_long_noncausal_mha__q8193_k12289_b3_c0_s1__h24kv24d64v64",
    "d64_direct_long_noncausal_mha__q9363_k24577_b7_c0_s1__h8kv8d64v64",
    "d64_direct_scope_controls__q769_k3073_b1_c0_s1__h24kv24d64v64",
    "d64_direct_scope_controls__q6657_k10241_b5_c0_s1__h16kv2d64v64",
    # D128 discovery
    "d128_1cta_long_noncausal_mha__q32768_k16384_b1_c0_s1__h16kv16d128v128",
    "d128_1cta_long_noncausal_mha__q4096_k24576_b4_c0_s1__h32kv32d128v128",
    "d128_1cta_scope_controls__q16384_k8192_b2_c0_s1__h16kv4d128v128",
    "d128_1cta_scope_controls__q8193_k16385_b3_c0_s1__h24kv3d128v128",
    # D128 boundary
    "d128_1cta_long_noncausal_mha__q32767_k9216_b1_c0_s1__h16kv16d128v128",
    "d128_1cta_long_noncausal_mha__q24577_k8193_b2_c0_s1__h16kv16d128v128",
    "d128_1cta_scope_controls__q511_k8192_b1_c0_s1__h32kv32d128v128",
    "d128_1cta_scope_controls__q511_k2048_b2_c0_s1__h16kv16d128v128",
    # D128 holdout
    "d128_1cta_long_noncausal_mha__q8193_k12289_b3_c0_s1__h24kv24d128v128",
    "d128_1cta_long_noncausal_mha__q9363_k24577_b7_c0_s1__h8kv8d128v128",
    "d128_1cta_scope_controls__q769_k3073_b1_c0_s1__h24kv24d128v128",
    "d128_1cta_scope_controls__q6657_k10241_b5_c0_s1__h16kv2d128v128",
)

SETTINGS = {
    "seed": 0,
    "rounds": 7,
    "iters_per_round": 31,
    "warmup_iters": 10,
    "clock_warmup_iters": 20,
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def checkpoint(payload: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(OUTPUT)


def paired_interval(row: dict) -> tuple[float, float, float]:
    logs = [
        math.log(baseline / automatic)
        for baseline, automatic in zip(
            row["baseline_round_medians_us"], row["config_none_round_medians_us"]
        )
    ]
    center = statistics.mean(logs)
    stderr = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    return (
        math.exp(center),
        math.exp(center - 1.96 * stderr),
        math.exp(center + 1.96 * stderr),
    )


def summarize(rows: list[dict]) -> dict:
    paired_logs = []
    for row in rows:
        paired_logs.append(
            [
                math.log(baseline / automatic)
                for baseline, automatic in zip(
                    row["baseline_round_medians_us"],
                    row["config_none_round_medians_us"],
                )
            ]
        )
    mean_log = statistics.mean(map(statistics.mean, paired_logs))
    variance = sum(
        statistics.variance(logs) / len(logs) if len(logs) > 1 else 0.0
        for logs in paired_logs
    ) / len(rows) ** 2
    stderr = math.sqrt(variance)
    ratios = [row["baseline_over_config_none"] for row in rows]
    return {
        "cells": len(rows),
        "geomean_paired_round_speedup": math.exp(mean_log),
        "geomean_paired_round_ci95": [
            math.exp(mean_log - 1.96 * stderr),
            math.exp(mean_log + 1.96 * stderr),
        ],
        "time_weighted_speedup": sum(row["baseline_median_us"] for row in rows)
        / sum(row["config_none_median_us"] for row in rows),
        "minimum_speedup": min(ratios),
        "maximum_speedup": max(ratios),
        "wins_neutral_losses": [
            sum(value > 1.01 for value in ratios),
            sum(0.99 <= value <= 1.01 for value in ratios),
            sum(value < 0.99 for value in ratios),
        ],
    }


def make_callables(case, old_baseline, dtype_name: str, seed: int):
    _, interface_module = bench.flash_modules()
    torch.manual_seed(seed)
    kwargs = bench.build_inputs(case, getattr(torch, dtype_name), torch.randn)

    def call_kwargs(config):
        return {
            **kwargs,
            "out": torch.empty(
                *kwargs["q"].shape[:-1],
                case.dv,
                device="cuda",
                dtype=getattr(torch, dtype_name),
            ),
            **bench.split_workspaces(case, kwargs, config),
        }

    baseline_kwargs = call_kwargs(old_baseline)
    automatic_kwargs = call_kwargs(old_baseline)

    def baseline():
        return interface_module._flash_attn_fwd(
            **baseline_kwargs, config=old_baseline
        )[0]

    def automatic():
        return interface_module._flash_attn_fwd(**automatic_kwargs)[0]

    return {"old_explicit_baseline": baseline, "config_none": automatic}


def measure(case, old_baseline, expected_auto, dtype_name: str) -> dict:
    callables = make_callables(case, old_baseline, dtype_name, SETTINGS["seed"])
    baseline_output = callables["old_explicit_baseline"]()
    automatic_output = callables["config_none"]()
    torch.testing.assert_close(
        automatic_output.float(), baseline_output.float(), atol=0.04, rtol=0.04
    )

    samples = {name: [] for name in callables}
    round_medians = {name: [] for name in callables}
    warm_clocks = bench.clock_warmup(
        dtype_name, SETTINGS["clock_warmup_iters"]
    )
    names = tuple(callables)
    for round_index in range(SETTINGS["rounds"]):
        order = names[round_index % 2 :] + names[: round_index % 2]
        for name in order:
            warm_clocks()
            stats = bench.benchmark_stats(
                callables[name],
                SETTINGS["iters_per_round"],
                SETTINGS["warmup_iters"],
            )
            samples[name].extend(stats.samples_us)
            round_medians[name].append(stats.median_us)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    row = {
        **asdict(case),
        "rule": "d64_direct_output" if case.d == 64 else "d128_1cta",
        "old_explicit_baseline_config": asdict(old_baseline),
        "selected_config_none_config": asdict(expected_auto),
        "selector_check": "config=None exactly matched the originally measured candidate",
        "direct_correctness": {
            "passed": True,
            "atol": 0.04,
            "rtol": 0.04,
        },
        "baseline_median_us": medians["old_explicit_baseline"],
        "config_none_median_us": medians["config_none"],
        "baseline_over_config_none": (
            medians["old_explicit_baseline"] / medians["config_none"]
        ),
        "baseline_samples_us": samples["old_explicit_baseline"],
        "config_none_samples_us": samples["config_none"],
        "baseline_round_medians_us": round_medians["old_explicit_baseline"],
        "config_none_round_medians_us": round_medians["config_none"],
    }
    paired, low, high = paired_interval(row)
    row["paired_round_speedup"] = paired
    row["paired_round_ci95"] = [low, high]
    return row


def main() -> None:
    settings, experiments, cases, correctness_cases = bench.load_campaign(CAMPAIGN)
    dtype_name = settings.get("dtype", "bfloat16")
    by_name = {case.name: case for case in cases}
    if len(SELECTED_NAMES) != len(set(SELECTED_NAMES)):
        raise AssertionError("duplicate selected confirmation names")
    missing = set(SELECTED_NAMES) - set(by_name)
    if missing:
        raise AssertionError(f"missing selected cells: {sorted(missing)}")
    selected = [by_name[name] for name in SELECTED_NAMES]
    source_payload = json.loads(SOURCE_RESULTS.read_text())
    source_rows = {row["name"]: row for row in source_payload["results"]}
    config_module, _ = bench.flash_modules()

    selector_checks = []
    old_baselines = {}
    expected_autos = {}
    for case in selected:
        source = source_rows[case.name]
        old_baseline = config_module.FwdConfig(**source["baseline_config"])
        measured_candidate = config_module.FwdConfig(**source["candidate_config"])
        automatic = config_module.select_fwd_config(
            bench.selector_inputs(case, dtype_name)
        )
        if automatic != measured_candidate:
            raise AssertionError(
                f"{case.name}: auto {automatic} != measured candidate {measured_candidate}"
            )
        resolved_baseline, resolved_candidate = bench.resolve_arms(
            case, experiments[case.experiment], dtype_name
        )
        if resolved_baseline.config != old_baseline:
            raise AssertionError(f"{case.name}: old baseline config drifted")
        if resolved_candidate.config != automatic:
            raise AssertionError(f"{case.name}: candidate config drifted")
        old_baselines[case.name] = old_baseline
        expected_autos[case.name] = automatic
        selector_checks.append(
            {
                "case": case.name,
                "passed": True,
                "old_baseline": asdict(old_baseline),
                "config_none": asdict(automatic),
            }
        )

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "repo_commit": git("rev-parse", "HEAD"),
            "repo_status": git("status", "--short"),
            "branch": git("branch", "--show-current"),
            "campaign": str(CAMPAIGN.relative_to(REPO)),
            "campaign_sha256": hashlib.sha256(CAMPAIGN.read_bytes()).hexdigest(),
            "source_results": str(SOURCE_RESULTS.relative_to(REPO)),
            "source_results_sha256": hashlib.sha256(SOURCE_RESULTS.read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cutlass_dsl_libs_cu13": importlib.metadata.version(
                "nvidia-cutlass-dsl-libs-cu13"
            ),
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "timing_contract": (
                "old exact explicit baseline versus post-selector config=None; "
                "fixed-pointer CUDA-graph replay; preallocated output; alternating "
                "paired arm order; 7 rounds; 31 samples/arm/round; 10 graph warmups; "
                "20 untimed 4096x4096 GEMMs before each arm"
            ),
        },
        "measurement_settings": SETTINGS,
        "selection": {
            "total": len(selected),
            "per_rule": {"d64_direct_output": 12, "d128_1cta": 12},
            "per_rule_phase": {"discovery": 4, "boundary": 4, "holdout": 4},
            "rationale": (
                "Balanced retained-cell sample across development interiors, explicit "
                "selector/wave boundaries, and independent odd-shape holdouts; includes "
                "MHA and legal packed GQA/MQA, batches 1-7, heads 8-32, long/short "
                "contexts, Q=257, K=2048/2049, and non-power-of-two lengths."
            ),
            "checks": selector_checks,
        },
        "fake_compile": {"completed": 0, "rows": []},
        "small_reference_correctness": [],
        "progress": {"completed": 0, "total": len(selected)},
        "summaries": {},
        "results": [],
    }
    checkpoint(payload)

    # Uses the benchmark runner's isolated FakeTensorMode subprocess barrier.
    payload["fake_compile"]["rows"] = bench.compile_barrier(
        selected, experiments, dtype_name, workers=16, seed=SETTINGS["seed"]
    )
    payload["fake_compile"]["completed"] = len(payload["fake_compile"]["rows"])
    checkpoint(payload)

    # Re-run every campaign's independent float32-reference correctness case on
    # the post-selector commit. Each checks old baseline, candidate, and config=None.
    payload["small_reference_correctness"] = bench.run_correctness(
        correctness_cases, experiments, dtype_name, SETTINGS["seed"]
    )
    checkpoint(payload)

    completed = {row["name"] for row in payload["results"]}
    for case in selected:
        if case.name in completed:
            continue
        row = measure(
            case,
            old_baselines[case.name],
            expected_autos[case.name],
            dtype_name,
        )
        payload["results"].append(row)
        payload["progress"]["completed"] = len(payload["results"])
        checkpoint(payload)
        print(
            f"[{len(payload['results'])}/{len(selected)}] {case.name}: "
            f"old/config=None={row['baseline_over_config_none']:.4f}x "
            f"paired={row['paired_round_speedup']:.4f}x "
            f"[{row['paired_round_ci95'][0]:.4f}, {row['paired_round_ci95'][1]:.4f}]"
        )
        torch.cuda.empty_cache()

    groups = {
        "d64_direct_output": [row for row in payload["results"] if row["d"] == 64],
        "d128_1cta": [row for row in payload["results"] if row["d"] == 128],
    }
    payload["summaries"] = {
        rule: {
            "all": summarize(rows),
            "by_phase": {
                phase: summarize([row for row in rows if row["phase"] == phase])
                for phase in ("discovery", "boundary", "holdout")
            },
        }
        for rule, rows in groups.items()
    }
    checkpoint(payload)
    print(json.dumps(payload["summaries"], indent=2))
    print(f"results={OUTPUT}")


if __name__ == "__main__":
    main()
