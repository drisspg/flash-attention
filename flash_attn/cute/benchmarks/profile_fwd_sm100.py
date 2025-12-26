#!/usr/bin/env python3
"""Profile FlashAttention SM100 Forward Kernel with Intra-Kernel Profiling.

Demonstrates using transformer_nuggets.cute.profiler to trace warp-specialized
execution of the SM100 flash attention forward kernel.

Usage:
    cd /home/dev/meta/flash-attention/flash_attn/cute/benchmarks
    python profile_fwd_sm100.py

    # View trace at https://ui.perfetto.dev/
"""

import sys
import os
from rich import print

# Add parent directory (flash_attn/cute) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cuda.bindings.driver as cuda
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from flash_fwd_sm100 import FlashAttentionForwardSm100, PROFILER_TAG_NAMES

# Import profiler
from transformer_nuggets.cute.profiler import (
    profile_session,
    group_by_unit,
    rename_processes,
)
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds


def to_cute_tensor(t, assumed_align=16, leading_dim=-1):
    """Convert torch tensor to cute tensor with dynamic layout marking."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def main():
    print("=" * 70)
    print("FlashAttention SM100 Forward - Intra-Kernel Profiling Demo")
    print("=" * 70)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")

    # Problem dimensions - sized to use all 148 SMs
    # Work items = batch * heads * (seqlen_q / 128) = 4 * 8 * 32 = 1024
    batch_size = 4
    seqlen_q = 4096  # 32 Q-tiles
    seqlen_k = 8192  # Long K for more iterations per tile
    num_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    device = "cuda"

    print("\nProblem size:")
    print(f"  batch_size = {batch_size}")
    print(f"  seqlen_q   = {seqlen_q}")
    print(f"  seqlen_k   = {seqlen_k}")
    print(f"  num_heads  = {num_heads}")
    print(f"  head_dim   = {head_dim}")
    print(f"  dtype      = {dtype}")

    # Allocate tensors (Q uses seqlen_q, K/V use seqlen_k)
    Q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch_size, seqlen_k, num_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(batch_size, seqlen_k, num_heads, head_dim, dtype=dtype, device=device)
    O = torch.zeros_like(Q)
    lse = torch.zeros(batch_size, num_heads, seqlen_q, dtype=torch.float32, device=device)

    softmax_scale = head_dim**-0.5

    # Get CUDA stream
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Convert to CUTE tensors
    Q_cute = to_cute_tensor(Q)
    K_cute = to_cute_tensor(K)
    V_cute = to_cute_tensor(V)
    O_cute = to_cute_tensor(O)
    lse_cute = to_cute_tensor(lse)

    # Create the forward operator (non-causal since seqlen_q != seqlen_k)
    print("\nCreating FlashAttentionForwardSm100...")
    fwd = FlashAttentionForwardSm100(
        head_dim=head_dim,
        is_causal=False,  # Can't use causal with different Q/K lengths
        is_persistent=True,
    )

    # Get number of SM workers
    import cutlass

    hardware_info = cutlass.utils.HardwareInfo()
    num_sm_workers = hardware_info.get_device_multiprocessor_count()

    # Warmup run (compiles kernel)
    print(f"\nWarmup run (compiles kernel, {num_sm_workers} SMs)...")
    fwd(Q_cute, K_cute, V_cute, O_cute, lse_cute, softmax_scale, stream)
    torch.cuda.synchronize()

    # =========================================================================
    # Benchmark: No Profiling (baseline)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Benchmarking WITHOUT profiling (baseline)...")
    print("-" * 70)

    def run_no_profiling():
        fwd(Q_cute, K_cute, V_cute, O_cute, lse_cute, softmax_scale, stream)

    baseline_us = benchmark_cuda_function_in_microseconds(run_no_profiling, NUM_ITERS=100)
    print(f"  Baseline time: {baseline_us:.3f} us")

    # =========================================================================
    # Benchmark: With Profiling
    # =========================================================================
    trace_path = "flash_fwd_trace.json"
    # 14 tags (6 warp roles + 8 per-tile subtags) * ~7 tiles/SM
    max_events_per_unit = 8192

    print("\n" + "-" * 70)
    print("Benchmarking WITH profiling...")
    print("-" * 70)

    print("\nProfiling configuration:")
    print(f"  num_sm_workers      = {num_sm_workers}")
    print(f"  max_events_per_unit = {max_events_per_unit}")
    print(f"  trace_path          = {trace_path}")
    print(f"  tags                = {PROFILER_TAG_NAMES}")

    # Create process names for all SM workers (zero-padded for correct sorting in Perfetto)
    num_digits = len(str(num_sm_workers - 1))  # e.g., 148 SMs -> 3 digits
    process_names = {i: f"SM {i:0{num_digits}d}" for i in range(num_sm_workers)}

    with profile_session(
        max_events_per_unit=max_events_per_unit,
        num_units=(num_sm_workers, "SM"),
        tag_names=PROFILER_TAG_NAMES,
        trace_path=trace_path,
        device=device,
        post_process_events=group_by_unit,
        post_process_trace=rename_processes(process_names),
    ) as (prof, tag_table):
        # Convert prof buffer to cute tensor
        prof_buf_cute = from_dlpack(prof.tensor)

        def run_with_profiling():
            fwd(
                Q_cute,
                K_cute,
                V_cute,
                O_cute,
                lse_cute,
                softmax_scale,
                stream,
                prof_buf=prof_buf_cute,
                prof_max_events=Int32(prof.max_events_per_unit),
            )

        # Only run 1 iteration when profiling to keep timestamps aligned
        profiled_us = benchmark_cuda_function_in_microseconds(run_with_profiling, NUM_ITERS=1)

    print(f"  Profiled time: {profiled_us:.3f} us")

    # =========================================================================
    # Overhead Analysis
    # =========================================================================
    overhead_us = profiled_us - baseline_us
    overhead_pct = 100.0 * overhead_us / baseline_us

    print("\n" + "=" * 70)
    print("PROFILING OVERHEAD ANALYSIS")
    print("=" * 70)
    print(f"  Baseline (no profiling):  {baseline_us:>10.3f} us")
    print(f"  With profiling:           {profiled_us:>10.3f} us")
    print(f"  Overhead:                 {overhead_us:>10.3f} us ({overhead_pct:+.2f}%)")
    print("=" * 70)

    # Verify output
    output_norm = O.float().norm().item()
    print(f"\nOutput tensor norm: {output_norm:.4f}")

    print(f"\nTrace written to: {os.path.abspath(trace_path)}")
    print("\nTo view the timeline:")
    print("  1. Open https://ui.perfetto.dev/")
    print("  2. Click 'Open trace file' and select flash_fwd_trace.json")
    print("\nIn the trace you'll see:")
    print(
        f"  - {num_sm_workers} SM workers as separate 'process' rows (SM 0 - SM {num_sm_workers - 1})"
    )
    print("  - Per-tile tags:")
    print("    * load_q - Q TMA loads (Q0 + Q1)")
    print("    * load_issue - Per K-block K+V TMA load (includes stall + issue)")
    print("    * mma_prologue/epilogue, mma_pv/qk (per GEMM iteration)")
    print("    * softmax0_compute, softmax1_compute (per K-block softmax_step)")
    print("    * tmem_load_s - Wait for S from MMA + TMEM load (per K-block)")
    print("    * tmem_store_p - Store P to TMEM + fence (per K-block)")
    print("    * correction_rescale (per K-block), correction_final (normalization + sO store)")
    print("    * epilogue_tile (after barrier wait)")
    print("  - Timing excludes barrier waits for accurate compute measurement")


if __name__ == "__main__":
    main()
