# B200 forward-selector review handoff

## Immutable collaboration state

- Branch: `fa4-b200-tuning`
- Commit: `0348c1b7dcb3c512c09a98a59149ee9dab6027b7`
- Remote: `origin/fa4-b200-tuning` points to the same commit.
- Base/numbered stack tip: `origin/drisspg/stack/56` remains `f16ee60a6a7dbde3c7f0498b16b4558d1579c331`.
- No `stack-pr` command was run; no numbered stack branch was modified.
- This evidence was generated without amending the commit. All review artifacts live under ignored `agent_space/`.

## Complete campaign

- Campaign: `benchmarks/configs/fwd_config_b200.yaml`
- Campaign SHA256: `7a84f7803b4b12e760ac1bb552129dfe29d0a0c88a78a9b43acf2ab84764907e`
- Complete 144-cell result: `agent_space/b200_fwd_config/full/results.json`
- Result SHA256: `86e7b207563da26581717da9bcdc75bb389e7624300b358bd33d15a2f4361334`
- Progress: 144/144; fake-compile specializations: 22/22; independent small-reference correctness: 4/4.
- Timing: 7 alternating paired rounds, 31 CUDA-graph samples per arm per round, 10 graph warmups, 20 clock-warmup GEMMs per arm.

### Frozen campaign summaries by phase

| Rule / phase | Cells | Geomean [paired-round 95% CI] | Weighted | Min | Max | W/N/L |
|---|---:|---:|---:|---:|---:|---:|
| D64 discovery | 20 | 1.1977 [1.1943, 1.2011] | 1.1920 | 1.1711 | 1.2156 | 20/0/0 |
| D64 boundary | 20 | 1.2144 [1.2117, 1.2172] | 1.1943 | 1.1745 | 1.2633 | 20/0/0 |
| D64 holdout | 20 | 1.2011 [1.1983, 1.2038] | 1.1864 | 1.1726 | 1.2640 | 20/0/0 |
| **D64 all** | **60** | **1.2044 [1.2026, 1.2061]** | **1.1907** | **1.1711** | **1.2640** | **60/0/0** |
| D128 discovery | 24 | 1.1477 [1.1436, 1.1517] | 1.1503 | 1.1047 | 1.1806 | 24/0/0 |
| D128 boundary | 24 | 1.2520 [1.2482, 1.2557] | 1.1747 | 1.1493 | 1.5030 | 24/0/0 |
| D128 holdout | 24 | 1.2192 [1.2160, 1.2225] | 1.1621 | 1.1178 | 1.5600 | 24/0/0 |
| **D128 all** | **72** | **1.2055 [1.2033, 1.2076]** | **1.1583** | **1.1047** | **1.5600** | **72/0/0** |

Full machine-readable summaries and every excluded timed control are in `campaign_review_summary.json`; the readable form is `campaign_review_summary.md`.

## Timed excluded controls

The only performance-timed excluded stratum is D64 causal: 12 cells, geomean 0.9933x [0.9894, 0.9972], weighted 0.9982x, range 0.9941-1.0071x, W/N/L 0/12/0. Eight medians are below 1.0 and three paired intervals are wholly below 1.0. All 12 cells, including the four medians at or above 1.0, are listed in `campaign_review_summary.{md,json}`.

FP16, SM103, Q<=256, K<2048, varlen, SplitKV, paging, sparse, QV/MLA, gather, modifiers, sinks, LSE/training, tile overrides, ratio-2 GQA, unpacked GQA, local attention, and unequal/unmeasured head dimensions are conservative unmeasured exclusions. They are not misrepresented as timed cells.

## Final old-baseline versus config=None confirmation

- Result: `final_7x31_baseline_vs_auto.json`
- SHA256: `e166828532008eb3bab061c4e6b0ffde2f9f40d5f25f5f64e72ecad58fe31617`
- Log: `final_7x31_baseline_vs_auto.log`
- 24 retained cells total: 12 D64 + 12 D128.
- Per rule: 4 discovery + 4 boundary + 4 independent holdout.
- Every cell compares the exact old explicit baseline config stored in the complete campaign result against post-selector `config=None`.
- Every arm has 7 alternating paired rounds x 31 fixed-pointer CUDA-graph samples = 217 samples.
- Post-selector fake-compile barrier: 12/12 unique specializations.
- Small independent float32-reference checks: 4/4.
- Direct old-baseline/config=None agreement on all timed cells: 24/24.
- `config=None` exactly equals the originally measured winning candidate on all 24 cells.

| Rule / phase | Cells | Geomean [paired-round 95% CI] | Weighted | Min | Max | W/N/L |
|---|---:|---:|---:|---:|---:|---:|
| D64 discovery | 4 | 1.2068 [1.2011, 1.2124] | 1.1981 | 1.1732 | 1.2208 | 4/0/0 |
| D64 boundary | 4 | 1.2138 [1.2055, 1.2222] | 1.1977 | 1.1878 | 1.2613 | 4/0/0 |
| D64 holdout | 4 | 1.2032 [1.1942, 1.2122] | 1.1835 | 1.1750 | 1.2375 | 4/0/0 |
| **D64 all** | **12** | **1.2079 [1.2034, 1.2124]** | **1.1928** | **1.1732** | **1.2613** | **12/0/0** |
| D128 discovery | 4 | 1.1377 [1.1308, 1.1445] | 1.1447 | 1.1120 | 1.1729 | 4/0/0 |
| D128 boundary | 4 | 1.2697 [1.2660, 1.2735] | 1.1717 | 1.1567 | 1.5003 | 4/0/0 |
| D128 holdout | 4 | 1.2280 [1.1997, 1.2570] | 1.1819 | 1.1319 | 1.3248 | 4/0/0 |
| **D128 all** | **12** | **1.2105 [1.2008, 1.2204]** | **1.1622** | **1.1120** | **1.5003** | **12/0/0** |

## Environment

- GPU: NVIDIA GB200 (SM100), 152 SMs, compute capability 10.0.
- SM clock: 120 MHz idle before the original campaign; 2062 MHz at campaign completion; maximum 2062 MHz.
- Power limit: 1120 W; original campaign telemetry was 201.01 W before and 264.91 W after.
- Driver: 580.126.20.
- Torch: 2.14.0.dev20260708+cu132.
- CUDA reported by Torch: 13.2.
- CUTLASS DSL/base/CUDA-13 libs: 4.6.0.dev0.
- Python: 3.12.13+meta.
- Full manifests: `environment_manifest.txt`, `pip_freeze.txt`, `gpu_before.txt`, `gpu_after.txt`, and `final_gpu_environment.txt`.

## Validation outputs

- `host_pytest.log`: 145 passed.
- `b200_runtime_correctness.log`: 2 passed (D64 and D128 `config=None` packed-GQA runtime checks); 16 upstream deprecation warnings.
- `ruff_and_diff_check.log`: required Ruff scope passed; `git diff --check` passed; branch clean.
- `run.log` / `launcher.log`: complete original fake-compile, correctness, and campaign output.
- `final_7x31_baseline_vs_auto.log`: fresh post-selector fake-compile, correctness, and timed confirmation output.

## Holdout rationale

The holdouts were frozen before promotion and were not used to choose candidate arms or policy boundaries. They use unseen batch/head combinations and mostly odd Q/K lengths rather than copies of discovery or +/-1 boundary triplets. The distribution is a stratified geometry envelope, not a production-frequency claim: MHA and legal packed GQA/MQA ratios 1/4/8/16, batch 1-7, varied heads, asymmetric Q/K, short wave-boundary and long prefill contexts. D64 MHA is a dense-head encoder/GPT-style geometry proxy; D128 MHA and GQA/MQA are proxies for scheduling geometry common in Llama/Mistral/Qwen-like decoder families. No model popularity or traffic mix is inferred.

D64 holdout: 20 cells = 12 long MHA + 4 short MHA + 4 long packed GQA/MQA. D128 holdout: 24 cells = 12 long MHA + 6 short MHA + 6 long packed GQA/MQA.
