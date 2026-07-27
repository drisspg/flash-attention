# B200 campaign review summary

## Frozen campaign phase summaries

### d64_direct_output

- **discovery:** 20 cells | geomean 1.1977x [1.1943, 1.2011] | weighted 1.1920x | min 1.1711x | max 1.2156x | W/N/L 20/0/0 | paired-CI-below-1 0
- **boundary:** 20 cells | geomean 1.2144x [1.2117, 1.2172] | weighted 1.1943x | min 1.1745x | max 1.2633x | W/N/L 20/0/0 | paired-CI-below-1 0
- **holdout:** 20 cells | geomean 1.2011x [1.1983, 1.2038] | weighted 1.1864x | min 1.1726x | max 1.2640x | W/N/L 20/0/0 | paired-CI-below-1 0
- **all:** 60 cells | geomean 1.2044x [1.2026, 1.2061] | weighted 1.1907x | min 1.1711x | max 1.2640x | W/N/L 60/0/0 | paired-CI-below-1 0

### d128_1cta

- **discovery:** 24 cells | geomean 1.1477x [1.1436, 1.1517] | weighted 1.1503x | min 1.1047x | max 1.1806x | W/N/L 24/0/0 | paired-CI-below-1 0
- **boundary:** 24 cells | geomean 1.2520x [1.2482, 1.2557] | weighted 1.1747x | min 1.1493x | max 1.5030x | W/N/L 24/0/0 | paired-CI-below-1 0
- **holdout:** 24 cells | geomean 1.2192x [1.2160, 1.2225] | weighted 1.1621x | min 1.1178x | max 1.5600x | W/N/L 24/0/0 | paired-CI-below-1 0
- **all:** 72 cells | geomean 1.2055x [1.2033, 1.2076] | weighted 1.1583x | min 1.1047x | max 1.5600x | W/N/L 72/0/0 | paired-CI-below-1 0

## Timed excluded controls: all 12 D64 causal cells

Aggregate: 12 cells | geomean 0.9933x [0.9894, 0.9972] | weighted 0.9982x | min 0.9941x | max 1.0071x | W/N/L 0/12/0 | paired-CI-below-1 3

- discovery B1 H16/KV16 Q32768 K8192: median 1.0051x; paired 1.0034x [0.9983, 1.0087]
- discovery B1 H32/KV32 Q16384 K16384: median 0.9959x; paired 0.9845x [0.9689, 1.0003]
- discovery B2 H16/KV16 Q16385 K12289: median 1.0053x; paired 1.0026x [0.9848, 1.0207]
- discovery B2 H32/KV32 Q8193 K24577: median 1.0005x; paired 1.0015x [0.9982, 1.0047]
- discovery B4 H16/KV16 Q8192 K32768: median 0.9956x; paired 0.9948x [0.9920, 0.9976]
- discovery B3 H24/KV24 Q8193 K12289: median 0.9941x; paired 0.9878x [0.9689, 1.0070]
- discovery B1 H40/KV40 Q13313 K10241: median 0.9962x; paired 0.9871x [0.9766, 0.9978]
- discovery B5 H16/KV16 Q6657 K9217: median 0.9972x; paired 0.9939x [0.9834, 1.0046]
- holdout B3 H20/KV20 Q9001 K14337: median 1.0071x; paired 0.9900x [0.9641, 1.0166]
- holdout B2 H24/KV24 Q11009 K13313: median 0.9992x; paired 0.9908x [0.9810, 1.0006]
- holdout B1 H48/KV48 Q11009 K9217: median 0.9999x; paired 0.9993x [0.9923, 1.0064]
- holdout B5 H12/KV12 Q8801 K18433: median 0.9944x; paired 0.9843x [0.9711, 0.9978]

Eight of these have median speedup below 1.0; all 12 are excluded because the causal stratum failed promotion.

## Conservative host-only exclusions (not timed performance cells)

- FP16 (separate dtype stratum)
- device_arch != 100, including SM103/GB300
- Q packed length <= 256
- max K length < 2048
- causal or local/windowed attention
- varlen paths
- SplitKV / num_splits != 1
- paged KV
- block-sparse attention
- QV/MLA
- gather KV
- score modifiers
- mask modifiers
- learnable sinks
- LSE-producing, preallocated-LSE, Flex/FLASH, and training/autograd paths
- requested tile_m or tile_n other than 128
- D != DV or D outside {64, 128}
- GQA ratio 2
- unpacked GQA; retained GQA ratios 4/8/16 require pack_gqa

## Independent-holdout rationale

- **Independence:** The holdout tables were frozen before promotion and were not used to choose the candidate arm or selector boundaries. They use unseen batch/head-count combinations and mostly odd/non-power-of-two Q/K lengths, rather than copies of the discovery or +/-1 boundary triplets.
- **Distribution:** The campaign is a stratified shape envelope, not a frequency-weighted claim: batch 1-7, heads 8-48 (and up to 64 for D128 controls), symmetric/asymmetric Q/K, short-prefill/wave-boundary and long-prefill contexts, and legal packed MHA/GQA/MQA ratios 1/4/8/16. Odd lengths model real request/padding residue and prevent power-of-two-only tuning.
- **Model-family proxies:** D64 MHA cells proxy dense-head encoder/GPT-style families and D64 serving variants; D128 MHA and ratios 4/8/16 proxy the scheduling geometry common in Llama/Mistral/Qwen-like decoder families and MQA-style deployments. These are geometry proxies only; no production model mix or traffic frequency is inferred.
- **D64 holdout:** 20 cells = 12 long MHA + 4 short MHA + 4 long packed GQA/MQA.
- **D128 holdout:** 24 cells = 12 long MHA + 6 short MHA + 6 long packed GQA/MQA.
