# B200 BF16 forward-config campaign

## d64_noncausal_output

60 cells (40 development, 20 holdout) | geomean 1.2044x [1.2026, 1.2061] | weighted 1.1907x | min 1.1711x | max 1.2640x | W/N/L 60/0/0 | holdout min 1.1726x

Paired intervals wholly below 1.0: 0

## d64_causal_exclusion

12 cells (8 development, 4 holdout) | geomean 0.9933x [0.9894, 0.9972] | weighted 0.9982x | min 0.9941x | max 1.0071x | W/N/L 0/12/0 | holdout min 0.9944x

Paired intervals wholly below 1.0: 3

## d128_noncausal_output

72 cells (48 development, 24 holdout) | geomean 1.2055x [1.2033, 1.2076] | weighted 1.1583x | min 1.1047x | max 1.5600x | W/N/L 72/0/0 | holdout min 1.1178x

Paired intervals wholly below 1.0: 0

## All losing cells

- discovery B1 H32/KV32 Q16384 K16384 D64 causal=True: median 0.9959x; paired 0.9845x [0.9689, 1.0003]
- discovery B4 H16/KV16 Q8192 K32768 D64 causal=True: median 0.9956x; paired 0.9948x [0.9920, 0.9976]
- discovery B3 H24/KV24 Q8193 K12289 D64 causal=True: median 0.9941x; paired 0.9878x [0.9689, 1.0070]
- discovery B1 H40/KV40 Q13313 K10241 D64 causal=True: median 0.9962x; paired 0.9871x [0.9766, 0.9978]
- discovery B5 H16/KV16 Q6657 K9217 D64 causal=True: median 0.9972x; paired 0.9939x [0.9834, 1.0046]
- holdout B2 H24/KV24 Q11009 K13313 D64 causal=True: median 0.9992x; paired 0.9908x [0.9810, 1.0006]
- holdout B1 H48/KV48 Q11009 K9217 D64 causal=True: median 0.9999x; paired 0.9993x [0.9923, 1.0064]
- holdout B5 H12/KV12 Q8801 K18433 D64 causal=True: median 0.9944x; paired 0.9843x [0.9711, 0.9978]
