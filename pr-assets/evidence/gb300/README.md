# GB300/SM103 selector evidence

The 303-cell `final_exact_results.json` is the exact-source superset run. Six dense MQA Q=1 cells failed the paired-interval gate, so the selector excludes Q=1 and `final_dense_narrow_results.json` reruns the retained 67-cell dense region. `final_summary.json` combines the first three policies from the superset with the narrowed dense policy for the 297 cells shown in the PR plot.

Both runs used seven alternating rounds, 31 fixed-pointer CUDA-graph samples per arm, preallocated outputs/workspaces, and 20 untimed BF16 4096x4096 GEMMs before each arm on an NVIDIA GB300 (SM103, 152 SMs). The complete compile/correctness records and raw samples are in the result JSON files.
