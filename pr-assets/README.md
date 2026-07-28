# FA4 typed-forward stack PR assets

These images are embedded in PRs #2732, #2733, #2735, #2738, and #2741.

- `typed-config-boundary.png`: typed config and code-generation projection model.
- `forward-runtime-flow.png`: `_flash_attn_fwd` selection/execution flow.
- `benchmark-contract.png`: explicit-config campaign workflow and timing contract.
- `gb300-policy-gains.png`: all 297 measured GB300/SM103 policy cells.
- `b200-policy-gains.png`: all 132 retained B200/SM100 cells plus 12 measured causal controls.

Regenerate with:

```bash
python generate_stack_pr_assets.py
```

The performance figures are Seaborn strip plots built directly from the archived raw result rows. Every circle is one timed workload cell, colored by campaign phase. The outlined diamond and error bar are the geomean and paired-round 95% timing interval recomputed from each cell's seven round medians; the `X` is the time-weighted aggregate. There are no synthetic points, fitted distributions, or smoothing.

The complete GB300 result JSONs, run logs, frozen campaign, and checksums are under `evidence/gb300/`. The B200 plot source is under `evidence/b200/`; the complete B200 package is also archived at evidence commit `0a9f4dc3944a00da16730e642318f39e95a981c1`.
