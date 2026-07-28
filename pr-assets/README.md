# FA4 typed-forward stack PR assets

These images are embedded in PRs #2732, #2733, #2735, #2738, and #2741.

- `typed-config-boundary.png`: typed config and code-generation projection model.
- `forward-runtime-flow.png`: `_flash_attn_fwd` selection/execution flow.
- `benchmark-contract.png`: explicit-config campaign workflow and timing contract.
- `gb300-policy-gains.png`: aggregate GB300/SM103 policy evidence.
- `b200-policy-gains.png`: aggregate B200/SM100 policy evidence and rejected causal control.

Regenerate with:

```bash
python generate_stack_pr_assets.py
```

The policy plots use `gb300_summary.json` and `b200_summary.json`. Thick bars are paired-round 95% timing intervals around the geomean; thin bars are the full range of per-cell median speedups. Diamonds are time-weighted speedups.

The complete GB300 result JSONs, run logs, frozen campaign, and checksums are under `evidence/gb300/`. The complete B200 package is archived separately at evidence commit `0a9f4dc3944a00da16730e642318f39e95a981c1`.
