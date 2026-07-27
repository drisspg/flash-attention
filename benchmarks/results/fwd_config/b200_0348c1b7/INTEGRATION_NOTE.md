# Stack-owner integration note

This is a review-only evidence branch. Its parent is the B200 policy commit
`0348c1b7dcb3c512c09a98a59149ee9dab6027b7`; do not cherry-pick this artifact
commit into the production stack.

The stack owner will add `total_mblocks >= 64` during integration and add
62/64-M-block host boundary tests. This narrows the rule to measured occupancy
without removing any of the 60 D64 or 72 D128 measured retained cells, so no GPU
timing rerun is required.

The stack owner will also move PyYAML into benchmark commit #2735, correct the
combined-control campaign comment, add phase and final config=None results to
the production commit message, run cross-architecture checks, and submit the
fifth PR with stack-pr.

The B200 evidence producer did not run stack-pr or modify numbered stack
branches.
