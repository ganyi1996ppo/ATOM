# MiniMax-M3.1 perf bundle (internal testing only — not a merge-ready submission)

Branch: `minimax-m31-perf-bundle` (built off `origin/main`, pushed to `fork` = ganyi1996ppo/ATOM).

This branch cherry-picks 4 independent PRs from the same optimization pass onto one
branch so a colleague can test them together without juggling 4 checkouts. It does
**not** change PR review/merge status — treat each PR as the source of truth for
review discussion; this branch is for running them together.

## Bundled PRs

1. **#3721** — DSpark draft: sharded greedy Markov block, one atomic-max launch per
   position (`atom/models/minimax_m3/packed_argmax.py`, Markov kernels in
   `atom/models/minimax_m3/dspark.py`). See PR body for the host-side latency numbers.
2. **#3722** — DSpark block: metadata and embedding merged into one kernel each,
   pages published with the step's sampling upload (`_block_meta_kernel`,
   `_block_embed_kernel` in `atom/models/minimax_m3/dspark.py`).
3. **#3723** — DSpark draft: Gemma RMSNorm writes the MXFP8 input of the next
   linear; one GEMM for all context K/V (`atom/models/minimax_m3/draft_fused.py`).
4. **#3724** — Step watchdog and GPU guard for Mono decode stalls
   (`atom/utils/gpu_guard.py`, robustness fix, not a perf change).

A real merge conflict occurred cherry-picking #3722 and #3723: both #3721 and #3722
add independent new Triton kernels / class methods into
`atom/models/minimax_m3/dspark.py` at adjacent insertion points, and #3723 adds its
own independent imports/methods at the same spots. All three conflicts were
purely additive (no two PRs touch the same existing line) — resolved by keeping
**both** sides' additions, concatenated. No logic was dropped or altered.

## Verification done here

- All touched/added files parse cleanly (`ast.parse`).
- `python3 -c "import atom"` succeeds with no error.
- Could **not** run the actual test suite or any GPU benchmark from this sandbox
  (no `torch`/cluster access here) — this is code-level verification only, not a
  functional or performance re-validation. Treat this branch as "applies cleanly
  and imports", not as "proven to reproduce the numbers" until someone runs it on
  the cluster.

## Required to actually reproduce this session's measured performance

**These 4 PRs alone will NOT reproduce the measured wins.** The actual measured
results (c1: +6.8% ix_p90 / +3.3% tput vs same-condition vLLM; c2: officially
confirmed 628.0 ix_p90 / 8222 tok/s/GPU vs published 621.1 / 7955) depend on
**runtime recipe/env flags that live on the cluster, not in this repo**:

- The C17S scheduler-fix flags used for the c2 win:
  `ATOM_P1_SCHED_SERIAL_MAXN=2`, `_AGE_TOKENS=256`, `_GAP_TOKENS=64`.
- The broader recipe chain: `env_armC17.sh -> env_armC16.sh -> env_armC15.sh ->
  env_armC14.sh`, including `ATOM_P1_MONO_MAX_ROWS=8` (gates the mono decode
  kernel off above 8 rows — confirmed necessary; mono measurably loses at 16
  rows+, see the G2 mono investigation notes).

These recipe files are under `/shared_nfs/ygan/p1/` on the Crusoe cluster this
session used, starting from `env_armC17.sh`. Whoever hands off this branch needs
to also hand over (or point at) that recipe directory — this repo alone is code,
not the full reproduction package.
