# GLM-5.2: TP-only short extends + prefill CUDA graphs under prefill-CP (B3 + P-small)

Engine-side patch against SGLang **v0.5.19** (tag `v0.5.19`, commit `0bcd822`) for
GLM-5.2 (`GlmMoeDsa`, NVFP4) served with TP4 + EP4 + prefill context parallelism.
It is **not** an upstream candidate as-is; see "Known gaps".

## What it does

In the validated prod recipe (`--enable-prefill-cp --attn-cp-size 4
--cp-strategy interleave`, EAGLE 5/1/6) at production cache-hit rates
(~0.57-0.66) only ~14% of prefill steps are short extends of <= 2048 new tokens
(ladder c=1: 186/1350 rank-0 prefill steps; all 186 replayed a graph, 0 of the
1164 CP-sharded steps did). ~85% of steps are long (> 2048 new tokens)
CP-sharded prefills, which this patch leaves unchanged. Short extends dominate
only in hot-prefix traffic (hit ~0.98): ~200-450 new tokens appended to a
10-14k-token cached prefix. Prefill-CP shards such a short batch across the 4
ranks exactly like a cold prefill, and the step becomes launch-paced (~3.6k
kernel launches; host CPU ~253 ms vs GPU busy ~137 ms per short step).

The patch adds two mechanisms, both off by default:

1. **B3 - TP-only short extends.** For an extend batch whose total number of
   new tokens is `<= SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS` (we use 2048), all
   three per-batch CP gates return "no CP" and the batch runs plain TP4. CP stays
   on for long / cold prefills.
   - `InterleaveCPStrategy.can_apply` (`layers/cp/interleave.py`)
   - `can_dsa_cp_split` and `can_dsa_prefill_cp_round_robin_split`
     (`layers/attention/dsa/utils.py`)

   All three gates consult the same helper, `dsa_prefill_cp_skip_short_extend`,
   so a batch is either fully sharded or fully unsharded. Flipping only two of
   the three (an earlier attempt, "B2") produced illegal memory accesses: the DSA
   metadata was sized for 1/4 of the tokens while `q`/`k`/`out_cache_loc` carried
   all of them.

2. **P-small - prefill CUDA graphs for TP-only batches under prefill-CP.**
   Upstream turns prefill CUDA graphs off for DSA models with prefill-CP
   (`arg_groups/model_hook.py`) and turns breakable graphs off under any CP
   (`arg_groups/cuda_graph_hook.py`). With B3 enabled this patch keeps
   `--cuda-graph-backend-prefill breakable`, clamps the capture buckets to
   `<= SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS`, and replays a graph only for a
   batch that every CP gate leaves TP-only (`PrefillCudaGraphRunner.can_run_graph`,
   `_prefill_cuda_graph_allows_context_parallel`). CP-sharded batches stay eager.
   Replay enters the per-layer `maybe_use_decode_attn_tp` (sliced attention
   heads) context that the eager layer forward would otherwise enter
   (`_dsa_prefill_cp_tp_only_attn_ctx`), and the DSA indexer takes the non-CP
   split-op path when `attn_cp_metadata is None`.

## Enabling

On **all ranks** of the server:

```bash
export SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS=2048
python -m sglang.launch_server ... \
  --tp-size 4 --ep-size 4 \
  --enable-prefill-cp --attn-cp-size 4 --cp-strategy interleave \
  --cuda-graph-backend-prefill breakable
```

B3 alone: set the env var and leave `--cuda-graph-backend-prefill` at its
default. P-small needs both the env var and the breakable (or full) prefill
graph backend; without the env var the upstream hooks disable prefill graphs
under prefill-CP exactly as before.

## Disabling

Unset `SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS` (or set it to `0`) and drop
`--cuda-graph-backend-prefill breakable`. With the env var unset / flag off the
engine's behaviour is byte-identical to pristine v0.5.19: every new code path
is behind `envs.SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS.get() > 0`.

## Validated configuration

- SGLang v0.5.19, B200 x4, GLM-5.2 (`GlmMoeDsa`) NVFP4.
- TP4 + EP4, `--enable-prefill-cp --attn-cp-size 4 --cp-strategy interleave`.
- EAGLE speculative decoding, steps/topk/draft-tokens = 5/1/6.
- `--mem-fraction-static 0.85`.
- `SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS=2048`, `--cuda-graph-backend-prefill breakable`.

## Building

Either:

- From this branch: `pip install -e python` (plus the usual `sgl-kernel` /
  FlashInfer wheels for v0.5.19), or
- Start from the `lmsysorg/sglang` v0.5.19 Docker image and apply the diff of
  this branch against `release/v0.5.19-base` on top of the installed
  `sglang` package (`git diff release/v0.5.19-base...HEAD -- python/ | git apply
  --directory=<site-packages path containing sglang/>`; the patch touches
  pure-Python files only, no kernel rebuild).

sha256 of the applied diff (`git diff release/v0.5.19-base...HEAD -- python/`,
i.e. the 8 engine files, excluding this README):

```
3e4b9c39526a9733c1d8f921cff7dd00d976759cd3f90cadf57d5147712d28a4
```

The source patch (`tp4opt_B3_plus_Psmall_combined.patch`, sha256
`e783e0afcf093fd41122b703f3dd0fd550a97e3fd2ec68ee80690f49347ecd68`) differs
from the applied diff only by two line-joins made by the repo's `black`
pre-commit hook (`model_hook.py`, `dsa_indexer.py`); the AST of all 8 files is
identical between the two.

## Log lines that prove it is active

Rank 0, at startup:

```
DSA prefill CP: keeping Backend.BREAKABLE prefill CUDA graph for TP-only extends <= SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS=2048; CP-sharded batches run eager; capture buckets=[...]
Breakable prefill CUDA graph kept under context parallel: extends <= 2048 tokens run TP-only and replay the graph; CP-sharded batches stay eager (buckets clamped to <= 2048).
```

Rank 0, on the first short extend (B3; logged once per process):

```
DSA prefill CP: extend batch with <N> new tokens <= SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS=2048 runs TP-only (no CP sharding)
```

On every graph replay (P-small):

```
prefill BCG replay bucket=<B> tokens=<N>
```

## Measured effects (staging, engine-side, same flags, B200 x4)

| Metric | Baseline (v0.5.19) | Patched |
|---|---|---|
| c=1 first token, hot 10-14k prefix (hit ~0.98) | 177 ms | 72 ms |
| Prefill-step p50, extends <= 2048 new tokens | - | -13% to -20% |
| Long cold prefills | unchanged | unchanged |
| Loaded p95 | unchanged | unchanged |
| EAGLE acceptance length | 4.61 | 4.56 |
| CUDA / NCCL faults over >6k requests | 0 | 0 |

Matched cache-hit rate (staging, engine-side, B200 x4, dense prod corpus):

| Run | Metric | Baseline (v0.5.19) | Patched | Delta |
|---|---|---|---|---|
| c=1, hit 0.571 | TTFT p50 / p95 | 182 / 405 ms | 183 / 402 ms | +-0 |
| c=1, hit 0.571 | E2E p95 | 962 ms | 916 ms | -5% |
| c=1, hit 0.571 | rps | - | - | +5% |
| c=2, hit 0.571 | TTFT p50 / p95 | 511 / 843 ms | 386 / 754 ms | -25% / -11% |
| c=2, hit 0.571 | E2E p95 | 1457 ms | 1425 ms | -2% |
| c=2, hit 0.571 | rps | - | - | +9% |

Output parity: `finish_reason`, `cached_tokens`, `tool_calls` and SSE structure
exact (40/40 paired requests); reasoning text differs within the
control-vs-control nondeterminism band (NVFP4 MoE + EAGLE).

## Known gaps (not fixed here)

- `dsa_prefill_cp_skip_short_extend` is not DSA-gated: with the env var set it
  would also flip the CP gates for other interleave-CP DSA/MLA models.
- The per-batch decision is rank-local (`extend_seq_lens_cpu` of the local
  batch); it is unsafe with DP-attention + CP, where ranks may disagree.
- Targets v0.5.19 only. It does **not** apply to current `main`: 7 of 8 files
  conflict, the CP-v1 gates were removed upstream and `main` has two CP gates,
  not three.
- Threshold 2048 covers only ~14% of prefill steps at production cache-hit
  rates (~85% are long CP-sharded prefills, unchanged); it was not swept.
- Graph memory cost (buckets up to 2048 tokens) was not measured in MB; no OOM
  at `--mem-fraction-static 0.85`.

## Review Q&A

| # | Question | Answer / evidence | Status |
|---|---|---|---|
| 1 | Output same as eager/CP path? | Structural parity exact (finish_reason, cached_tokens, tool_calls, SSE shape, 40/40); reasoning text differs within control-vs-control nondeterminism band (NVFP4 MoE + EAGLE). No bitwise test; upstream has none for prefill graphs either. | answered, not bitwise |
| 2 | Model quality changed? | Same weights/KV/sampling; only partitioning + graph replay change. Accept len 4.56 vs 4.61. No formal eval run. | partially open |
| 3 | Why safe when B2 crashed? | B2 flipped 2 of 3 CP gates -> DSA metadata sized for 1/4 tokens, OOB writes. B3 flips all 3 together; 0 CUDA/NCCL faults over >6k requests incl. forced cold-CP->short-extend transitions. | answered |
| 4 | Why does upstream disable this? | PR #23351 (PCG only when not CP), PR #29458 (CP all_gather replay size mismatch). Neither applies to fully TP-only batches; CP-sharded batches stay eager here. | answered |
| 5 | Other models/shapes? | Not DSA-gated (would flip other interleave-CP DSA/MLA models); rank-local decision -> unsafe with DP-attention + CP. Long-cold/RAG: no effect (and at prod hit ~0.57-0.66 that is ~85% of steps); high batch: benefit fades. | open |
| 6 | EAGLE interaction? | Draft/verify untouched; acceptance unchanged; prefill graphs capture target model only. | answered |
| 7 | Kill switch? | Env unset / flag off = byte-identical engine. | answered |
| 8 | What it buys? | c=1 first token 177->72 ms on hot prefix (hit ~0.98); short-extend prefill step -13...-20%; at matched prod hit 0.571: c=1 TTFT +-0, E2E p95 -5%, rps +5%; c=2 TTFT p50/p95 -25%/-11%, E2E p95 -2%, rps +9%; long cold prefills unchanged. | answered |
| 9 | Graph memory cost? | Buckets up to 2048 tokens; no OOM at mem 0.85; exact MB not recorded. | partially open |
| 10 | Applies to main? | No - 7/8 files conflict; main has 2 CP gates. v0.5.19 only. | answered |
| 11 | Threshold 2048? | Covers ~14% of prefill steps at prod cache-hit rates (186/1350 in the c=1 ladder; ~85% of steps are long CP-sharded prefills); not swept. | open |
