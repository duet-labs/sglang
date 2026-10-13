# GLM-5.2: TP-only short extends + prefill CUDA graphs under prefill-CP (B3 + P-small)

Engine-side patch for GLM-5.2 (`GlmMoeDsa`, NVFP4) served with TP4 + EP4 +
prefill context parallelism. This branch is the port to upstream `main`
(base `7224fce`, 2026-10-12); the validated original targets v0.5.19 and lives
in `duet-labs/sglang` PR #1 (branch `devin/glm52-psmall-prefill-cp-tp-only-graphs`).

**Status: ported, not yet GPU-validated on `main`.** Every number below was
measured on the v0.5.19 branch. It is **not** an upstream candidate as-is;
see "Known gaps".

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
   new tokens is `<= SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS` (we use 2048),
   every per-batch CP gate returns "no CP" and the batch runs plain TP4. CP
   stays on for long / cold prefills. On `main` there are two such gates:
   - `InterleaveCPStrategy.can_apply` (`layers/cp/interleave.py`) - drives
     `is_cp_active`, hence `prepare_cp_forward` / `attn_cp_metadata`, the eager
     runner's CP boundary and the decode-attn-TP context;
   - `can_dsa_prefill_cp_interleave` (`layers/attention/dsa/utils.py`) - drives
     the DSA attention-metadata sharding and padding in `dsa_backend.py`
     (`cal_padded_tokens`, `pad_dsa_cache_seqlens`).

   Both consult the same helper, `dsa_prefill_cp_skip_short_extend`, so a batch
   is either fully sharded or fully unsharded. The derived checks
   (`dsa_use_prefill_cp`, `attn_cp_metadata is None`) follow from the first
   gate. On v0.5.19 there were three gates (the CP-v1 `can_dsa_cp_split` and
   `can_dsa_prefill_cp_round_robin_split` have since been removed upstream);
   flipping only two of the three (an earlier attempt, "B2") produced illegal
   memory accesses: the DSA metadata was sized for 1/4 of the tokens while
   `q`/`k`/`out_cache_loc` carried all of them.

2. **P-small - prefill CUDA graphs for TP-only batches under prefill-CP.**
   Upstream turns prefill CUDA graphs off for DSA models with prefill-CP
   (`arg_groups/model_hook.py`) and turns breakable graphs off under CP unless
   `supports_prefill_cp_bcg` (`arg_groups/cuda_graph_hook.py`), which does not
   cover the `dsa` interleave recipe. With B3 enabled this patch keeps
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
engine's behaviour is byte-identical to the base commit: every new code path
is behind `envs.SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS.get() > 0`.

## Validated configuration (v0.5.19 branch, PR #1)

- SGLang v0.5.19, B200 x4, GLM-5.2 (`GlmMoeDsa`) NVFP4.
- TP4 + EP4, `--enable-prefill-cp --attn-cp-size 4 --cp-strategy interleave`.
- EAGLE speculative decoding, steps/topk/draft-tokens = 5/1/6.
- `--mem-fraction-static 0.85`.
- `SGLANG_DSA_PREFILL_CP_MIN_EXTEND_TOKENS=2048`, `--cuda-graph-backend-prefill breakable`.

The same recipe has **not** been run on this `main` port yet.

## Building

From this branch: `pip install -e python` (plus the `sgl-kernel` / FlashInfer
wheels matching upstream `main` at the base commit). The patch touches
pure-Python files only, no kernel rebuild; `git diff main...HEAD -- python/ |
git apply --directory=<site-packages path containing sglang/>` also works on
top of an installed `main`-based `sglang` package of the same commit.

sha256 of the applied diff (`git diff main...HEAD -- python/`, i.e. the 8
engine files, excluding this README and the env-var docs row):

```
6e8b14f5e855d42b6bbd9fb5916d348488b5df91ca590cbcacdc6b3f127e18e1
```

The v0.5.19 source patch (`tp4opt_B3_plus_Psmall_combined.patch`, sha256
`e783e0afcf093fd41122b703f3dd0fd550a97e3fd2ec68ee80690f49347ecd68`) does not
apply to `main` (7 of 8 files conflict); this branch re-implements the same
logic against `main`'s two CP gates and current `PrefillCudaGraphRunner`.

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

## Measured effects (v0.5.19 branch; staging, engine-side, same flags, B200 x4)

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

- **Untested on `main`.** The port is by inspection of `main`'s CP code paths;
  it has not been built or run on a GPU. `main`-only paths that the v0.5.19
  validation never exercised include `enable_cp_tp_group_sharing`, the GLM
  `prepare_cp_inputs` CP boundary in the eager runner, DSA cache layer split,
  and the `dsv4` interleave BCG path (`supports_prefill_cp_bcg`), which must
  stay mutually exclusive with this hybrid.
- `dsa_prefill_cp_skip_short_extend` is not DSA-gated: with the env var set it
  would also flip the CP gates for other interleave-CP DSA/MLA models.
- The per-batch decision is rank-local (`extend_seq_lens_cpu` of the local
  batch); it is unsafe with DP-attention + CP, where ranks may disagree.
- Threshold 2048 covers only ~14% of prefill steps at production cache-hit
  rates (~85% are long CP-sharded prefills, unchanged); it was not swept.
- Graph memory cost (buckets up to 2048 tokens) was not measured in MB; no OOM
  at `--mem-fraction-static 0.85` on v0.5.19.

## Review Q&A

| # | Question | Answer / evidence | Status |
|---|---|---|---|
| 1 | Output same as eager/CP path? | Structural parity exact (finish_reason, cached_tokens, tool_calls, SSE shape, 40/40); reasoning text differs within control-vs-control nondeterminism band (NVFP4 MoE + EAGLE). No bitwise test; upstream has none for prefill graphs either. (v0.5.19) | answered, not bitwise |
| 2 | Model quality changed? | Same weights/KV/sampling; only partitioning + graph replay change. Accept len 4.56 vs 4.61. No formal eval run. | partially open |
| 3 | Why safe when B2 crashed? | B2 flipped 2 of 3 CP gates on v0.5.19 -> DSA metadata sized for 1/4 tokens, OOB writes. B3 flips every gate together; on `main` both gates (`InterleaveCPStrategy.can_apply`, `can_dsa_prefill_cp_interleave`) consult the same helper. 0 CUDA/NCCL faults over >6k requests on v0.5.19 incl. forced cold-CP->short-extend transitions; not yet re-run on `main`. | answered on v0.5.19, open on main |
| 4 | Why does upstream disable this? | PR #23351 (PCG only when not CP), PR #29458 (CP all_gather replay size mismatch). Neither applies to fully TP-only batches; CP-sharded batches stay eager here. | answered |
| 5 | Other models/shapes? | Not DSA-gated (would flip other interleave-CP DSA/MLA models); rank-local decision -> unsafe with DP-attention + CP. Long-cold/RAG: no effect (and at prod hit ~0.57-0.66 that is ~85% of steps); high batch: benefit fades. | open |
| 6 | EAGLE interaction? | Draft/verify untouched; acceptance unchanged; prefill graphs capture target model only. | answered |
| 7 | Kill switch? | Env unset / flag off = byte-identical engine. | answered |
| 8 | What it buys? | c=1 first token 177->72 ms on hot prefix (hit ~0.98); short-extend prefill step -13...-20%; at matched prod hit 0.571: c=1 TTFT +-0, E2E p95 -5%, rps +5%; c=2 TTFT p50/p95 -25%/-11%, E2E p95 -2%, rps +9%; long cold prefills unchanged. (v0.5.19) | answered |
| 9 | Graph memory cost? | Buckets up to 2048 tokens; no OOM at mem 0.85; exact MB not recorded. | partially open |
| 10 | Applies to main? | This branch is the `main` port (base `7224fce`). The v0.5.19 patch did not apply (7/8 files conflict; main has 2 CP gates, not 3), so the logic was re-implemented; not GPU-validated yet. | ported, untested |
| 11 | Threshold 2048? | Covers ~14% of prefill steps at prod cache-hit rates (186/1350 in the c=1 ladder; ~85% of steps are long CP-sharded prefills); not swept. | open |
