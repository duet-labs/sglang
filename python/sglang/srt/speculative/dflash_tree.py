"""Tree proposals for DFLASH (DFlash2 selector drafts).

The selector scores K x K transitions between adjacent proposal slots
(``scores[b, e, p, c]``: candidate ``c`` at predicted position ``e`` after
candidate ``p`` at ``e - 1``; ``scores[b, 0, 0, :]`` is the row after the verified
anchor). The chain draft walks one path through that lattice. This module
instead picks the ``num_nodes - 1`` highest prefix-mass lattice nodes (best
first) and lays them out as an EAGLE-shaped tree, so the stock tree verify
kernels (full attention ``custom_mask``, GDN/KDA ``retrieve_*`` recurrences,
``verify_tree_greedy`` / ``tree_speculative_sampling_target_only``) run as-is.

Node 0 is the anchor (root). Nodes are numbered in selection order, so every
parent index is smaller than its children, which the tree recurrence kernels
require. Layouts follow ``sgl_build_tree_kernel_efficient``:
``retrieve_index[b, i] = b * N + i``; ``retrieve_next_token[b, p]`` is the
lowest-numbered child of ``p``; ``retrieve_next_sibling[b, i]`` the next
higher-numbered child of the same parent; ``positions[b * N + i] = seq_len[b]
+ depth(i)``; ``tree_mask`` is the FULL_MASK layout (per request a
``[N, seq_len + N]`` bool block; prefix columns True; tree columns True for the
node itself and its ancestors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

NEG_INF = float("-inf")


@dataclass
class DFlashTree:
    num_nodes: int
    draft_tokens: torch.Tensor  # [bs * N] int64, node 0 = anchor
    positions: torch.Tensor  # [bs * N] int64
    tree_mask: torch.Tensor  # flat bool, FULL_MASK layout
    retrieve_index: torch.Tensor  # [bs, N] int64
    retrieve_next_token: torch.Tensor  # [bs, N] int64
    retrieve_next_sibling: torch.Tensor  # [bs, N] int64
    parents: torch.Tensor  # [bs, N] int64, -1 for the root
    depth: torch.Tensor  # [bs, N] int64, 0 for the root
    node_pos: torch.Tensor  # [bs, N] int64, lattice position e (= depth - 1), -1 root
    node_cand: torch.Tensor  # [bs, N] int64, candidate rank c in the lattice, -1 root
    log_mass: torch.Tensor  # [bs, N] float32, cumulative log-prob, 0 root
    ancestor_mask: torch.Tensor  # [bs, N, N] bool, [x, j] = j is x or an ancestor of x
    pre_node: Optional[torch.Tensor] = None  # [bs, N] int64 preorder node ids (-1 pad)
    pre_pop: Optional[torch.Tensor] = None  # [bs, N] int64 stack pops before visiting pre_node[t]


def _select_nodes(
    *,
    logp: torch.Tensor,
    num_nodes: int,
    force_spine: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Best-first node selection. Returns parents, depth, node_pos, node_cand,
    log_mass, each [bs, N]."""
    bs, P, K, _ = logp.shape
    device = logp.device
    N = num_nodes
    bidx = torch.arange(bs, device=device)
    parents = torch.full((bs, N), -1, dtype=torch.int64, device=device)
    depth = torch.zeros((bs, N), dtype=torch.int64, device=device)
    node_pos = torch.full((bs, N), -1, dtype=torch.int64, device=device)
    node_cand = torch.full((bs, N), -1, dtype=torch.int64, device=device)
    log_mass = torch.zeros((bs, N), dtype=torch.float32, device=device)
    # cand_mass[b, n, c]: prefix mass of the (not yet selected) child c of node n.
    cand_mass = torch.full((bs, N, K), NEG_INF, dtype=torch.float32, device=device)
    cand_mass[:, 0, :] = logp[:, 0, 0, :]
    num_forced = min(P, N - 1) if force_spine else 0
    for m in range(1, N):
        if m <= num_forced:
            # Greedy spine: extend the previous spine node (m - 1) by its best child.
            row = cand_mass[:, m - 1, :]
            c_star = row.argmax(dim=-1)
            n_star = torch.full_like(c_star, m - 1)
            mass = row[bidx, c_star]
        else:
            flat = cand_mass.view(bs, -1).argmax(dim=-1)
            n_star = flat // K
            c_star = flat % K
            mass = cand_mass.view(bs, -1)[bidx, flat]
        e = depth[bidx, n_star]  # lattice position of the new node
        parents[:, m] = n_star
        depth[:, m] = e + 1
        node_pos[:, m] = e
        node_cand[:, m] = c_star
        log_mass[:, m] = mass
        cand_mass[bidx, n_star, c_star] = NEG_INF
        # Children of the new node live at lattice position e + 1 (if any).
        has_child = e + 1 < P
        e_next = torch.clamp(e + 1, max=P - 1)
        child_rows = logp[bidx, e_next, c_star, :] + mass[:, None]
        cand_mass[:, m, :] = torch.where(has_child[:, None], child_rows, NEG_INF)
    return parents, depth, node_pos, node_cand, log_mass


def _ancestor_mask(parents: torch.Tensor) -> torch.Tensor:
    bs, N = parents.shape
    device = parents.device
    anc = torch.eye(N, dtype=torch.bool, device=device).expand(bs, N, N).clone()
    cur = torch.arange(N, device=device).expand(bs, N).clone()
    bidx = torch.arange(bs, device=device)[:, None].expand(bs, N)
    xidx = torch.arange(N, device=device)[None, :].expand(bs, N)
    for _ in range(N - 1):
        cur = parents[bidx, cur.clamp(min=0)]
        valid = cur >= 0
        anc[bidx[valid], xidx[valid], cur[valid]] = True
        if not bool(valid.any()):
            break
    return anc


def _retrieve_links(parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bs, N = parents.shape
    device = parents.device
    next_token = torch.full((bs, N), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((bs, N), -1, dtype=torch.int64, device=device)
    bidx = torch.arange(bs, device=device)
    for i in range(N - 1, 0, -1):
        p = parents[:, i]
        prev = next_token[bidx, p]
        next_token[bidx, p] = i
        next_sibling[:, i] = prev
    return next_token, next_sibling


def _preorder_host(next_token, next_sibling, parents):
    """Reference preorder (DFS) walk; pre_pop[t] = pops before visiting pre_node[t]."""
    bs, N = parents.shape
    nt, ns, pc = next_token.cpu().tolist(), next_sibling.cpu().tolist(), parents.cpu().tolist()
    pre_node = torch.full((bs, N), -1, dtype=torch.int64)
    pre_pop = torch.zeros((bs, N), dtype=torch.int64)
    for b in range(bs):
        cur, depth, t = 0, 0, 0
        while cur >= 0 and t < N:
            pre_node[b, t] = cur
            if nt[b][cur] >= 0:
                depth += 1
                cur = nt[b][cur]
            else:
                nxt, pops = ns[b][cur], 0
                while nxt < 0 and depth > 0:
                    depth -= 1
                    pops += 1
                    cur = pc[b][cur]
                    nxt = ns[b][cur]
                cur = nxt
                if t + 1 < N:
                    pre_pop[b, t + 1] = pops
            t += 1
    return pre_node.to(parents.device), pre_pop.to(parents.device)


def build_dflash_tree(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_nodes: int,
    force_spine: bool = False,
    seq_lens_sum: Optional[int] = None,
    tree_mask_buf: Optional[torch.Tensor] = None,
) -> DFlashTree:
    bs, P, K = candidate_ids.shape
    N = int(num_nodes)
    assert scores.shape == (bs, P, K, K), scores.shape
    assert N >= 2
    device = candidate_ids.device
    logp = torch.log_softmax(scores.float(), dim=-1)
    parents, depth, node_pos, node_cand, log_mass = _select_nodes(
        logp=logp, num_nodes=N, force_spine=force_spine
    )
    bidx = torch.arange(bs, device=device)[:, None].expand(bs, N)
    tokens = candidate_ids[bidx, node_pos.clamp(min=0), node_cand.clamp(min=0)]
    tokens[:, 0] = anchor_ids.to(tokens.dtype)
    seq_lens = seq_lens.to(torch.int64)
    positions = (seq_lens[:, None] + depth).reshape(-1)
    retrieve_index = (
        torch.arange(bs, device=device)[:, None] * N
        + torch.arange(N, device=device)[None, :]
    )
    next_token, next_sibling = _retrieve_links(parents)
    anc = _ancestor_mask(parents)

    if seq_lens_sum is None:
        seq_lens_sum = int(seq_lens.sum().item())
    total = seq_lens_sum * N + N * N * bs
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        assert tree_mask.numel() >= total
        tree_mask[:total].fill_(True)
    else:
        tree_mask = torch.ones((total,), dtype=torch.bool, device=device)
    prefix = torch.cumsum(seq_lens, dim=0) - seq_lens  # [bs]
    base = N * N * torch.arange(bs, device=device) + prefix * N  # [bs]
    row_len = seq_lens + N  # [bs]
    x = torch.arange(N, device=device)
    idx = (
        base[:, None, None]
        + x[None, :, None] * row_len[:, None, None]
        + seq_lens[:, None, None]
        + x[None, None, :]
    )  # [bs, N(x), N(j)]
    tree_mask.view(-1)[idx.reshape(-1)] = anc.reshape(-1)

    pre_node, pre_pop = _preorder_host(next_token, next_sibling, parents)
    return DFlashTree(
        num_nodes=N,
        draft_tokens=tokens.reshape(-1),
        positions=positions,
        tree_mask=tree_mask,
        retrieve_index=retrieve_index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
        parents=parents,
        depth=depth,
        node_pos=node_pos,
        node_cand=node_cand,
        log_mass=log_mass,
        ancestor_mask=anc,
        pre_node=pre_node,
        pre_pop=pre_pop,
    )


# ---------------------------------------------------------------------------
# Fused Triton builder: one program per request does the best-first selection,
# the child/sibling links, positions and the tree columns of the FULL mask.
# Same contract as `build_dflash_tree` (which stays the reference / CPU path).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


if _HAS_TRITON:

    @triton.jit
    def _row_logp(scores_ptr, b, e, p, P: tl.constexpr, K: tl.constexpr):
        offs = tl.arange(0, K)
        v = tl.load(scores_ptr + ((b * P + e) * K + p) * K + offs).to(tl.float32)
        m = tl.max(v, 0)
        lse = m + tl.log(tl.sum(tl.exp(v - m), 0))
        return v - lse

    @triton.jit
    def _dflash_tree_kernel(
        scores_ptr,  # [bs, P, K, K]
        cand_ptr,  # [bs, P, K] int64
        anchor_ptr,  # [bs] int64
        seq_lens_ptr,  # [bs] int64
        seq_prefix_ptr,  # [bs] int64 (exclusive cumsum of seq_lens)
        ws_ptr,  # [bs, N*K] fp32 scratch
        tokens_ptr,  # [bs*N] int64
        positions_ptr,  # [bs*N] int64
        parents_ptr,  # [bs, N] int64
        depth_ptr,  # [bs, N] int64
        node_pos_ptr,  # [bs, N] int64
        node_cand_ptr,  # [bs, N] int64
        mass_ptr,  # [bs, N] fp32
        next_token_ptr,  # [bs, N] int64
        next_sibling_ptr,  # [bs, N] int64
        pre_node_ptr,  # [bs, N] int64
        pre_pop_ptr,  # [bs, N] int64
        retrieve_index_ptr,  # [bs, N] int64
        tree_mask_ptr,  # flat bool, FULL layout, prefix columns pre-filled True
        anc_ptr,  # [bs, N, N] bool
        P: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        NKP2: tl.constexpr,
        NP2: tl.constexpr,
        NUM_FORCED: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        neg_inf = float("-inf")
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, NP2)
        mn = offs_n < N
        offs_nk = tl.arange(0, NKP2)
        ws = ws_ptr + b * (N * K)
        tl.store(ws + offs_nk, tl.full([NKP2], neg_inf, tl.float32), mask=offs_nk < N * K)
        tl.store(ws + offs_k, _row_logp(scores_ptr, b, 0, 0, P, K))
        node = b * N
        tl.store(parents_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(depth_ptr + node + offs_n, tl.zeros([NP2], tl.int64), mask=mn)
        tl.store(node_pos_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(node_cand_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(mass_ptr + node + offs_n, tl.zeros([NP2], tl.float32), mask=mn)
        tl.store(next_token_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(next_sibling_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(retrieve_index_ptr + node + offs_n, node + offs_n, mask=mn)
        tl.store(tokens_ptr + node, tl.load(anchor_ptr + b))
        for m in tl.static_range(1, N):
            tl.debug_barrier()
            if m <= NUM_FORCED:
                row = tl.load(ws + (m - 1) * K + offs_k)
                c_star = tl.argmax(row, 0).to(tl.int64)
                mass = tl.max(row, 0)
                n_star = tl.full([], m - 1, tl.int64)
            else:
                vals = tl.load(ws + offs_nk, mask=offs_nk < N * K, other=neg_inf)
                flat = tl.argmax(vals, 0).to(tl.int64)
                mass = tl.max(vals, 0)
                n_star = flat // K
                c_star = flat % K
            e = tl.load(depth_ptr + node + n_star)
            tl.store(parents_ptr + node + m, n_star)
            tl.store(depth_ptr + node + m, e + 1)
            tl.store(node_pos_ptr + node + m, e)
            tl.store(node_cand_ptr + node + m, c_star)
            tl.store(mass_ptr + node + m, mass)
            tl.store(tokens_ptr + node + m, tl.load(cand_ptr + (b * P + e) * K + c_star))
            tl.store(ws + n_star * K + c_star, neg_inf)
            has_child = (e + 1) < P
            e_next = tl.minimum(e + 1, P - 1)
            child = _row_logp(scores_ptr, b, e_next, c_star, P, K) + mass
            child = tl.where(has_child, child, neg_inf)
            tl.store(ws + m * K + offs_k, child)
        # child / sibling links (lowest-numbered child first)
        for i in tl.static_range(N - 1, 0, -1):
            tl.debug_barrier()
            p = tl.load(parents_ptr + node + i)
            prev = tl.load(next_token_ptr + node + p)
            tl.store(next_token_ptr + node + p, tl.full([], i, tl.int64))
            tl.store(next_sibling_ptr + node + i, prev)
        tl.debug_barrier()
        # preorder walk for the DFS verify kernel
        tl.store(pre_node_ptr + node + offs_n, tl.full([NP2], -1, tl.int64), mask=mn)
        tl.store(pre_pop_ptr + node + offs_n, tl.zeros([NP2], tl.int64), mask=mn)
        cur = tl.full([], 0, tl.int64)
        dep = tl.full([], 0, tl.int64)
        for t in tl.static_range(N):
            if cur >= 0:
                tl.store(pre_node_ptr + node + t, cur)
                ch = tl.load(next_token_ptr + node + cur)
                if ch >= 0:
                    dep += 1
                    cur = ch
                else:
                    nxt = tl.load(next_sibling_ptr + node + cur)
                    pops = tl.full([], 0, tl.int64)
                    for _d in tl.static_range(P + 1):
                        if nxt < 0 and dep > 0:
                            dep -= 1
                            pops += 1
                            cur = tl.load(parents_ptr + node + cur)
                            nxt = tl.load(next_sibling_ptr + node + cur)
                    cur = nxt
                    if t + 1 < N:
                        tl.store(pre_pop_ptr + node + t + 1, pops)
        tl.debug_barrier()
        seq_len = tl.load(seq_lens_ptr + b)
        depth_v = tl.load(depth_ptr + node + offs_n, mask=mn, other=0)
        tl.store(positions_ptr + node + offs_n, seq_len + depth_v, mask=mn)
        # tree columns of the FULL mask + ancestor matrix
        base = N * N * b + tl.load(seq_prefix_ptr + b) * N
        row_len = seq_len + N
        for x in tl.static_range(N):
            row = base + x * row_len + seq_len
            tl.store(tree_mask_ptr + row + offs_n, tl.zeros([NP2], tl.int1), mask=mn)
            tl.store(anc_ptr + (b * N + x) * N + offs_n, tl.zeros([NP2], tl.int1), mask=mn)
            tl.debug_barrier()
            cur = tl.full([], x, tl.int64)
            for _ in tl.static_range(P + 1):
                valid = cur >= 0
                safe = tl.maximum(cur, 0)
                tl.store(tree_mask_ptr + row + safe, tl.full([], 1, tl.int1), mask=valid)
                tl.store(anc_ptr + (b * N + x) * N + safe, tl.full([], 1, tl.int1), mask=valid)
                nxt = tl.load(parents_ptr + node + safe, mask=valid, other=-1)
                cur = tl.where(valid, nxt, -1)


def build_dflash_tree_triton(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_nodes: int,
    force_spine: bool = False,
    seq_lens_sum: Optional[int] = None,
    tree_mask_buf: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> DFlashTree:
    if not (_HAS_TRITON and candidate_ids.is_cuda):
        return build_dflash_tree(
            candidate_ids=candidate_ids, scores=scores, anchor_ids=anchor_ids,
            seq_lens=seq_lens, num_nodes=num_nodes, force_spine=force_spine,
            seq_lens_sum=seq_lens_sum, tree_mask_buf=tree_mask_buf,
        )
    bs, P, K = candidate_ids.shape
    N = int(num_nodes)
    assert scores.shape == (bs, P, K, K), scores.shape
    assert K == _next_pow2(K), "K must be a power of two"
    device = candidate_ids.device
    scores = scores.contiguous()
    candidate_ids = candidate_ids.contiguous().to(torch.int64)
    seq_lens = seq_lens.to(torch.int64)
    if seq_lens_sum is None:
        seq_lens_sum = int(seq_lens.sum().item())
    total = seq_lens_sum * N + N * N * bs
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        assert tree_mask.numel() >= total
        tree_mask[:total].fill_(True)
    else:
        tree_mask = torch.ones((total,), dtype=torch.bool, device=device)
    if workspace is None or workspace.numel() < bs * N * K:
        workspace = torch.empty((bs * N * K,), dtype=torch.float32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    tokens = torch.empty((bs * N,), **i64)
    positions = torch.empty((bs * N,), **i64)
    parents = torch.empty((bs, N), **i64)
    depth = torch.empty((bs, N), **i64)
    node_pos = torch.empty((bs, N), **i64)
    node_cand = torch.empty((bs, N), **i64)
    log_mass = torch.empty((bs, N), dtype=torch.float32, device=device)
    next_token = torch.empty((bs, N), **i64)
    next_sibling = torch.empty((bs, N), **i64)
    retrieve_index = torch.empty((bs, N), **i64)
    pre_node = torch.empty((bs, N), **i64)
    pre_pop = torch.empty((bs, N), **i64)
    anc = torch.empty((bs, N, N), dtype=torch.bool, device=device)
    seq_prefix = torch.cumsum(seq_lens, dim=0) - seq_lens
    num_forced = min(P, N - 1) if force_spine else 0
    _dflash_tree_kernel[(bs,)](
        scores, candidate_ids, anchor_ids.to(torch.int64).contiguous(), seq_lens.contiguous(),
        seq_prefix.contiguous(), workspace, tokens, positions, parents, depth, node_pos,
        node_cand, log_mass, next_token, next_sibling, pre_node, pre_pop, retrieve_index, tree_mask, anc,
        P=P, K=K, N=N, NKP2=_next_pow2(N * K), NP2=_next_pow2(N), NUM_FORCED=num_forced,
    )
    return DFlashTree(
        num_nodes=N, draft_tokens=tokens, positions=positions, tree_mask=tree_mask,
        retrieve_index=retrieve_index, retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling, parents=parents, depth=depth,
        node_pos=node_pos, node_cand=node_cand, log_mass=log_mass, ancestor_mask=anc,
        pre_node=pre_node, pre_pop=pre_pop,
    )
