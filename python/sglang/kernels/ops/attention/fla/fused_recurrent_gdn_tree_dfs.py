"""DFS tree verify for GDN: walks the draft tree with an on-chip ancestor
stack instead of writing one recurrent-state snapshot per node.

Same math as fused_recurrent_gated_delta_rule_update_fwd_kernel; per node
h_node = update(h_parent, node); o_node = h_node . q_node. Nothing is written
except o; the accepted path is committed by replay (see dflash tree worker).
Tree layout: retrieve_next_token / retrieve_next_sibling / retrieve_parent
as in EAGLE (node 0 = root/anchor, -1 = none). MAX_DEPTH bounds the stack.
"""
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp


@triton.jit
def _stack_push(h, depth, s0, s1, s2, s3, s4, s5, s6, s7):
    s0 = tl.where(depth == 0, h, s0)
    s1 = tl.where(depth == 1, h, s1)
    s2 = tl.where(depth == 2, h, s2)
    s3 = tl.where(depth == 3, h, s3)
    s4 = tl.where(depth == 4, h, s4)
    s5 = tl.where(depth == 5, h, s5)
    s6 = tl.where(depth == 6, h, s6)
    s7 = tl.where(depth == 7, h, s7)
    return s0, s1, s2, s3, s4, s5, s6, s7


@triton.jit
def _stack_peek(depth, h0, s0, s1, s2, s3, s4, s5, s6, s7):
    # state of the ancestor at depth-1 (h0 when depth == 0)
    h = tl.where(depth == 1, s0, h0)
    h = tl.where(depth == 2, s1, h)
    h = tl.where(depth == 3, s2, h)
    h = tl.where(depth == 4, s3, h)
    h = tl.where(depth == 5, s4, h)
    h = tl.where(depth == 6, s5, h)
    h = tl.where(depth == 7, s6, h)
    h = tl.where(depth == 8, s7, h)
    return h


@triton.jit
def fused_recurrent_gdn_tree_dfs_kernel(
    q, k, v, g, beta, o,
    h0_source, h0_indices,
    next_token, next_sibling, parent,
    pre_node, pre_pop, row_map, h0_ptrs,
    q_prev, k_prev, v_prev, g_prev, beta_prev, prev_row, prev_chain,
    scale, stride_h0, sq, sk, sv,
    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    S: tl.constexpr,
    USE_PREORDER: tl.constexpr,
    WRITE_FINAL_STATE: tl.constexpr,
    USE_ROW_MAP: tl.constexpr,
    USE_PTR_TABLE: tl.constexpr,
    USE_PREFIX: tl.constexpr,
    LEAF_SKIP: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)  # grid x carries B*S*HV (no 65535 limit)
    i_n, i_s, i_hv = i_nh // (S * HV), (i_nh // HV) % S, i_nh % HV
    if USE_ROW_MAP:
        i_row = tl.load(row_map + i_n).to(tl.int64)  # storage row of this logical row
    else:
        i_row = i_n.to(tl.int64)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    h0 = tl.zeros([BV, BK], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    # int64 offsets: layer-stacked pools exceed 2^31 elements
    h0_off = tl.maximum(idx, 0).to(tl.int64) * stride_h0.to(tl.int64) + (i_hv * K * V).to(tl.int64)
    if USE_PTR_TABLE:
        # per-row base pointer (e.g. that row's layer state tensor); idx is the slot within it
        h0_base = tl.load(h0_ptrs + i_n).to(tl.pointer_type(h0_source.dtype.element_ty))
        p_h0 = h0_base + h0_off + o_v[:, None] * K + o_k[None, :]
    else:
        p_h0 = h0_source + h0_off + o_v[:, None] * K + o_k[None, :]
    if idx >= 0:
        h0 += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    s0 = tl.zeros([BV, BK], dtype=tl.float32)
    s1 = tl.zeros([BV, BK], dtype=tl.float32)
    s2 = tl.zeros([BV, BK], dtype=tl.float32)
    s3 = tl.zeros([BV, BK], dtype=tl.float32)
    s4 = tl.zeros([BV, BK], dtype=tl.float32)
    s5 = tl.zeros([BV, BK], dtype=tl.float32)
    s6 = tl.zeros([BV, BK], dtype=tl.float32)
    s7 = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_PREFIX:
        # deferred commit: replay the previous step's accepted path (a chain in the
        # previous stash) from the persistent state, then commit it before the tree
        prow = tl.load(prev_row + i_n).to(tl.int64)
        if prow >= 0:
            b_c = h0
            for t in range(0, T):
                node = tl.load(prev_chain + i_n * T + t).to(tl.int64)
                if node >= 0:
                    prw = (prow * T + node).to(tl.int64)
                    c_k = tl.load(k_prev + prw * H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
                    c_v = tl.load(v_prev + prw * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
                    c_g = tl.load(g_prev + prw * HV + i_hv).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        c_k = c_k / (tl.sqrt(tl.sum(c_k * c_k) + 1e-6))
                    b_c *= exp(c_g)
                    c_v -= tl.sum(b_c * c_k[None, :], 1)
                    if IS_BETA_HEADWISE:
                        c_beta = tl.load(beta_prev + prw * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
                    else:
                        c_beta = tl.load(beta_prev + prw * HV + i_hv).to(tl.float32)
                    c_v *= c_beta
                    b_c += c_v[:, None] * c_k[None, :]
            if idx >= 0:
                tl.store(p_h0, b_c.to(p_h0.dtype.element_ty), mask=mask_h)
            h0 = b_c

    tree = i_n * T  # per-tree arrays (next_token / next_sibling / parent)
    seg = (i_n * S + i_s) * T  # per-segment preorder arrays (pre_node / pre_pop)
    cur = tl.full([], 0, tl.int64)
    depth = tl.full([], 0, tl.int64)
    b_par = h0
    b_h = h0
    for t in range(0, T):
        if USE_PREORDER:
            cur = tl.load(pre_node + seg + t).to(tl.int64)
            npop = tl.load(pre_pop + seg + t).to(tl.int64)
            if npop > 0:
                depth -= npop
                b_par = _stack_peek(depth, h0, s0, s1, s2, s3, s4, s5, s6, s7)
        if cur >= 0:
            row = (i_row * T + cur).to(tl.int64)
            p_q = q + row * sq + i_h * K + o_k
            p_k = k + row * sk + i_h * K + o_k
            p_v = v + row * sv + i_hv * V + o_v
            p_o = o + row * HV * V + i_hv * V + o_v
            p_g = g + row * HV + i_hv
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
            b_q = b_q * scale
            if IS_BETA_HEADWISE:
                b_beta = tl.load(beta + row * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
            else:
                b_beta = tl.load(beta + row * HV + i_hv).to(tl.float32)
            child = tl.load(next_token + tree + cur).to(tl.int64)
            if LEAF_SKIP and child < 0:
                # leaf: output from the parent state without materialising this
                # node's state: o = e^g (h q) + v' (k.q),  v' = beta (v - e^g (h k))
                e_g = exp(b_g)
                hq = tl.sum(b_par * b_q[None, :], 1)
                hk = tl.sum(b_par * b_k[None, :], 1)
                b_v = (b_v - e_g * hk) * b_beta
                b_o = e_g * hq + b_v * tl.sum(b_k * b_q)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            else:
                b_h = b_par
                b_h *= exp(b_g)
                b_v -= tl.sum(b_h * b_k[None, :], 1)
                b_v *= b_beta
                b_h += b_v[:, None] * b_k[None, :]
                b_o = tl.sum(b_h * b_q[None, :], 1)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            # advance in preorder
            if child >= 0:
                s0, s1, s2, s3, s4, s5, s6, s7 = _stack_push(b_h, depth, s0, s1, s2, s3, s4, s5, s6, s7)
                depth += 1
                cur = child
                b_par = b_h
            elif USE_PREORDER:
                pass
            else:
                nxt = tl.load(next_sibling + tree + cur).to(tl.int64)
                popped = tl.full([], 0, tl.int64)
                for _d in tl.static_range(MAX_DEPTH):
                    if nxt < 0 and depth > 0:
                        depth -= 1
                        popped += 1
                        cur = tl.load(parent + tree + cur).to(tl.int64)
                        nxt = tl.load(next_sibling + tree + cur).to(tl.int64)
                if popped > 0:
                    b_par = _stack_peek(depth, h0, s0, s1, s2, s3, s4, s5, s6, s7)
                cur = nxt  # -1 when the tree is exhausted
    if WRITE_FINAL_STATE:
        # commit-by-replay: the preorder was the accepted chain; b_h is its end state
        if idx >= 0:
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gdn_tree_dfs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor, initial_state_indices: torch.Tensor,
    retrieve_next_token: torch.Tensor, retrieve_next_sibling: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    max_depth: int = 8,
    pre_node: Optional[torch.Tensor] = None,
    pre_pop: Optional[torch.Tensor] = None,
    write_final_state: bool = False,
    row_map: Optional[torch.Tensor] = None,
    h0_ptrs: Optional[torch.Tensor] = None,
    prev_inputs: Optional[tuple] = None,
    prev_row: Optional[torch.Tensor] = None,
    prev_chain: Optional[torch.Tensor] = None,
    row_strides: Optional[tuple] = None,
    leaf_skip: bool = False,
) -> torch.Tensor:
    """q/k: [B, T, H, K]; v/g/beta: [B, T, HV, V] / [B, T, HV] / [B, T, HV(,V)].
    Tree arrays: [B, T] int. Returns o [B, T, HV, V]; no state is written."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if row_map is not None:
        B = row_map.shape[0]
    S = pre_node.shape[1] if (pre_node is not None and pre_node.dim() == 3) else 1
    BK = triton.next_power_of_2(K)
    assert BK == K, "K must be a power of two"
    import os
    BV = min(triton.next_power_of_2(V), int(os.environ.get('SGLANG_DFS_BV', '4')))
    NV = triton.cdiv(V, BV)
    assert max_depth <= 8, "register stack supports depth <= 8"
    o = torch.empty_like(v)
    grid = (B * S * HV, NV)
    fused_recurrent_gdn_tree_dfs_kernel[grid](
        q, k, v, g, beta, o,
        initial_state_source, initial_state_indices,
        retrieve_next_token.contiguous(), retrieve_next_sibling.contiguous(),
        retrieve_parent_token.contiguous(),
        pre_node if pre_node is not None else retrieve_parent_token,
        pre_pop if pre_pop is not None else retrieve_parent_token,
        row_map if row_map is not None else retrieve_parent_token,
        h0_ptrs if h0_ptrs is not None else retrieve_parent_token,
        *(prev_inputs if prev_inputs is not None else (q, k, v, g, beta)),
        prev_row if prev_row is not None else retrieve_parent_token,
        prev_chain if prev_chain is not None else retrieve_parent_token,
        scale=scale, stride_h0=initial_state_source.stride(0),
        sq=row_strides[0] if row_strides else H * K, sk=row_strides[1] if row_strides else H * K,
        sv=row_strides[2] if row_strides else HV * V, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        MAX_DEPTH=max_depth,
        S=S,
        USE_PREORDER=pre_node is not None,
        WRITE_FINAL_STATE=write_final_state,
        USE_ROW_MAP=row_map is not None,
        USE_PTR_TABLE=h0_ptrs is not None,
        USE_PREFIX=prev_chain is not None,
        LEAF_SKIP=leaf_skip and not write_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=int(os.environ.get('SGLANG_DFS_WARPS', '1')),
    )
    return o


def build_dfs_segments(pre_node: torch.Tensor, pre_pop: torch.Tensor, parents: torch.Tensor, depth: torch.Tensor, S: int):
    """Split each tree's preorder [B, T] into S contiguous chunks; segment s =
    ancestors(first chunk node) + chunk. Returns pre_node_seg / pre_pop_seg [B, S, T]."""
    B, T = pre_node.shape
    pn, pp, pc, dp = pre_node.cpu().tolist(), pre_pop.cpu().tolist(), parents.cpu().tolist(), depth.cpu().tolist()
    seg_node = torch.full((B, S, T), -1, dtype=torch.int64)
    seg_pop = torch.zeros((B, S, T), dtype=torch.int64)
    for b in range(B):
        order = [x for x in pn[b] if x >= 0]
        n = len(order)
        bounds = [round(i * n / S) for i in range(S + 1)]
        for s_ in range(S):
            lo, hi = bounds[s_], bounds[s_ + 1]
            if lo >= hi:
                continue
            first = order[lo]
            path = []
            p = pc[b][first]
            while p >= 0:
                path.append(p)
                p = pc[b][p]
            path.reverse()
            seq = path + order[lo:hi]
            assert len(seq) <= T, "segment longer than T"
            pops = [0] * len(path) + [0] + [pp[b][t] for t in range(lo + 1, hi)]
            for t, x in enumerate(seq):
                seg_node[b, s_, t] = x
                seg_pop[b, s_, t] = pops[t]
    return seg_node.to(pre_node.device), seg_pop.to(pre_node.device)
