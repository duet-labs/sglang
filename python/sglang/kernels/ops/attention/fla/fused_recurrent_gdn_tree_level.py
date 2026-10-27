# Depth-parallel (level-by-level) tree verify for the gated delta rule.
# One launch per tree depth: every node of that depth is an independent
# program reading its parent's state from a ping-pong scratch (depth-1) and
# writing its own state to scratch (depth). Serial length per layer = depth,
# independent of the number of tree nodes; no per-node persistent snapshots.
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def fused_recurrent_gdn_tree_level_kernel(
    q, k, v, g, beta, o,
    h0_source, h0_indices, stride_h0,
    level_nodes, level_pslot,  # [B, W] for this depth: node id (-1 pad), parent slot
    scratch_in, scratch_out,  # [B, W, HV, V, K] fp32
    q_prev, k_prev, v_prev, g_prev, beta_prev, prev_row, prev_chain,
    scale, sq, sk, sv,
    T: tl.constexpr, W: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_ROOT: tl.constexpr, USE_PREFIX: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_BETA_HEADWISE: tl.constexpr,
):
    i_bw, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_w = i_bw // W, i_bw % W
    i_h = i_hv // (HV // H)
    node = tl.load(level_nodes + i_n * W + i_w).to(tl.int64)
    if node >= 0:
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]
        if IS_ROOT:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            h0_off = tl.maximum(idx, 0).to(tl.int64) * stride_h0.to(tl.int64) + (i_hv * K * V).to(tl.int64)
            p_h0 = h0_source + h0_off + o_v[:, None] * K + o_k[None, :]
            b_par = tl.zeros([BV, BK], dtype=tl.float32)
            if idx >= 0:
                b_par += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
            if USE_PREFIX:
                prow = tl.load(prev_row + i_n).to(tl.int64)
                if prow >= 0:
                    for t in range(0, T):
                        pn = tl.load(prev_chain + i_n * T + t).to(tl.int64)
                        if pn >= 0:
                            prw = (prow * T + pn).to(tl.int64)
                            c_k = tl.load(k_prev + prw * H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
                            c_v = tl.load(v_prev + prw * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
                            c_g = tl.load(g_prev + prw * HV + i_hv).to(tl.float32)
                            if USE_QK_L2NORM_IN_KERNEL:
                                c_k = c_k / (tl.sqrt(tl.sum(c_k * c_k) + 1e-6))
                            b_par *= tl.exp(c_g)
                            c_v -= tl.sum(b_par * c_k[None, :], 1)
                            if IS_BETA_HEADWISE:
                                c_beta = tl.load(beta_prev + prw * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
                            else:
                                c_beta = tl.load(beta_prev + prw * HV + i_hv).to(tl.float32)
                            c_v *= c_beta
                            b_par += c_v[:, None] * c_k[None, :]
                    if idx >= 0:
                        tl.store(p_h0, b_par.to(p_h0.dtype.element_ty), mask=mask_h)
        else:
            pslot = tl.load(level_pslot + i_n * W + i_w).to(tl.int64)
            p_in = scratch_in + ((i_n * W + pslot) * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            b_par = tl.load(p_in, mask=mask_h, other=0).to(tl.float32)
        row = (i_n * T + node).to(tl.int64)
        b_q = tl.load(q + row * sq + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + row * sk + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(v + row * sv + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(g + row * HV + i_hv).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_par
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(beta + row * HV * V + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(beta + row * HV + i_hv).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        p_o = o + row * HV * V + i_hv * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_out = scratch_out + ((i_n * W + i_w) * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
        tl.store(p_out, b_h, mask=mask_h)


def build_level_metadata(parents: torch.Tensor, depth: torch.Tensor, W: int):
    """parents/depth: [B, N] (root has parent -1, depth 0). Returns
    level_nodes [B, D, W], level_pslot [B, D, W] (-1 padded) and D."""
    B, N = parents.shape
    D = int(depth.max().item()) + 1
    pc, dc = parents.cpu().tolist(), depth.cpu().tolist()
    nodes = torch.full((B, D, W), -1, dtype=torch.int64)
    pslot = torch.full((B, D, W), -1, dtype=torch.int64)
    for b in range(B):
        slot_of = {}
        cnt = [0] * D
        for n in range(N):
            d = dc[b][n]
            if d < 0:
                continue
            j = cnt[d]
            assert j < W, f"level width {j + 1} exceeds W={W}"
            cnt[d] += 1
            nodes[b, d, j] = n
            slot_of[n] = j
            pslot[b, d, j] = slot_of[pc[b][n]] if pc[b][n] >= 0 else -1
    return nodes.to(parents.device), pslot.to(parents.device), D


def fused_recurrent_gdn_tree_level(
    q, k, v, g, beta, scale,
    initial_state_source, initial_state_indices,
    level_nodes, level_pslot, num_levels: int,
    scratch: torch.Tensor,  # [2, B, W, HV, V, K] fp32 (per-layer shared)
    use_qk_l2norm_in_kernel: bool = False,
    prev_inputs: Optional[tuple] = None, prev_row=None, prev_chain=None,
    row_strides: Optional[tuple] = None, BV: int = 4, num_warps: int = 1,
):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    W = level_nodes.shape[2]
    BK = triton.next_power_of_2(K)
    NV = triton.cdiv(V, BV)
    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    prev = prev_inputs if prev_inputs is not None else (q, k, v, g, beta)
    pr = prev_row if prev_row is not None else initial_state_indices
    pcn = prev_chain if prev_chain is not None else initial_state_indices
    sq = row_strides[0] if row_strides else H * K
    sk = row_strides[1] if row_strides else H * K
    sv = row_strides[2] if row_strides else HV * V
    grid = (B * W, HV, NV)
    # per-level [B, W] slices must be contiguous for the kernel's flat indexing
    ln_d = level_nodes.permute(1, 0, 2).contiguous()
    lp_d = level_pslot.permute(1, 0, 2).contiguous()
    for d in range(num_levels):
        fused_recurrent_gdn_tree_level_kernel[grid](
            q, k, v, g, beta, o,
            initial_state_source, initial_state_indices, initial_state_source.stride(0),
            ln_d[d], lp_d[d],
            scratch[(d + 1) % 2], scratch[d % 2],
            *prev, pr, pcn,
            scale=scale, sq=sq, sk=sk, sv=sv,
            T=T, W=W, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
            IS_ROOT=d == 0, USE_PREFIX=prev_chain is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_BETA_HEADWISE=beta.ndim == v.ndim,
            num_warps=num_warps,
        )
    return o
