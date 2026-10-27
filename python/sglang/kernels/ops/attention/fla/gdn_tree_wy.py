# Tree-masked WY/UT closed form of the gated delta rule for draft-tree verification
# (cf. STree 2505.14969, SpecLA 2607.16673, Bole 2608.01651, TreeWY 2608.20961).
# One Triton launch per layer, one program per (batch, value head): reads the pre-tree
# state once, verifies every draft node of the tree in parallel (no per-node states),
# and emits the pseudo-value factors U used to reconstruct the accepted state on commit.
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_wy_kernel(
    Q, Kraw, V, G, BETA, S0, S0IDX, AMASK, AIMASK, O, U, CG,
    stride_qn, stride_kn, stride_vn, stride_on, stride_s0, scale,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, DEPTH: tl.constexpr, PREC: tl.constexpr,
    L2NORM: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    o_n = tl.arange(0, NP)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, DV)
    m_n = o_n < N
    row = i_b * N + o_n                                   # token rows of this request
    b_q = tl.load(Q + row[:, None] * stride_qn + i_h * K + o_k[None, :], mask=m_n[:, None], other=0).to(tl.float32)
    b_k = tl.load(Kraw + row[:, None] * stride_kn + i_h * K + o_k[None, :], mask=m_n[:, None], other=0).to(tl.float32)
    if L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, 1) + 1e-6)[:, None]
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)[:, None]
    b_q = b_q * scale
    b_v = tl.load(V + row[:, None] * stride_vn + i_hv * DV + o_v[None, :], mask=m_n[:, None], other=0).to(tl.float32)
    g = tl.load(G + row * HV + i_hv, mask=m_n, other=0.0).to(tl.float32)
    bt = tl.load(BETA + row * HV + i_hv, mask=m_n, other=0.0).to(tl.float32)
    mm = m_n[:, None] & m_n[None, :]
    am = tl.load(AMASK + i_b * N * N + o_n[:, None] * N + o_n[None, :], mask=mm, other=0).to(tl.float32)
    aim = tl.load(AIMASK + i_b * N * N + o_n[:, None] * N + o_n[None, :], mask=mm, other=0).to(tl.float32)
    cg = tl.sum(aim * g[None, :], 1)                     # cumulative log-decay along ancestors incl. self
    gt = tl.exp(cg)
    # pre-tree state of this (request, head): pool stores h[v, k]; we need S0^T as [K, DV]
    idx = tl.maximum(tl.load(S0IDX + i_b).to(tl.int64), 0)  # CUDA-graph pad rows carry -1; their output is discarded
    p_s0 = S0 + idx * stride_s0 + i_hv * K * DV + o_v[None, :] * K + o_k[:, None]
    s0t = tl.load(p_s0).to(tl.float32)                   # [K, DV]  (s0t[k, v] = h[v, k])
    s0q = tl.dot(b_q, s0t, input_precision=PREC)         # [NP, DV] rows = (S0 q_t)^T
    s0k = tl.dot(b_k, s0t, input_precision=PREC)
    # G_t / G_i only on ancestor pairs: unmasked differences can overflow exp() to inf (inf*0 = nan)
    ratio = tl.exp(tl.where(aim > 0, cg[:, None] - cg[None, :], 0.0))
    kk = tl.dot(b_k, tl.trans(b_k), input_precision=PREC)
    kq = tl.dot(b_q, tl.trans(b_k), input_precision=PREC)
    gm = bt[:, None] * ratio * kk * am                   # strictly lower-triangular (ancestors precede)
    r = bt[:, None] * (b_v - gt[:, None] * s0k)
    u = r
    acc = r
    for _ in tl.static_range(DEPTH):                     # (I + Gm)^-1 = sum_j (-Gm)^j, nilpotent in depth
        acc = -tl.dot(gm, acc, input_precision=PREC)
        u += acc
    c = ratio * kq * aim
    o = gt[:, None] * s0q + tl.dot(c, u, input_precision=PREC)
    tl.store(O + row[:, None] * stride_on + i_hv * DV + o_v[None, :], o.to(O.dtype.element_ty), mask=m_n[:, None])
    base = (i_b * HV + i_hv) * N
    tl.store(U + base * DV + o_n[:, None] * DV + o_v[None, :], u, mask=m_n[:, None])
    tl.store(CG + base + o_n, cg, mask=m_n)


def gdn_tree_wy_verify(q, k, v, g, beta, scale, S0, s0_idx, anc_strict, anc_incl, depth, l2norm=True):
    """q,k [B*N,H,K] (any row stride); v [B*N,HV,V]; g,beta [B*N,HV]; S0 pool [S,HV,K?,V?] fp32 with
    s0_idx [B]; anc_strict/anc_incl [B,N,N] int8/bool; depth = max tree depth (root 0).
    Returns o [B*N,HV,V] (v.dtype) and factors U [B,HV,N,V] fp32, CG [B,HV,N] fp32."""
    B, N = anc_strict.shape[0], anc_strict.shape[1]
    H, K = q.shape[-2], q.shape[-1]
    HV, DV = v.shape[-2], v.shape[-1]
    o = torch.empty((B * N, HV, DV), dtype=v.dtype, device=v.device)
    U = torch.empty((B, HV, N, DV), dtype=torch.float32, device=v.device)
    CG = torch.empty((B, HV, N), dtype=torch.float32, device=v.device)
    NP = max(16, triton.next_power_of_2(N))
    am = anc_strict if anc_strict.dtype == torch.int8 else anc_strict.to(torch.int8)
    aim = anc_incl if anc_incl.dtype == torch.int8 else anc_incl.to(torch.int8)
    _tree_wy_kernel[(B * HV,)](
        q, k, v, g, beta, S0, s0_idx, am.contiguous(), aim.contiguous(), o, U, CG,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0), S0.stride(0), scale,
        N=N, NP=NP, K=K, DV=DV, H=H, HV=HV, DEPTH=int(depth),
        PREC=os.environ.get("SGLANG_WY_PREC", "tf32"), L2NORM=l2norm,
        num_warps=int(os.environ.get("SGLANG_WY_WARPS", "8")),
    )
    return o, U, CG


def gdn_tree_wy_commit(S0, U, CG, Kn, anc_incl, acc_node):
    """State of accepted node a: S_a = G_a S0 + sum_{i<=a} (G_a/G_i) u_i k_i^T (pool layout h[v,k]).
    S0 [B,HV,V,K] (gathered rows), U [B,HV,N,V], CG [B,HV,N], Kn [B,HV,N,K] (normalised keys)."""
    B, HV, N, DV = U.shape
    bi = torch.arange(B, device=U.device)
    w = anc_incl[bi, acc_node].float().unsqueeze(1)                        # [B,1,N]
    cga = CG[bi, :, acc_node]                                               # [B,HV]
    ratio = torch.exp((cga.unsqueeze(-1) - CG).masked_fill(w == 0, 0.0)) * w   # [B,HV,N]; avoid exp overflow off-path
    return torch.exp(cga)[..., None, None] * S0 + torch.matmul((ratio.unsqueeze(-1) * U).transpose(-1, -2), Kn)


@triton.jit
def _tree_wy_commit_kernel(
    U, CG, Kraw, AIMASK, ACC, S0, S0IDX, stride_kn, stride_s0,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, PREC: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    o_n = tl.arange(0, NP)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, DV)
    m_n = o_n < N
    a = tl.load(ACC + i_b).to(tl.int32)
    w = tl.load(AIMASK + i_b * N * N + a * N + o_n, mask=m_n, other=0).to(tl.float32)   # ancestors of a incl. a
    base = (i_b * HV + i_hv) * N
    cg = tl.load(CG + base + o_n, mask=m_n, other=0.0)
    cga = tl.sum(tl.where(o_n == a, cg, 0.0), 0)
    ratio = tl.exp(tl.where(w > 0, cga - cg, 0.0)) * w                                  # G_a / G_i on the path
    row = i_b * N + o_n
    b_k = tl.load(Kraw + row[:, None] * stride_kn + i_h * K + o_k[None, :], mask=m_n[:, None], other=0).to(tl.float32)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)[:, None]
    b_u = tl.load(U + base * DV + o_n[:, None] * DV + o_v[None, :], mask=m_n[:, None], other=0.0)
    idx = tl.maximum(tl.load(S0IDX + i_b).to(tl.int64), 0)  # CUDA-graph pad rows carry -1; their output is discarded
    p_s0 = S0 + idx * stride_s0 + i_hv * K * DV + o_v[:, None] * K + o_k[None, :]       # pool tile h[v, k]
    s0 = tl.load(p_s0).to(tl.float32)
    s_a = tl.exp(cga) * s0 + tl.dot(tl.trans(b_u * ratio[:, None]), b_k, input_precision=PREC)
    tl.store(p_s0, s_a.to(p_s0.dtype.element_ty))


def gdn_tree_wy_commit_inplace(U, CG, kraw, anc_incl_i8, acc_node, ssm_states, s0_idx):
    """In-place: ssm_states[s0_idx[b]] <- exact state of accepted node acc_node[b].
    U [B,HV,N,V] fp32, CG [B,HV,N] fp32, kraw [B*N,H,K] (raw keys, any row stride),
    anc_incl_i8 [B,N,N] int8, acc_node [B] int, s0_idx [B] int."""
    B, HV, N, DV = U.shape
    H, K = kraw.shape[-2], kraw.shape[-1]
    NP = max(16, triton.next_power_of_2(N))
    _tree_wy_commit_kernel[(B * HV,)](
        U, CG, kraw, anc_incl_i8, acc_node, ssm_states, s0_idx, kraw.stride(0), ssm_states.stride(0),
        N=N, NP=NP, K=K, DV=DV, H=H, HV=HV, PREC=os.environ.get("SGLANG_WY_PREC", "tf32"), num_warps=4,
    )

@triton.jit
def _tree_wy_commit_all_kernel(
    U, CG, Kraw, AIMASK, ACC, POOL, LAYER_OFF, S0IDX, stride_ul, stride_cgl, stride_kl, stride_kn, stride_pl, stride_s0,
    B: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, DV: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, PREC: tl.constexpr,
):
    # one launch for every layer: program = (layer, batch, value head); stashes are stacked [L, cap, ...]
    pid = tl.program_id(0)
    i_l = pid // (B * HV)
    i_bh = pid % (B * HV)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    U = U + i_l.to(tl.int64) * stride_ul
    CG = CG + i_l.to(tl.int64) * stride_cgl
    Kraw = Kraw + i_l.to(tl.int64) * stride_kl
    S0 = POOL + tl.load(LAYER_OFF + i_l).to(tl.int64) * stride_pl
    o_n = tl.arange(0, NP)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, DV)
    m_n = o_n < N
    a = tl.load(ACC + i_b).to(tl.int32)
    w = tl.load(AIMASK + i_b * N * N + a * N + o_n, mask=m_n, other=0).to(tl.float32)
    base = (i_b * HV + i_hv) * N
    cg = tl.load(CG + base + o_n, mask=m_n, other=0.0)
    cga = tl.sum(tl.where(o_n == a, cg, 0.0), 0)
    ratio = tl.exp(tl.where(w > 0, cga - cg, 0.0)) * w
    row = i_b * N + o_n
    b_k = tl.load(Kraw + row[:, None] * stride_kn + i_h * K + o_k[None, :], mask=m_n[:, None], other=0).to(tl.float32)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)[:, None]
    b_u = tl.load(U + base * DV + o_n[:, None] * DV + o_v[None, :], mask=m_n[:, None], other=0.0)
    idx = tl.maximum(tl.load(S0IDX + i_b).to(tl.int64), 0)
    p_s0 = S0 + idx * stride_s0 + i_hv * K * DV + o_v[:, None] * K + o_k[None, :]
    s0 = tl.load(p_s0).to(tl.float32)
    s_a = tl.exp(cga) * s0 + tl.dot(tl.trans(b_u * ratio[:, None]), b_k, input_precision=PREC)
    tl.store(p_s0, s_a.to(p_s0.dtype.element_ty))


def gdn_tree_wy_commit_all(U, CG, k, anc_incl_i8, acc_node, pool, layer_off, s0_idx, bs):
    """Single launch over all stashed layers. U [L,cap,HV,N,V] fp32, CG [L,cap,HV,N] fp32,
    k [L,cap,N,H,K], pool [Lp,S,HV,K,V] (temporal state pool), layer_off [L] pool-layer index per stash slot."""
    L, cap, HV, N, DV = U.shape
    H, K = k.shape[-2], k.shape[-1]
    NP = max(16, triton.next_power_of_2(N))
    _tree_wy_commit_all_kernel[(L * bs * HV,)](
        U, CG, k, anc_incl_i8, acc_node, pool, layer_off, s0_idx,
        U.stride(0), CG.stride(0), k.stride(0), k.stride(2), pool.stride(0), pool.stride(1),
        B=bs, N=N, NP=NP, K=K, DV=DV, H=H, HV=HV, PREC=os.environ.get("SGLANG_WY_PREC", "tf32"), num_warps=4,
    )
