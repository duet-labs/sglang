"""Fused page-table ancestor filter for DFLASH tree verify on the non-kpool DSA path.

One program per verify row: drops entries that are (a) verify slots of nodes not on the
row's root path, (b) at positions >= the row's causal length, or (c) negative, and
compacts the survivors to the row front in original order (tail = -1). Same semantics
as the torch reference in dsa_backend._dflash_tree_filter_page_table.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_tree_filter_kernel(
    pt_ptr, out_ptr, slots_ptr, anc_ptr, seqlen_ptr, vlen_ptr,
    W, R, stride_pt, stride_out,
    N: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr, HAS_SEQ: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= R:
        # graph-padded rows beyond the verify rows: copy through unchanged
        for start in range(0, W, BLOCK):
            pos = start + tl.arange(0, BLOCK)
            pm = pos < W
            v = tl.load(pt_ptr + row * stride_pt + pos, mask=pm, other=-1)
            tl.store(out_ptr + row * stride_out + pos, v, mask=pm)
        if HAS_SEQ:
            tl.store(vlen_ptr + row, tl.load(seqlen_ptr + row).to(tl.int32))
        else:
            tl.store(vlen_ptr + row, W.to(tl.int32))
        return
    b = row // N
    j = tl.arange(0, NP)
    jm = j < N
    sl = tl.load(slots_ptr + b * N + j, mask=jm, other=-2).to(tl.int64)
    an = tl.load(anc_ptr + row * N + j, mask=jm, other=1)
    banned = tl.where(an != 0, -2, sl)
    if HAS_SEQ:
        seql = tl.load(seqlen_ptr + row).to(tl.int32)
    else:
        seql = W
    # FA3 reads only [0, count) with count <= seql, so scan/fill just the first seql columns.
    if HAS_SEQ:
        wlim = tl.minimum(seql, W)
    else:
        wlim = W
    count = tl.zeros((), dtype=tl.int32)
    for start in range(0, wlim, BLOCK):
        pos = start + tl.arange(0, BLOCK)
        pm = pos < wlim
        v = tl.load(pt_ptr + row * stride_pt + pos, mask=pm, other=-1)
        hit = tl.sum((v.to(tl.int64)[:, None] == banned[None, :]).to(tl.int32), 1) > 0
        keep = pm & (v >= 0) & (pos < seql) & (hit == 0)
        ki = keep.to(tl.int32)
        idx = count + tl.cumsum(ki, 0) - ki
        tl.store(out_ptr + row * stride_out + idx, v, mask=keep)
        count += tl.sum(ki, 0)
    # tail = -1 (out is a reused buffer, not a clone)
    for start in range(0, wlim, BLOCK):
        pos = start + tl.arange(0, BLOCK)
        tm = (pos < wlim) & (pos >= count)
        tl.store(out_ptr + row * stride_out + pos, tl.full((BLOCK,), -1, tl.int32), mask=tm)
    tl.store(vlen_ptr + row, count)


_BUFS = {}


def _bufs(page_table_1):
    key = (tuple(page_table_1.shape), page_table_1.dtype, page_table_1.device)
    b = _BUFS.get(key)
    if b is None:
        b = (
            torch.empty_like(page_table_1),
            torch.empty(page_table_1.shape[0], dtype=torch.int32, device=page_table_1.device),
        )
        _BUFS[key] = b
    return b


def dsa_tree_filter_page_table(page_table_1, rows, num_nodes, slots, anc_rows, cache_seqlens):
    """page_table_1: [>=rows, W] int32; slots: [bs, N] cache locs of the verify nodes;
    anc_rows: [rows, N] bool/uint8 ancestor matrix; cache_seqlens: [>=rows] int32 or None.
    Returns (out, valid_len[page_table rows]) with out[:rows] filtered+compacted, out[rows:] copied unchanged.
    `out` / `valid_len` are per-shape reused buffers (one launch per layer; the consumer
    runs on the same stream before the next layer's filter overwrites them)."""
    out, valid_len = _bufs(page_table_1)
    anc_u8 = anc_rows.contiguous().view(torch.uint8) if anc_rows.dtype == torch.bool else anc_rows.contiguous()
    W = page_table_1.shape[1]
    has_seq = cache_seqlens is not None
    _dsa_tree_filter_kernel[(page_table_1.shape[0],)](
        page_table_1, out, slots.contiguous(), anc_u8,
        cache_seqlens if has_seq else valid_len, valid_len,
        W, rows, page_table_1.stride(0), out.stride(0),
        N=num_nodes, NP=triton.next_power_of_2(num_nodes), BLOCK=1024, HAS_SEQ=has_seq,
        num_warps=4,
    )
    # valid_len covers every page-table row (pad rows carry their seqlen), so the
    # backend can hand it to the attention kernels in place of cache_seqlens.
    return out, valid_len


@triton.jit
def _dsa_tree_transform_filter_kernel(
    topk_ptr, pt_ptr, out_ptr, slots_ptr, anc_ptr, seqlen_ptr, vlen_ptr,
    TOPK, R, stride_topk, stride_pt, stride_out,
    N: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr, EXPANDED: tl.constexpr, TAIL: tl.constexpr,
):
    """Fused top-k page-table transform + ancestor filter: out[row] = compacted
    page_table[req, topk[row]] keeping only entries that are valid (topk >= 0),
    at positions < seqlen[row] and not a non-ancestor verify slot. One program per row."""
    row = tl.program_id(0)
    b = row // N
    j = tl.arange(0, NP)
    jm = j < N
    real = row < R
    sl = tl.load(slots_ptr + b * N + j, mask=jm & real, other=-2).to(tl.int32)
    an = tl.load(anc_ptr + row * N + j, mask=jm & real, other=1)
    banned = tl.where(an != 0, -2, sl)
    seql = tl.load(seqlen_ptr + row).to(tl.int32)
    wlim = tl.minimum(seql, TOPK)
    if EXPANDED:
        prow = row
    else:
        prow = b
    count = tl.zeros((), dtype=tl.int32)
    for start in range(0, wlim, BLOCK):
        pos = start + tl.arange(0, BLOCK)
        pm = pos < wlim
        ti = tl.load(topk_ptr + row * stride_topk + pos, mask=pm, other=-1)
        tm = pm & (ti >= 0)
        v = tl.load(pt_ptr + prow * stride_pt + ti, mask=tm, other=-1)
        hit = tl.sum((v[:, None] == banned[None, :]).to(tl.int32), 1) > 0
        keep = tm & (v >= 0) & (hit == 0)
        ki = keep.to(tl.int32)
        idx = count + tl.cumsum(ki, 0) - ki
        tl.store(out_ptr + row * stride_out + idx, v, mask=keep)
        count += tl.sum(ki, 0)
    if TAIL:  # FA3 reads only [0, count); tail fill is for reference checks
        for start in range(0, wlim, BLOCK):
            pos = start + tl.arange(0, BLOCK)
            tm = (pos < wlim) & (pos >= count)
            tl.store(out_ptr + row * stride_out + pos, tl.full((BLOCK,), -1, tl.int32), mask=tm)
    tl.store(vlen_ptr + row, count)


def dsa_tree_transform_filter(topk_indices, page_table_req, rows, num_nodes, slots, anc_rows, cache_seqlens, expanded=True, tail=False):
    """Replaces transform_index_page_table_* + dsa_tree_filter_page_table for tree verify.
    topk_indices: [>=rows, TOPK] int32 (logical positions, -1 invalid); page_table_req: [bs, W] int32;
    cache_seqlens: [>=rows] int32 per verify row. Returns (out [rows_total, TOPK], valid_len [rows_total])."""
    rows_total = topk_indices.shape[0]
    TOPK = topk_indices.shape[1]
    key = ("tf", rows_total, TOPK, topk_indices.device)
    bufs = _BUFS.get(key)
    if bufs is None:
        bufs = (
            torch.full((rows_total, TOPK), -1, dtype=torch.int32, device=topk_indices.device),
            torch.empty(rows_total, dtype=torch.int32, device=topk_indices.device),
        )
        _BUFS[key] = bufs
    out, valid_len = bufs
    anc_u8 = anc_rows.contiguous().view(torch.uint8) if anc_rows.dtype == torch.bool else anc_rows.contiguous()
    _dsa_tree_transform_filter_kernel[(rows_total,)](
        topk_indices, page_table_req, out, slots.contiguous(), anc_u8, cache_seqlens, valid_len,
        TOPK, rows, topk_indices.stride(0), page_table_req.stride(0), out.stride(0),
        N=num_nodes, NP=triton.next_power_of_2(num_nodes), BLOCK=1024, EXPANDED=expanded, TAIL=tail, num_warps=4,
    )
    return out, valid_len
