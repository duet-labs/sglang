"""Fused finalize for DFLASH tree verify: accepted-path KV/predict/hidden compaction in
two launches (+ the pool's own move) instead of ~12 eager torch ops."""
import torch
import triton
import triton.language as tl


@triton.jit
def _locs_kernel(
    acc_idx_ptr, acc_len_ptr, out_loc_ptr, r2t_ptr, req_idx_ptr, seq_ptr, tgt_ptr, src_ptr,
    r2t_stride, S: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.arange(0, S)
    L = tl.load(acc_len_ptr + b)
    valid = k < L
    node = tl.load(acc_idx_ptr + b * S + k, mask=valid, other=0)
    node = tl.maximum(node, 0).to(tl.int64)
    src = tl.load(out_loc_ptr + node, mask=valid, other=0)
    req = tl.load(req_idx_ptr + b).to(tl.int64)
    seq = tl.load(seq_ptr + b).to(tl.int64)
    tgt = tl.load(r2t_ptr + req * r2t_stride + seq + k, mask=valid, other=0)
    tl.store(tgt_ptr + b * S + k, tgt.to(tl.int64))
    tl.store(src_ptr + b * S + k, src.to(tl.int64))


@triton.jit
def _compact_kernel(
    x_ptr, out_ptr, acc_idx_ptr, H, S: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.program_id(1)
    row = b * N + k
    if k < S:
        node = tl.load(acc_idx_ptr + b * S + k)
        node = tl.maximum(node, 0).to(tl.int64)
    else:
        node = row.to(tl.int64)
    for start in range(0, H, BLOCK):
        h = start + tl.arange(0, BLOCK)
        m = h < H
        v = tl.load(x_ptr + node * H + h, mask=m)
        tl.store(out_ptr + row * H + h, v, mask=m)


_HBUF = {}


def finalize_accept_tree_path_fused(batch, accept_index, accept_lens, predict, logits_output, bs, *, token_to_kv_pool_allocator, num_draft_tokens):
    """Same contract as eagle_worker_common._finalize_accept_tree_path."""
    S = accept_index.shape[1]
    N = num_draft_tokens
    dev = accept_index.device
    acc_idx = accept_index[:bs].contiguous()
    tgt = torch.empty(bs * S, dtype=torch.int64, device=dev)
    src = torch.empty(bs * S, dtype=torch.int64, device=dev)
    r2t = batch.req_to_token_pool.req_to_token
    _locs_kernel[(bs,)](
        acc_idx, accept_lens[:bs].contiguous(), batch.out_cache_loc, r2t,
        batch.req_pool_indices[:bs], batch.seq_lens[:bs], tgt, src, r2t.stride(0), S=S,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(tgt, src)

    pred_out = torch.empty_like(predict)
    _compact_kernel[(bs, N)](predict, pred_out, acc_idx, 1, S=S, N=N, BLOCK=1)
    hid = logits_output.hidden_states
    if hid is not None:
        key = (tuple(hid.shape), hid.dtype, dev)
        out = _HBUF.get(key)
        if out is None:
            out = torch.empty_like(hid)
            _HBUF[key] = out
        H = hid.shape[1]
        _compact_kernel[(bs, N)](hid, out, acc_idx, H, S=S, N=N, BLOCK=1024)
        logits_output.hidden_states = out
    return pred_out
