"""Move whole rows (tokens) between slots in every layer's buffer with two launches
(gather all layers -> tmp, scatter) instead of a per-layer Python loop of
``buf[tgt] = buf[src]``. Buffers are addressed through a device array of base
pointers and viewed as [num_slots, row_bytes] uint8."""
import torch
import triton
import triton.language as tl

_PTRS = {}


@triton.jit
def _row_move_kernel(ptrs_ptr, loc_ptr, tmp_ptr, n_tokens, ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr, GATHER: tl.constexpr):
    t = tl.program_id(0)
    layer = tl.program_id(1)
    base = tl.load(ptrs_ptr + layer).to(tl.pointer_type(tl.uint8))
    loc = tl.load(loc_ptr + t).to(tl.int64)
    row = base + loc * ROW_BYTES
    tmp = tmp_ptr + (layer * n_tokens + t) * ROW_BYTES
    for start in range(0, ROW_BYTES, BLOCK):
        o = start + tl.arange(0, BLOCK)
        m = o < ROW_BYTES
        if GATHER:
            tl.store(tmp + o, tl.load(row + o, mask=m), mask=m)
        else:
            tl.store(row + o, tl.load(tmp + o, mask=m), mask=m)


def move_rows_all_layers(bufs, tgt_loc, src_loc):
    bufs = [b for b in bufs if b.numel() > 0]
    n = src_loc.numel()
    if n == 0 or not bufs:
        return
    b0 = bufs[0]
    row_bytes = b0[0].numel() * b0.element_size()
    key = (id(bufs[0]), len(bufs))
    ptrs = _PTRS.get(key)
    if ptrs is None:
        ptrs = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=b0.device)
        _PTRS[key] = ptrs
    tmp = torch.empty((len(bufs), n, row_bytes), dtype=torch.uint8, device=b0.device)
    src = src_loc.to(torch.int64).contiguous()
    tgt = tgt_loc.to(torch.int64).contiguous()
    BLOCK = min(1024, triton.next_power_of_2(row_bytes))
    _row_move_kernel[(n, len(bufs))](ptrs, src, tmp, n, ROW_BYTES=row_bytes, BLOCK=BLOCK, GATHER=True)
    _row_move_kernel[(n, len(bufs))](ptrs, tgt, tmp, n, ROW_BYTES=row_bytes, BLOCK=BLOCK, GATHER=False)
