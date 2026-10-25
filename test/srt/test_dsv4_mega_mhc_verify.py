"""GPU unit test for SGLANG_OPT_DSV4_MEGA_MHC_VERIFY: the DeepGEMM mega-mHC boundary
(``mhc_mega_boundary``) against the stock 4-kernel decode/verify path for small row
counts (1..127 rows), on random data shaped like DeepSeek-V4.1-Flash (hidden 5120, hc 4).

Run on an SM100 host:  python -m pytest -q test/srt/test_dsv4_mega_mhc_verify.py
"""

import pytest
import torch

HIDDEN, HC, ITERS, RMS_EPS, HC_EPS = 5120, 4, 20, 1e-6, 1e-6
ROWS = [1, 8, 9, 16, 17, 64, 127]


def _reference(x, residual, pre, post, comb, fn, scale, base, w):
    from sglang.kernels.ops.layernorm.mhc import _mhc_post_torch, hc_mix_stats_sinkhorn

    updated = _mhc_post_torch(x, residual, post.unsqueeze(-1), comb)  # bf16, stock math
    combined = (pre.unsqueeze(-1) * updated.float()).sum(dim=1)
    var = combined.pow(2).mean(dim=-1, keepdim=True)
    normalized = (combined * torch.rsqrt(var + RMS_EPS) * w.float()).to(torch.bfloat16)
    nxt = hc_mix_stats_sinkhorn(
        updated.flatten(1), fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS
    )
    return updated, normalized, nxt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an SM100 GPU")
@pytest.mark.parametrize("rows", ROWS)
def test_mega_boundary_matches_stock_path(rows):
    from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary

    torch.manual_seed(rows)
    dev = "cuda"
    x = torch.randn(rows, HIDDEN, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(rows, HC, HIDDEN, device=dev, dtype=torch.bfloat16)
    pre = torch.rand(rows, HC, device=dev) + 0.5
    post = torch.rand(rows, HC, device=dev) + 0.5
    comb = torch.softmax(torch.randn(rows, HC, HC, device=dev), dim=-1)
    fn = torch.randn(HC * HC + 2 * HC, HC * HIDDEN, device=dev) * 0.02  # fp32, as make_hc_mixing_params
    scale = torch.rand(3, device=dev) + 0.5
    base = torch.randn(HC * HC + 2 * HC, device=dev) * 0.1
    w = torch.rand(HIDDEN, device=dev, dtype=torch.bfloat16) + 0.5

    upd, norm, nxt_mega = mhc_mega_boundary(
        x, residual, pre, post, comb, fn, scale, base, w, RMS_EPS, HC_EPS, RMS_EPS, ITERS
    )
    upd_ref, norm_ref, nxt_ref = _reference(x, residual, pre, post, comb, fn, scale, base, w)

    torch.testing.assert_close(upd.float(), upd_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(norm.float(), norm_ref.float(), rtol=2e-2, atol=2e-2)
    for got, ref, name in zip(nxt_mega, nxt_ref, ("pre", "post", "comb")):
        torch.testing.assert_close(
            got.float().flatten(), ref.float().flatten(), rtol=1e-2, atol=1e-3, msg=name
        )
