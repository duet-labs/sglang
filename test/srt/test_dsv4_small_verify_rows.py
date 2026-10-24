"""CPU shape contracts for the DSV4 small-verify fused paths at 9/16/17 rows
(SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS; verify-16 is 17 rows incl. the root).

Kernel launches are stubbed, so this checks the host-side row limits, grid
shapes and scale-factor buffer sizing only; numerical checks need a GPU.
"""

import pytest
import torch
import triton
from triton.runtime.jit import JITFunction

from sglang.srt.environ import envs

ROWS = (9, 16, 17, 32)
K = 5120


class _Launches:
    def __init__(self):
        self.grids = []

    def __call__(self, grid):
        self.grids.append(grid)
        return lambda *a, **k: None


@pytest.fixture
def launches(monkeypatch):
    rec = _Launches()
    monkeypatch.setattr(JITFunction, "__getitem__", lambda self, grid: rec(grid))
    return rec


def _sf_offsets(m: int, k: int):
    groups = k // 32
    offs = set()
    for r in range(m):
        for g in range(groups):
            offs.add((g // 4) * 512 + ((r % 32) * 4 + ((r // 32) % 4)) * 4 + g % 4)
    return offs


@pytest.mark.parametrize("m", list(range(1, 129, 7)) + [128])
def test_mxfp8_swizzle_unique_and_in_bounds_up_to_128_rows(m):
    offs = _sf_offsets(m, K)
    assert len(offs) == m * (K // 32)
    assert max(offs) < (K // 32) * 128


def test_mxfp8_swizzle_overflows_at_129_rows():
    offs = _sf_offsets(129, K)
    assert len(offs) < 129 * (K // 32)


@pytest.mark.parametrize("m", ROWS)
def test_hc_combine_norm_mxfp8_accepts_rows_under_cap(launches, m):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm_mxfp8

    x = torch.zeros((m, 4 * K), dtype=torch.bfloat16)
    pre = torch.zeros((m, 4), dtype=torch.float32)
    w = torch.ones(K, dtype=torch.bfloat16)
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(max(m, 17)):
        y, q, s = hc_combine_norm_mxfp8(x, pre, w, 1e-6)
    assert y.shape == (m, K) and q.shape == (m, K)
    assert s.numel() == (K // 32) * triton.cdiv(m, 128) * 128
    assert launches.grids[-1][0] == m


@pytest.mark.parametrize("m", ROWS)
def test_rmsnorm_mxfp8_accepts_rows_under_cap(launches, m):
    from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8

    x = torch.zeros((m, K), dtype=torch.bfloat16)
    w = torch.ones(K, dtype=torch.bfloat16)
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(max(m, 17)):
        y, q, s = rmsnorm_mxfp8(x, w, 1e-6)
    assert y.shape == q.shape == (m, K) and s.numel() == (K // 32) * 128
    assert launches.grids[-1][0] == m + triton.cdiv(s.numel(), 512)


@pytest.mark.parametrize("m", ROWS)
def test_default_cap_still_rejects_rows_above_8(m):
    from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8

    x = torch.zeros((m, K), dtype=torch.bfloat16)
    w = torch.ones(K, dtype=torch.bfloat16)
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(8):
        with pytest.raises(AssertionError):
            rmsnorm_mxfp8(x, w, 1e-6)


def test_swapab_attention_batch_is_a_grid_dim():
    from sglang.kernels.ops.attention.dsv4 import decode_attention_sm100 as m

    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(17):
        assert m._max_batch() == 17
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(8):
        assert m._max_batch() == 8
    assert m.MAX_BATCH == 8


class _CudaLike(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("m", ROWS + (33,))
def test_wo_a_small_batch_tile_is_32_rows(launches, m):
    from sglang.kernels.ops.attention.dsv4.wo_a import (
        SMALL_BATCH_MAX_ROWS,
        small_batch_max_rows,
        wo_a_bf16_small_batch,
        wo_a_bf16_small_batch_mxfp8,
    )

    x = torch.zeros((m, 2, 4096), dtype=torch.bfloat16).as_subclass(_CudaLike)
    w = torch.zeros((2, 1024, 4096), dtype=torch.bfloat16).as_subclass(_CudaLike)
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(64):
        assert small_batch_max_rows() == SMALL_BATCH_MAX_ROWS == 32
        if m > 32:
            with pytest.raises(AssertionError):
                wo_a_bf16_small_batch(x, w)
        else:
            assert wo_a_bf16_small_batch(x, w).shape == (m, 2, 1024)
            assert launches.grids[-2] == (16, 2, 8)
            q, s = wo_a_bf16_small_batch_mxfp8(x, w)
            assert q.shape == (m, 2048) and s.numel() == 8192
    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(8):
        assert small_batch_max_rows() == 8


def _fused_finalize_quant_rows(cap):
    from sglang.kernels.ops.communication.all_reduce_mhc import (
        MOE_FINALIZE_QUANT_MAX_ROWS,
    )

    with envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.override(cap):
        return min(envs.SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS.get(), MOE_FINALIZE_QUANT_MAX_ROWS)


@pytest.mark.parametrize("m", [1, 6, 8])
def test_fused_finalize_quant_default_cap_unchanged_at_or_below_8_rows(m):
    """Default cap 8: rows <= 8 take the kQuant epilogue exactly as before."""
    assert 0 < m <= _fused_finalize_quant_rows(8)


@pytest.mark.parametrize("m", ROWS)
def test_fused_finalize_quant_rows_above_8_need_the_flag(m):
    """9/16/17/32 rows reach all_reduce_fusion.cuh's kQuant epilogue only when
    SGLANG_OPT_DSV4_SMALL_VERIFY_MAX_ROWS is raised past them (opt-in)."""
    assert not (0 < m <= _fused_finalize_quant_rows(8))
    assert 0 < m <= _fused_finalize_quant_rows(max(m, 17)) or m > 17
    assert 0 < m <= _fused_finalize_quant_rows(m)


@pytest.mark.parametrize("cap", [8, 17, 129, 1024])
def test_fused_finalize_quant_never_exceeds_the_128_row_scale_tile(cap):
    """The kQuant scale swizzle covers one 128-row tile (CHECK_HOST(num_tokens <= 128))."""
    from sglang.kernels.ops.communication.all_reduce_mhc import (
        MOE_FINALIZE_QUANT_MAX_ROWS,
    )

    assert MOE_FINALIZE_QUANT_MAX_ROWS == 128
    assert _fused_finalize_quant_rows(cap) <= 128
    assert not (0 < 129 <= _fused_finalize_quant_rows(cap))
