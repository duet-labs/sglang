from typing import Optional, Tuple, Union

import msgspec
import logging
import os
import torch

logger = logging.getLogger(__name__)

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CAUSAL_CONV1D_FWD_BLOCK_M,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
    build_verify_intermediate_state_indices,
    pp_spec_stable_rows_enabled,
    select_verify_intermediate_state_indices,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import is_pin_memory_available, rank0_log

_is_hip = is_hip()

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip() or is_xpu():
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        can_use_fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkvzba_split_reshape_cat_contiguous,
    )

MAX_FUSED_QKV_SPLIT_DIM = 8192
_fused_decode_proj_conv_logged = False
_fused_decode_proj_conv_fallback_logged = False
_fused_decode_proj_conv_layers_logged: set[int] = set()
_fused_decode_real_tensor_verified_layers: set[int] = set()
_fused_decode_log_layer_hits = envs.SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS.get()
_fused_decode_verify_real_tensors = (
    envs.SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS.get()
)


class GDNMISMetadata(msgspec.Struct, frozen=True):
    query_token_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    query_seq_lens_cpu: list[int]
    query_request_indices: torch.Tensor
    item_token_indices: torch.Tensor
    item_cu_seqlens: torch.Tensor
    item_seq_lens_cpu: list[int]
    item_request_indices: torch.Tensor


def build_gdn_mis_metadata(forward_batch: ForwardBatch) -> GDNMISMetadata:
    """Build compact query/item segments from request-local MIS delimiters."""
    if not forward_batch.is_prefill_only:
        raise ValueError("GDN MIS is only supported for prefill-only requests")

    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if any(int(prefix_len) != 0 for prefix_len in prefix_lens):
        raise ValueError("GDN MIS does not support cached prefixes")

    seq_lens = forward_batch.extend_seq_lens_cpu
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    delimiter_indices = forward_batch.multi_item_delimiter_indices
    if delimiter_indices is None or len(delimiter_indices) != len(seq_lens):
        raise ValueError("GDN MIS requires delimiter indices for every request")
    if sum(seq_lens) > forward_batch.input_ids.numel():
        raise ValueError("GDN MIS sequence lengths exceed the input tokens")

    query_token_indices: list[int] = []
    query_seq_lens_cpu: list[int] = []
    query_request_indices: list[int] = []
    item_token_indices: list[int] = []
    item_seq_lens_cpu: list[int] = []
    item_request_indices: list[int] = []

    request_start = 0
    for request_idx, (seq_len, request_delimiters) in enumerate(
        zip(seq_lens, delimiter_indices)
    ):
        delimiters = [int(index) for index in request_delimiters.tolist()]
        if len(delimiters) < 2:
            raise ValueError("GDN MIS requires at least two delimiters per request")
        if any(
            current >= following
            for current, following in zip(delimiters, delimiters[1:])
        ):
            raise ValueError("GDN MIS delimiter indices must be strictly increasing")
        if delimiters[0] < 0 or delimiters[-1] >= seq_len:
            raise ValueError("GDN MIS delimiter index is outside the request")
        if delimiters[-1] != seq_len - 1:
            raise ValueError("GDN MIS final delimiter must be the last request token")

        query_len = delimiters[0]
        if query_len > 0:
            query_token_indices.extend(range(request_start, request_start + query_len))
            query_seq_lens_cpu.append(query_len)
            query_request_indices.append(request_idx)

        branch_ends = delimiters[1:] + [seq_len]
        for branch_start, branch_end in zip(delimiters, branch_ends):
            branch_len = branch_end - branch_start
            item_token_indices.extend(
                range(request_start + branch_start, request_start + branch_end)
            )
            item_seq_lens_cpu.append(branch_len)
            item_request_indices.append(request_idx)

        request_start += seq_len

    device = forward_batch.input_ids.device

    def _indices(values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
        result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        if lengths:
            result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
                dim=0
            )
        return result

    return GDNMISMetadata(
        query_token_indices=_indices(query_token_indices, torch.int64),
        query_cu_seqlens=_cu_seqlens(query_seq_lens_cpu),
        query_seq_lens_cpu=query_seq_lens_cpu,
        query_request_indices=_indices(query_request_indices, torch.int64),
        item_token_indices=_indices(item_token_indices, torch.int64),
        item_cu_seqlens=_cu_seqlens(item_seq_lens_cpu),
        item_seq_lens_cpu=item_seq_lens_cpu,
        item_request_indices=_indices(item_request_indices, torch.int64),
    )


def validate_gdn_mis_backend(prefill_backend: LinearAttnKernelBackend) -> None:
    if not get_exec().features.enable_mis:
        return
    if not prefill_backend.is_triton():
        raise ValueError(
            "GDN multi-item scoring requires the Triton linear-attention prefill "
            "backend. Set --linear-attn-prefill-backend triton."
        )
    if get_memory().enable_page_major_kv_layout:
        raise ValueError("GDN multi-item scoring does not support page-major layout")


if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def flashinfer_gdn_prefill_default(model_runner: ModelRunner) -> Optional[str]:
    """FlashInfer for the narrow SM90/SM100 GDN prefill domains we validated, else None."""
    sm_major = torch.cuda.get_device_capability()[0] if is_cuda() else 0
    if (
        get_exec().mamba.linear_attn_prefill_backend is not None
        or get_exec().mamba.linear_attn_backend != "triton"
        or get_exec().deterministic.enable_deterministic_inference
        or get_memory().enable_page_major_kv_layout
        or sm_major not in (9, 10)
    ):
        return None

    # SM100 runs the CUDA>=13 CuTe-DSL chunk kernel on a bf16 state pool;
    # SM90 runs the fused Hopper kernel on an fp32 state pool and tolerates
    # larger chunks. Everything outside these validated domains keeps Triton.
    cuda_version = torch.version.cuda
    if sm_major == 10:
        if cuda_version is None or int(cuda_version.split(".", 1)[0]) < 13:
            return None
        max_chunk = 8192
        expected_state_dtype = torch.bfloat16
    else:
        max_chunk = 32768
        expected_state_dtype = torch.float32

    chunk_size = get_schedule().chunked_prefill_size
    config = hybrid_gdn_config(model_runner.model_config)
    if (
        get_schedule().enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= max_chunk
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != expected_state_dtype
    ):
        return None

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if not is_flashinfer_gdn_prefill_available():
        return None

    rank0_log(f"Defaulting SM{sm_major}0 GDN prefill backend to FlashInfer.")
    return "flashinfer"


def _validate_gdn_linear_attn_backends(backends: LinearAttnBackends) -> None:
    if (
        get_exec().deterministic.enable_deterministic_inference
        and backends.prefill.is_flashinfer()
    ):
        raise ValueError(
            "FlashInfer GDN prefill is not supported with "
            "--enable-deterministic-inference. Use "
            "--linear-attn-prefill-backend triton."
        )


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: Optional[LinearAttnKernelBackend] = None,
    ):
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # The fused SYCL kernel is dispatched via XpuGDNAttnBackend.forward_fused_gdn,
            # outside this dispatcher; Triton is the dispatcher-level kernel for requests
            # that hook doesn't handle (e.g. verify).
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        elif decode_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # See the decode branch above: intel_xpu uses Triton as its
            # dispatcher-level fallback kernel.
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        elif prefill_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel. An explicitly configured verify backend wins; the
        # historical auto rule (FlashInfer when the selected FlashInfer kernel
        # supports MTP verify) only applies when no explicit choice was made.
        # SM90 FlashInfer verify requires a fp32 SSM state, so e.g.
        # --mamba-ssm-dtype bfloat16 setups must be able to force Triton here.
        if verify_backend is not None and verify_backend.is_triton():
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False
        elif (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
            self.verify_kernel_is_flashinfer = True
        else:
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    @property
    def extend_uses_state_checkpoints(self) -> bool:
        return self.extend_kernel.uses_state_checkpoints

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        tree_pre_node = kwargs.pop("tree_pre_node", None)
        tree_pre_pop = kwargs.pop("tree_pre_pop", None)
        dfs_layer_id = kwargs.pop("dfs_layer_id", None)
        tree_next_token = kwargs.pop("tree_next_token", None)
        tree_next_sibling = kwargs.pop("tree_next_sibling", None)
        if (
            kwargs.get("retrieve_parent_token") is not None
            and os.environ.get("SGLANG_DFLASH_TREE_WY_VERIFY") == "1"
        ):
            if _WY.get("am") is None:
                N = int(kwargs["cache_steps"])
                _WY["am"] = torch.zeros((256, N, N), dtype=torch.int8, device=q.device)
                _WY["aim"] = torch.zeros((256, N, N), dtype=torch.int8, device=q.device)
            return self._target_verify_tree_wy(
                A_log, dt_bias, q, k, v, a, b, ssm_states=ssm_states,
                cache_indices=cache_indices, dfs_layer_id=dfs_layer_id,
                query_start_loc=query_start_loc, **kwargs,
            )
        if (
            kwargs.get("retrieve_parent_token") is not None
            and tree_pre_node is not None
            and os.environ.get("SGLANG_DFLASH_TREE_DFS_VERIFY") == "1"
        ):
            return self._target_verify_tree_dfs(
                A_log, dt_bias, q, k, v, a, b, ssm_states=ssm_states,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                tree_pre_node=tree_pre_node, tree_pre_pop=tree_pre_pop,
                dfs_layer_id=dfs_layer_id, tree_next_token=tree_next_token,
                tree_next_sibling=tree_next_sibling, **kwargs,
            )
        verify_kernel = self._get_target_verify_kernel(
            kwargs.get("retrieve_parent_token")
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def _target_verify_tree_wy(
        self, A_log, dt_bias, q, k, v, a, b, *, ssm_states, cache_indices, dfs_layer_id, cache_steps,
        query_start_loc, retrieve_parent_token, **kwargs,
    ):
        """Closed-form (tree-masked WY) verify: all draft nodes in parallel from one read of
        the pre-tree state; factors stored per layer for the commit (SGLANG_DFLASH_TREE_WY_VERIFY=1)."""
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        from sglang.kernels.ops.attention.fla.gdn_tree_wy import gdn_tree_wy_verify

        T = int(cache_steps)
        H, K = q.shape[-2], q.shape[-1]
        rows = q.numel() // (H * K)
        bs = rows // T  # padded batch during CUDA-graph capture/replay; masks of pad rows are zero
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        q2, k2, v2 = q.view(rows, *q.shape[-2:]), k.view(rows, *k.shape[-2:]), v.view(rows, *v.shape[-2:])
        o, U, CG = gdn_tree_wy_verify(
            q2, k2, v2, g.view(rows, -1), beta.view(rows, -1), q.shape[-1] ** -0.5,
            ssm_states, cache_indices[:bs], _WY["am"][:bs], _WY["aim"][:bs], 8,
        )
        st = _WY["layers"].setdefault(dfs_layer_id, {})
        st["ssm_states"] = ssm_states
        if "U" not in st or st["U"].shape[0] < bs or st["U"].shape[2] != T:
            if torch.cuda.is_current_stream_capturing() and "U" in st:
                raise RuntimeError("WY factor stash too small during CUDA graph capture")
            cap = max(bs, cache_indices.shape[0])
            # per-layer stashes are slices of one stacked buffer so the commit is a single launch
            pool = getattr(ssm_states, "_base", None)
            stk = _WY.get("stack")
            if stk is None or stk["U"].shape[1] < cap or stk["U"].shape[3] != T:
                L = pool.shape[0] if pool is not None and pool.dim() == 5 else 64
                stk = dict(
                    U=torch.empty((L, cap, *U.shape[1:]), dtype=U.dtype, device=U.device),
                    CG=torch.empty((L, cap, *CG.shape[1:]), dtype=CG.dtype, device=CG.device),
                    k=torch.empty((L, cap, T, *k.shape[-2:]), dtype=k.dtype, device=k.device),
                    layer_off=torch.zeros((L,), dtype=torch.int64, device=U.device),
                    pool=pool if pool is not None and pool.dim() == 5 else None,
                    slot_of={}, n=0,
                )
                _WY["stack"] = stk
            if dfs_layer_id not in stk["slot_of"]:
                li = stk["n"]
                stk["slot_of"][dfs_layer_id] = li
                stk["n"] = li + 1
                if stk["pool"] is not None:
                    off = (ssm_states.data_ptr() - stk["pool"].data_ptr()) // (stk["pool"].stride(0) * stk["pool"].element_size())
                    stk["layer_off"][li] = off
            li = stk["slot_of"][dfs_layer_id]
            st["U"], st["CG"], st["k"] = stk["U"][li], stk["CG"][li], stk["k"][li]
        st["U"][:bs].copy_(U)
        st["CG"][:bs].copy_(CG)
        st["k"][:bs].copy_(k2.view(bs, T, *k.shape[-2:]))
        if os.environ.get("SGLANG_DFLASH_TREE_WY_DEBUG") == "1":
            st["dbg"] = dict(q=q2.clone().view(bs, T, *q.shape[-2:]), v=v2.clone().view(bs, T, *v.shape[-2:]),
                             g=g.view(bs, T, -1).clone(), beta=beta.view(bs, T, -1).clone(), o=o.clone().view(bs, T, *o.shape[-2:]),
                             cache_indices=cache_indices[:bs].clone(), parents=retrieve_parent_token[:bs].clone())
        if (
            os.environ.get("SGLANG_DFLASH_TREE_WY_DEBUG") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and _WY["aim"][:bs].any()
        ):
            _WY["dbg"] = _WY.get("dbg", 0) + 1
            if _WY["dbg"] <= 96:
                # WY mode shrinks the intermediate-state buffer to one step: give the
                # snapshot-based reference its own full scratch buffer
                dbg_kwargs = dict(kwargs)
                dbg_kwargs["intermediate_states_buffer"] = torch.zeros(
                    (bs, T, *ssm_states.shape[1:]), dtype=torch.float32, device=q.device
                )
                dbg_kwargs["intermediate_state_indices"] = torch.arange(bs, device=q.device, dtype=torch.int32)
                ref = self._get_target_verify_kernel(retrieve_parent_token).target_verify(
                    A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b, ssm_states=ssm_states.clone(),
                    cache_indices=cache_indices, query_start_loc=query_start_loc,
                    retrieve_parent_token=retrieve_parent_token, cache_steps=cache_steps, **dbg_kwargs,
                )
                d = (ref.float().reshape(-1) - o.float().reshape(-1)).abs()
                if int(_WY["am"][:bs].sum().item()) == 0:
                    logger.info("WY debug degenerate tree parents=%s", retrieve_parent_token[:bs].tolist())
                logger.info(
                    "WY debug layer=%s bs=%s T=%s ref.shape=%s o.shape=%s max|d|=%.3e mean|d|=%.3e ref|mean|=%.3e am.sum=%d aim.sum=%d",
                    dfs_layer_id, bs, T, tuple(ref.shape), tuple(o.shape), d.max().item(), d.mean().item(),
                    ref.float().abs().mean().item(), int(_WY["am"][:bs].sum().item()), int(_WY["aim"][:bs].sum().item()),
                )
        return o

    def _target_verify_tree_dfs(
        self, A_log, dt_bias, q, k, v, a, b, *, ssm_states, cache_indices,
        query_start_loc, tree_pre_node, tree_pre_pop, retrieve_parent_token, cache_steps,
        dfs_layer_id=None, tree_next_token=None, tree_next_sibling=None, **kwargs,
    ):
        """DFS tree verify: no per-node state snapshots; the accepted path is
        committed by replay (SGLANG_DFLASH_TREE_DFS_VERIFY=1)."""
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        from sglang.kernels.ops.attention.fla.fused_recurrent_gdn_tree_dfs import (
            fused_recurrent_gdn_tree_dfs,
        )

        T = int(cache_steps)
        bs = tree_pre_node.shape[0]
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        q4 = q.view(bs, T, *q.shape[-2:])
        k4 = k.view(bs, T, *k.shape[-2:])
        v4 = v.view(bs, T, *v.shape[-2:])
        g4, beta4 = g.view(bs, T, -1), beta.view(bs, T, -1)
        li = _ensure_dfs_stash(dfs_layer_id, q4, k4, v4, g4, beta4, ssm_states, cache_indices, cache_indices.shape[0])
        st = _DFS_STACK
        # deferred commit: the kernel first replays last step's accepted path
        # (rows of the previous stash) and commits it, then walks the new tree
        o = fused_recurrent_gdn_tree_dfs(
            q4, k4, v4, g4, beta4,
            q.shape[-1] ** -0.5, ssm_states, cache_indices[:bs],
            tree_next_token[:bs], tree_next_sibling[:bs], retrieve_parent_token[:bs],
            use_qk_l2norm_in_kernel=True,
            pre_node=tree_pre_node[:bs], pre_pop=tree_pre_pop[:bs],
            prev_inputs=(st["q"][li], st["k"][li], st["v"][li], st["g"][li], st["beta"][li]),
            prev_row=st["prev_row"], prev_chain=st["prev_chain"],
            row_strides=(q4.stride(1), k4.stride(1), v4.stride(1)),
        )
        for key, src in (("q", q4), ("k", k4), ("v", v4), ("g", g4), ("beta", beta4)):
            st[key][li, :bs].copy_(src)
        if os.environ.get("SGLANG_DFLASH_TREE_DFS_DEBUG") == "1" and not torch.cuda.is_current_stream_capturing():
            ssm_c1, ssm_c2 = ssm_states.clone(), ssm_states.clone()
            ref = self._get_target_verify_kernel(retrieve_parent_token).target_verify(
                A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b, ssm_states=ssm_c1,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                retrieve_parent_token=retrieve_parent_token, cache_steps=cache_steps, **kwargs,
            )
            d = (ref.float().reshape(-1) - o.float().reshape(-1)).abs()
            from sglang.kernels.ops.attention.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule_update_fwd,
            )

            scratch = torch.zeros((bs, T, *ssm_states.shape[1:]), dtype=torch.float32, device=q.device)
            ref2 = fused_recurrent_gated_delta_rule_update_fwd(
                q4, k4, v4, g4, beta4, q.shape[-1] ** -0.5, ssm_c2, cache_indices[:bs],
                use_qk_l2norm_in_kernel=True, disable_state_update=True,
                intermediate_states_buffer=scratch,
                intermediate_state_indices=torch.arange(bs, device=q.device, dtype=torch.int32),
                cache_steps=T, retrieve_parent_token=retrieve_parent_token[:bs],
            )
            d2 = (ref.float().reshape(-1) - ref2.float().reshape(-1)).abs()
            d3 = (o.float().reshape(-1) - ref2.float().reshape(-1)).abs()
            # float64 torch tree reference on the same inputs
            with torch.no_grad():
                par = retrieve_parent_token[:bs].to(torch.int64)
                HVn, Kd, Vd = ssm_states.shape[1], ssm_states.shape[2], ssm_states.shape[3]
                Hn = q4.shape[2]
                qd = torch.nn.functional.normalize(q4.double(), dim=-1, eps=0.0) if False else q4.double() / torch.sqrt((q4.double() ** 2).sum(-1, keepdim=True) + 1e-6)
                kd = k4.double() / torch.sqrt((k4.double() ** 2).sum(-1, keepdim=True) + 1e-6)
                qd = qd * (q.shape[-1] ** -0.5)
                rep = HVn // Hn
                qd = qd.repeat_interleave(rep, dim=2); kd = kd.repeat_interleave(rep, dim=2)
                vd, gd, bd = v4.double(), g4.double(), beta4.double()
                h0 = ssm_states[cache_indices[:bs].long()].double()  # [bs, HV, K, V]? check layout
                o_t = torch.zeros_like(vd)
                hs = [None] * T
                for t in range(T):
                    hp = h0 if t == 0 else hs[int(par[0, t].item())]
                    # upstream layout: h[hv, k, v] with b_h indexed [v, k] tiles -> state[hv, v, k]? use [HV, K, V] as stored
                    h = hp * torch.exp(gd[:, t])[:, :, None, None]
                    kv = torch.einsum("bhkv,bhk->bhv", h, kd[:, t])
                    dv = (vd[:, t] - kv) * bd[:, t][:, :, None]
                    h = h + torch.einsum("bhk,bhv->bhkv", kd[:, t], dv)
                    hs[t] = h
                    o_t[:, t] = torch.einsum("bhkv,bhk->bhv", h, qd[:, t])
                h0T = h0.transpose(-1, -2)
                o_tT = torch.zeros_like(vd); hsT = [None] * T
                for t in range(T):
                    hp = h0T if t == 0 else hsT[int(par[0, t].item())]
                    h = hp * torch.exp(gd[:, t])[:, :, None, None]
                    kv = torch.einsum("bhkv,bhk->bhv", h, kd[:, t])
                    dv = (vd[:, t] - kv) * bd[:, t][:, :, None]
                    h = h + torch.einsum("bhk,bhv->bhkv", kd[:, t], dv)
                    hsT[t] = h
                    o_tT[:, t] = torch.einsum("bhkv,bhk->bhv", h, qd[:, t])
            dr = (ref.double().reshape(-1) - o_t.reshape(-1)).abs().max().item()
            dr2 = (ref2.double().reshape(-1) - o_t.reshape(-1)).abs().max().item()
            drT = (ref.double().reshape(-1) - o_tT.reshape(-1)).abs().max().item()
            dr2T = (ref2.double().reshape(-1) - o_tT.reshape(-1)).abs().max().item()
            logger.info(
                "DFS debug3 layer=%s torch-tree-ref: |ref-torch|=%.3e |ref2-torch|=%.3e | state transposed: |ref-torchT|=%.3e |ref2-torchT|=%.3e",
                dfs_layer_id, dr, dr2, drT, dr2T,
            )
            logger.info(
                "DFS debug2 layer=%s |ref-ref2(update kernel, my g/beta)|=%.3e  |dfs-ref2|=%.3e  a%s b%s g4 mean=%.3e beta4 mean=%.3e",
                dfs_layer_id, d2.max().item(), d3.max().item(), tuple(a.shape), tuple(b.shape),
                g4.float().mean().item(), beta4.float().mean().item(),
            )
            logger.info(
                "DFS debug layer=%s bs=%d T=%d q%s k%s v%s ssm%s strides=%s dtype=%s ref%s | max|diff|=%.3e mean=%.3e ref_absmean=%.3e pre_node=%s pre_pop=%s",
                dfs_layer_id, bs, T, tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(ssm_states.shape),
                ssm_states.stride(), ssm_states.dtype, tuple(ref.shape), d.max().item(), d.mean().item(),
                ref.float().abs().mean().item(), tree_pre_node[0].tolist(), tree_pre_pop[0].tolist(),
            )
        return o.view(bs * T, *o.shape[-2:])

    def target_verify_supports_strided_qkv(
        self, retrieve_parent_token: Optional[torch.Tensor]
    ) -> bool:
        verify_kernel = self._get_target_verify_kernel(retrieve_parent_token)
        return (
            getattr(verify_kernel, "supports_strided_target_verify_qkv", False) is True
        )

    def _get_target_verify_kernel(self, retrieve_parent_token: Optional[torch.Tensor]):
        # Tree drafts use Triton even when linear MTP verification uses FlashInfer.
        return (
            self.tree_verify_kernel
            if retrieve_parent_token is not None
            else self.verify_kernel
        )


_DFS_VERIFY_STASH: dict = {}  # layer_id -> (ssm_states, cache_indices); tensors live in _DFS_STACK
_WY: dict = {"layers": {}, "am": None, "aim": None, "bs": 0}  # closed-form tree verify state


def dflash_tree_wy_prepare(ancestor_mask: torch.Tensor) -> None:
    """Before the verify forward: inclusive ancestor mask [bs,N,N] (bool) -> graph-static int8
    strict/inclusive masks read by every GDN layer's WY kernel."""
    bs, N = ancestor_mask.shape[0], ancestor_mask.shape[1]
    if _WY["am"] is None or _WY["am"].shape[0] < bs or _WY["am"].shape[1] != N:
        cap = max(bs, 256)
        _WY["am"] = torch.zeros((cap, N, N), dtype=torch.int8, device=ancestor_mask.device)
        _WY["aim"] = torch.zeros((cap, N, N), dtype=torch.int8, device=ancestor_mask.device)
    eye = torch.eye(N, dtype=torch.bool, device=ancestor_mask.device)
    _WY["aim"][:bs].copy_(ancestor_mask.to(torch.int8))
    _WY["am"][:bs].copy_((ancestor_mask & ~eye).to(torch.int8))
    _WY["bs"] = bs


_WY_COMMIT_STATIC: dict = {}


def _wy_commit_graphed(accept_index, accept_lens, mamba_idx, bs, N):
    """All layers' in-place WY commits (48 Triton launches + the accepted-node index prep) replayed
    from one CUDA graph per batch size; stashes/masks are graph-static buffers."""
    from sglang.kernels.ops.attention.fla.gdn_tree_wy import gdn_tree_wy_commit_inplace

    key = (bs, accept_index.shape[1], N)
    st = _WY_COMMIT_STATIC.get(key)
    dev = accept_index.device
    if st is None:
        st = dict(
            acc=torch.zeros((bs, accept_index.shape[1]), dtype=torch.int64, device=dev),
            lens=torch.zeros((bs,), dtype=torch.int64, device=dev),
            midx=torch.zeros((bs,), dtype=torch.int32, device=dev),
            node=torch.zeros((bs,), dtype=torch.int32, device=dev),
            graph=None, warm=0,
        )
        _WY_COMMIT_STATIC[key] = st
    st["acc"].copy_(accept_index)
    st["lens"].copy_(accept_lens[:bs])
    st["midx"].copy_(mamba_idx[:bs])

    def _launch_all():
        bi = torch.arange(bs, device=dev)
        last = (st["lens"] - 1).clamp(min=0)
        st["node"].copy_(st["acc"][bi, last] % N)
        stk = _WY.get("stack")
        if stk is not None and stk["pool"] is not None and stk["n"] == len(_WY["layers"]) \
                and os.environ.get("SGLANG_WY_COMMIT_PER_LAYER") != "1":
            from sglang.kernels.ops.attention.fla.gdn_tree_wy import gdn_tree_wy_commit_all
            L = stk["n"]
            gdn_tree_wy_commit_all(stk["U"][:L], stk["CG"][:L], stk["k"][:L], _WY["aim"][:bs], st["node"],
                                   stk["pool"], stk["layer_off"][:L], st["midx"], bs)
            return
        for lid, ls in _WY["layers"].items():
            U, CG = ls["U"][:bs], ls["CG"][:bs]
            gdn_tree_wy_commit_inplace(U, CG, ls["k"][:bs].view(bs * U.shape[2], *ls["k"].shape[-2:]),
                                       _WY["aim"][:bs], st["node"], ls["ssm_states"], st["midx"])

    if st["graph"] is not None:
        st["graph"].replay()
    elif st["warm"] < 2 or torch.cuda.is_current_stream_capturing():
        st["warm"] += 1
        _launch_all()
    else:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _launch_all()
        st["graph"] = g
        g.replay()


def dflash_tree_wy_commit(accept_index: torch.Tensor, accept_lens: torch.Tensor, mamba_idx: torch.Tensor) -> None:
    """After accept: write the accepted node's exact state for every GDN layer from the stored
    factors (S_a = G_a S0 + sum_{i<=a} (G_a/G_i) u_i k_i^T); the per-node SSM scatter is skipped."""
    from sglang.kernels.ops.attention.fla.gdn_tree_wy import gdn_tree_wy_commit, gdn_tree_wy_commit_inplace
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import DFS_SSM_COMMITTED

    DFS_SSM_COMMITTED[0] = True
    if not _WY["layers"]:
        return
    bs = accept_index.shape[0]
    N = _WY["am"].shape[1]
    if os.environ.get("SGLANG_DFLASH_TREE_WY_DEBUG") != "1" and os.environ.get("SGLANG_DFLASH_TREE_WY_EAGER_COMMIT") != "1":
        return _wy_commit_graphed(accept_index, accept_lens, mamba_idx, bs, N)
    bi = torch.arange(bs, device=accept_index.device)
    last = (accept_lens[:bs].to(torch.int64) - 1).clamp(min=0)
    acc_node = (accept_index[bi, last] % N).to(torch.int64)
    aim = _WY["aim"][:bs].bool()
    idx = mamba_idx[:bs].to(torch.int64)
    for lid, st in _WY["layers"].items():
        U, CG, kraw, ssm = st["U"][:bs], st["CG"][:bs], st["k"][:bs], st["ssm_states"]
        HV = U.shape[1]
        if "dbg" not in st:
            gdn_tree_wy_commit_inplace(U, CG, st["k"][:bs].view(bs * U.shape[2], *st["k"].shape[-2:]),
                                       _WY["aim"][:bs], acc_node.to(torch.int32), ssm, idx.to(torch.int32))
            continue
        kn = kraw.float().repeat_interleave(HV // kraw.shape[2], dim=2)
        kn = (kn / torch.sqrt((kn * kn).sum(-1, keepdim=True) + 1e-6)).permute(0, 2, 1, 3)
        S0 = ssm[idx]
        new_state = gdn_tree_wy_commit(S0, U, CG, kn, aim, acc_node).to(ssm.dtype)
        if "dbg" in st and lid in (0, 24):
            from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update_fwd

            d = st["dbg"]; T = U.shape[2]
            scratch = torch.zeros((bs, T, *ssm.shape[1:]), dtype=torch.float32, device=U.device)
            ssm_c = ssm.clone()
            ref = fused_recurrent_gated_delta_rule_update_fwd(
                d["q"], kraw, d["v"], d["g"], d["beta"], d["q"].shape[-1] ** -0.5, ssm_c, d["cache_indices"],
                use_qk_l2norm_in_kernel=True, disable_state_update=True, intermediate_states_buffer=scratch,
                intermediate_state_indices=torch.arange(bs, device=U.device, dtype=torch.int32),
                cache_steps=T, retrieve_parent_token=d["parents"],
            ).view(bs, T, HV, -1)
            do = (ref.float() - d["o"].float()).abs()
            ref_state = scratch[bi, acc_node]
            ds = (ref_state - new_state.float()).abs()
            logger.info(
                "WY commit debug layer=%d bs=%d acc_node=%s idx=%s cache_idx=%s |o-ref| max=%.3e mean=%.3e ref|o|=%.3e | state |d| max=%.3e mean=%.3e ref|S|=%.3e |S0-Sref|=%.3e",
                lid, bs, acc_node.tolist(), idx.tolist(), d["cache_indices"].tolist(), do.max().item(), do.mean().item(),
                ref.float().abs().mean().item(), ds.max().item(), ds.mean().item(), ref_state.abs().mean().item(),
                (ref_state - S0.float()).abs().max().item(),
            )
        ssm[idx] = new_state


_DFS_COMMIT_STATIC: dict = {}  # bs -> static commit inputs + captured graph
_DFS_STACK: dict = {}  # "q","k","v","g","beta": [L, cap, T, ...] stacked over mamba layers; "pool": temporal pool


def _ensure_dfs_stash(layer_id, q4, k4, v4, g4, beta4, ssm_states, cache_indices, cap):
    """Persistent per-layer buffers holding the verify-window inputs of the last
    step (read by the next step's deferred commit) + pending-commit tables."""
    bs, T = q4.shape[0], q4.shape[1]
    pool = getattr(ssm_states, "_base", None)
    if pool is None or pool.dim() != 5:
        raise RuntimeError("DFS verify needs ssm_states to be a layer view of the temporal pool")
    L, S = pool.shape[0], pool.shape[1]
    if not _DFS_STACK or _DFS_STACK["q"].shape[1] < bs or _DFS_STACK["q"].shape[2] != T:
        if torch.cuda.is_current_stream_capturing() and _DFS_STACK:
            raise RuntimeError("DFS verify stash too small during CUDA graph capture")
        n = max(bs, cap)
        dev = q4.device
        mk = lambda t: torch.empty((L, n, T, *t.shape[2:]), dtype=t.dtype, device=dev)
        _DFS_STACK.update(
            q=mk(q4), k=mk(k4), v=mk(v4), g=mk(g4), beta=mk(beta4), pool=pool,
            prev_row=torch.full((n,), -1, dtype=torch.int64, device=dev),
            prev_chain=torch.full((n, T), -1, dtype=torch.int64, device=dev),
            pend_row=torch.full((S,), -1, dtype=torch.int64, device=dev),
            pend_chain=torch.full((S, T), -1, dtype=torch.int64, device=dev),
            layer_of={},
        )
        logger.info("DFS verify stash allocated: L=%d cap=%d T=%d slots=%d", L, n, T, S)
    off = (ssm_states.data_ptr() - pool.data_ptr()) // (pool.stride(0) * pool.element_size())
    li = int(off)
    _DFS_STACK["layer_of"][layer_id] = li
    _DFS_VERIFY_STASH[layer_id] = (ssm_states, cache_indices)
    return li


def dflash_tree_dfs_prepare(mamba_idx: torch.Tensor) -> None:
    """Before the verify forward: gather each request's pending accepted path
    (from its slot) into the graph-static prev_row / prev_chain buffers."""
    if not _DFS_STACK:
        return
    bs = mamba_idx.shape[0]
    st = _DFS_STACK
    st["prev_row"][:bs].copy_(st["pend_row"][mamba_idx])
    st["prev_chain"][:bs].copy_(st["pend_chain"][mamba_idx])


def dflash_tree_dfs_record(accept_index: torch.Tensor, accept_lens: torch.Tensor, mamba_idx: torch.Tensor) -> None:
    """After accept: remember the accepted path per slot; it is committed by the
    next verify launch (deferred commit). The per-node SSM scatter is skipped."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import DFS_SSM_COMMITTED

    DFS_SSM_COMMITTED[0] = True
    if not _DFS_STACK:
        return
    st = _DFS_STACK
    T = st["q"].shape[2]
    bs, ncol = accept_index.shape
    pos = torch.arange(ncol, device=accept_index.device).unsqueeze(0)
    keep = (accept_index >= 0) & (pos < accept_lens[:bs].to(torch.int64).unsqueeze(1))
    nodes = torch.where(keep, accept_index % T, torch.full_like(accept_index, -1)).to(torch.int64)
    chain = torch.full((bs, T), -1, dtype=torch.int64, device=accept_index.device)
    chain[:, : min(ncol, T)] = nodes[:, :T]
    st["pend_chain"][mamba_idx] = chain
    st["pend_row"][mamba_idx] = torch.arange(bs, device=accept_index.device)


def dflash_tree_dfs_clear(mamba_idx: torch.Tensor) -> None:
    """Slots (re)initialised by a prefill carry no pending commit."""
    if _DFS_STACK:
        _DFS_STACK["pend_row"][mamba_idx] = -1


def dflash_tree_dfs_commit(
    accept_index: torch.Tensor, accept_lens: torch.Tensor, cache_indices: torch.Tensor
) -> int:
    """Commit-by-replay for DFS tree verify: re-run the accepted path of every
    mamba layer through the recurrence from the persistent SSM state and write
    the end state back. One launch per layer, captured in a per-bs CUDA graph."""
    from sglang.kernels.ops.attention.fla.fused_recurrent_gdn_tree_dfs import (
        fused_recurrent_gdn_tree_dfs,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import DFS_SSM_COMMITTED

    if not _DFS_STACK:
        return 0
    DFS_SSM_COMMITTED[0] = True
    T = _DFS_STACK["q"].shape[2]
    bs, ncol = accept_index.shape
    dev = accept_index.device
    key = (bs, ncol)
    st = _DFS_COMMIT_STATIC.get(key)
    if st is None:
        st = dict(
            acc=torch.full((bs, ncol), -1, dtype=torch.int64, device=dev),
            lens=torch.zeros((bs,), dtype=torch.int64, device=dev),
            chain=torch.full((bs, T), -1, dtype=torch.int64, device=dev),
            nt=torch.full((bs, T), -1, dtype=torch.int64, device=dev),
            nt_ext=torch.full((bs, T + 1), -1, dtype=torch.int64, device=dev),
            zeros=torch.zeros((bs, T), dtype=torch.int64, device=dev),
            cidx=torch.zeros((bs,), dtype=torch.int64, device=dev),
            pos=torch.arange(ncol, device=dev).unsqueeze(0),
            graph=None, warm=0,
        )
        _DFS_COMMIT_STATIC[key] = st
    # dynamic inputs -> static buffers (the rest is fixed-shape and graph-captured)
    st["acc"].copy_(accept_index)
    st["lens"].copy_(accept_lens[:bs])
    st["cidx"].copy_(cache_indices[:bs])

    def _prep():
        keep = (st["acc"] >= 0) & (st["pos"] < st["lens"].unsqueeze(1))
        nodes = torch.where(keep, st["acc"] % T, torch.full_like(st["acc"], -1))
        st["chain"].fill_(-1)
        st["chain"][:, :ncol] = nodes[:, :T] if ncol > T else nodes
        src, dst = st["chain"][:, :-1], st["chain"][:, 1:]
        ok = (src >= 0) & (dst >= 0)
        st["nt_ext"].fill_(-1)
        st["nt_ext"].scatter_(1, torch.where(ok, src, torch.full_like(src, T)), torch.where(ok, dst, torch.full_like(dst, -1)))
        st["nt"].copy_(st["nt_ext"][:, :T])

    def _launch_all():
        _prep()
        for layer_id, (ssm_states, _ci) in _DFS_VERIFY_STASH.items():
            li = _DFS_STACK["layer_of"][layer_id]
            q4 = _DFS_STACK["q"][li, :bs]
            fused_recurrent_gdn_tree_dfs(
                q4, _DFS_STACK["k"][li, :bs], _DFS_STACK["v"][li, :bs],
                _DFS_STACK["g"][li, :bs], _DFS_STACK["beta"][li, :bs], q4.shape[-1] ** -0.5,
                ssm_states, st["cidx"], st["nt"], st["chain"], st["chain"],
                use_qk_l2norm_in_kernel=True, pre_node=st["chain"], pre_pop=st["zeros"],
                write_final_state=True,
            )

    if st["graph"] is not None:
        st["graph"].replay()
    elif st["warm"] < 2 or torch.cuda.is_current_stream_capturing():
        st["warm"] += 1
        _launch_all()
    else:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _launch_all()
        st["graph"] = g
        g.replay()
    return len(_DFS_VERIFY_STASH)


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False
    supports_mis: bool = True
    requires_contiguous_prefill_state: bool = True

    def __init__(self, model_runner: ModelRunner):
        _validate_gdn_linear_attn_backends(model_runner.linear_attn_backends)
        super().__init__(model_runner)
        self.enable_mis = get_exec().features.enable_mis
        self.mis_metadata: Optional[GDNMISMetadata] = None
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert self.conv_states_shape[-1] < FLA_CHUNK_SIZE, (
                f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"
            )

        backends = model_runner.linear_attn_backends
        validate_gdn_mis_backend(backends.prefill)
        self.linear_attn_backends = backends
        self.kernel_dispatcher = GDNKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )
        # Sized past the pool for attn_tp-padded warmup/MLP-sync batches (see helper).
        self.verify_intermediate_state_indices = (
            build_verify_intermediate_state_indices(
                self.req_to_token_pool.size,
                model_runner.device,
            )
        )
        self._use_strided_target_verify_qkv = False

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self._init_target_verify_qkv_routing(forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._init_target_verify_qkv_routing(forward_batch)
        self.mis_metadata = None
        if forward_batch.multi_item_delimiter_indices is not None:
            if not self.enable_mis:
                raise ValueError("GDN MIS metadata requires --enable-mis")
            self.mis_metadata = build_gdn_mis_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            if getattr(self.forward_metadata, "mamba_track_mask_indices", None) is None:
                self.forward_metadata.mamba_track_mask_indices = (
                    forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
            if self.kernel_dispatcher.extend_uses_state_checkpoints:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    maybe_build_flashinfer_checkpoint_plan,
                )

                maybe_build_flashinfer_checkpoint_plan(
                    forward_batch, self.forward_metadata, self.device
                )
        self._init_prefill_conv_block_table(forward_batch)

    def _init_prefill_conv_block_table(self, forward_batch: ForwardBatch) -> None:
        metadata = self.forward_metadata
        metadata.conv_block_table = None
        if (
            not is_cuda()
            or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
            or self.mis_metadata is not None
            or forward_batch.tbo_parent_token_range is not None
        ):
            return
        seq_lens = forward_batch.extend_seq_lens_cpu
        if seq_lens is None or len(seq_lens) != metadata.query_start_loc.numel() - 1:
            return
        block_counts = [
            (length + CAUSAL_CONV1D_FWD_BLOCK_M - 1) // CAUSAL_CONV1D_FWD_BLOCK_M
            for length in seq_lens
        ]
        # Amortize table construction only when most of the rectangular grid
        # would be empty: a 4x ratio means over 75% of its blocks do no work.
        # This conservative crossover avoids charging mildly ragged batches
        # for the host table and H2D copy, measured at ~280 us per forward.
        if not block_counts or len(block_counts) * max(block_counts) <= 4 * sum(
            block_counts
        ):
            return

        # Build once per forward for all GDN layers. A rectangular launch on a
        # mixed batch otherwise schedules every decode row for the longest prefill.
        pairs = [
            (seq, block)
            for seq, count in enumerate(block_counts)
            for block in range(count)
        ]
        metadata.conv_block_table = torch.tensor(
            pairs, dtype=torch.int32, device="cpu", pin_memory=is_pin_memory_available()
        ).to(self.device, non_blocking=True)

    def _init_target_verify_qkv_routing(self, forward_batch: ForwardBatch) -> None:
        # CUDA-graph metadata leaves mode and draft count at their defaults.
        if not forward_batch.forward_mode.is_target_verify():
            self._use_strided_target_verify_qkv = False
            return

        metadata = self.forward_metadata
        mamba_pool = self.req_to_token_pool.mamba_pool
        mamba_cache = mamba_pool.mamba_cache
        is_gdn_replayssm = not getattr(mamba_pool, "replayssm_is_kda", False)
        use_replayssm_fold = (
            mamba_cache.replayssm_rawv is not None
            and getattr(mamba_pool, "replayssm_spec_fold", False)
            and is_gdn_replayssm
        )
        use_replayssm_spec = (
            mamba_cache.replayssm_d is not None
            and getattr(mamba_pool, "replayssm_cache_base", None) is not None
            and is_gdn_replayssm
        )
        self._use_strided_target_verify_qkv = self._target_verify_supports_strided_qkv(
            retrieve_parent_token=metadata.retrieve_parent_token,
            use_replayssm_fold=use_replayssm_fold,
            use_replayssm_spec=use_replayssm_spec,
            ssm_dtype=mamba_cache.temporal.dtype,
            draft_token_num=forward_batch.spec_info.draft_token_num,
        )

    def _replayssm_fold_uses_cutedsl(
        self, ssm_dtype: torch.dtype, draft_token_num: int
    ) -> bool:
        return (
            self.kernel_dispatcher.verify_kernel_is_flashinfer
            and ssm_dtype == torch.bfloat16
            and draft_token_num >= 3
        )

    def _target_verify_supports_strided_qkv(
        self,
        *,
        retrieve_parent_token: Optional[torch.Tensor],
        use_replayssm_fold: bool,
        use_replayssm_spec: bool,
        ssm_dtype: torch.dtype,
        draft_token_num: int,
    ) -> bool:
        # ReplaySSM Triton routes accept strides; the CuTeDSL fold does not.
        if use_replayssm_fold:
            return not self._replayssm_fold_uses_cutedsl(ssm_dtype, draft_token_num)
        if use_replayssm_spec:
            return True
        return self.kernel_dispatcher.target_verify_supports_strided_qkv(
            retrieve_parent_token
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        global _fused_decode_proj_conv_fallback_logged
        global _fused_decode_proj_conv_logged
        global _fused_decode_proj_conv_layers_logged
        global _fused_decode_real_tensor_verified_layers

        if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        return_z = False
        conv_already_applied = False
        if isinstance(mixed_qkv, tuple):
            if len(mixed_qkv) != 2:
                raise ValueError(
                    "Fused GDN decode projection input must be "
                    "(projected_qkvz, projected_ba)"
                )
            projected_qkvz, projected_ba = mixed_qkv
            eligible, eligibility_reason = (
                can_use_fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                )
            )
            if eligible:
                qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
                fused_backend = "triton_direct_oracle_exact"
                if not _fused_decode_proj_conv_logged:
                    rank0_log("Using fused GDN decode QKVZ/BA unpack + indexed Conv1D.")
                    _fused_decode_proj_conv_logged = True
                if (
                    _fused_decode_log_layer_hits or _fused_decode_verify_real_tensors
                ) and layer.layer_id not in _fused_decode_proj_conv_layers_logged:
                    rank0_log(
                        "GDN_FUSED_DECODE_BACKEND "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        f"batch={projected_qkvz.shape[0]} "
                        f"qkv_dim={qkv_dim} state_shape={tuple(conv_states.shape)} "
                        f"state_indices_dtype={cache_indices.dtype}"
                    )
                    _fused_decode_proj_conv_layers_logged.add(layer.layer_id)

                # Compare real activations against the direct-Triton update on a
                # compact state copy, leaving the live cache to the candidate.
                verify_real_tensors = (
                    _fused_decode_verify_real_tensors
                    and layer.layer_id not in _fused_decode_real_tensor_verified_layers
                )
                if verify_real_tensors:
                    if bool(torch.any(cache_indices < 0).item()):
                        raise AssertionError(
                            "Real-tensor GDN fusion verification requires "
                            "non-padding cache indices"
                        )
                    ref_indices = torch.arange(
                        cache_indices.numel(),
                        device=cache_indices.device,
                        dtype=torch.int32,
                    )
                    ref_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    ref_mixed_qkv, ref_z, ref_b, ref_a = (
                        fused_qkvzba_split_reshape_cat_contiguous(
                            projected_qkvz,
                            projected_ba,
                            layer.num_q_heads,
                            layer.num_v_heads,
                            layer.head_q_dim,
                            layer.head_v_dim,
                        )
                    )
                    ref_mixed_qkv = causal_conv1d_update(
                        ref_mixed_qkv,
                        ref_state,
                        layer.conv_weights,
                        layer.bias,
                        layer.activation,
                        conv_state_indices=ref_indices,
                    )

                mixed_qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    cache_indices,
                    qkv_dim=qkv_dim,
                    v_dim=layer.v_dim,
                    num_v_heads=layer.num_v_heads,
                    head_v_dim=layer.head_v_dim,
                    activation=layer.activation,
                )
                if verify_real_tensors:
                    candidate_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    named_pairs = (
                        ("qkv", mixed_qkv, ref_mixed_qkv),
                        ("z", z, ref_z),
                        ("b", b, ref_b),
                        ("a", a, ref_a),
                        ("state", candidate_state, ref_state),
                    )
                    report = []
                    mismatch = False
                    for tensor_name, candidate, reference in named_pairs:
                        diff = (candidate.float() - reference.float()).abs()
                        nonzero = int(torch.count_nonzero(diff).item())
                        mismatch |= nonzero != 0
                        report.append(
                            f"{tensor_name}_nonzero={nonzero}/"
                            f"{diff.numel()} {tensor_name}_max="
                            f"{diff.max().item()}"
                        )
                    rank0_log(
                        "GDN_FUSED_REAL_TENSOR_PARITY "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        + " ".join(report)
                    )
                    _fused_decode_real_tensor_verified_layers.add(layer.layer_id)
                    if mismatch:
                        raise AssertionError(
                            "GDN fused real-tensor parity failed at "
                            f"layer_id={layer.layer_id}; " + " ".join(report)
                        )
                conv_already_applied = True
            else:
                if not _fused_decode_proj_conv_fallback_logged:
                    rank0_log(
                        "Falling back from fused GDN decode projection/Conv1D: "
                        f"{eligibility_reason}"
                    )
                    _fused_decode_proj_conv_fallback_logged = True
                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                    projected_qkvz,
                    projected_ba,
                    layer.num_q_heads,
                    layer.num_v_heads,
                    layer.head_q_dim,
                    layer.head_v_dim,
                )
            return_z = True
        else:
            assert isinstance(mixed_qkv, torch.Tensor)

        if not conv_already_applied:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return (core_attn_out, z) if return_z else core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )

        return (core_attn_out, z) if return_z else core_attn_out

    def _convolve_prefill(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        block_table = self.forward_metadata.conv_block_table
        conv_kwargs = {"block_table": block_table} if block_table is not None else {}
        return causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=forward_batch.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            **conv_kwargs,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]

    @staticmethod
    def _prefill_gates(
        layer: RadixLinearAttention, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        if _is_hip and seq_len == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token
        tree_pre_node = getattr(forward_metadata, "tree_pre_node", None)
        tree_pre_pop = getattr(forward_metadata, "tree_pre_pop", None)

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if self.mis_metadata is not None:
            if is_target_verify:
                raise ValueError("GDN MIS does not support target verify")
            return self._forward_extend_mis(
                layer=layer,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = select_verify_intermediate_state_indices(
                self.verify_intermediate_state_indices,
                forward_batch.req_pool_indices,
                cache_indices[: query_start_loc.shape[0] - 1] >= 0,
                self.req_to_token_pool.size,
            )
            mamba_pool = self.req_to_token_pool.mamba_pool
            use_replayssm_fold = (
                mamba_cache_params.replayssm_rawv is not None
                and getattr(mamba_pool, "replayssm_spec_fold", False)
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )
            use_replayssm_spec = (
                mamba_cache_params.replayssm_d is not None
                and getattr(mamba_pool, "replayssm_cache_base", None) is not None
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. Run them on contiguous per-sequence copies (identity-indexed) and
        # scatter the result back. No-op for the default contiguous pool.
        # CPU kernels (causal_conv1d_fwd_cpu, chunk_gated_delta_rule_cpu) use
        # proper indexed writes and handle non-contiguous pools directly via
        # cache_indices, so the gather/scatter round-trip is unnecessary on CPU.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_state_gather = (
            self.requires_contiguous_prefill_state
            and (not is_target_verify)
            and (not is_cpu())
            and (not conv_states.is_contiguous() or not ssm_states.is_contiguous())
        )
        if needs_state_gather:
            conv_states_contig = conv_states[cache_indices].contiguous()
            ssm_states_contig = ssm_states[cache_indices].contiguous()
            state_cache_indices = torch.arange(
                cache_indices.shape[0],
                device=cache_indices.device,
                dtype=cache_indices.dtype,
            )
        else:
            conv_states_contig = conv_states
            ssm_states_contig = ssm_states
            state_cache_indices = cache_indices

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    forward_metadata.track_conv_indices
                ].transpose(1, 2)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = self._convolve_prefill(
                layer,
                forward_batch,
                mixed_qkv,
                conv_states_contig,
                state_cache_indices,
            )

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (
            (is_cuda() or is_hip() or is_xpu())
            and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM
            and not self._use_strided_target_verify_qkv
        ):
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            # ReplaySSM verify protocols: fold-every-commit (ring-write during
            # verify, fold on commit), circular ring, or the snapshotting
            # fallback when neither ring is allocated.
            if use_replayssm_fold:
                core_attn_out = self._replayssm_fold_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    layer_cache=mamba_cache_params,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    retrieve_parent_token=retrieve_parent_token,
                )
            elif use_replayssm_spec:
                core_attn_out = self._replayssm_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    mamba_pool=mamba_pool,
                    layer_cache=mamba_cache_params,
                    cache_indices=cache_indices,
                    replay_indices=forward_batch.req_pool_indices,
                    query_start_loc=query_start_loc,
                    draft_token_num=forward_batch.spec_info.draft_token_num,
                )
            else:
                # The recurrent fallback needs the per-draft snapshots, which
                # the pool gates OFF under --enable-linear-replayssm-spec (the
                # same flag that makes `use_replayssm_spec` true above), so
                # this branch is unreachable with a None buffer by
                # construction -- keep it loud rather than silently frozen.
                assert intermediate_state_cache is not None, (
                    "recurrent target_verify fallback requires intermediate_ssm, "
                    "which is not allocated under --enable-linear-replayssm-spec"
                )
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                    tree_pre_node=tree_pre_node,
                    tree_pre_pop=tree_pre_pop,
                    dfs_layer_id=layer.layer_id,
                    tree_next_token=retrieve_next_token,
                    tree_next_sibling=forward_metadata.retrieve_next_sibling,
                    stable_rows=pp_spec_stable_rows_enabled(),
                )
        else:
            g, beta = self._prefill_gates(layer, a, b)
            h_track_buf = None
            if (
                forward_metadata.has_mamba_track_mask
                and forward_metadata.track_ssm_h_src.numel() > 0
                and self.kernel_dispatcher.extend_kernel.supports_track_state_snapshot
            ):
                # Snapshot-capable kernels can supply just the selected FP32
                # checkpoint instead of materializing all per-chunk states.
                assert forward_metadata.track_chunk_idx is not None
                h_track_buf = torch.empty(
                    (len(cache_indices), *ssm_states.shape[1:]),
                    dtype=torch.float32,
                    device=ssm_states.device,
                )
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states_contig,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),
                num_state_checkpoints=forward_metadata.num_state_checkpoints,
                state_checkpoint_every_n_tokens=(
                    forward_metadata.state_checkpoint_every_n_tokens
                ),
                output=kwargs.get("linear_attn_output"),
                layer_id=layer.layer_id,
                extend_prefix_lens=forward_batch.extend_prefix_lens,
                track_state=h_track_buf,
                track_chunk_idx=(
                    forward_metadata.track_chunk_idx
                    if h_track_buf is not None
                    else None
                ),
            )

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if needs_state_gather:
                # Scatter the in-place-updated contiguous copies back to the
                # strided envelope pool (advanced indexing handles the strides).
                conv_states[cache_indices] = conv_states_contig
                ssm_states[cache_indices] = ssm_states_contig

            if forward_metadata.has_mamba_track_mask:
                self._track_mamba_state_extend(
                    forward_batch,
                    h,
                    ssm_states,
                    forward_metadata,
                    h_track_buf=h_track_buf,
                )

        return core_attn_out

    def _forward_extend_mis(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self.mis_metadata
        assert metadata is not None

        conv_states[cache_indices] = 0
        ssm_states[cache_indices] = 0
        output = mixed_qkv.new_zeros(
            1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
        )

        if metadata.query_token_indices.numel() > 0:
            query_cache_indices = cache_indices[metadata.query_request_indices]
            query_output = self._forward_mis_segments(
                layer=layer,
                mixed_qkv=mixed_qkv[metadata.query_token_indices],
                a=a[metadata.query_token_indices],
                b=b[metadata.query_token_indices],
                conv_states=conv_states,
                conv_cache_indices=query_cache_indices,
                has_initial_states=torch.zeros(
                    len(metadata.query_seq_lens_cpu),
                    dtype=torch.bool,
                    device=mixed_qkv.device,
                ),
                query_start_loc=metadata.query_cu_seqlens,
                seq_lens_cpu=metadata.query_seq_lens_cpu,
                ssm_states=ssm_states,
                ssm_cache_indices=query_cache_indices,
                inplace_update=True,
            )
            output[:, metadata.query_token_indices] = query_output

        item_ssm_indices = cache_indices[metadata.item_request_indices]
        item_conv_states = conv_states[item_ssm_indices]
        item_conv_indices = torch.arange(
            item_ssm_indices.shape[0],
            dtype=cache_indices.dtype,
            device=cache_indices.device,
        )
        item_output = self._forward_mis_segments(
            layer=layer,
            mixed_qkv=mixed_qkv[metadata.item_token_indices],
            a=a[metadata.item_token_indices],
            b=b[metadata.item_token_indices],
            conv_states=item_conv_states,
            conv_cache_indices=item_conv_indices,
            has_initial_states=torch.ones(
                len(metadata.item_seq_lens_cpu),
                dtype=torch.bool,
                device=mixed_qkv.device,
            ),
            query_start_loc=metadata.item_cu_seqlens,
            seq_lens_cpu=metadata.item_seq_lens_cpu,
            ssm_states=ssm_states,
            ssm_cache_indices=item_ssm_indices,
            inplace_update=False,
        )
        output[:, metadata.item_token_indices] = item_output
        return output

    def _forward_mis_segments(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        conv_cache_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: list[int],
        ssm_states: torch.Tensor,
        ssm_cache_indices: torch.Tensor,
        inplace_update: bool,
    ) -> torch.Tensor:
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=conv_cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
        ).transpose(0, 1)

        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            num_tokens = mixed_qkv.shape[0]
            query = query.view(1, num_tokens, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, num_tokens, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, num_tokens, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        output, _, _ = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=ssm_cache_indices,
            query_start_loc=query_start_loc,
            inplace_update=inplace_update,
        )
        return output

    def _replayssm_fold_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_cache: "MambaPool.SpeculativeState",
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        retrieve_parent_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring-writing verify; the commit fold replays the accepted prefix
        into ``temporal``. Uses the vendored CuTe DSL MTP kernel when the
        dispatcher selected the FlashInfer bf16-state verify, else the Triton
        recurrent kernel (both store the same raw window)."""
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert retrieve_parent_token is None, (
            "ReplaySSM fold-every-commit supports a linear draft chain only "
            "(topk <= 1); EAGLE tree verify must use the recurrent verify."
        )
        seq_len = query.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size
        if self._replayssm_fold_uses_cutedsl(ssm_states.dtype, draft_token_num):
            from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import (
                gated_delta_rule_mtp,
            )

            num_v_heads = value.shape[2]
            head_v_dim = value.shape[3]
            out = gated_delta_rule_mtp(
                A_log=layer.A_log.detach(),
                a=a.view(batch_size, draft_token_num, num_v_heads),
                dt_bias=layer.dt_bias.detach(),
                q=query.view(batch_size, draft_token_num, *query.shape[2:]),
                k=key.view(batch_size, draft_token_num, *key.shape[2:]),
                v=value.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
                b=b.view(batch_size, draft_token_num, num_v_heads),
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                cache_ring=True,
                replayssm_rawv=layer_cache.replayssm_rawv,
                replayssm_rawk=layer_cache.replayssm_rawk,
                replayssm_g=layer_cache.replayssm_g,
                replayssm_beta=layer_cache.replayssm_beta,
            )
            return out.view(1, seq_len, num_v_heads, head_v_dim)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            cache_ring=True,
            replayssm_rawv=layer_cache.replayssm_rawv,
            replayssm_rawk=layer_cache.replayssm_rawk,
            replayssm_g=layer_cache.replayssm_g,
            replayssm_beta=layer_cache.replayssm_beta,
        )

    def _replayssm_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        mamba_pool: MambaPool,
        layer_cache: "MambaPool.SpeculativeState",
        cache_indices: torch.Tensor,
        replay_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        """ReplaySSM GDN spec-verify (Part B of #28511).

        Reconstructs the verify output for the whole draft window from the frozen
        checkpoint (``temporal``) + the per-slot circular ``(d, k, g)`` ring, and
        appends this window's drafts to the compact ``(d, k, g)`` rings. BF16
        checkpoints also keep low parts for compensated materialization. The rings
        are per-layer (sliced via ``mamba2_layer_cache``), while the cursors
        (write_pos, cache_base, is_flush) are request-slot pool attributes shared
        by all GDN layers of the step. The cursors advance once per accepted step;
        here we only read them and write this step's ring entries. GDN has K == V,
        so ``temporal``
        ([slots, HV, K, V]) is consumed directly as the kernel's [slots, HV, V, K]
        checkpoint.
        """
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        H, K = layer.num_k_heads, layer.head_k_dim
        HV, V = layer.num_v_heads, layer.head_v_dim
        # q/k/v may be [1, seq, *] (fallback split) or [seq, *] (fused split);
        # derive the packed token count from numel so both layouts flatten.
        seq_len = query.numel() // (H * K)
        q = query.reshape(seq_len, H, K)
        k = key.reshape(seq_len, H, K)
        v = value.reshape(seq_len, HV, V)
        a = a.reshape(seq_len, HV)
        b = b.reshape(seq_len, HV)
        d_cache = layer_cache.replayssm_d  # [slots, HV, L, V]
        max_cache_len = d_cache.shape[-2]  # ring length L
        out = q.new_empty(seq_len, HV, V)
        gdn_replayssm_spec_decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            checkpoint_state=layer_cache.temporal,
            d_cache=d_cache,
            k_cache=layer_cache.replayssm_k,
            g_cache=layer_cache.replayssm_g,
            # BF16 compact D/K low parts; None for fp32 checkpoints.
            rawv_cache=layer_cache.replayssm_rawv,
            rawk_cache=layer_cache.replayssm_rawk,
            beta_cache=layer_cache.replayssm_beta,
            out=out,
            query_start_loc=query_start_loc,
            ssm_state_indices=cache_indices,
            replay_indices=replay_indices,
            # Request-slot cursors live on the pool (shared across all GDN layers)
            # and advance only after acceptance, so the decode-path
            # forward_metadata.replayssm_write_pos snapshot is not used here.
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            max_cache_len=max_cache_len,
            max_spec_len=draft_token_num,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            # SGLang marks invalid/padding requests with a negative mamba slot
            # index (valid slots start at 0), so the kernel's "null block"
            # sentinel is -1, not the vLLM default of 0.
            null_block_id=-1,
            # Capacity folds are committed once across every GDN layer after
            # acceptance; the active path is therefore a single launch/layer.
            launch_mode="verify",
        )
        # Match the recurrent target_verify output shape (== value.shape).
        return out.reshape(value.shape)
