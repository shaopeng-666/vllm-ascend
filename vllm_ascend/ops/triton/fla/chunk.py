# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors
import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context

from .chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .utils import input_guard, prepare_chunk_indices, prepare_final_chunk_indices


def _as_host_tuple(values):
    if values is None:
        return None
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(int(v) for v in values)
    if isinstance(values, torch.Tensor):
        return tuple(int(v) for v in values.detach().cpu().reshape(-1).tolist())
    return tuple(int(v) for v in values)


def _prepare_chunk_indices_if_needed(cu_seqlens, chunk_indices, chunk_size: int):
    if cu_seqlens is None or chunk_indices is not None:
        return chunk_indices
    if isinstance(cu_seqlens, torch.Tensor):
        return prepare_chunk_indices(cu_seqlens, chunk_size)
    return None


def solve_tril(
    A: torch.Tensor,
    cu_seqlens=None,
    chunk_indices_large_block=None,
    chunk_indices_bt=None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    del chunk_indices_large_block
    output_dtype = A.dtype if output_dtype is None else output_dtype
    A_for_kernel = A.to(output_dtype).contiguous()
    if cu_seqlens is None:
        return torch.ops._C_ascend.npu_solve_tri(A_for_kernel, layout="bsnd")

    chunk_size = A_for_kernel.shape[-1]
    if cu_seqlens is not None and chunk_indices_bt is None:
        chunk_indices_bt = prepare_chunk_indices(cu_seqlens, chunk_size)

    A_tnd = A_for_kernel.reshape(-1, A_for_kernel.shape[-2], A_for_kernel.shape[-1])
    out = torch.ops._C_ascend.npu_solve_tri(
        A_tnd,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_bt,
        layout="tnd",
    )
    return out.reshape_as(A_for_kernel)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens=None,
    chunk_indices=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = A.shape[-1]
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    k_hf = k.contiguous()
    v_hf = v.contiguous()
    beta_hf = beta.to(g_cumsum.dtype).contiguous()
    A_hf = A.contiguous()
    g_hf = g_cumsum.contiguous()
    w, u = torch.ops._C_ascend.npu_recompute_wu_fwd(
        k_hf,
        v_hf,
        beta_hf,
        A_hf,
        g_hf,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return w.contiguous(), u.contiguous()


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del chunk_offsets
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k.to(torch.bfloat16).transpose(1, 2).contiguous(),
        w.to(torch.bfloat16).transpose(1, 2).contiguous(),
        u.to(torch.bfloat16).transpose(1, 2).contiguous(),
        g=None if g is None else g.transpose(1, 2).contiguous(),
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    return h.transpose(1, 2).contiguous(), v_new.transpose(1, 2).contiguous(), final_state


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens=None,
    chunk_size: int = 64,
    chunk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    del chunk_offsets
    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    h_for_kernel = h.transpose(1, 2).contiguous() if h.dim() == 5 else h
    out = torch.ops._C_ascend.chunk_fwd_o(
        q.to(torch.bfloat16).transpose(1, 2).contiguous(),
        k.to(torch.bfloat16).transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        h_for_kernel,
        scale,
        g=None if g is None else g.transpose(1, 2).contiguous(),
        g_gamma=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    return out.transpose(1, 2).contiguous()


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
):
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes
    chunk_size = 64
    block_indices_cumsum = None if prebuilt_meta is None else prebuilt_meta.block_indices_cumsum
    cu_seqlens_host = None if prebuilt_meta is None else prebuilt_meta.cu_seqlens_host
    chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64
    chunk_indices_chunk64_host = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64_host
    chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_offsets_chunk64
    update_chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.update_chunk_offsets_chunk64
    final_chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.final_chunk_indices_chunk64
    chunk_indices_large_block = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_large_block

    cu_seqlens = None if cu_seqlens is None else cu_seqlens.to(torch.int64)
    if cu_seqlens is not None and chunk_indices_chunk64 is None and chunk_indices_chunk64_host is None:
        chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = None if chunk_indices_chunk64 is None else chunk_indices_chunk64.to(torch.int64)
    if cu_seqlens_host is None and cu_seqlens is not None:
        cu_seqlens_host = _as_host_tuple(cu_seqlens)
    if chunk_indices_chunk64_host is None and chunk_indices is not None:
        chunk_indices_chunk64_host = _as_host_tuple(chunk_indices)
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_indices=block_indices_cumsum,
    )

    g_hf = g.transpose(1, 2).contiguous()
    beta_hf = beta.transpose(1, 2).contiguous()
    v_hf = v.contiguous()

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta_hf,
        g_cumsum=g_hf,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens_host,
        chunk_indices_large_block=chunk_indices_large_block,
        chunk_indices_bt=chunk_indices_chunk64_host,
        output_dtype=k.dtype,
    )

    A_hf = A.transpose(1, 2).contiguous()

    w, u = recompute_w_u_fwd(
        k=k,
        v=v_hf,
        beta=beta_hf,
        A=A_hf,
        g_cumsum=g_hf,
        cu_seqlens=cu_seqlens_host,
        chunk_indices=chunk_indices_chunk64_host,
    )

    q_ascendc = q.to(torch.bfloat16).contiguous()
    k_ascendc = k.to(torch.bfloat16).contiguous()
    w_ascendc = w.to(torch.bfloat16).contiguous()
    u_ascendc = u.to(torch.bfloat16).contiguous()
    g_ascendc = g_hf.contiguous()

    h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_ascendc,
        w_ascendc,
        u_ascendc,
        g=g_ascendc,
        gk=None,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=cu_seqlens_host,
        chunk_indices=chunk_indices_chunk64_host,
        use_exp2=False,
        transpose_state_layout=False,
    )

    if get_pcp_group().world_size > 1:
        # When integrating mtp, since `mix_qkv` has been split, `num_decode`
        # cannot be directly obtained from the metadata and needs to be recalculated.
        actual_num_decodes = getattr(prebuilt_meta, "num_decodes", None)
        if actual_num_decodes is None:
            actual_num_decodes = num_decodes
        h_update = chunk_gated_delta_rule_fwd_hupdate(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
            chunk_offsets=chunk_offsets_chunk64,
            update_chunk_offsets=update_chunk_offsets_chunk64,
            num_decodes=actual_num_decodes,
        )
        all_final_state = get_pcp_group().all_gather(final_state.unsqueeze(0), 0)
        final_chunk_indices = final_chunk_indices_chunk64
        if final_chunk_indices is None:
            final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, chunk_size)
        final_h_update = h_update[:, final_chunk_indices, :, :, :]
        all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)

        updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
        updated_state[0, ...] = all_final_state[0]
        for i in range(1, get_pcp_group().world_size):
            # correct_i = all_final_state[i] + Φ_i · (correct_{i-1} - s0) = Φ_i · correct_{i-1} + p_i
            updated_final_state = all_final_state[i] + torch.matmul(
                all_final_h_update[i, ...], updated_state[i - 1, ...] - initial_state
            )
            updated_state[i, ...] = updated_final_state

        final_state = updated_state[-1, ...]

        if get_pcp_group().rank_in_group == 0:
            updated_h_state = torch.zeros_like(final_state)
        else:
            updated_h_state = updated_state[get_pcp_group().rank_in_group - 1, ...]

        if get_pcp_group().rank_in_group > 0:
            rerun_initial_state = initial_state.clone()
            if cu_seqlens is not None:
                _ns_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                prefill_seq_offset = int(((_ns_lens > 0) & (_ns_lens <= 1)).sum().item())
            else:
                prefill_seq_offset = num_decodes
            prefill_slice = slice(prefill_seq_offset, final_state.shape[0])
            rerun_initial_state[prefill_slice] = updated_h_state[prefill_slice]
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=g,
                initial_state=rerun_initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices_chunk64,
                chunk_offsets=chunk_offsets_chunk64,
            )
            h = h.transpose(1, 2).contiguous()
            v_new = v_new.transpose(1, 2).contiguous()

    o_ascendc = torch.ops._C_ascend.chunk_fwd_o(
        q_ascendc,
        k_ascendc,
        v_new,
        h,
        scale,
        g=g_ascendc,
        g_gamma=None,
        cu_seqlens=cu_seqlens_host,
        chunk_indices=chunk_indices_chunk64_host,
        chunk_size=64,
        transpose_state_layout=False,
    )

    o = o_ascendc.to(torch.bfloat16).transpose(1, 2).contiguous()
    return o, final_state


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        prebuilt_meta=None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        o, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
        )
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Gated Delta Rule chunk forward (head-first prefill path).

    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` (head-first).
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` (head-first).
        v (torch.Tensor):
            values of shape `[B, T, H, V]` (time-first, transposed internally).
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` (time-first, transposed internally).
        beta (torch.Tensor):
            betas of shape `[B, T, H]` (time-first, transposed internally).
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, defaults to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to q/k inside the kernel. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        prebuilt_meta,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
