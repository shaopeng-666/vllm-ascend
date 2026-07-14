# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs = tl.load(X + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def fused_l2norm_fwd_kernel(X1, Y1, X2, Y2, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs1 = tl.load(X1 + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square1 = xs1 * xs1
        square_sum1 = tl.sum(square1, 1)[:, None]
        rsqrt1 = tl.rsqrt(square_sum1 + eps)
        tl.store(Y1 + (rindex + N * row_idx), xs1 * rsqrt1, xmask)

        xs2 = tl.load(X2 + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square2 = xs2 * xs2
        square_sum2 = tl.sum(square2, 1)[:, None]
        rsqrt2 = tl.rsqrt(square_sum2 + eps)
        tl.store(Y2 + (rindex + N * row_idx), xs2 * rsqrt2, xmask)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    MBLOCK = 69
    # M, N = x.shape
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    l2norm_fwd_kernel2_loop[grid](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return y.view(x_shape_og)


def fused_l2norm_fwd(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6):
    """
    Fused L2 normalization for q and k tensors in a single kernel launch.
    Both tensors must have the same shape and dtype.
    
    Args:
        q: query tensor of shape [B, H, T, K] (head-first)
        k: key tensor of shape [B, H, T, K] (head-first)
        eps: epsilon for numerical stability
    
    Returns:
        Tuple of (normalized_q, normalized_k) with same shapes as inputs
    """
    assert q.shape == k.shape, f"q and k must have same shape, got {q.shape} vs {k.shape}"
    assert q.dtype == k.dtype, f"q and k must have same dtype, got {q.dtype} vs {k.dtype}"
    
    q_shape_og = q.shape
    q_flat = q.reshape(-1, q.shape[-1])
    k_flat = k.reshape(-1, k.shape[-1])
    
    y_q = torch.empty_like(q_flat)
    y_k = torch.empty_like(k_flat)
    
    assert y_q.stride(-1) == 1
    assert y_k.stride(-1) == 1
    
    T, D = q_flat.shape[0], q_flat.shape[-1]
    
    MAX_FUSED_SIZE = 65536 // q_flat.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"fused_l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")
    
    MBLOCK = 69
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    
    fused_l2norm_fwd_kernel[grid](
        X1=q_flat,
        Y1=y_q,
        X2=k_flat,
        Y2=y_k,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )
    
    return y_q.view(q_shape_og), y_k.view(q_shape_og)
