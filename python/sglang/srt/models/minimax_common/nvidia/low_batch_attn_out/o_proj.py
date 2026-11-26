# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM's vllm/models/minimax_m3/nvidia/specialized_decode/attention/o_proj.py.
"""o_proj of the low-batch attention: FlashInfer's low-latency MXFP8 GEMM.

``LowLatencyBlockscaledGemmKernel`` (the kernel behind ``mm_mxfp8``'s
``cutedsl_low_latency`` backend) compiled with an 8-, 16- or 32-row activation
tile and one fixed tactic per tile. The kernel computes ``out^T``: its M is
the weight's N (6144), its N the step's rows. Only the activation (B) load warp
waits for the previous grid; the weight (A) load warp fills its stages without
waiting. With K = 2048 and split-K 2 every CTA's whole weight slice fits in its
four stages, so under PDL the weights are resident before the activation is.
"""

from __future__ import annotations

from functools import cache

import torch

# (cta_k, ab_stages, sfb_tmem_stages, split_k) per activation tile.
TACTICS = {8: (256, 4, 1, 2), 16: (256, 4, 1, 2), 32: (128, 10, 1, 1)}
MAX_ROWS = 32


def tile_rows(rows: int) -> int:
    return 8 if rows <= 8 else 16 if rows <= 16 else 32


def _ptr(dtype, address: int, align: int):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _kernel(tile: int, device_index: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
        LowLatencyBlockscaledGemmKernel,
    )

    cta_k, ab_stages, sfb_stages, split_k = TACTICS[tile]
    gemm = LowLatencyBlockscaledGemmKernel(
        acc_dtype=cutlass.Float32,
        mma_tiler_mnk=(128, tile, cta_k),
        num_ab_stage=ab_stages,
        num_sfb_tmem_stage=sfb_stages,
        sf_vec_size=32,
        use_pdl=True,
        split_k=split_k,
    )
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    with torch.cuda.device(device_index):
        return cute.compile(
            gemm,
            _ptr(f8, 16, 16),
            _ptr(sf, 32, 32),
            _ptr(f8, 16, 16),
            _ptr(sf, 16, 16),
            _ptr(bf16, 16, 16),
            None,
            None,
            tuple(cutlass.Int32(v) for v in (128, tile, cta_k, 1)),
            make_fake_stream(),
        )


def compile_all(device_index: int) -> None:
    for tile in TACTICS:
        _kernel(tile, device_index)


def o_proj_gemm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    act: torch.Tensor,
    act_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out`` [rows, n] BF16 = ``act`` [rows, k] E4M3 (UE8M0 ``act_scale`` in
    the 128x4 swizzled layout of one 128-row tile) times ``weight`` [n, k]^T
    (K-major E4M3, ``weight_scale`` 128x4 swizzled), on the current stream."""
    import cuda.bindings.driver as cuda
    import cutlass

    n, k = weight.shape
    rows = act.shape[0]
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    _kernel(tile_rows(rows), weight.device.index)(
        _ptr(f8, weight.data_ptr(), 16),
        _ptr(sf, weight_scale.data_ptr(), 32),
        _ptr(f8, act.data_ptr(), 16),
        _ptr(sf, act_scale.data_ptr(), 16),
        _ptr(bf16, out.data_ptr(), 16),
        None,
        None,
        tuple(cutlass.Int32(v) for v in (n, rows, k, 1)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
