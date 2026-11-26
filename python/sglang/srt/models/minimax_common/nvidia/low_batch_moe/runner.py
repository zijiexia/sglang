# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's vllm/models/minimax_m3/nvidia/specialized_decode/moe/
# __init__.py (LowBatchFfn), Copyright contributors to the vLLM project.
"""Routed half of the low-batch FFN: front (routing + per-token NVFP4 input +
routing record) -> FC1 (SwiGLU-OAI, per-tile NVFP4 of h) -> FC2, one launch
each. Returns FlashInfer's ``do_finalize=False`` triple
``(gemm2_permuted, expert_weights, expanded_idx_to_permuted_idx)``."""

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from ._record import (
    EXPERTS,
    HIDDEN,
    MAX_ROWS,
    TILE,
    TOP_K,
    cta_words,
    max_ctas,
    route_words,
    width,
)
from .front_cute import LowBatchFront, h_tile_bytes
from .gemm_cute import RoutedFc1, RoutedFc2

FC1_CTA_K = 256
FC2_CTA_K = 256
FC1_STAGES = 8
FC2_STAGES = 4
# Above this many rows FC2 starts on each token tile as FC1 completes it.
OVERLAP_ROWS = 8


def _ptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype, align: int = 16):
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache
def _front(device_index: int, built: int):
    with torch.cuda.device(device_index):
        return cute.compile(
            LowBatchFront(built),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            Float32(1.0),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint16),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc1(device_index: int, columns: int, overlap: bool):
    with torch.cuda.device(device_index):
        return cute.compile(
            RoutedFc1(HIDDEN, columns, overlap, FC1_CTA_K, FC1_STAGES),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Uint32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint32),
            _fake(cutlass.Uint8),
            _fake(cutlass.Float32),
            _fake(cutlass.Int32),
            _fake(cutlass.Int32),
            Int32(1),
            Int32(EXPERTS),
            Int32(4),
            Int32(TILE * TOP_K),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _fc2(device_index: int, columns: int, overlap: bool):
    with torch.cuda.device(device_index):
        return cute.compile(
            RoutedFc2(
                HIDDEN,
                columns,
                2 * columns // 128 if overlap else 0,
                FC2_CTA_K,
                FC2_STAGES,
            ),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.BFloat16),
            _fake(cutlass.Int32),
            Int32(TILE * TOP_K),
            Int32(EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


class LowBatchRoutedMoe:
    """Bound to one layer's TRT-LLM-gen-prepared NVFP4 expert weights."""

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        alpha1,
        alpha2,
        bias,
        routed_scaling_factor: float,
    ):
        self.w13, self.w13_scale = w13, w13_scale.view(torch.uint8)
        self.w2, self.w2_scale = w2, w2_scale.view(torch.uint8)
        self.alpha1 = alpha1.float().contiguous()
        self.alpha2 = alpha2.float().contiguous()
        self.bias = bias.float().contiguous()
        self.scale = float(routed_scaling_factor)
        self.columns = w2.shape[-1] * 2
        self.device_index = w13.device.index
        for built in (8, 16, 32):
            _front(self.device_index, built)
        for overlap in (False, True):
            _fc1(self.device_index, self.columns, overlap)
            _fc2(self.device_index, self.columns, overlap)

    def run(self, x: torch.Tensor, logits: torch.Tensor):
        rows = x.size(0)
        assert 1 <= rows <= MAX_ROWS and x.dtype == torch.bfloat16 and x.is_contiguous()
        logits = logits.float().contiguous()
        device = x.device
        tiles = max_ctas(rows, TOP_K, EXPERTS)
        h_rows = TILE * tiles
        words = Int32(cta_words(tiles))
        weights = torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device)
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        counts = route_words(tiles)
        record = torch.empty(
            counts + cta_words(tiles), dtype=torch.int32, device=device
        )
        x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        x_scale = torch.empty((rows, HIDDEN // 16), dtype=torch.uint8, device=device)
        x_pt = torch.empty(rows, dtype=torch.float32, device=device)
        c = self.columns
        h = torch.empty((h_rows, c), dtype=torch.bfloat16, device=device)
        h_quant = torch.empty((h_rows, c // 2), dtype=torch.uint8, device=device)
        h_scale = torch.empty(tiles * h_tile_bytes(c), dtype=torch.uint8, device=device)
        h_pt = torch.empty(h_rows, dtype=torch.float32, device=device)
        output = torch.empty((h_rows, HIDDEN), dtype=torch.bfloat16, device=device)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        ctas = Int32(_sm_count(self.device_index))
        _front(self.device_index, width(rows))(
            _ptr(cutlass.Float32, logits),
            _ptr(cutlass.Float32, self.bias),
            Float32(self.scale),
            _ptr(cutlass.Uint32, x),
            _ptr(cutlass.Uint16, weights),
            _ptr(cutlass.Uint32, x_quant),
            _ptr(cutlass.Uint8, x_scale),
            _ptr(cutlass.Float32, x_pt),
            _ptr(cutlass.Int32, permuted),
            _ptr(cutlass.Int32, record),
            Int32(rows),
            stream,
        )
        overlap = rows > OVERLAP_ROWS
        _fc1(self.device_index, c, overlap)(
            _ptr(cutlass.Float4E2M1FN, self.w13, 32),
            _ptr(cutlass.Float8E4M3FN, self.w13_scale),
            _ptr(cutlass.Float4E2M1FN, x_quant, 32),
            _ptr(cutlass.Uint8, x_quant),
            _ptr(cutlass.Uint32, x_scale),
            _ptr(cutlass.Float32, x_pt),
            _ptr(cutlass.Float32, self.alpha1),
            _ptr(cutlass.Uint32, h),
            _ptr(cutlass.Uint32, h_quant),
            _ptr(cutlass.Uint8, h_scale),
            _ptr(cutlass.Float32, h_pt),
            _ptr(cutlass.Int32, record),
            _ptr(cutlass.Int32, record[counts:]),
            Int32(rows),
            Int32(EXPERTS),
            words,
            Int32(h_rows),
            ctas,
            stream,
        )
        _fc2(self.device_index, c, overlap)(
            _ptr(cutlass.Float4E2M1FN, self.w2, 32),
            _ptr(cutlass.Float8E4M3FN, self.w2_scale),
            _ptr(cutlass.Float4E2M1FN, h_quant, 32),
            _ptr(cutlass.Float8E4M3FN, h_scale),
            _ptr(cutlass.Float32, h_pt),
            _ptr(cutlass.Float32, self.alpha2),
            _ptr(cutlass.BFloat16, output),
            _ptr(cutlass.Int32, record),
            Int32(h_rows),
            Int32(EXPERTS),
            words,
            ctas,
            stream,
        )
        return output, weights, permuted
