# SPDX-License-Identifier: Apache-2.0
"""Binds the low-latency o_proj to a ``MiniMaxM3Attention`` layer.

``SGLANG_MINIMAX_M3_LOW_BATCH_OPROJ=1`` runs o_proj of the 1-32 row calls of a
TP4 MiniMax-M3.1-Flash layer through the low-latency MXFP8 GEMM; any other
layer, backend or row count keeps the layer's own linear method.
``SGLANG_MINIMAX_M3_LOW_BATCH_OPROJ_COMPARE=1`` also runs the stock o_proj on
eager (non-captured) calls and logs the output error.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MAX_ROWS = 32
HIDDEN = 6144
Q_SIZE = 2048  # 16 query heads x 128 per TP4 rank
_compare_calls = 0


def enabled() -> bool:
    return envs.SGLANG_MINIMAX_M3_LOW_BATCH_OPROJ.get()


class LowBatchOProj(NamedTuple):
    weight: torch.Tensor  # [HIDDEN, Q_SIZE] E4M3, K-major
    weight_scale: torch.Tensor  # 128x4 swizzled UE8M0

    def run(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """o_proj's BF16 output [rows, HIDDEN] of the attention output ``x``:
        BF16 [rows, Q_SIZE], or already MXFP8 as ``(data, swizzled scale)``."""
        from .o_proj import o_proj_gemm

        if isinstance(x, tuple):
            act, act_scale = x
        else:
            from flashinfer import mxfp8_quantize

            act, act_scale = mxfp8_quantize(
                x.reshape(-1, Q_SIZE),
                is_sf_swizzled_layout=True,
                alignment=32,
                enable_pdl=True,
            )
        out = torch.empty(act.shape[0], HIDDEN, dtype=torch.bfloat16, device=act.device)
        o_proj_gemm(self.weight, self.weight_scale, act, act_scale, out)
        return out


def _unsupported_reason(o_proj) -> Optional[str]:
    if getattr(o_proj, "bias", None) is not None:
        return "o_proj has a bias"
    weight = getattr(o_proj, "weight", None)
    scale = getattr(o_proj, "weight_scale_inv_swizzled", None)
    if weight is None or scale is None:
        return "o_proj is not a FlashInfer (cute-dsl / cutlass) MXFP8 linear"
    if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (HIDDEN, Q_SIZE):
        return f"o_proj weight {tuple(weight.shape)} {weight.dtype} (TP4 only)"
    if not weight.is_contiguous() or weight.data_ptr() % 16:
        return "o_proj weight is not contiguous and 16-byte aligned"
    if scale.numel() != HIDDEN * Q_SIZE // 32 or scale.data_ptr() % 32:
        return f"o_proj scale has {scale.numel()} bytes"
    return None


def get_low_batch_o_proj(attn) -> Optional[LowBatchOProj]:
    """The layer's low-latency o_proj, built on first use; None when the
    layer does not fit (logged once per layer). Build it outside graph capture:
    the first call compiles the three tiles."""
    built = getattr(attn, "_low_batch_o_proj", False)
    if built is not False:
        return built
    reason = _unsupported_reason(attn.o_proj)
    if reason is not None:
        logger.warning(
            "MiniMax-M3 low-batch o_proj off for layer %s: %s",
            attn.attn.layer_id,
            reason,
        )
        attn._low_batch_o_proj = None
        return None
    from .o_proj import compile_all

    weight = attn.o_proj.weight.data
    compile_all(weight.device.index)
    attn._low_batch_o_proj = LowBatchOProj(
        weight, attn.o_proj.weight_scale_inv_swizzled.data.view(torch.uint8)
    )
    return attn._low_batch_o_proj


def log_compare(attn, rows: int, out: torch.Tensor, ref: torch.Tensor) -> None:
    """Log the low-latency o_proj against the stock one (eager calls only)."""
    global _compare_calls
    _compare_calls += 1
    if _compare_calls > 400 and _compare_calls % 100:
        return
    a, b = out.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    logger.info(
        "m3_lowbatch_oproj compare layer=%s rows=%d cos=%.6f max_abs=%.4g ref_max=%.4g",
        attn.attn.layer_id,
        rows,
        cos.item(),
        (a - b).abs().max().item(),
        b.abs().max().item(),
    )
