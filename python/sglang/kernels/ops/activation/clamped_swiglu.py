"""Model-neutral entry point for the fused clamped-SwiGLU kernel.

``out = silu(min(gate, limit)) * clamp(up, -limit, limit)`` where ``limit`` is
``swiglu_limit``. The CUDA implementation currently lives with the DeepSeek V4
kernel sources (``deepseek_v4/silu_and_mul_masked_post_quant.cuh``); this
module is the model-neutral import site for generic consumers such as the
GGUF MoE method, so they do not reach into a model-specific namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul_clamp(
    input: torch.Tensor, output: torch.Tensor, swiglu_limit: float
) -> None:
    from sglang.kernels.ops.attention.dsv4.moe import silu_and_mul_clamp as _impl

    _impl(input, output, swiglu_limit)
