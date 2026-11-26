# SPDX-License-Identifier: Apache-2.0
"""Binds :class:`LowBatchRoutedMoe` to a ``MiniMaxM3MoE`` layer.

``SGLANG_MINIMAX_M3_LOW_BATCH_MOE=1`` routes the 1-32 row deferred-finalize
calls of a TP4 MiniMax-M3.1-Flash layer through the low-batch front + FC1 +
FC2 kernels; any other layer or call keeps the FlashInfer TRT-LLM-gen path.
``SGLANG_MINIMAX_M3_LOW_BATCH_MOE_COMPARE=1`` also runs the stock path on
eager (non-captured) calls and logs the finalized-output error.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from .runner import LowBatchRoutedMoe

logger = logging.getLogger(__name__)

MAX_ROWS = 32
_compare_calls = 0


def enabled() -> bool:
    return envs.SGLANG_MINIMAX_M3_LOW_BATCH_MOE.get()


def _unsupported_reason(moe) -> Optional[str]:
    from ._record import EXPERTS, HIDDEN, INTERMEDIATE_PER_RANK, TOP_K

    config, experts = moe.config, moe.experts
    if (
        config.num_local_experts,
        config.num_experts_per_tok,
        config.hidden_size,
    ) != (EXPERTS, TOP_K, HIDDEN):
        return "layer shape"
    if moe.e_score_correction_bias is None:
        return "no routing bias"
    if moe.num_fused_shared_experts:
        return "fused shared experts"
    if getattr(experts, "num_local_experts", EXPERTS) != EXPERTS:
        return "experts are sharded (EP)"
    w13, w2 = experts.w13_weight, experts.w2_weight
    columns = w2.shape[-1] * 2
    if columns != INTERMEDIATE_PER_RANK:
        return f"{columns} intermediate columns per rank (TP4 only)"
    if tuple(w13.shape) != (EXPERTS, 2 * columns, HIDDEN // 2) or w13.data_ptr() % 32:
        return f"w13 layout {tuple(w13.shape)}"
    if tuple(w2.shape) != (EXPERTS, HIDDEN, columns // 2) or w2.data_ptr() % 32:
        return f"w2 layout {tuple(w2.shape)}"
    up = getattr(experts, "g1_alphas_up", None)
    if up is not None and not torch.equal(up, experts.g1_alphas):
        return "w1 and w3 global scales differ"
    return None


def get_low_batch_moe(moe) -> Optional[LowBatchRoutedMoe]:
    """The layer's low-batch routed MoE, built on first use; None when the
    feature is off or the layer does not fit (logged once per layer)."""
    built = getattr(moe, "_low_batch_moe", False)
    if built is not False:
        return built
    reason = _unsupported_reason(moe)
    if reason is not None:
        logger.warning(
            "MiniMax-M3 low-batch MoE off for layer %s: %s", moe.layer_id, reason
        )
        moe._low_batch_moe = None
        return None
    from .runner import LowBatchRoutedMoe

    e = moe.experts
    moe._low_batch_moe = LowBatchRoutedMoe(
        e.w13_weight.data,
        e.w13_weight_scale.data,
        e.w2_weight.data,
        e.w2_weight_scale.data,
        e.g1_alphas.data,
        e.g2_alphas.data,
        moe.e_score_correction_bias.data,
        moe.routed_scaling_factor,
    )
    return moe._low_batch_moe


def _finalize(gemm2, weights, permuted) -> torch.Tensor:
    m, k = weights.shape
    rows = gemm2[permuted.reshape(-1).long()].view(m, k, -1).float()
    return (rows * weights.float().unsqueeze(-1)).sum(1)


def log_compare(moe, rows: int, out, ref) -> None:
    """Finalized low-batch output ``out`` against the stock deferred output."""
    global _compare_calls
    k = out[1].shape[1]
    a = _finalize(*out)
    b = _finalize(
        ref.gemm2_out.view(-1, ref.gemm2_out.shape[-1]),
        ref.expert_weights.view(-1, k)[:rows],
        ref.expanded_idx_to_permuted_idx.view(-1, k)[:rows],
    )
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    _compare_calls += 1
    if _compare_calls <= 200 or cos < 0.999:
        logger.info(
            "low-batch MoE compare layer=%s rows=%d max_abs=%.4g ref_max=%.4g cos=%.6f",
            moe.layer_id,
            rows,
            (a - b).abs().max().item(),
            b.abs().max().item(),
            cos,
        )
