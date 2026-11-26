# SPDX-License-Identifier: Apache-2.0
"""Debug check of the fused attention front against the stock kernels
(``SGLANG_MINIMAX_M3_LOW_BATCH_FRONT_COMPARE=1``, eager calls only).

After a front launch, the stock chain runs on the same input: finalize +
all-reduce + residual add + norm, the fused QKV + index GEMM, the grouped
head norms + RoPE with FP8 Q, and the NVFP4 stores into scratch pages. The
log line gives the residual's and the dequantized K/V/index-K pages' cosine
and max error, the FP8 queries' cosine, and the fraction of identical NVFP4
bytes. Every TP rank runs it at the same calls (the stock all-reduce is a
collective).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

PAGE = 128
_calls: dict = {}  # logged calls per row count
_E2M1 = None


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[n, 64] E2M1 bytes (low nibble first) and [n, 8] E4M3 scales -> [n, 128]."""
    global _E2M1
    if _E2M1 is None or _E2M1.device != codes.device:
        mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        _E2M1 = torch.tensor(mag + [-m for m in mag], device=codes.device)
    c = codes.to(torch.int64)
    vals = torch.stack((_E2M1[c & 15], _E2M1[c >> 4]), dim=-1).reshape(c.shape[0], 128)
    sf = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sf


def _stats(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return f"cos={cos:.6f} max_abs={(a - b).abs().max().item():.4g}"


def log(layer, owed, residual, forward_batch, out, r_mid, kv, ik, slots) -> None:
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope_grouped_fp8,
    )
    from sglang.kernels.ops.communication.all_reduce_fusion import (
        moe_finalize_all_reduce_residual_norm,
    )
    from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import nvfp4_store_kv_index
    from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
    from sglang.srt.models import minimax_m3 as m3
    from sglang.srt.runtime_context import get_parallel

    attn = layer.self_attn
    rows = owed.m
    top_k = owed.expert_weights.shape[-1]
    normed, res_ref = moe_finalize_all_reduce_residual_norm(
        owed.routed_output,
        owed.permuted_indices.reshape(-1),
        owed.expert_weights,
        top_k,
        owed.gated_shared_output,
        residual,
        fused_norm_gamma(layer.input_layernorm),
        layer.input_layernorm.variance_epsilon,
        world_size=get_parallel().tp_size,
        hidden_dim=residual.shape[-1],
        residual_out=torch.empty_like(residual),
    )
    fused = attn.fused_qkv_index_proj(normed)
    q8 = fused.new_empty(
        (rows, attn.num_heads, attn.head_dim), dtype=torch.float8_e4m3fn
    )
    iq8 = fused.new_empty(
        (rows, attn.num_idx_heads, attn.idx_head_dim), dtype=torch.float8_e4m3fn
    )
    minimax_qknorm_rope_grouped_fp8(
        fused,
        attn._qknorm_groups(),
        attn.rotary_emb.cos_sin_cache,
        forward_batch.positions,
        attn.q_norm.variance_epsilon,
        fp8_a=q8,
        group_a=m3._QKNORM_GROUP_MAIN_Q,
        fp8_b=iq8,
        group_b=m3._QKNORM_GROUP_INDEX_Q,
    )
    nq = attn.num_heads * 128
    k = fused[:, nq : nq + 128].reshape(rows, 1, 128)
    v = fused[:, nq + 128 : nq + 256].reshape(rows, 1, 128)
    idx_k = fused[:, -128:]

    # Stock stores into scratch pages: token i at page i, its own row.
    dev = fused.device
    sk = torch.zeros(rows, 1, PAGE, 64, dtype=torch.uint8, device=dev)
    sks = torch.zeros(rows, 1, PAGE, 8, dtype=torch.uint8, device=dev)
    sv, svs = torch.zeros_like(sk), torch.zeros_like(sks)
    sik = torch.zeros(rows, PAGE, 64, dtype=torch.uint8, device=dev)
    siks = torch.zeros(rows, PAGE, 8, dtype=torch.uint8, device=dev)
    row = slots % PAGE
    loc = torch.arange(rows, device=dev) * PAGE + row
    one = torch.ones(1, dtype=torch.float32, device=dev)
    nvfp4_store_kv_index(
        k, v, idx_k, sk, sks, sv, svs, sik, siks, loc, one, page_size=PAGE
    )

    # The front's bytes, read back from the pool.
    from .layer import sparse_kv_pool

    kv_pool = sparse_kv_pool(forward_batch)
    _, pv, pks, pvs = kv_pool.main_pool.get_raw_kv_buffers(attn.attn.layer_id)
    _, piks = kv_pool.index_k_pool.get_raw_k_buffers(
        kv_pool.index_k_layer_id_mapping[attn.attn.layer_id]
    )
    page = slots // PAGE
    ar = torch.arange(rows, device=dev)
    got = {
        "K": (kv[page, 0, row], pks[page, 0, row]),
        "V": (pv[page, 0, row], pvs[page, 0, row]),
        "iK": (ik[page, row], piks[page, row]),
    }
    ref = {
        "K": (sk[ar, 0, row], sks[ar, 0, row]),
        "V": (sv[ar, 0, row], svs[ar, 0, row]),
        "iK": (sik[ar, row], siks[ar, row]),
    }
    parts = [f"residual {_stats(r_mid, res_ref)}"]
    parts.append(f"q {_stats(out.q_fp8.float(), q8.reshape(rows, -1).float())}")
    parts.append(f"iq {_stats(out.index_q_fp8.float(), iq8.reshape(rows, -1).float())}")
    for name in ("K", "V", "iK"):
        gc, gs = got[name]
        rc, rs = ref[name]
        same = (
            (torch.cat((gc, gs), -1) == torch.cat((rc, rs), -1)).float().mean().item()
        )
        parts.append(
            f"{name} {_stats(_dequant(gc, gs), _dequant(rc, rs))} same={same:.4f}"
        )
    _calls[rows] = _calls.get(rows, 0) + 1
    cos_r = torch.nn.functional.cosine_similarity(
        r_mid.float().flatten(), res_ref.float().flatten(), dim=0
    ).item()
    if _calls[rows] <= 200 or cos_r < 0.999:
        logger.info(
            "fused front compare layer=%s rows=%d %s",
            layer.layer_id,
            rows,
            " | ".join(parts),
        )
