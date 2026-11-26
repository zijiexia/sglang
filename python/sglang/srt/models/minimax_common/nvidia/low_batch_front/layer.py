# SPDX-License-Identifier: Apache-2.0
"""Binds the fused attention front to MiniMax-M3 decoder layers.

``SGLANG_MINIMAX_M3_LOW_BATCH_FRONT=1``: on a sparse attention layer whose
input is the previous MoE layer's deferred finalize, one launch
(``front.QkvFrontStage``) replaces, for decode and target-verify batches of
1-16 rows:

* the MoE finalize + TP all-reduce + residual add + Gemma RMSNorm
  (``moe_finalize_all_reduce_residual_norm``),
* the MXFP8 quantization of the normed row and the fused QKV + index GEMM,
* the per-head Gemma RMSNorm + RoPE with the FP8 Q / index-Q outputs
  (``fused_gemma_qknorm_rope``),
* the NVFP4 K/V and index-K stores (``nvfp4_store_kv_index``).

The fronts are built once after weight loading, collectively over the TP
group (``build_fronts``): every rank allocates the same peer memory and keeps
only the layers that built on every rank. Any other layer, batch or shape
keeps the stock path; the per-call decision reads host state only and is the
same on every rank.

``SGLANG_MINIMAX_M3_LOW_BATCH_FRONT_COMPARE=1`` also runs the stock kernels
on eager front calls (scratch cache rows, no side effects) and logs the
error of the residual, Q, index-Q and the stored K/V/index-K pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

HIDDEN = 6144
HEAD_DIM = 128
PAGE = 128
BUCKETS = (8, 16)
MAX_ROWS = 16
MAX_TOP_K = 8
_ARENA_ALIGN = 256


def enabled() -> bool:
    return envs.SGLANG_MINIMAX_M3_LOW_BATCH_FRONT.get()


@dataclass
class FrontOutput:
    """What the fused front hands the attention in place of its normed input:
    the E4M3 queries ([rows, nq * 128], [rows, niq * 128]); K/V and index-K
    are already in the cache."""

    q_fp8: torch.Tensor
    index_q_fp8: torch.Tensor
    is_low_batch_front_output = True  # read by MiniMaxM3Attention.forward_prepare

    @property
    def shape(self):
        return self.q_fp8.shape


@dataclass
class _LayerFront:
    stages: dict
    weights: dict
    q_fp8: torch.Tensor  # e4m3 [16, nq * 128]
    index_q: torch.Tensor  # e4m3 [16, niq * 128]


class FusedFronts:
    """The built fronts of a model's local decoder layers (keyed by layer)."""

    def __init__(self, layers: dict[int, _LayerFront], transport):
        self.layers = layers
        self.transport = transport


_fronts: Optional[FusedFronts] = None
_checked_pool: set = set()


def _unsupported_reason(prev, layer) -> Optional[str]:
    attn = layer.self_attn
    if not getattr(attn, "is_sparse_attention_layer", False):
        return "dense attention"
    if not getattr(prev, "is_layer_sparse", False):
        return "the previous layer is not a MoE layer"
    if not getattr(prev.mlp, "supports_deferred_finalize", False):
        return "the previous MoE does not defer its finalize"
    if not getattr(layer, "use_gemma_norm", False):
        return "not a Gemma norm"
    if attn._fused_qkv_index is None:
        return "no fused QKV + index projection"
    if not attn._fp8_q_epilogue_ok:
        return "no FP8 Q epilogue (NVFP4 native stack)"
    if not attn.disable_index_value:
        return "index value heads"
    if (attn.head_dim, attn.idx_head_dim, attn.num_kv_heads) != (HEAD_DIM, HEAD_DIM, 1):
        return f"head geometry {attn.head_dim}/{attn.idx_head_dim}/{attn.num_kv_heads}"
    if not attn.rotary_emb.is_neox_style:
        return "non-NeoX RoPE"
    cs = attn.rotary_emb.cos_sin_cache
    if cs.dtype != torch.float32 or not cs.is_contiguous():
        return f"RoPE table {cs.dtype}"
    return None


def _candidates(model) -> list:
    layers = list(model.layers)
    out = []
    for prev, layer in zip(layers, layers[1:]):
        if not hasattr(layer, "self_attn") or not hasattr(prev, "mlp"):
            continue
        reason = _unsupported_reason(prev, layer)
        if reason is None:
            out.append(layer)
        elif getattr(layer.self_attn, "is_sparse_attention_layer", False):
            logger.info(
                "MiniMax-M3 fused attention front off for layer %s: %s",
                layer.layer_id,
                reason,
            )
    return out


def _qkv_operands(attn) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused projection's e4m3 weight [N, K] and its UE8M0 scales in the
    F8_128x4 swizzle (rows padded to 128), from the canonical [N, K / 32]."""
    from flashinfer import block_scale_interleave

    proj = attn._fused_qkv_index
    w = proj.weight.data
    s = getattr(proj, "weight_scale_inv", None)
    if hasattr(proj, "weight_scale_inv_shuffled"):
        raise ValueError("the QKV weight is TRT-LLM shuffled")
    if w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_contiguous():
        raise ValueError(
            f"QKV weight {w.dtype} {tuple(w.shape)} (MXFP8 [N, K] expected)"
        )
    if s is None:
        raise ValueError("no QKV weight scales")
    n, k = w.shape
    s = s.data.contiguous().view(torch.uint8).reshape(n, k // 32)
    sf = block_scale_interleave(s).contiguous().view(torch.uint8).reshape(-1)
    want = (n + 127) // 128 * 128 * ((k // 32 + 3) // 4 * 4)
    if sf.numel() != want:
        raise ValueError(f"{sf.numel()} swizzled scale bytes, expected {want}")
    return w, sf


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


def _build_layer(layer, transport) -> _LayerFront:
    from .front import QkvFrontStage

    attn = layer.self_attn
    heads = attn.num_heads + 2 * attn.num_kv_heads + attn.num_idx_heads + 1
    w, sf = _qkv_operands(attn)
    if tuple(w.shape) != (heads * HEAD_DIM, HIDDEN):
        raise ValueError(
            f"QKV weight {tuple(w.shape)}, expected [{heads * HEAD_DIM}, {HIDDEN}]"
        )
    rope = attn.rotary_emb
    weights = dict(
        qkv=w,
        qkv_sf=sf,
        input_norm=_bf16(layer.input_layernorm.weight),
        q_norm=_bf16(attn.q_norm.weight),
        k_norm=_bf16(attn.k_norm.weight),
        iq_norm=_bf16(attn.index_q_norm.weight),
        ik_norm=_bf16(attn.index_k_norm.weight),
        cos_sin=rope.cos_sin_cache,  # fp32, read in place
    )
    q_scale = getattr(attn.attn, "q_scale_float", None)
    kwargs = dict(
        transport=transport,
        nq=attn.num_heads,
        nkv=attn.num_kv_heads,
        niq=attn.num_idx_heads,
        rot=rope.rotary_dim,
        block_size=PAGE,
        kv_cache_dtype="nvfp4",
        indexer_kv_dtype="nvfp4",
        prepushed=False,
        eps=layer.input_layernorm.variance_epsilon,
        # The stock grouped kernel norms all four head branches with q_norm's epsilon.
        head_eps=attn.q_norm.variance_epsilon,
        q_scale=1.0 if q_scale is None else float(q_scale),
        with_step_meta=False,
    )
    stages = {b: QkvFrontStage(max_rows=b, **kwargs) for b in BUCKETS}
    for stage in stages.values():
        stage.precompile()  # outside graph capture; nothing is launched
    dev = w.device
    return _LayerFront(
        stages=stages,
        weights=weights,
        q_fp8=torch.zeros(
            MAX_ROWS, attn.num_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
        ),
        index_q=torch.zeros(
            MAX_ROWS,
            attn.num_idx_heads * HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=dev,
        ),
    )


def build_fronts(model) -> None:
    """Build the fronts of ``model`` (a ``MiniMaxM3Model``). Collective over
    the TP group: every rank calls it after weight loading and before graph
    capture; a layer that fails on any rank stays stock on every rank."""
    global _fronts
    if not enabled():
        return
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    group = parallel.tp_group
    world, rank = group.world_size, group.rank_in_group
    cpu_group = group.cpu_group

    def all_ranks(flags: list[bool]) -> list[bool]:
        t = torch.tensor([int(f) for f in flags], dtype=torch.int32)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
        return [bool(v) for v in t.tolist()]

    def off(why: str) -> None:
        logger.warning(
            "SGLANG_MINIMAX_M3_LOW_BATCH_FRONT=1 has no effect: %s; the sparse "
            "layers keep the stock boundary, QKV and KV stores.",
            why,
        )

    layers = _candidates(model)
    ready = True
    try:
        from . import _arch, _dtypes, front

        _arch.device_spec()
        front._ext(_dtypes.CacheFormat.NVFP4, world)
        if parallel.attn_tp_size != parallel.tp_size or not 1 < world <= 16:
            raise ValueError(f"tensor parallel {world} with attention TP only")
    except Exception as err:
        logger.warning("MiniMax-M3 fused attention front unavailable: %s", err)
        ready = False
    if not all_ranks([ready and bool(layers)])[0]:
        off("the kernel cannot run on some rank, or no sparse layer qualifies")
        return

    from ._comm import make_transport
    from ._exchange import exchange_words, push_workspace

    capacity = sum(4 * exchange_words(world, b) + _ARENA_ALIGN for b in BUCKETS)
    try:
        transport = make_transport(
            "symm_mem",
            group=group.device_group,
            rank=rank,
            world=world,
            capacity=capacity,
        )
        for b in BUCKETS:
            push_workspace(transport, b, "front")
        peer_ok = True
    except Exception:
        logger.exception("MiniMax-M3 fused attention front: peer memory setup failed")
        transport, peer_ok = None, False
    if not all_ranks([peer_ok])[0]:
        off("peer memory setup failed on some rank")
        return

    built: dict[int, _LayerFront] = {}
    oks = []
    for layer in layers:
        try:
            built[id(layer)] = _build_layer(layer, transport)
            oks.append(True)
        except Exception:
            logger.exception(
                "MiniMax-M3 fused attention front did not build for layer %s",
                layer.layer_id,
            )
            oks.append(False)
    for layer, ok in zip(layers, all_ranks(oks)):
        if not ok:
            built.pop(id(layer), None)
    if not built:
        off("the front did not build on some rank")
        return
    _fronts = FusedFronts(built, transport)
    logger.info(
        "MiniMax-M3 fused attention front (finalize + all-reduce + norm, QKV, head "
        "norms, RoPE, NVFP4 K/V and index-K stores in one launch) built for %d "
        "sparse layers, 1-%d rows, %s transport of %d ranks.",
        len(built),
        MAX_ROWS,
        transport.kind,
        world,
    )


def has_front(layer) -> bool:
    return _fronts is not None and id(layer) in _fronts.layers


def _pool_views(kv_pool, layer_id: int):
    """The layer's K payload and index-K payload views (the kernel's page-row
    bases), after checking once per pool that the scales and V sit where the
    kernel writes them."""
    main, index = kv_pool.main_pool, kv_pool.index_k_pool
    k, v, ks, vs = main.get_raw_kv_buffers(layer_id)
    ik, iks = index.get_raw_k_buffers(kv_pool.index_k_layer_id_mapping[layer_id])
    key = id(kv_pool)
    if key not in _checked_pool:
        b = PAGE * HEAD_DIM // 2  # one plane's codes
        base, ibase = k.data_ptr(), ik.data_ptr()
        ok = (
            main.page_size == PAGE
            and tuple(k.shape[1:]) == (1, PAGE, HEAD_DIM // 2)
            and ks.data_ptr() == base + b
            and v.data_ptr() == base + b + PAGE * 8
            and vs.data_ptr() == base + 2 * b + PAGE * 8
            and tuple(ks.stride()[1:]) == (PAGE * 8, 8, 1)
            and tuple(vs.stride()[1:]) == (PAGE * 8, 8, 1)
            and iks.data_ptr() == ibase + b
            and tuple(iks.stride()[1:]) == (8, 1)
        )
        if not ok:
            raise ValueError("NVFP4 pool layout differs from the front's page rows")
        # The pool's outer scales are fixed ones (unit-scale MSA-v1 NVFP4).
        _checked_pool.add(key)
    return k, ik


def sparse_kv_pool(forward_batch=None):
    """The NVFP4 MiniMax-M3 pool (main K/V and index-K) of the active attention
    backend, or None. A wrapping backend (hybrid or speculative) holds the
    sparse one as an attribute; the pool object is fixed, so it is kept."""
    global _pool
    if _pool is not None:
        return _pool
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    if not has_forward_context():
        return None
    seen, todo = set(), [get_forward_context().attn_backend]
    while todo:
        b = todo.pop()
        if b is None or id(b) in seen or len(seen) > 64:
            continue
        seen.add(id(b))
        pool = getattr(b, "kv_pool", None)
        if isinstance(pool, MiniMaxM3Nvfp4KVPool):
            _pool = pool
            return pool
        for v in vars(b).values() if hasattr(b, "__dict__") else ():
            if hasattr(v, "init_forward_metadata"):
                todo.append(v)
            elif isinstance(v, (list, tuple)):
                todo.extend(x for x in v if hasattr(x, "init_forward_metadata"))
    return None


_pool = None


_declined: set = set()


def _decline(why: str) -> None:
    """Log each reason the front leaves a batch to the stock path, once."""
    if why not in _declined:
        _declined.add(why)
        logger.info("MiniMax-M3 fused attention front declined a batch: %s", why)
    return None


def try_run(layer, owed, residual: torch.Tensor, forward_batch):
    """The fused front of ``layer`` on the current stream: (FrontOutput, new
    residual), or None (nothing launched) for the stock path. ``owed`` is the
    previous MoE's ``MoeDeferredFinalize``."""
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )

    if _fronts is None:
        return _decline("not built")
    lf = _fronts.layers.get(id(layer))
    if lf is None or not isinstance(owed, MoeDeferredFinalize):
        return _decline("layer has no front or input is not a deferred MoE finalize")
    rows = owed.m
    positions = forward_batch.positions
    loc = forward_batch.out_cache_loc
    if not 0 < rows <= MAX_ROWS:
        return None  # the stock path serves larger batches
    if not (
        residual is not None
        and residual.dtype == torch.bfloat16
        and tuple(residual.shape) == (rows, HIDDEN)
    ):
        return _decline("residual shape or dtype")
    if positions is None or positions.shape[0] != rows or loc is None:
        return _decline("positions or cache locations do not match the rows")
    routed = owed.routed_output
    perm = owed.permuted_indices
    weights = owed.expert_weights
    top_k = weights.shape[-1]
    shared = owed.gated_shared_output
    if not (
        routed.dtype == torch.bfloat16
        and routed.dim() == 2
        and routed.stride() == (HIDDEN, 1)
        and perm.dtype == torch.int32
        and weights.dtype in (torch.float32, torch.bfloat16)
        and perm.numel() == rows * top_k
        and weights.numel() == rows * top_k
        and 0 < top_k <= MAX_TOP_K
        and (
            shared is None
            or (
                shared.dtype == torch.bfloat16 and tuple(shared.shape) == (rows, HIDDEN)
            )
        )
    ):
        return _decline("MoE handoff layout")
    from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

    from ._exchange import bucket
    from .front import FinalizeIn

    attn = layer.self_attn
    kv_pool = sparse_kv_pool(forward_batch)
    if kv_pool is None:
        return _decline("no NVFP4 MiniMax-M3 KV pool behind the attention backend")
    write_loc, _, _ = unwrap_write_loc(loc)
    slots = write_loc.reshape(-1)
    if slots.dtype != torch.int64:
        slots = slots.to(torch.int64)
    if positions.dtype != torch.int64:
        return _decline("positions are not int64")
    kv, ik = _pool_views(kv_pool, attn.attn.layer_id)
    w = lf.weights
    finalize = FinalizeIn(
        fc2=routed,
        perm=perm.reshape(-1).contiguous(),
        scales=weights.reshape(-1).contiguous(),
        topk=top_k,
    )
    compare = (
        envs.SGLANG_MINIMAX_M3_LOW_BATCH_FRONT_COMPARE.get()
        and not torch.cuda.is_current_stream_capturing()
    )
    # A fresh residual, as the stock boundary's: hidden-state captures may keep it.
    r_mid = torch.empty_like(residual)
    lf.stages[bucket(rows)](
        rows,
        residual=residual.contiguous(),
        norm_w=w["input_norm"],
        out_residual=r_mid,
        w=w["qkv"],
        wsf=w["qkv_sf"],
        q_w=w["q_norm"],
        k_w=w["k_norm"],
        iq_w=w["iq_norm"],
        ik_w=w["ik_norm"],
        cos_sin=w["cos_sin"],
        positions=positions,
        slot_mapping=slots,
        index_slot_mapping=slots,
        kv_cache=kv,
        index_cache=ik,
        q_fp8=lf.q_fp8[:rows],
        index_q=lf.index_q[:rows],
        x2=None if shared is None else shared.contiguous(),
        finalize=finalize,
        single_round=True,
    )
    attn._mark_sparse_kv_cached_by_fusion(forward_batch, attn.attn.layer_id)
    out = FrontOutput(lf.q_fp8[:rows], lf.index_q[:rows])
    if compare:
        from . import compare as _compare

        _compare.log(layer, owed, residual, forward_batch, out, r_mid, kv, ik, slots)
    return out, r_mid
