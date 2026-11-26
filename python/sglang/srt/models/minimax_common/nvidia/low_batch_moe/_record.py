# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (vllm/models/deepseek_v41/nvidia/specialized_decode/moe/
# routed_expert_gemm.py and vllm/models/minimax_m3/nvidia/specialized_decode/
# __init__.py), Copyright contributors to the vLLM project.
"""Shape constants and routing-record sizing of the MiniMax-M3 low-batch MoE."""

HIDDEN = 6144
EXPERTS = 128
TOP_K = 4
INTERMEDIATE_PER_RANK = 768  # TP4
TILE = 8
MAX_ROWS = 32
WIDTHS = (8, 16, 32)


def width(rows: int) -> int:
    return next(w for w in WIDTHS if rows <= w)


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    return 4 + 3 * cta_words(ctas) + TILE * ctas
