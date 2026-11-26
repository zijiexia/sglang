# SPDX-License-Identifier: Apache-2.0
"""Fused attention front (1-16 rows) of MiniMax-M3.1-Flash at TP4 on SM100.

The CuTe DSL kernel and its exchange/transport helpers are ported from vLLM's
MiniMax-M3 ``specialized_decode`` front (vllm-project/vllm#66324), Apache-2.0,
Copyright contributors to the vLLM project. Each ported file keeps its
upstream header and names its source; the NVFP4 epilogue writes SGLang's
pool layout and block encoding.

``layer.py`` binds it to the model (``SGLANG_MINIMAX_M3_LOW_BATCH_FRONT``);
the kernel modules import ``cutlass`` and are loaded only when it is on.
"""
