# SPDX-License-Identifier: Apache-2.0
"""Low-batch (1-32 rows) attention output of MiniMax-M3.1-Flash at TP4 on SM100.

o_proj runs FlashInfer's low-latency MXFP8 GEMM with a fixed tactic per
activation tile, the way vLLM's MiniMax-M3 ``specialized_decode`` attention
does (vllm-project/vllm, Apache-2.0, Copyright contributors to the vLLM
project). Under PDL only its activation loads wait for the producer kernel, so
the 12 MB of o_proj weights per rank stream in while the attention (or the
input quantize) is still running.

Import is deferred to first use (``layer.get_low_batch_o_proj``): the package
pulls in ``cutlass`` and compiles kernels for the current device.
"""
