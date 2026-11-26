# SPDX-License-Identifier: Apache-2.0
"""Low-batch (1-32 rows) routed MoE of MiniMax-M3.1-Flash at TP4 on SM100.

The CuTe DSL kernels are ported from vLLM's MiniMax-M3 ``specialized_decode``
path (vllm-project/vllm#66324) and its DeepSeek-V4.1 low-batch MoE skeleton,
Apache-2.0, Copyright contributors to the vLLM project. Each ported file keeps
its upstream header and names its source.

Import is deferred to first use (``layer.get_low_batch_moe``): the package
pulls in ``cutlass`` and compiles kernels for the current device.
"""
