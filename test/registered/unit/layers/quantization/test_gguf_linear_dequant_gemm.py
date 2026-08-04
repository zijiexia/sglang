"""GGUF linear dequant+GEMM routing tests.

fused_mul_mat_gguf routes large-M batches to dequant + bf16 GEMM via
_use_linear_dequant_gemm, which gates on the measured (backend, quant type)
allowlist, a per-device-capability M threshold
(_resolve_linear_dequant_gemm_min_tokens: SM121 = 60, general CUDA = 512),
and a dense-scratch cap. Graph safety is by construction (the predicate is
a pure function of static per-graph values), pinned here by a capture-and-
replay test at the DSpark verify width. Numerical agreement is checked for
both routes against explicit dequantize + matmul.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import unittest

import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import quantize

from sglang.srt.layers.quantization import gguf as gguf_quant
from sglang.test.test_utils import CustomTestCase

# The device's active threshold (60 on SM121, 512 elsewhere): boundary tests
# must hold on every CUDA CI runner, so they key off the resolved value.
_MIN = gguf_quant._linear_dequant_gemm_min_tokens()


def _q8_0_qweight(
    rows: int, cols: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A CUDA Q8_0 packed weight plus its exact dequantized reference."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols), dtype=np.float32)
    packed = quantize(w, GGMLQuantizationType.Q8_0)
    from gguf.quants import dequantize

    ref = dequantize(packed, GGMLQuantizationType.Q8_0)
    return (
        torch.from_numpy(packed).cuda(),
        torch.from_numpy(ref.astype(np.float32)).cuda(),
    )


class TestThresholdPolicy(CustomTestCase):
    """The pure per-capability policy, independent of the host device."""

    def test_sm121_uses_measured_low_m_threshold(self):
        self.assertEqual(
            gguf_quant._resolve_linear_dequant_gemm_min_tokens((12, 1)),
            gguf_quant._LINEAR_DEQUANT_GEMM_MIN_TOKENS_SM121,
        )

    def test_other_capabilities_keep_general_policy(self):
        for cap in [(8, 0), (9, 0), (10, 0), (12, 0)]:
            self.assertEqual(
                gguf_quant._resolve_linear_dequant_gemm_min_tokens(cap),
                gguf_quant._LINEAR_DEQUANT_GEMM_MIN_TOKENS,
            )

    def test_general_policy_is_prefill_scale(self):
        self.assertEqual(gguf_quant._LINEAR_DEQUANT_GEMM_MIN_TOKENS, 512)
        self.assertEqual(gguf_quant._LINEAR_DEQUANT_GEMM_MIN_TOKENS_SM121, 60)


class TestLinearDequantGemmPredicate(CustomTestCase):
    """Predicate gates at the device's resolved threshold."""

    def _x(self, m: int) -> torch.Tensor:
        return torch.zeros(m, 64, dtype=torch.bfloat16, device="cuda")

    def setUp(self):
        self.qweight, _ = _q8_0_qweight(64, 64)
        self.q8_0 = int(GGMLQuantizationType.Q8_0)

    def test_m_boundary(self):
        self.assertFalse(
            gguf_quant._use_linear_dequant_gemm(
                self._x(_MIN - 1), self.qweight, self.q8_0
            )
        )
        self.assertTrue(
            gguf_quant._use_linear_dequant_gemm(self._x(_MIN), self.qweight, self.q8_0)
        )

    def test_type_allowlist(self):
        q2k = int(GGMLQuantizationType.Q2_K)
        self.assertFalse(
            gguf_quant._use_linear_dequant_gemm(self._x(_MIN), self.qweight, q2k)
        )

    def test_scratch_cap_excludes_lm_head_sized_weights(self):
        # DS4-Flash lm_head is 129280x4096 -> 1.06GB bf16, over the 256MB cap.
        # The predicate only reads .shape, so meta tensors suffice.
        import gguf as gguf_pkg

        block_size, type_size = gguf_pkg.GGML_QUANT_SIZES[self.q8_0]
        lm_head_like = torch.empty(
            129280, 4096 // block_size * type_size, dtype=torch.uint8, device="meta"
        )
        self.assertFalse(
            gguf_quant._use_linear_dequant_gemm(self._x(_MIN), lm_head_like, self.q8_0)
        )
        # And the largest real DS4-Flash prefill linear (wq_b 32768x1024,
        # 64MB bf16) stays under the cap.
        wq_b_like = torch.empty(
            32768, 1024 // block_size * type_size, dtype=torch.uint8, device="meta"
        )
        self.assertTrue(
            gguf_quant._use_linear_dequant_gemm(self._x(_MIN), wq_b_like, self.q8_0)
        )

    def test_predicate_is_static_per_shape(self):
        # Graph safety is by construction: the predicate depends only on
        # static per-graph values, so capture and replay cannot diverge.
        x = self._x(_MIN)
        first = gguf_quant._use_linear_dequant_gemm(x, self.qweight, self.q8_0)
        self.assertTrue(first)
        for _ in range(3):
            self.assertEqual(
                gguf_quant._use_linear_dequant_gemm(x, self.qweight, self.q8_0),
                first,
            )


class TestLinearDequantGemmNumerics(CustomTestCase):
    ROWS, COLS = 512, 256

    def setUp(self):
        self.qweight, self.ref_w = _q8_0_qweight(self.ROWS, self.COLS, seed=7)
        self.q8_0 = int(GGMLQuantizationType.Q8_0)

    def _run(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (torch.randn(m, self.COLS, dtype=torch.float32, device="cuda") * 0.1).to(
            torch.bfloat16
        )
        y = gguf_quant.fused_mul_mat_gguf(x, self.qweight, self.q8_0)
        ref = x.to(torch.float32) @ self.ref_w.T
        return y.to(torch.float32), ref

    def test_empty_input(self):
        y = gguf_quant.fused_mul_mat_gguf(
            torch.zeros(0, self.COLS, dtype=torch.bfloat16, device="cuda"),
            self.qweight,
            self.q8_0,
        )
        self.assertEqual(tuple(y.shape), (0, self.ROWS))

    def test_gemm_route_matches_explicit_dequant_matmul(self):
        # M = threshold -> dequant+GEMM route; only bf16 rounding vs the
        # fp32 reference on exactly-dequantized weights.
        y, ref = self._run(_MIN)
        cos = torch.nn.functional.cosine_similarity(
            y.reshape(-1), ref.reshape(-1), dim=0
        )
        self.assertGreater(cos.item(), 0.999)

    def test_mmq_route_agrees_below_threshold(self):
        # M = threshold-1 -> MMQ route; wider tolerance (Q8_1-quantized
        # activations), same reference.
        y, ref = self._run(_MIN - 1)
        cos = torch.nn.functional.cosine_similarity(
            y.reshape(-1), ref.reshape(-1), dim=0
        )
        self.assertGreater(cos.item(), 0.99)

    def test_graph_capture_replay(self):
        # The DSpark target-verify graph captures this route at M=72 on
        # SM121. Pin: (1) the route is numerically correct when captured
        # and replayed, and (2) replays are self-consistent. Skipped where
        # the device threshold keeps M=72 on MMQ (general CUDA policy).
        m = 72
        x_static = (
            torch.randn(m, self.COLS, dtype=torch.float32, device="cuda") * 0.1
        ).to(torch.bfloat16)
        if not gguf_quant._use_linear_dequant_gemm(x_static, self.qweight, self.q8_0):
            self.skipTest("device threshold keeps M=72 on MMQ (non-SM121)")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                gguf_quant.fused_mul_mat_gguf(x_static, self.qweight, self.q8_0)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_static = gguf_quant.fused_mul_mat_gguf(x_static, self.qweight, self.q8_0)
        g.replay()
        torch.cuda.synchronize()
        ref = x_static.to(torch.float32) @ self.ref_w.T
        cos = torch.nn.functional.cosine_similarity(
            y_static.to(torch.float32).reshape(-1), ref.reshape(-1), dim=0
        )
        self.assertGreater(cos.item(), 0.999)
        first = y_static.clone()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(first, y_static))

    def test_routes_agree_with_each_other(self):
        x = (torch.randn(_MIN, self.COLS, dtype=torch.float32, device="cuda") * 0.1).to(
            torch.bfloat16
        )
        y_gemm = gguf_quant.fused_mul_mat_gguf(x, self.qweight, self.q8_0)
        # Force the MMQ route on the identical input via the type allowlist.
        orig = gguf_quant._LINEAR_DEQUANT_GEMM_TYPES
        gguf_quant._LINEAR_DEQUANT_GEMM_TYPES = frozenset()
        try:
            y_mmq = gguf_quant.fused_mul_mat_gguf(x, self.qweight, self.q8_0)
        finally:
            gguf_quant._LINEAR_DEQUANT_GEMM_TYPES = orig
        cos = torch.nn.functional.cosine_similarity(
            y_gemm.to(torch.float32).reshape(-1),
            y_mmq.to(torch.float32).reshape(-1),
            dim=0,
        )
        self.assertGreater(cos.item(), 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
