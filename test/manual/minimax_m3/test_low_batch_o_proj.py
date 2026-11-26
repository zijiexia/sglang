"""Numerical test of the MiniMax-M3 low-latency MXFP8 o_proj (SM100 only).

A random o_proj weight [6144, 2048] quantized to MXFP8 the way SGLang loads it
(canonical UE8M0 scales, then ``block_scale_interleave``), and BF16 attention
outputs at every row count the kernel serves. The output is checked against
the FP32 product of the dequantized operands and against FlashInfer's
``mm_mxfp8`` (the stock SGLang o_proj path), under eager launches and a CUDA
graph replay.

    python3 test/manual/minimax_m3/test_low_batch_o_proj.py
"""

import unittest

import torch

N, K = 6144, 2048


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _dequant(data: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """FP32 value of E4M3 ``data`` [m, k] with canonical UE8M0 ``scale_u8``
    [m, k // 32]."""
    m, k = data.shape
    scale = torch.exp2(scale_u8[:m].float() - 127.0)
    return (data.float().view(m, k // 32, 32) * scale.unsqueeze(-1)).view(m, k)


@unittest.skipUnless(_sm100(), "needs an SM100 GPU")
class TestLowBatchOProj(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer import block_scale_interleave, mxfp8_quantize

        from sglang.srt.models.minimax_common.nvidia.low_batch_attn_out.layer import (
            LowBatchOProj,
        )

        torch.manual_seed(0)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
        w_q, w_scale = mxfp8_quantize(w, is_sf_swizzled_layout=False, alignment=32)
        w_scale = w_scale.view(torch.uint8).view(N, K // 32)
        cls.w_ref = _dequant(w_q, w_scale)
        cls.w_q = w_q
        cls.w_swizzled = block_scale_interleave(w_scale.contiguous()).view(torch.uint8)
        cls.o_proj = LowBatchOProj(w_q, cls.w_swizzled)

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer import mxfp8_quantize

        x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=32)
        x_scale = x_scale.view(torch.uint8).view(-1, K // 32)
        return _dequant(x_q, x_scale) @ self.w_ref.t()

    def _stock(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer import mm_mxfp8, mxfp8_quantize

        x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        return mm_mxfp8(
            x_q,
            self.w_q.t(),
            x_scale,
            self.w_swizzled,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
        )

    def _check(self, out: torch.Tensor, x: torch.Tensor) -> None:
        ref = self._reference(x)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0
        )
        self.assertGreater(cos.item(), 0.9999)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
        stock = self._stock(x).float()
        torch.testing.assert_close(out.float(), stock, atol=2e-2, rtol=2e-2)

    def test_eager(self):
        for rows in (1, 2, 7, 8, 9, 16, 17, 24, 32):
            with self.subTest(rows=rows):
                x = torch.randn(rows, K, dtype=torch.bfloat16, device="cuda")
                out = self.o_proj.run(x)
                torch.cuda.synchronize()
                self.assertEqual(tuple(out.shape), (rows, N))
                self._check(out, x)

    def test_prequantized_input(self):
        from flashinfer import mxfp8_quantize

        x = torch.randn(16, K, dtype=torch.bfloat16, device="cuda")
        x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        out = self.o_proj.run((x_q, x_scale.view(torch.uint8)))
        torch.cuda.synchronize()
        self._check(out, x)

    def test_cuda_graph(self):
        for rows in (1, 16, 32):
            with self.subTest(rows=rows):
                x = torch.randn(rows, K, dtype=torch.bfloat16, device="cuda")
                self.o_proj.run(x)  # compile and warm up outside the capture
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.o_proj.run(x)
                x.copy_(torch.randn_like(x))
                graph.replay()
                torch.cuda.synchronize()
                self._check(out, x)


if __name__ == "__main__":
    unittest.main()
