"""Smoke test of the MiniMax-M3 low-batch routed MoE kernels (SM100 only).

Random NVFP4 weights in the TRT-LLM-gen layout; checks the routing weights
(renormalized top-4 times the routed scale), the permuted indices and that
every routed output row is finite, at each row count the kernels are built
for. The numerical match against the FlashInfer path is checked in a server
with SGLANG_MINIMAX_M3_LOW_BATCH_MOE_COMPARE=1.

    python3 test/manual/minimax_m3/test_low_batch_moe.py
"""

import unittest

import torch


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@unittest.skipUnless(_sm100(), "needs an SM100 GPU")
class TestLowBatchMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models.minimax_common.nvidia.low_batch_moe.runner import (
            LowBatchRoutedMoe,
        )

        torch.manual_seed(0)
        d, e, h, i = "cuda", 128, 6144, 768
        w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device=d)
        w13_scale = torch.full((e, 2 * i, h // 16), 0x38, dtype=torch.uint8, device=d)
        w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device=d)
        w2_scale = torch.full((e, h, i // 16), 0x38, dtype=torch.uint8, device=d)
        alpha = torch.full((e,), 1e-3, device=d)
        cls.bias = torch.randn(e, device=d) * 0.01
        cls.moe = LowBatchRoutedMoe(
            w13, w13_scale, w2, w2_scale, alpha, alpha, cls.bias, 2.0
        )
        cls.experts, cls.hidden = e, h

    def test_rows(self):
        for rows in (1, 7, 8, 16, 24, 32):
            with self.subTest(rows=rows):
                x = torch.randn(rows, self.hidden, dtype=torch.bfloat16, device="cuda")
                logits = torch.randn(rows, self.experts, device="cuda")
                out, weights, permuted = self.moe.run(x, logits)
                torch.cuda.synchronize()
                self.assertEqual(tuple(weights.shape), (rows, 4))
                torch.testing.assert_close(
                    weights.float().sum(1),
                    torch.full((rows,), 2.0, device="cuda"),
                    atol=1e-2,
                    rtol=0,
                )
                # Weights: the top-4 of sigmoid(logit) + bias, renormalized
                # unbiased scores times the routed scale.
                scores = torch.sigmoid(logits)
                idx = (scores + self.bias).topk(4, dim=1).indices
                s = scores.gather(1, idx)
                ref = (2.0 * s / s.sum(1, keepdim=True)).sort(1).values
                torch.testing.assert_close(
                    weights.float().sort(1).values, ref, atol=2e-2, rtol=0
                )
                flat = permuted.reshape(-1).long()
                self.assertEqual(flat.unique().numel(), rows * 4)
                self.assertTrue(bool((flat < out.shape[0]).all()))
                self.assertTrue(bool(torch.isfinite(out[flat]).all()))


if __name__ == "__main__":
    unittest.main()
