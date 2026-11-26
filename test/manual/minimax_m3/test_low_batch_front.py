"""Single-GPU check of the MiniMax-M3 fused attention front (SM100 only).

One rank (loopback transport, so the all-reduce is the identity) with random
MXFP8 QKV weights. The reference is the stock chain on the same input: the
finalize + residual add + Gemma RMSNorm in FP32, the QKV projection on the
dequantized weight, SGLang's grouped head norms + RoPE with FP8 Q
(``minimax_qknorm_rope_grouped_fp8``) and its NVFP4 store
(``nvfp4_store_kv_index``) into a pool laid out as SGLang's MiniMax-M3.1
NVFP4 page rows. Checks the residual, Q / index-Q and the dequantized K, V
and index-K pages at every row count the front serves. A second reference
runs the stock norms, RoPE and store on the front's own projection rows
(``qkv_out``): its FP8 Q / index-Q and NVFP4 K, V and index-K bytes must
then equal the stock kernels' exactly.

    python3 test/manual/minimax_m3/test_low_batch_front.py
"""

import unittest

import torch

H, D, NQ, NIQ, PAGE, ROT, TOPK = 6144, 128, 16, 1, 128, 64, 4
HEADS = NQ + 2 + NIQ + 1
MAX_POS, PAGES = 4096, 64
EPS = 1e-6


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _e2m1_table(dev):
    mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(mag + [-m for m in mag], device=dev)


def _dequant(codes, scales):
    t = _e2m1_table(codes.device)
    c = codes.to(torch.int64)
    vals = torch.stack((t[c & 15], t[c >> 4]), dim=-1).reshape(c.shape[0], D)
    return vals * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


class _Pool:
    """SGLang's MiniMax-M3.1 NVFP4 page rows for one layer: main [K codes |
    K scales | V codes | V scales], index [codes | scales]."""

    def __init__(self, dev):
        b, s = PAGE * D // 2, PAGE * 8
        self.main = torch.zeros(PAGES, 2 * (b + s), dtype=torch.uint8, device=dev)
        self.index = torch.zeros(PAGES, b + s, dtype=torch.uint8, device=dev)
        m, i = self.main, self.index
        self.k = m[:, :b].view(PAGES, 1, PAGE, D // 2)
        self.ks = m[:, b : b + s].view(PAGES, 1, PAGE, 8)
        self.v = m[:, b + s : 2 * b + s].view(PAGES, 1, PAGE, D // 2)
        self.vs = m[:, 2 * b + s :].view(PAGES, 1, PAGE, 8)
        self.ik = i[:, :b].view(PAGES, PAGE, D // 2)
        self.iks = i[:, b:].view(PAGES, PAGE, 8)

    def rows(self, slots):
        p, r = slots // PAGE, slots % PAGE
        return {
            "K": (self.k[p, 0, r], self.ks[p, 0, r]),
            "V": (self.v[p, 0, r], self.vs[p, 0, r]),
            "iK": (self.ik[p, r], self.iks[p, r]),
        }


@unittest.skipUnless(_sm100(), "needs an SM100 GPU")
class TestLowBatchFront(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer import block_scale_interleave

        from sglang.srt.models.minimax_common.nvidia.low_batch_front._comm import (
            make_transport,
        )
        from sglang.srt.models.minimax_common.nvidia.low_batch_front._exchange import (
            exchange_words,
        )
        from sglang.srt.models.minimax_common.nvidia.low_batch_front.front import (
            QkvFrontStage,
        )

        torch.manual_seed(0)
        dev = "cuda"
        n = HEADS * D
        # MXFP8 weight: e4m3 values of magnitude ~1, UE8M0 scales 2^-6 .. 2^-4.
        cls.w = (torch.randn(n, H, device=dev) * 0.5).to(torch.float8_e4m3fn)
        cls.w_sf = torch.randint(121, 124, (n, H // 32), dtype=torch.uint8, device=dev)
        sf = block_scale_interleave(cls.w_sf).contiguous().view(torch.uint8).reshape(-1)
        cls.w_deq = cls.w.float() * torch.exp2(
            cls.w_sf.float() - 127
        ).repeat_interleave(32, dim=1)
        cls.norm_w = (torch.randn(H, device=dev) * 0.1).bfloat16()
        cls.head_w = [(torch.randn(D, device=dev) * 0.1).bfloat16() for _ in range(4)]
        inv = 1.0 / (10000 ** (torch.arange(0, ROT, 2, device=dev).float() / ROT))
        ang = torch.arange(MAX_POS, device=dev).float()[:, None] * inv[None, :]
        cls.cos_sin = torch.cat((ang.cos(), ang.sin()), dim=-1)  # fp32 [pos, ROT]
        transport = make_transport(
            "loopback", rank=0, world=1, capacity=4 * exchange_words(1, 16) * 2 + 512
        )
        cls.stages = {
            b: QkvFrontStage(
                transport=transport,
                nq=NQ,
                nkv=1,
                niq=NIQ,
                rot=ROT,
                block_size=PAGE,
                kv_cache_dtype="nvfp4",
                indexer_kv_dtype="nvfp4",
                max_rows=b,
                eps=EPS,
                head_eps=EPS,
                with_step_meta=False,
            )
            for b in (8, 16)
        }
        for st in cls.stages.values():
            st.precompile()
        cls.sf = sf

    def _reference(
        self, rows, residual, fc2, perm, scales, x2, positions, slots, qkv=None
    ):
        from sglang.kernels.ops.attention.minimax_qknorm_rope import (
            minimax_qknorm_rope_grouped_fp8,
        )
        from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import (
            nvfp4_store_kv_index,
        )

        part = (fc2[perm.long()].float() * scales.float()[:, None]).view(
            rows, TOPK, H
        ).sum(1) + x2.float()
        r = (part + residual.float()).bfloat16()
        rf = r.float()
        normed = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + EPS)
        normed = (normed * (1 + self.norm_w.float())).bfloat16()
        if qkv is None:
            qkv = (normed.float() @ self.w_deq.t()).bfloat16().contiguous()
        else:
            qkv = qkv.clone()
        q8 = torch.empty(rows, NQ, D, dtype=torch.float8_e4m3fn, device="cuda")
        iq8 = torch.empty(rows, NIQ, D, dtype=torch.float8_e4m3fn, device="cuda")
        groups = [
            (self.head_w[0], 0, NQ),
            (self.head_w[1], NQ, 1),
            (self.head_w[2], NQ + 2, NIQ),
            (self.head_w[3], NQ + 2 + NIQ, 1),
        ]
        minimax_qknorm_rope_grouped_fp8(
            qkv,
            groups,
            self.cos_sin,
            positions,
            EPS,
            fp8_a=q8,
            group_a=0,
            fp8_b=iq8,
            group_b=2,
        )
        pool = _Pool("cuda")
        k = qkv[:, NQ * D : (NQ + 1) * D].reshape(rows, 1, D)
        v = qkv[:, (NQ + 1) * D : (NQ + 2) * D].reshape(rows, 1, D)
        ik = qkv[:, -D:]
        one = torch.ones(1, device="cuda")
        nvfp4_store_kv_index(
            k,
            v,
            ik,
            pool.k,
            pool.ks,
            pool.v,
            pool.vs,
            pool.ik,
            pool.iks,
            slots,
            one,
            page_size=PAGE,
        )
        return r, q8.reshape(rows, -1), iq8.reshape(rows, -1), pool

    def test_rows(self):
        from sglang.srt.models.minimax_common.nvidia.low_batch_front.front import (
            FinalizeIn,
        )

        dev = "cuda"
        for step, rows in enumerate((1, 3, 4, 8, 9, 12, 16)):
            with self.subTest(rows=rows):
                residual = torch.randn(rows, H, device=dev).bfloat16()
                fc2 = (torch.randn(rows * TOPK + 5, H, device=dev) * 0.5).bfloat16()
                perm = torch.randperm(rows * TOPK + 5, device=dev)[: rows * TOPK].int()
                scales = torch.rand(rows * TOPK, device=dev).bfloat16()
                x2 = (torch.randn(rows, H, device=dev) * 0.5).bfloat16()
                positions = torch.randint(0, MAX_POS, (rows,), device=dev)
                slots = torch.randperm(PAGES * PAGE, device=dev)[:rows]
                pool = _Pool(dev)
                r_mid = torch.empty_like(residual)
                q8 = torch.zeros(rows, NQ * D, dtype=torch.float8_e4m3fn, device=dev)
                iq8 = torch.zeros(rows, NIQ * D, dtype=torch.float8_e4m3fn, device=dev)
                qkv_out = torch.empty(rows, HEADS * D, dtype=torch.bfloat16, device=dev)
                stage = self.stages[8 if rows <= 8 else 16]
                # Twice: the second call runs on the other exchange parity.
                for _ in range(2):
                    stage(
                        rows,
                        residual=residual,
                        norm_w=self.norm_w,
                        out_residual=r_mid,
                        w=self.w,
                        wsf=self.sf,
                        q_w=self.head_w[0],
                        k_w=self.head_w[1],
                        iq_w=self.head_w[2],
                        ik_w=self.head_w[3],
                        cos_sin=self.cos_sin,
                        positions=positions,
                        slot_mapping=slots,
                        index_slot_mapping=slots,
                        kv_cache=pool.k,
                        index_cache=pool.ik,
                        q_fp8=q8,
                        index_q=iq8,
                        x2=x2,
                        finalize=FinalizeIn(fc2, perm, scales, TOPK),
                        single_round=True,
                        qkv_out=qkv_out,
                    )
                torch.cuda.synchronize()
                r_ref, q_ref, iq_ref, ref = self._reference(
                    rows, residual, fc2, perm, scales, x2, positions, slots
                )
                torch.testing.assert_close(
                    r_mid.float(), r_ref.float(), atol=0.07, rtol=0.02
                )
                msg = [f"rows={rows} residual cos={_cos(r_mid, r_ref):.6f}"]
                for name, a, b in (("q", q8, q_ref), ("iq", iq8, iq_ref)):
                    c = _cos(a.float(), b.float())
                    msg.append(f"{name} cos={c:.5f}")
                    self.assertGreater(c, 0.995, name)
                got, want = pool.rows(slots), ref.rows(slots)
                for name in ("K", "V", "iK"):
                    gc, gs = got[name]
                    wc, ws = want[name]
                    c = _cos(_dequant(gc, gs), _dequant(wc, ws))
                    same = (
                        (torch.cat((gc, gs), -1) == torch.cat((wc, ws), -1))
                        .float()
                        .mean()
                        .item()
                    )
                    msg.append(f"{name} cos={c:.5f} same={same:.3f}")
                    self.assertGreater(c, 0.98, name)
                # The epilogue alone: the stock norms, RoPE and stores on the
                # front's own projection rows.
                _, q_e, iq_e, ref_e = self._reference(
                    rows, residual, fc2, perm, scales, x2, positions, slots, qkv_out
                )
                for name, a, b in (("q", q8, q_e), ("iq", iq8, iq_e)):
                    same = (
                        (a.view(torch.uint8) == b.view(torch.uint8))
                        .float()
                        .mean()
                        .item()
                    )
                    msg.append(f"epi {name} same={same:.4f}")
                    self.assertEqual(same, 1.0, f"epilogue {name} bytes")
                want = ref_e.rows(slots)
                for name in ("K", "V", "iK"):
                    gc, gs = got[name]
                    wc, ws = want[name]
                    same = (
                        (torch.cat((gc, gs), -1) == torch.cat((wc, ws), -1))
                        .float()
                        .mean()
                        .item()
                    )
                    msg.append(f"epi {name} same={same:.4f}")
                    self.assertEqual(same, 1.0, f"epilogue {name} bytes")
                print(" ".join(msg))


if __name__ == "__main__":
    unittest.main()
