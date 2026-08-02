"""Unit tests for the sglang-side GGUF architecture interop.

Builds a tiny synthetic ``deepseek4`` GGUF (the llama.cpp architecture of
DeepSeek V4 checkpoints) and exercises the ``gguf_arch`` registry dispatch
plus the deepseek4 adapter: config builder, tokenizer builder, tensor-rule
map, and weights iterator.
"""

import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import quantize

from sglang.srt.utils.hf_transformers.gguf_arch import (
    get_gguf_arch_adapter,
    read_gguf_architecture,
)
from sglang.srt.utils.hf_transformers.gguf_deepseek4 import (
    _MOE_EXPS_RE,
    GGUF_ARCH,
    build_config_from_gguf,
    build_tokenizer_from_gguf,
    deepseek4_gguf_weights_iterator,
    deepseek4_tensor_rules,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Tiny deepseek4 dims: hidden=64, 3 layers with compress ratios [0, 4, 128],
# 4 experts top-2, 1 hash layer, 4 heads x head_dim 32 (rope 8), q_lora=32,
# o_groups=2 x o_lora=32, indexer 4 heads x 16, moe_inter=64, vocab=32.
H, L, E, HD, NH, QL, OG, OL, IH, IHD, MI, V = 64, 3, 4, 32, 4, 32, 2, 32, 4, 16, 64, 32
RATIOS = [0, 4, 128, 0]  # length L + 1, like real files (tail = MTP slot)
NUM_HASH_LAYERS = 1


def _write_tiny_deepseek4_gguf(path: str) -> None:
    w = gguf.GGUFWriter(path, GGUF_ARCH)
    w.add_uint32("deepseek4.block_count", L)
    w.add_uint32("deepseek4.context_length", 4096)
    w.add_uint32("deepseek4.embedding_length", H)
    w.add_uint32("deepseek4.attention.head_count", NH)
    w.add_uint32("deepseek4.attention.head_count_kv", 1)
    w.add_string("deepseek4.rope.scaling.type", "yarn")
    w.add_float32("deepseek4.rope.scaling.factor", 4.0)
    w.add_uint32("deepseek4.rope.scaling.original_context_length", 1024)
    w.add_float32("deepseek4.rope.scaling.yarn_beta_fast", 32.0)
    w.add_float32("deepseek4.rope.scaling.yarn_beta_slow", 1.0)
    w.add_float32("deepseek4.rope.freq_base", 10000.0)
    w.add_float32("deepseek4.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32("deepseek4.expert_used_count", 2)
    w.add_uint32("deepseek4.expert_gating_func", 4)
    w.add_uint32("deepseek4.attention.key_length", HD)
    w.add_uint32("deepseek4.attention.value_length", HD)
    w.add_uint32("deepseek4.vocab_size", V)
    w.add_uint32("deepseek4.rope.dimension_count", 8)
    w.add_uint32("deepseek4.attention.q_lora_rank", QL)
    w.add_uint32("deepseek4.attention.output_lora_rank", OL)
    w.add_uint32("deepseek4.attention.output_group_count", OG)
    w.add_array("deepseek4.attention.compress_ratios", RATIOS)
    w.add_float32("deepseek4.attention.compress_rope_freq_base", 160000.0)
    w.add_uint32("deepseek4.expert_feed_forward_length", MI)
    w.add_uint32("deepseek4.expert_count", E)
    w.add_uint32("deepseek4.expert_shared_count", 1)
    w.add_float32("deepseek4.expert_weights_scale", 1.5)
    w.add_uint32("deepseek4.hash_layer_count", NUM_HASH_LAYERS)
    w.add_bool("deepseek4.expert_weights_norm", True)
    w.add_array("deepseek4.swiglu_clamp_exp", [10.0] * L)
    w.add_uint32("deepseek4.attention.sliding_window", 128)
    w.add_uint32("deepseek4.attention.indexer.head_count", IH)
    w.add_uint32("deepseek4.attention.indexer.key_length", IHD)
    w.add_uint32("deepseek4.attention.indexer.top_k", 8)
    w.add_uint32("deepseek4.nextn_predict_layers", 1)
    w.add_uint32("deepseek4.hyper_connection.count", 4)
    w.add_uint32("deepseek4.hyper_connection.sinkhorn_iterations", 20)
    w.add_float32("deepseek4.hyper_connection.epsilon", 1e-6)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "joyai-llm")
    toks = ["<s>", "</s>", "t", "o", "k", "to", "tok"] + [f"x{i}" for i in range(V - 7)]
    w.add_array("tokenizer.ggml.tokens", toks)
    w.add_array("tokenizer.ggml.token_type", [3, 3] + [1] * (V - 2))
    w.add_array("tokenizer.ggml.merges", ["t o", "to k"])
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_string("tokenizer.chat_template", "{{messages}}")

    rng = np.random.default_rng(0)

    def q8(name, shape):
        data = rng.standard_normal(shape, dtype=np.float32)
        q = quantize(data, gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(
            name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0
        )

    def f16(name, shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float16))

    def f32(name, shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32))

    def i32(name, shape):
        w.add_tensor(name, rng.integers(0, E, shape).astype(np.int32))

    f16("token_embd.weight", (V, H))
    q8("output.weight", (V, H))
    f32("output_norm.weight", (H,))
    f32("output_hc_base.weight", (4,))
    f16("output_hc_fn.weight", (4, 4 * H))
    f32("output_hc_scale.weight", (1,))
    for l in range(L):
        p = f"blk.{l}."
        f32(p + "attn_norm.weight", (H,))
        f32(p + "ffn_norm.weight", (H,))
        q8(p + "attn_q_a.weight", (QL, H))
        f32(p + "attn_q_a_norm.weight", (QL,))
        q8(p + "attn_q_b.weight", (NH * HD, QL))
        q8(p + "attn_kv.weight", (HD, H))
        f32(p + "attn_kv_a_norm.weight", (HD,))
        q8(p + "attn_output_a.weight", (OG * OL, NH * HD // OG))
        q8(p + "attn_output_b.weight", (H, OG * OL))
        f32(p + "attn_sinks.weight", (NH,))
        for hc in ("hc_attn", "hc_ffn"):
            f32(p + hc + "_base.weight", (24,))
            f16(p + hc + "_fn.weight", (24, 4 * H))
            f32(p + hc + "_scale.weight", (3,))
        f16(p + "ffn_gate_inp.weight", (E, H))
        if l < NUM_HASH_LAYERS:
            i32(p + "ffn_gate_tid2eid.weight", (V, 2))
        else:
            f32(p + "exp_probs_b.bias", (E,))
        for x in ("gate", "up"):
            q8(p + f"ffn_{x}_exps.weight", (E, MI, H))
        q8(p + "ffn_down_exps.weight", (E, H, MI))
        for x, shp in (("gate", (MI, H)), ("up", (MI, H)), ("down", (H, MI))):
            q8(p + f"ffn_{x}_shexp.weight", shp)
        ratio = RATIOS[l]
        if ratio in (4, 128):
            coff = 2 if ratio == 4 else 1
            f16(p + "attn_compressor_kv.weight", (coff * HD, H))
            f16(p + "attn_compressor_gate.weight", (coff * HD, H))
            f16(p + "attn_compressor_ape.weight", (ratio, coff * HD))
            f32(p + "attn_compressor_norm.weight", (HD,))
        if ratio == 4:
            f16(p + "indexer.attn_q_b.weight", (IH * IHD, QL))
            f16(p + "indexer.proj.weight", (IH, H))
            f16(p + "indexer_compressor_kv.weight", (2 * IHD, H))
            f16(p + "indexer_compressor_gate.weight", (2 * IHD, H))
            f16(p + "indexer_compressor_ape.weight", (4, 2 * IHD))
            f32(p + "indexer_compressor_norm.weight", (IHD,))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _write_minimal_gguf(path: str, tensors: list) -> None:
    """A deepseek4-arch GGUF carrying only the given (name, array) tensors."""
    w = gguf.GGUFWriter(path, GGUF_ARCH)
    for name, array, raw_dtype in tensors:
        if raw_dtype is None:
            w.add_tensor(name, array)
        else:
            w.add_tensor(name, array, raw_shape=array.shape, raw_dtype=raw_dtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class TestGGUFDeepseek4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.gguf_path = str(Path(cls._tmpdir.name) / "tiny_deepseek4.gguf")
        _write_tiny_deepseek4_gguf(cls.gguf_path)

    def test_read_architecture(self):
        self.assertEqual(read_gguf_architecture(self.gguf_path), GGUF_ARCH)

    def test_arch_registry_dispatch(self):
        """The three GGUF dispatch sites key on this registry; a deepseek4
        file must resolve to the adapter and unknown architectures must not
        (they fall through to the transformers GGUF path)."""
        adapter = get_gguf_arch_adapter(self.gguf_path)
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.gguf_arch, GGUF_ARCH)
        self.assertIs(adapter.build_config, build_config_from_gguf)

        other_arch_path = str(Path(self._tmpdir.name) / "other_arch.gguf")
        w = gguf.GGUFWriter(other_arch_path, "llama")
        w.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        self.assertIsNone(get_gguf_arch_adapter(other_arch_path))
        # Unreadable files probe to None rather than raising.
        self.assertIsNone(get_gguf_arch_adapter(str(Path(self._tmpdir.name) / "nope")))

    def test_build_config(self):
        config = build_config_from_gguf(self.gguf_path)
        self.assertEqual(config.model_type, "deepseek_v4")
        self.assertEqual(config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(config.num_hidden_layers, L)
        self.assertEqual(config.hidden_size, H)
        self.assertEqual(config.head_dim, HD)
        self.assertEqual(config.qk_rope_head_dim, 8)
        self.assertEqual(config.q_lora_rank, QL)
        self.assertEqual(config.o_groups, OG)
        self.assertEqual(config.o_lora_rank, OL)
        self.assertEqual(config.n_routed_experts, E)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_intermediate_size, MI)
        self.assertEqual(config.compress_ratios, RATIOS)
        self.assertEqual(config.compress_rope_theta, 160000.0)
        self.assertEqual(config.num_hash_layers, NUM_HASH_LAYERS)
        self.assertEqual(config.scoring_func, "sqrtsoftplus")
        self.assertEqual(config.topk_method, "noaux_tc")
        self.assertEqual(config.routed_scaling_factor, 1.5)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.swiglu_limit, 10.0)
        self.assertEqual(config.sliding_window, 128)
        self.assertEqual(config.index_topk, 8)
        self.assertEqual(config.index_n_heads, IH)
        self.assertEqual(config.index_head_dim, IHD)
        self.assertEqual(config.max_position_embeddings, 4096)
        self.assertEqual(config.rope_scaling["type"], "yarn")
        self.assertEqual(config.rope_scaling["original_max_position_embeddings"], 1024)
        self.assertEqual(config.num_nextn_predict_layers, 1)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.torch_dtype, torch.bfloat16)
        # GGUF quantization comes from the server-args coupling, never from
        # a checkpoint quantization_config.
        self.assertFalse(getattr(config, "quantization_config", None))

    def test_get_config_dispatch(self):
        """get_config on a .gguf path must route to the deepseek4 builder."""
        from sglang.srt.utils.hf_transformers.config import get_config

        config = get_config(self.gguf_path, trust_remote_code=False)
        self.assertEqual(config.model_type, "deepseek_v4")
        self.assertEqual(config.architectures, ["DeepseekV4ForCausalLM"])

    def test_build_tokenizer(self):
        tokenizer = build_tokenizer_from_gguf(self.gguf_path)
        self.assertEqual(tokenizer.bos_token, "<s>")
        self.assertEqual(tokenizer.eos_token, "</s>")
        self.assertEqual(tokenizer.chat_template, "{{messages}}")
        self.assertEqual(tokenizer.vocab_size, V)
        # "tok" is index 6 of the synthetic vocab, reachable through the
        # merges ["t o", "to k"] under the DeepSeek pre-tokenizer.
        self.assertEqual(tokenizer.encode("tok", add_special_tokens=False), [6])

    def test_tensor_rules_cover_all_tensors(self):
        config = build_config_from_gguf(self.gguf_path)
        rules = deepseek4_tensor_rules(config)
        tensor_names = {t.name for t in gguf.GGUFReader(self.gguf_path).tensors}
        uncovered = {
            n
            for n in tensor_names
            if n not in rules and _MOE_EXPS_RE.fullmatch(n) is None
        }
        unused = set(rules) - tensor_names
        self.assertFalse(uncovered)
        self.assertFalse(unused)

    def test_weights_iterator(self):
        config = build_config_from_gguf(self.gguf_path)
        pairs = list(deepseek4_gguf_weights_iterator(self.gguf_path, config))
        names = [name for name, _ in pairs]
        weights = dict(pairs)

        # Two-pass protocol: every qweight_type marker precedes every weight.
        type_positions = [i for i, n in enumerate(names) if n.endswith(".qweight_type")]
        weight_positions = [
            i for i, n in enumerate(names) if not n.endswith(".qweight_type")
        ]
        self.assertLess(max(type_positions), min(weight_positions))

        # Unquantized tensors keep their names and dtypes.
        self.assertEqual(weights["embed.weight"].dtype, torch.float16)
        self.assertEqual(weights["norm.weight"].dtype, torch.float32)
        self.assertEqual(weights["layers.0.ffn.gate.tid2eid"].dtype, torch.int32)
        self.assertEqual(weights["layers.0.ffn.gate.tid2eid"].shape, (V, 2))
        self.assertEqual(weights["layers.1.ffn.gate.bias"].dtype, torch.float32)
        self.assertIn("layers.0.hc_attn_fn", weights)
        self.assertIn("layers.0.attn.attn_sink", weights)
        self.assertIn("layers.1.attn.compressor.wkv.weight", weights)
        self.assertIn("layers.1.attn.compressor.wgate.weight", weights)
        self.assertIn("layers.1.attn.compressor.ape", weights)
        self.assertIn("layers.1.attn.indexer.compressor.ape", weights)
        self.assertIn("layers.1.attn.indexer.weights_proj.weight", weights)
        self.assertIn("layers.2.attn.compressor.wkv.weight", weights)
        self.assertNotIn("layers.2.attn.indexer.wq_b.qweight", weights)

        # Quantized linears go through .qweight / .qweight_type.
        self.assertIn("layers.0.attn.wq_a.qweight", weights)
        self.assertIn("layers.0.ffn.shared_experts.w1.qweight", weights)
        self.assertEqual(
            int(weights["layers.0.attn.wq_a.qweight_type"]),
            int(gguf.GGMLQuantizationType.Q8_0),
        )
        # The lm_head targets the post-remap name directly.
        self.assertIn("lm_head.qweight", weights)
        self.assertNotIn("head.qweight", weights)

        # F16 quant payloads (indexer wq_b) are normalized to bf16.
        self.assertEqual(
            weights["layers.1.attn.indexer.wq_b.qweight"].dtype, torch.bfloat16
        )
        self.assertEqual(
            int(weights["layers.1.attn.indexer.wq_b.qweight_type"]),
            int(gguf.GGMLQuantizationType.BF16),
        )

        # wo_a is dequantized to bf16 with the original .weight name.
        wo_a = weights["layers.0.attn.wo_a.weight"]
        self.assertEqual(wo_a.dtype, torch.bfloat16)
        self.assertEqual(wo_a.shape, (OG * OL, NH * HD // OG))
        # Q8_0 round-trip of standard-normal data stays approximately normal.
        self.assertGreater(wo_a.float().abs().mean().item(), 0.5)
        self.assertLess(wo_a.float().abs().max().item(), 6.0)

        # Packed experts follow the generic per-expert convention.
        for expert_id in range(E):
            self.assertIn(
                f"model.layers.0.mlp.experts.{expert_id}.gate_proj.qweight", weights
            )
            self.assertIn(
                f"model.layers.2.mlp.experts.{expert_id}.down_proj.qweight_type",
                weights,
            )

    def test_weights_iterator_rejects_unknown_tensor(self):
        """A tensor outside the vendored rule map must fail loudly, not load
        partially — silent skips would surface as garbage outputs."""
        config = build_config_from_gguf(self.gguf_path)
        bogus_path = str(Path(self._tmpdir.name) / "bogus_tensor.gguf")
        _write_minimal_gguf(
            bogus_path,
            [("blk.0.bogus.weight", np.zeros((4, 4), dtype=np.float32), None)],
        )
        with self.assertRaisesRegex(ValueError, "Unexpected tensor"):
            # The pre-scan runs on first next(); the function is a generator.
            next(deepseek4_gguf_weights_iterator(bogus_path, config))

    def test_weights_iterator_rejects_mixed_gate_up_quant(self):
        """FusedMoE fuses gate+up under one w13 quant type; a file mixing
        types per layer must be rejected before any weight is staged."""
        config = build_config_from_gguf(self.gguf_path)
        mixed_path = str(Path(self._tmpdir.name) / "mixed_exps.gguf")
        gate = quantize(
            np.random.default_rng(1).standard_normal((E, MI, H), dtype=np.float32),
            gguf.GGMLQuantizationType.Q8_0,
        )
        up = np.zeros((E, MI, H), dtype=np.float16)
        _write_minimal_gguf(
            mixed_path,
            [
                ("blk.0.ffn_gate_exps.weight", gate, gguf.GGMLQuantizationType.Q8_0),
                ("blk.0.ffn_up_exps.weight", up, None),
            ],
        )
        with self.assertRaisesRegex(ValueError, "quantization type"):
            next(deepseek4_gguf_weights_iterator(mixed_path, config))

    def test_tokenizer_rejects_unknown_pre_tokenizer(self):
        """An unrecognized tokenizer.ggml.pre must fail loudly: falling back
        to the generic GPT-2 split regex silently mis-tokenizes."""
        weird_path = str(Path(self._tmpdir.name) / "weird_pre.gguf")
        w = gguf.GGUFWriter(weird_path, GGUF_ARCH)
        w.add_string("tokenizer.ggml.model", "gpt2")
        w.add_string("tokenizer.ggml.pre", "some-new-pre")
        w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "t", "o", "to"])
        w.add_array("tokenizer.ggml.token_type", [3, 3, 1, 1, 1])
        w.add_array("tokenizer.ggml.merges", ["t o"])
        w.add_uint32("tokenizer.ggml.bos_token_id", 0)
        w.add_uint32("tokenizer.ggml.eos_token_id", 1)
        w.add_tensor("token_embd.weight", np.zeros((5, 4), dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        with self.assertRaisesRegex(ValueError, "tokenizer.ggml.pre"):
            build_tokenizer_from_gguf(weird_path)

    def test_weights_iterator_rejects_missing_tensors(self):
        """An incomplete file (wrong variant, truncated conversion) must fail
        fast instead of loading partially with uninitialized parameters."""
        config = build_config_from_gguf(self.gguf_path)
        sparse_path = str(Path(self._tmpdir.name) / "missing_tensors.gguf")
        _write_minimal_gguf(
            sparse_path,
            [("token_embd.weight", np.zeros((V, H), dtype=np.float16), None)],
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            next(deepseek4_gguf_weights_iterator(sparse_path, config))


if __name__ == "__main__":
    unittest.main()
