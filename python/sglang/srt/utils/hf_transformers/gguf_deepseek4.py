"""GGUF support for DeepSeek V4 checkpoints (llama.cpp architecture ``deepseek4``).

Neither transformers' GGUF integration (``GGUF_SUPPORTED_ARCHITECTURES``) nor
the PyPI ``gguf`` tensor name map knows this architecture, so sglang carries
the interop itself:

- :func:`build_config_from_gguf` maps ``deepseek4.*`` GGUF metadata onto the
  registered ``deepseek_v4`` HF config class, mirroring the keys of the
  official ``deepseek-ai/DeepSeek-V4-Flash`` ``config.json``.
- :func:`build_tokenizer_from_gguf` drives transformers' GGUF GPT-2 tokenizer
  converter directly (the embedded tokenizer is ``tokenizer.ggml.model =
  "gpt2"``), bypassing the per-architecture support gate that rejects
  ``deepseek4``.
- :func:`deepseek4_gguf_weights_iterator` yields weights under the
  DeepSeek-native checkpoint names that ``DeepseekV4ForCausalLM.load_weights``
  already accepts (via ``remap_weight_name_to_dpsk_hf_format``). Packed routed
  experts follow the generic per-expert
  ``model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.qweight`` convention
  of :func:`sglang.srt.model_loader.weight_utils.gguf_quant_weights_iterator`.

The tensor inventory below was verified against
``antirez/deepseek-v4-gguf`` (``DeepSeek-V4-Flash-IQ2XXS-...-0731.gguf``) and
llama.cpp master's ``DeepseekV4Model`` converter, which keeps the original
DeepSeek-native tensor names (no permute/split transforms besides the
per-expert packing handled here).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Generator, Literal, Sequence, Tuple

import msgspec

if TYPE_CHECKING:
    import gguf
    import torch
    from transformers import PretrainedConfig, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

GGUF_ARCH = "deepseek4"
MODEL_TYPE = "deepseek_v4"

# llama.cpp gguf-py ExpertGatingFuncType values.
_EXPERT_GATING_FUNC_NAMES = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}

_MOE_EXPS_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
_MOE_SHARD_NAMES = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


class GGUFTensorRule(msgspec.Struct, frozen=True):
    """How one non-expert GGUF tensor maps into the checkpoint namespace.

    - ``quant``: a GGUF-quantized linear/lm-head weight; yielded as packed
      bytes under ``<target minus .weight>.qweight`` plus a ``.qweight_type``.
    - ``raw``: an unquantized tensor (F16/F32/I32) owned by a plain parameter
      or an unquantized linear; yielded verbatim under ``target``.
    - ``dequant_bf16``: dequantized to bf16 at load time; used for ``wo_a``,
      which the model builds as an unquantized bf16 linear when the fp8 wo_a
      GEMM path is off (the only mode GGUF supports).
    """

    target: str
    mode: Literal["quant", "raw", "dequant_bf16"]


def read_gguf_architecture(gguf_path: str) -> str:
    """Return ``general.architecture`` of a GGUF file."""
    import gguf

    reader = gguf.GGUFReader(gguf_path)
    return _read_field(reader, "general.architecture")


def is_deepseek4_gguf(gguf_path: str) -> bool:
    """True when the GGUF's architecture is ``deepseek4``.

    A probe, not a validator: unreadable or arch-less files answer False so
    callers fall through to the legacy transformers GGUF path (and keep its
    error surface).
    """
    try:
        return read_gguf_architecture(gguf_path) == GGUF_ARCH
    except Exception as e:
        logger.debug("Could not read GGUF architecture from %s: %s", gguf_path, e)
        return False


def _read_field(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"GGUF file is missing required metadata key {key!r}.")
    return field.contents()


def _read_arch_field(reader: gguf.GGUFReader, key: str):
    return _read_field(reader, f"{GGUF_ARCH}.{key}")


def _uniform_scalar(values: Sequence[float], key: str) -> float:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(
            f"GGUF key {key!r} carries non-uniform per-layer values {sorted(distinct)}; "
            "the deepseek_v4 config only supports a scalar."
        )
    return next(iter(distinct))


def build_config_from_gguf(gguf_path: str) -> PretrainedConfig:
    """Build the registered ``deepseek_v4`` HF config from GGUF metadata.

    Field names mirror the official ``deepseek-ai/DeepSeek-V4-Flash``
    ``config.json``. ``quantization_config`` is deliberately absent: GGUF
    quantization comes from the ``--quantization gguf`` server-args coupling.
    """
    import gguf

    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

    reader = gguf.GGUFReader(gguf_path)
    arch = _read_field(reader, "general.architecture")
    if arch != GGUF_ARCH:
        raise ValueError(f"Expected GGUF architecture {GGUF_ARCH!r}, got {arch!r}.")

    gating_func = int(_read_arch_field(reader, "expert_gating_func"))
    if gating_func not in _EXPERT_GATING_FUNC_NAMES:
        raise ValueError(f"Unknown deepseek4 expert_gating_func {gating_func}.")

    rope_scaling_type = _read_arch_field(reader, "rope.scaling.type")
    if rope_scaling_type != "yarn":
        raise ValueError(
            f"Expected yarn rope scaling for deepseek4 GGUF, got {rope_scaling_type!r}."
        )

    config_cls = _CONFIG_REGISTRY[MODEL_TYPE]
    config = config_cls(
        architectures=["DeepseekV4ForCausalLM"],
        bos_token_id=int(_read_field(reader, "tokenizer.ggml.bos_token_id")),
        eos_token_id=int(_read_field(reader, "tokenizer.ggml.eos_token_id")),
        head_dim=int(_read_arch_field(reader, "attention.key_length")),
        hidden_act="silu",
        hidden_size=int(_read_arch_field(reader, "embedding_length")),
        index_head_dim=int(_read_arch_field(reader, "attention.indexer.key_length")),
        index_n_heads=int(_read_arch_field(reader, "attention.indexer.head_count")),
        index_topk=int(_read_arch_field(reader, "attention.indexer.top_k")),
        max_position_embeddings=int(_read_arch_field(reader, "context_length")),
        moe_intermediate_size=int(
            _read_arch_field(reader, "expert_feed_forward_length")
        ),
        n_routed_experts=int(_read_arch_field(reader, "expert_count")),
        n_shared_experts=int(_read_arch_field(reader, "expert_shared_count")),
        norm_topk_prob=bool(_read_arch_field(reader, "expert_weights_norm")),
        num_attention_heads=int(_read_arch_field(reader, "attention.head_count")),
        num_experts_per_tok=int(_read_arch_field(reader, "expert_used_count")),
        num_hidden_layers=int(_read_arch_field(reader, "block_count")),
        num_hash_layers=int(_read_arch_field(reader, "hash_layer_count")),
        num_key_value_heads=int(_read_arch_field(reader, "attention.head_count_kv")),
        num_nextn_predict_layers=int(_read_arch_field(reader, "nextn_predict_layers")),
        o_groups=int(_read_arch_field(reader, "attention.output_group_count")),
        o_lora_rank=int(_read_arch_field(reader, "attention.output_lora_rank")),
        q_lora_rank=int(_read_arch_field(reader, "attention.q_lora_rank")),
        qk_rope_head_dim=int(_read_arch_field(reader, "rope.dimension_count")),
        rms_norm_eps=float(
            _read_arch_field(reader, "attention.layer_norm_rms_epsilon")
        ),
        rope_scaling={
            "beta_fast": _read_arch_field(reader, "rope.scaling.yarn_beta_fast"),
            "beta_slow": _read_arch_field(reader, "rope.scaling.yarn_beta_slow"),
            "factor": _read_arch_field(reader, "rope.scaling.factor"),
            "original_max_position_embeddings": int(
                _read_arch_field(reader, "rope.scaling.original_context_length")
            ),
            "type": "yarn",
        },
        rope_theta=_read_arch_field(reader, "rope.freq_base"),
        routed_scaling_factor=float(_read_arch_field(reader, "expert_weights_scale")),
        scoring_func=_EXPERT_GATING_FUNC_NAMES[gating_func],
        sliding_window=int(_read_arch_field(reader, "attention.sliding_window")),
        swiglu_limit=_uniform_scalar(
            _read_arch_field(reader, "swiglu_clamp_exp"),
            f"{GGUF_ARCH}.swiglu_clamp_exp",
        ),
        tie_word_embeddings=False,
        topk_method="noaux_tc",
        torch_dtype="bfloat16",
        use_cache=True,
        vocab_size=int(_read_arch_field(reader, "vocab_size")),
        compress_rope_theta=float(
            _read_arch_field(reader, "attention.compress_rope_freq_base")
        ),
        compress_ratios=[
            int(r) for r in _read_arch_field(reader, "attention.compress_ratios")
        ],
        hc_eps=float(_read_arch_field(reader, "hyper_connection.epsilon")),
        hc_mult=int(_read_arch_field(reader, "hyper_connection.count")),
        hc_sinkhorn_iters=int(
            _read_arch_field(reader, "hyper_connection.sinkhorn_iterations")
        ),
    )
    if len(config.compress_ratios) < config.num_hidden_layers:
        raise ValueError(
            f"compress_ratios has {len(config.compress_ratios)} entries for "
            f"{config.num_hidden_layers} layers."
        )
    config._name_or_path = str(gguf_path)
    return config


def build_tokenizer_from_gguf(gguf_path: str, **kwargs) -> PreTrainedTokenizerFast:
    """Build a fast tokenizer from the GGUF's embedded GPT-2-style BPE.

    transformers' own GGUF tokenizer path rejects unknown architectures before
    consulting ``tokenizer.ggml.model``, so drive its converter directly.
    """
    import gguf
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    reader = gguf.GGUFReader(gguf_path)
    tokenizer_model = _read_field(reader, "tokenizer.ggml.model")
    if tokenizer_model != "gpt2":
        raise ValueError(
            f"deepseek4 GGUF tokenizer type {tokenizer_model!r} is not supported; "
            "pass --tokenizer-path (e.g. deepseek-ai/DeepSeek-V4-Flash) instead."
        )

    prefix = "tokenizer.ggml."
    tokenizer_dict = {
        field.name[len(prefix) :]: field.contents()
        for field in reader.fields.values()
        if field.name.startswith(prefix)
    }
    # transformers' converters key the tokenizer type under "tokenizer_type".
    tokenizer_dict["tokenizer_type"] = tokenizer_dict.pop("model")

    fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(
        tokenizer_model, tokenizer_dict
    )
    pre = tokenizer_dict.get("pre")
    if pre == "joyai-llm":
        # The generic GPT-2 converter installs the classic GPT-2 split regex;
        # DeepSeek's BPE uses its own pre-tokenization (this is the
        # pre_tokenizer of deepseek-ai/DeepSeek-V4-Flash tokenizer.json,
        # matching llama.cpp's "joyai-llm" pre). Verified token-exact against
        # the official tokenizer.
        fast_tokenizer.pre_tokenizer = _deepseek_pre_tokenizer()
    else:
        logger.warning(
            "Unknown tokenizer.ggml.pre %r in deepseek4 GGUF; keeping the "
            "generic GPT-2 pre-tokenizer. For guaranteed tokenization parity, "
            "pass --tokenizer-path (e.g. deepseek-ai/DeepSeek-V4-Flash).",
            pre,
        )
    tokens = tokenizer_dict["tokens"]
    chat_template_field = reader.get_field("tokenizer.chat_template")
    if chat_template_field is not None:
        kwargs.setdefault("chat_template", chat_template_field.contents())
    kwargs.update(additional_kwargs)
    kwargs.setdefault("bos_token", tokens[int(tokenizer_dict["bos_token_id"])])
    kwargs.setdefault("eos_token", tokens[int(tokenizer_dict["eos_token_id"])])
    if "padding_token_id" in tokenizer_dict:
        kwargs.setdefault("pad_token", tokens[int(tokenizer_dict["padding_token_id"])])
    logger.info("Loaded the tokenizer embedded in the deepseek4 GGUF.")
    return PreTrainedTokenizerFast(tokenizer_object=fast_tokenizer, **kwargs)


def _deepseek_pre_tokenizer():
    from tokenizers import Regex, pre_tokenizers

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(r"\p{N}{1,3}"), behavior="isolated", invert=False
            ),
            pre_tokenizers.Split(
                Regex("[一-龥぀-ゟ゠-ヿ]+"),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.Split(
                Regex(
                    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"""
                    r"""|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"""
                    r"""| ?[\p{P}\p{S}]+[\r\n]*"""
                    r"""|\s*[\r\n]+"""
                    r"""|\s+(?!\S)"""
                    r"""|\s+"""
                ),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=True, use_regex=False
            ),
        ]
    )


def deepseek4_tensor_rules(hf_config: PretrainedConfig) -> dict[str, GGUFTensorRule]:
    """The gguf-name -> DeepSeek-native-name map for all non-routed-expert tensors.

    Targets are the names llama.cpp converted from, which
    ``remap_weight_name_to_dpsk_hf_format`` maps onto module parameters.
    Bare-parameter targets (``ape``, ``attn_sink``, ``tid2eid``, ``hc_*``)
    carry no ``.weight`` suffix even though the GGUF names do.
    """
    quant = lambda target: GGUFTensorRule(target=target, mode="quant")
    raw = lambda target: GGUFTensorRule(target=target, mode="raw")

    rules = {
        "token_embd.weight": raw("embed.weight"),
        # remap_weight_name_to_dpsk_hf_format only rewrites the exact name
        # "head.weight", which a ".qweight"-suffixed yield would miss — target
        # the post-remap name directly.
        "output.weight": quant("lm_head.weight"),
        "output_norm.weight": raw("norm.weight"),
        "output_hc_fn.weight": raw("hc_head_fn"),
        "output_hc_base.weight": raw("hc_head_base"),
        "output_hc_scale.weight": raw("hc_head_scale"),
    }

    compress_ratios = hf_config.compress_ratios
    num_hash_layers = hf_config.num_hash_layers
    for l in range(hf_config.num_hidden_layers):
        g = f"blk.{l}."
        t = f"layers.{l}."
        rules[g + "attn_norm.weight"] = raw(t + "attn_norm.weight")
        rules[g + "ffn_norm.weight"] = raw(t + "ffn_norm.weight")
        rules[g + "attn_q_a.weight"] = quant(t + "attn.wq_a.weight")
        rules[g + "attn_q_a_norm.weight"] = raw(t + "attn.q_norm.weight")
        rules[g + "attn_q_b.weight"] = quant(t + "attn.wq_b.weight")
        rules[g + "attn_kv.weight"] = quant(t + "attn.wkv.weight")
        rules[g + "attn_kv_a_norm.weight"] = raw(t + "attn.kv_norm.weight")
        # wo_a is an unquantized bf16 linear in the (GGUF-required)
        # non-fp8-wo_a mode; dequantize its GGML-quantized data at load time.
        rules[g + "attn_output_a.weight"] = GGUFTensorRule(
            target=t + "attn.wo_a.weight", mode="dequant_bf16"
        )
        rules[g + "attn_output_b.weight"] = quant(t + "attn.wo_b.weight")
        rules[g + "attn_sinks.weight"] = raw(t + "attn.attn_sink")
        for hc in ("hc_attn", "hc_ffn"):
            rules[g + hc + "_fn.weight"] = raw(t + hc + "_fn")
            rules[g + hc + "_base.weight"] = raw(t + hc + "_base")
            rules[g + hc + "_scale.weight"] = raw(t + hc + "_scale")
        rules[g + "ffn_gate_inp.weight"] = raw(t + "ffn.gate.weight")
        if l < num_hash_layers:
            rules[g + "ffn_gate_tid2eid.weight"] = raw(t + "ffn.gate.tid2eid")
        else:
            rules[g + "exp_probs_b.bias"] = raw(t + "ffn.gate.bias")
        rules[g + "ffn_gate_shexp.weight"] = quant(t + "ffn.shared_experts.w1.weight")
        rules[g + "ffn_up_shexp.weight"] = quant(t + "ffn.shared_experts.w3.weight")
        rules[g + "ffn_down_shexp.weight"] = quant(t + "ffn.shared_experts.w2.weight")
        ratio = compress_ratios[l]
        if ratio in (4, 128):
            c = t + "attn.compressor."
            rules[g + "attn_compressor_kv.weight"] = raw(c + "wkv.weight")
            rules[g + "attn_compressor_gate.weight"] = raw(c + "wgate.weight")
            rules[g + "attn_compressor_ape.weight"] = raw(c + "ape")
            rules[g + "attn_compressor_norm.weight"] = raw(c + "norm.weight")
        if ratio == 4:
            i = t + "attn.indexer."
            rules[g + "indexer.attn_q_b.weight"] = quant(i + "wq_b.weight")
            rules[g + "indexer.proj.weight"] = raw(i + "weights_proj.weight")
            rules[g + "indexer_compressor_kv.weight"] = raw(i + "compressor.wkv.weight")
            rules[g + "indexer_compressor_gate.weight"] = raw(
                i + "compressor.wgate.weight"
            )
            rules[g + "indexer_compressor_ape.weight"] = raw(i + "compressor.ape")
            rules[g + "indexer_compressor_norm.weight"] = raw(
                i + "compressor.norm.weight"
            )
    return rules


def _quant_target_name(rule: GGUFTensorRule, suffix: str) -> str:
    base, _, tail = rule.target.rpartition(".")
    assert tail == "weight", rule
    return f"{base}.{suffix}"


def _normalized_quant_type(
    tensor_type: gguf.GGMLQuantizationType,
) -> gguf.GGMLQuantizationType:
    """F16 payloads are cast to bf16 on yield; report the matching type."""
    import gguf

    if tensor_type.name == "F16":
        return gguf.GGMLQuantizationType.BF16
    return tensor_type


def deepseek4_gguf_weights_iterator(
    gguf_file: str, hf_config: PretrainedConfig
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs for a deepseek4 GGUF.

    Mirrors ``gguf_quant_weights_iterator``'s two-pass protocol (all
    ``qweight_type`` markers first, then the weights) and its per-expert
    emission for the packed routed-expert tensors.
    """
    import gguf
    import torch
    from gguf.quants import dequantize

    rules = deepseek4_tensor_rules(hf_config)
    reader = gguf.GGUFReader(gguf_file)

    unquantized_type_names = {"F32", "F16", "BF16", "I32"}
    moe_exps_types: dict[int, dict[str, object]] = {}
    for tensor in reader.tensors:
        moe_match = _MOE_EXPS_RE.fullmatch(tensor.name)
        if moe_match is not None:
            moe_exps_types.setdefault(int(moe_match.group(1)), {})[
                moe_match.group(2)
            ] = tensor.tensor_type
        elif tensor.name not in rules:
            raise ValueError(
                f"Unexpected tensor {tensor.name!r} in deepseek4 GGUF; if the "
                "GGUF layout changed, update deepseek4_tensor_rules()."
            )
    for layer_id, types in moe_exps_types.items():
        # FusedMoE fuses gate+up under a single w13 quant type.
        if "gate" in types and "up" in types and types["gate"] != types["up"]:
            raise ValueError(
                f"blk.{layer_id}: ffn_gate_exps is {types['gate'].name} but "
                f"ffn_up_exps is {types['up'].name}; the fused w13 GGUF MoE "
                "weight requires one quantization type for both."
            )
    # Fail fast on incomplete files (wrong variant, bad conversion) instead of
    # loading partially and warning about uninitialized parameters.
    present = {t.name for t in reader.tensors}
    missing = sorted(set(rules) - present) + sorted(
        f"blk.{l}.ffn_{proj}_exps.weight"
        for l in range(hf_config.num_hidden_layers)
        for proj in ("gate", "up", "down")
        if proj not in moe_exps_types.get(l, {})
    )
    if missing:
        raise ValueError(
            f"deepseek4 GGUF is missing {len(missing)} expected tensors "
            f"(first few: {missing[:5]})."
        )

    # First pass: quantization-type markers.
    for tensor in reader.tensors:
        moe_match = _MOE_EXPS_RE.fullmatch(tensor.name)
        if moe_match is not None:
            layer_id, proj = int(moe_match.group(1)), moe_match.group(2)
            num_experts = tensor.data.shape[0]
            for expert_id in range(num_experts):
                yield (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}."
                    f"{_MOE_SHARD_NAMES[proj]}.qweight_type",
                    torch.tensor(_normalized_quant_type(tensor.tensor_type)),
                )
            continue
        rule = rules[tensor.name]
        if rule.mode == "quant":
            yield (
                _quant_target_name(rule, "qweight_type"),
                torch.tensor(_normalized_quant_type(tensor.tensor_type)),
            )

    # Second pass: the weights themselves.
    for tensor in reader.tensors:
        moe_match = _MOE_EXPS_RE.fullmatch(tensor.name)
        if moe_match is not None:
            layer_id, proj = int(moe_match.group(1)), moe_match.group(2)
            num_experts = tensor.data.shape[0]
            for expert_id in range(num_experts):
                # Zero-copy views into the file mmap: the per-expert shards
                # are staged in FusedMoE data containers until materialization
                # and copying them here would hold the whole ~75GB expert
                # payload in anonymous host memory (OOM on unified-memory
                # boxes). mmap pages are clean and evictable instead.
                expert_weight = _from_readonly_numpy(tensor.data[expert_id])
                if tensor.tensor_type.name == "F16":
                    expert_weight = expert_weight.to(torch.bfloat16)
                yield (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}."
                    f"{_MOE_SHARD_NAMES[proj]}.qweight",
                    expert_weight,
                )
            continue
        rule = rules[tensor.name]
        if rule.mode == "quant":
            weight = torch.tensor(tensor.data)
            if tensor.tensor_type.name == "F16":
                # The model runs bf16 activations and fused_mul_mat_gguf's
                # unquantized path is a plain ``x @ qweight.T``, so F16
                # payloads (the indexer wq_b) are stored as bf16.
                weight = weight.to(torch.bfloat16)
            yield _quant_target_name(rule, "qweight"), weight
        elif rule.mode == "dequant_bf16":
            if tensor.tensor_type.name in unquantized_type_names:
                weight = torch.tensor(tensor.data)
            else:
                weight = torch.from_numpy(
                    dequantize(tensor.data, tensor.tensor_type).copy()
                )
            yield rule.target, weight.to(torch.bfloat16)
        else:
            yield rule.target, torch.tensor(tensor.data)


def _from_readonly_numpy(array) -> torch.Tensor:
    """torch.from_numpy on a read-only mmap view, without the per-call warning.

    The resulting tensor is only ever read (stacked/copied into materialized
    parameters), never written.
    """
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        return torch.from_numpy(array)
