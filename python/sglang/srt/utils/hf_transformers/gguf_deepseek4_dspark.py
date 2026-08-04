"""GGUF support for the DeepSeek V4 DSpark drafter (arch ``deepseek4-dspark``).

The drafter checkpoint (e.g. ``bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF``)
carries three DSpark stages that structurally mirror DeepSeek-V4-Flash target
layers, plus the stage-independent heads (``main_proj``, ``markov_w1/w2``,
``conf_proj``, ``hc_head_*``, ``norm``). Its GGUF metadata holds only the
``deepseek4.dspark.*`` keys — no base model dims — so the config builder pins
the DeepSeek-V4-Flash geometry and cross-checks every pinned value that has a
tensor to check it against, failing loudly on any drafter that deviates.

The weights iterator emits the ``mtp.<stage>.<dpsk-native>`` names that
``DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name`` accepts.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Generator, Tuple

from sglang.srt.utils.hf_transformers.gguf_deepseek4 import (
    GGUFTensorRule,
    _from_readonly_numpy,
    _normalized_quant_type,
    _quant_target_name,
)

if TYPE_CHECKING:
    import gguf
    import torch
    from transformers import PretrainedConfig, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

GGUF_ARCH = "deepseek4-dspark"

_KV_PREFIX = "deepseek4.dspark."

# DeepSeek-V4-Flash base geometry (deepseek-ai/DeepSeek-V4-Flash config.json).
# The drafter is distilled from and structurally locked to this target; every
# value with a corresponding drafter tensor is shape-checked in
# build_config_from_gguf.
_FLASH_BASE_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "max_position_embeddings": 1048576,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_key_value_heads": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-6,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 129280,
    "compress_rope_theta": 160000.0,
    "hc_eps": 1e-6,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    # The drafter attends with compress_ratio 0 everywhere.
    "num_hash_layers": 0,
}


def _read_field(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"GGUF file is missing required metadata key {key!r}.")
    return field.contents()


def _torch_shape(tensor) -> tuple:
    # gguf reader tensor.data is torch-ordered; for quantized tensors the last
    # dim is the packed byte width, so only the leading dims are logical.
    return tuple(tensor.data.shape)


def build_config_from_gguf(gguf_path: str) -> PretrainedConfig:
    """Build the drafter's HF config from GGUF metadata + pinned Flash dims."""
    import gguf

    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

    reader = gguf.GGUFReader(gguf_path)
    arch = _read_field(reader, "general.architecture")
    if arch != GGUF_ARCH:
        raise ValueError(f"Expected GGUF architecture {GGUF_ARCH!r}, got {arch!r}.")

    layer_count = int(_read_field(reader, _KV_PREFIX + "layer_count"))
    target_layers = [int(v) for v in _read_field(reader, _KV_PREFIX + "target_layers")]
    expert_count = int(_read_field(reader, _KV_PREFIX + "expert_count"))

    tensors = {t.name: t for t in reader.tensors}

    def _check(name: str, expect: tuple, what: str) -> None:
        got = _torch_shape(tensors[name])
        if got != expect:
            raise ValueError(
                f"DSpark drafter {what} mismatch: {name} has torch shape "
                f"{got}, expected {expect}. This drafter does not match the "
                "pinned DeepSeek-V4-Flash geometry."
            )

    def _check_rows(name: str, expect_rows: int, what: str) -> None:
        # Quantized tensors pack the trailing dim into bytes; only the row
        # count is directly comparable.
        got = _torch_shape(tensors[name])[0]
        if got != expect_rows:
            raise ValueError(
                f"DSpark drafter {what} mismatch: {name} has {got} rows, "
                f"expected {expect_rows}. This drafter does not match the "
                "pinned DeepSeek-V4-Flash geometry."
            )

    base = dict(_FLASH_BASE_CONFIG)
    hidden = base["hidden_size"]
    _check("dspark.main_norm.weight", (hidden,), "hidden_size")
    _check(
        "dspark.markov_w1.weight",
        (base["vocab_size"], int(_read_field(reader, _KV_PREFIX + "markov_rank"))),
        "vocab_size/markov_rank",
    )
    _check_rows("dspark.0.attn_q_a.weight", base["q_lora_rank"], "q_lora_rank")
    _check_rows("dspark.0.attn_kv.weight", base["head_dim"], "head_dim")
    _check_rows(
        "dspark.0.attn_q_b.weight",
        base["num_attention_heads"] * base["head_dim"],
        "num_attention_heads",
    )
    _check_rows(
        "dspark.0.attn_output_a.weight",
        base["o_groups"] * base["o_lora_rank"],
        "o_groups/o_lora_rank",
    )
    _check(
        "dspark.hc_head_fn.weight",
        (base["hc_mult"], base["hc_mult"] * hidden),
        "hc_mult",
    )
    exps = _torch_shape(tensors["dspark.0.ffn_gate_exps.weight"])
    if exps[0] != expert_count or exps[1] != base["moe_intermediate_size"]:
        raise ValueError(
            f"DSpark drafter expert geometry mismatch: ffn_gate_exps {exps} vs "
            f"expert_count={expert_count}, "
            f"moe_intermediate_size={base['moe_intermediate_size']}."
        )
    if len(target_layers) != layer_count:
        raise ValueError(
            f"DSpark drafter target_layers {target_layers} does not match "
            f"layer_count={layer_count}."
        )

    config_cls = _CONFIG_REGISTRY["deepseek_v4"]
    config = config_cls(
        **base,
        num_hidden_layers=layer_count,
        n_routed_experts=expert_count,
        # DSpark stage attention asserts compress_ratio == 0 on every stage.
        compress_ratios=[0] * layer_count,
        # DSpark bundling markers: checkpoint_bundles_dspark_draft keys on
        # these, remapping the draft arch to DeepseekV4ForCausalLMDSpark.
        dspark_block_size=int(_read_field(reader, _KV_PREFIX + "block_size")),
        dspark_markov_rank=int(_read_field(reader, _KV_PREFIX + "markov_rank")),
        dspark_noise_token_id=int(_read_field(reader, _KV_PREFIX + "noise_token_id")),
        dspark_target_layer_ids=target_layers,
        enable_confidence_head=True,
    )
    config._name_or_path = str(gguf_path)
    return config


def build_tokenizer_from_gguf(gguf_path: str, **kwargs) -> PreTrainedTokenizerFast:
    raise ValueError(
        "The deepseek4-dspark GGUF is a draft checkpoint and carries no "
        "tokenizer; the tokenizer comes from the target model."
    )


def dspark_tensor_rules(hf_config: PretrainedConfig) -> dict[str, GGUFTensorRule]:
    """gguf-name -> ``mtp.<stage>.<dpsk-native>`` map for non-expert tensors.

    Targets feed ``_remap_dspark_weight_name``: ``mtp.N.attn.* -> stages.N.
    self_attn.*``, ``mtp.N.markov_head.* -> markov_head.*`` (stage-dropped),
    etc. Bare parameters (``attn_sink``, ``hc_*``) carry no ``.weight``.
    """
    quant = lambda target: GGUFTensorRule(target=target, mode="quant")
    raw = lambda target: GGUFTensorRule(target=target, mode="raw")

    last = hf_config.num_hidden_layers - 1
    rules = {
        # Stage-0-owned projections of the target hidden state.
        "dspark.main_proj.weight": quant("mtp.0.main_proj.weight"),
        "dspark.main_norm.weight": raw("mtp.0.main_norm.weight"),
        # Stage-independent heads; the remapper drops the stage index.
        "dspark.markov_w1.weight": raw("mtp.0.markov_head.markov_w1.weight"),
        "dspark.markov_w2.weight": raw("mtp.0.markov_head.markov_w2.weight"),
        "dspark.conf_proj.weight": raw("mtp.0.confidence_head.proj.weight"),
        # Final-stage collapse head and norm.
        "dspark.hc_head_fn.weight": raw(f"mtp.{last}.hc_head_fn"),
        "dspark.hc_head_base.weight": raw(f"mtp.{last}.hc_head_base"),
        "dspark.hc_head_scale.weight": raw(f"mtp.{last}.hc_head_scale"),
        "dspark.norm.weight": raw(f"mtp.{last}.norm.weight"),
    }
    for n in range(hf_config.num_hidden_layers):
        g = f"dspark.{n}."
        t = f"mtp.{n}."
        rules[g + "attn_norm.weight"] = raw(t + "attn_norm.weight")
        rules[g + "ffn_norm.weight"] = raw(t + "ffn_norm.weight")
        rules[g + "attn_q_a.weight"] = quant(t + "attn.wq_a.weight")
        rules[g + "attn_q_a_norm.weight"] = raw(t + "attn.q_norm.weight")
        rules[g + "attn_q_b.weight"] = quant(t + "attn.wq_b.weight")
        rules[g + "attn_kv.weight"] = quant(t + "attn.wkv.weight")
        rules[g + "attn_kv_a_norm.weight"] = raw(t + "attn.kv_norm.weight")
        # DSpark attention is built with wo_a_fp8=False and no quant config on
        # wo_a (deepseek_v4_dspark.py), so dequantize like the target's wo_a.
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
        rules[g + "exp_probs_b.bias"] = raw(t + "ffn.gate.bias")
        rules[g + "ffn_gate_shexp.weight"] = quant(t + "ffn.shared_experts.w1.weight")
        rules[g + "ffn_up_shexp.weight"] = quant(t + "ffn.shared_experts.w3.weight")
        rules[g + "ffn_down_shexp.weight"] = quant(t + "ffn.shared_experts.w2.weight")
    return rules


_MOE_EXPS_RE = re.compile(r"dspark\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
_MOE_SHARD_NAMES = {"gate": "w1", "up": "w3", "down": "w2"}


def dspark_gguf_weights_iterator(
    gguf_file: str, hf_config: PretrainedConfig
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs for a deepseek4-dspark GGUF.

    Same two-pass qweight_type/qweight protocol as the deepseek4 iterator;
    packed experts are emitted per expert as ``mtp.{N}.ffn.experts.{E}.w{1,3,
    2}.weight`` so the remapper's ``.w1. -> .gate_proj.`` rewrite feeds
    FusedMoE's expert mapping.
    """
    import gguf
    import torch
    from gguf.quants import dequantize

    rules = dspark_tensor_rules(hf_config)
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
                f"Unexpected tensor {tensor.name!r} in deepseek4-dspark GGUF; "
                "update dspark_tensor_rules()."
            )
    for stage_id, types in moe_exps_types.items():
        # FusedMoE fuses gate+up under a single w13 quant type (mirrors the
        # target iterator's check): mixed types would both be written to the
        # one w13_qweight_type marker and dequantize each other's format.
        if "gate" in types and "up" in types and types["gate"] != types["up"]:
            raise ValueError(
                f"dspark.{stage_id}: ffn_gate_exps is {types['gate'].name} "
                f"but ffn_up_exps is {types['up'].name}; the fused w13 GGUF "
                "MoE weight requires one quantization type for both."
            )
    present = {t.name for t in reader.tensors}
    missing = sorted(set(rules) - present)
    if missing:
        raise ValueError(
            f"deepseek4-dspark GGUF is missing {len(missing)} expected tensors "
            f"(first few: {missing[:5]})."
        )

    def _moe_names(name: str):
        m = _MOE_EXPS_RE.fullmatch(name)
        if m is None:
            return None
        return int(m.group(1)), _MOE_SHARD_NAMES[m.group(2)]

    for tensor in reader.tensors:
        moe = _moe_names(tensor.name)
        if moe is not None:
            stage, proj = moe
            for expert_id in range(tensor.data.shape[0]):
                yield (
                    f"mtp.{stage}.ffn.experts.{expert_id}.{proj}.qweight_type",
                    torch.tensor(_normalized_quant_type(tensor.tensor_type)),
                )
            continue
        rule = rules[tensor.name]
        if rule.mode == "quant":
            yield (
                _quant_target_name(rule, "qweight_type"),
                torch.tensor(_normalized_quant_type(tensor.tensor_type)),
            )

    for tensor in reader.tensors:
        moe = _moe_names(tensor.name)
        if moe is not None:
            stage, proj = moe
            for expert_id in range(tensor.data.shape[0]):
                expert_weight = _from_readonly_numpy(tensor.data[expert_id])
                if tensor.tensor_type.name == "F16":
                    expert_weight = expert_weight.to(torch.bfloat16)
                yield (
                    f"mtp.{stage}.ffn.experts.{expert_id}.{proj}.qweight",
                    expert_weight,
                )
            continue
        rule = rules[tensor.name]
        if rule.mode == "quant":
            weight = torch.tensor(tensor.data)
            if tensor.tensor_type.name == "F16":
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
