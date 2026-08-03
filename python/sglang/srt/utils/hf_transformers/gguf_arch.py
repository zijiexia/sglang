"""Registry of GGUF architectures that sglang handles itself.

transformers' GGUF integration supports a fixed set of ``general.architecture``
values; checkpoints outside that set (currently only ``deepseek4``) get an
sglang-side adapter that supplies the three interop pieces the serving stack
needs: an HF config builder, a tokenizer builder, and a weights iterator.

Dispatch sites (``get_config``, ``get_tokenizer``, ``GGUFModelLoader``) consult
this registry instead of importing per-architecture modules directly; adding a
new architecture means registering one adapter here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Generator, Optional, Tuple

import msgspec

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class GGUFArchAdapter(msgspec.Struct, frozen=True):
    """Interop hooks for one GGUF architecture.

    - ``gguf_arch``: the ``general.architecture`` value this adapter serves.
    - ``build_config(gguf_path)``: GGUF metadata -> HF config. Must set
      ``tie_word_embeddings`` itself: adapter loads skip the generic loader's
      extra-tensor tie detection.
    - ``build_tokenizer(gguf_path, **kwargs)``: the file's embedded tokenizer.
    - ``weights_iterator(gguf_path, hf_config)``: (name, tensor) pairs in the
      naming convention the model's ``load_weights`` accepts.
    """

    gguf_arch: str
    build_config: Callable[[str], PretrainedConfig]
    build_tokenizer: Callable[..., PreTrainedTokenizerFast]
    weights_iterator: Callable[
        [str, PretrainedConfig], Generator[Tuple[str, torch.Tensor], None, None]
    ]


def read_gguf_architecture(gguf_path: str) -> str:
    """Return ``general.architecture`` of a GGUF file."""
    import gguf

    reader = gguf.GGUFReader(gguf_path)
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError(
            f"GGUF file {gguf_path!r} is missing the general.architecture key."
        )
    return field.contents()


def get_gguf_arch_adapter(gguf_path: str) -> Optional[GGUFArchAdapter]:
    """The adapter for a GGUF file's architecture, or None.

    A probe, not a validator: unreadable or arch-less files answer None so
    callers fall through to the transformers GGUF path (and keep its error
    surface).
    """
    try:
        arch = read_gguf_architecture(gguf_path)
    except Exception as e:
        logger.debug("Could not read GGUF architecture from %s: %s", gguf_path, e)
        return None
    return _adapters().get(arch)


_ADAPTERS: dict[str, GGUFArchAdapter] = {}


def register_gguf_arch_adapter(adapter: GGUFArchAdapter) -> None:
    existing = _ADAPTERS.get(adapter.gguf_arch)
    if existing is not None and existing != adapter:
        raise ValueError(
            f"GGUF architecture {adapter.gguf_arch!r} is already registered."
        )
    _ADAPTERS[adapter.gguf_arch] = adapter


def _adapters() -> dict[str, GGUFArchAdapter]:
    _register_builtin_adapters()
    return _ADAPTERS


_builtin_adapters_registered = False


def _register_builtin_adapters() -> None:
    global _builtin_adapters_registered
    if _builtin_adapters_registered:
        return

    from sglang.srt.utils.hf_transformers import gguf_deepseek4, gguf_deepseek4_dspark

    register_gguf_arch_adapter(
        GGUFArchAdapter(
            gguf_arch=gguf_deepseek4.GGUF_ARCH,
            build_config=gguf_deepseek4.build_config_from_gguf,
            build_tokenizer=gguf_deepseek4.build_tokenizer_from_gguf,
            weights_iterator=gguf_deepseek4.deepseek4_gguf_weights_iterator,
        )
    )
    register_gguf_arch_adapter(
        GGUFArchAdapter(
            gguf_arch=gguf_deepseek4_dspark.GGUF_ARCH,
            build_config=gguf_deepseek4_dspark.build_config_from_gguf,
            build_tokenizer=gguf_deepseek4_dspark.build_tokenizer_from_gguf,
            weights_iterator=gguf_deepseek4_dspark.dspark_gguf_weights_iterator,
        )
    )
    # Latched only after registration succeeds, so a failed import re-raises
    # on the next probe instead of poisoning the registry into fall-through.
    _builtin_adapters_registered = True
