from __future__ import annotations

from torch import nn

from .block import Block, CausalSelfAttention, MLP, RMSNorm
from __future__ import annotations

from torch import nn

from .attention import (
    SlidingWindowCausalSelfAttention,
    SlidingWindowGQACausalSelfAttention,
    XSACausalSelfAttention,
    XSAGQACausalSelfAttention,
    XSASlidingWindowCausalSelfAttention,
    XSASlidingWindowGQACausalSelfAttention,
    
)

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "gqa": CausalSelfAttention,
    "baseline": CausalSelfAttention,
    "swa": SlidingWindowCausalSelfAttention,
    "gqa_swa": SlidingWindowGQACausalSelfAttention,
    "xsa": XSACausalSelfAttention,
    "xsa_gqa": XSAGQACausalSelfAttention,
    "xsa_swa": XSASlidingWindowCausalSelfAttention,
    "xsa_gqa_swa": XSASlidingWindowGQACausalSelfAttention,
}

MLP_REGISTRY: dict[str, type[nn.Module]] = {
    "relu2": MLP,
    "baseline": MLP,
}

NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "rmsnorm": RMSNorm,
    "baseline": RMSNorm,
}

BLOCK_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": Block,
}


def get_attention_cls(name: str) -> type[nn.Module]:
    if name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type '{name}'. Available: {sorted(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[name]


def get_mlp_cls(name: str) -> type[nn.Module]:
    if name not in MLP_REGISTRY:
        raise ValueError(
            f"Unknown mlp type '{name}'. Available: {sorted(MLP_REGISTRY)}"
        )
    return MLP_REGISTRY[name]


def get_norm_cls(name: str) -> type[nn.Module]:
    if name not in NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm type '{name}'. Available: {sorted(NORM_REGISTRY)}"
        )
    return NORM_REGISTRY[name]


def get_block_cls(name: str) -> type[nn.Module]:
    if name not in BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block type '{name}'. Available: {sorted(BLOCK_REGISTRY)}"
        )
    return BLOCK_REGISTRY[name]