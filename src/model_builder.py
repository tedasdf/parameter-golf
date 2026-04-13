import torch
import torch.nn as nn
from config import ModelConfig
from model.block import CastedLinear, restore_low_dim_params_to_fp32
from model.gpt import GPT

# ============================================================
# Model build
# ============================================================

def apply_model_dtype(model: nn.Module, dtype_name: str) -> nn.Module:
    dtype_name = dtype_name.lower()
    if dtype_name == "float32":
        return model.float()
    if dtype_name == "float16":
        return model.half()
    if dtype_name == "bfloat16":
        return model.bfloat16()
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention_shapes(model_cfg: ModelConfig) -> tuple[int, int]:
    if model_cfg.auto_derive_attention_shapes:
        if model_cfg.model_dim % model_cfg.head_dim != 0:
            raise ValueError(
                f"model_dim={model_cfg.model_dim} must be divisible by head_dim={model_cfg.head_dim}"
            )
        num_heads = model_cfg.model_dim // model_cfg.head_dim
        if num_heads % model_cfg.gqa_ratio != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by gqa_ratio={model_cfg.gqa_ratio}"
            )
        num_kv_heads = num_heads // model_cfg.gqa_ratio
        return num_heads, num_kv_heads
    return model_cfg.num_heads, model_cfg.num_kv_heads


def build_model(cfg: ModelConfig, device: torch.device) -> nn.Module:
    num_heads, num_kv_heads = resolve_attention_shapes(cfg)

    model = GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=cfg.mlp_mult,
        tie_embeddings=cfg.tie_embeddings,
        tied_embed_init_std=cfg.tied_embed_init_std,
        logit_softcap=cfg.logit_softcap,
        rope_base=cfg.rope_base,
        qk_gain_init=cfg.qk_gain_init,
        block_type=cfg.block_type,
        attention_type=cfg.attention_type,
        mlp_type=cfg.mlp_type,
        norm_type=cfg.norm_type,
        window_size=cfg.window_size,
    )

    model = model.to(device)
    model = apply_model_dtype(model, cfg.dtype)

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()

    restore_low_dim_params_to_fp32(model)
    return model