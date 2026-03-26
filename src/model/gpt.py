from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .block import CastedLinear, RMSNorm, restore_low_dim_params_to_fp32
from .registry import (
    get_attention_cls,
    get_block_cls,
    get_mlp_cls,
    get_norm_cls,
)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        block_type: str = "baseline",
        attention_type: str = "gqa",
        mlp_type: str = "relu2",
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")

        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap

        block_cls = get_block_cls(block_type)
        attention_cls = get_attention_cls(attention_type)
        mlp_cls = get_mlp_cls(mlp_type)
        norm_cls = get_norm_cls(norm_type)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )

        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    attention_cls=attention_cls,
                    mlp_cls=mlp_cls,
                    norm_cls=norm_cls,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = norm_cls()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)

        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


__all__ = [
    "GPT",
    "CastedLinear",
    "restore_low_dim_params_to_fp32",
    "RMSNorm",
]