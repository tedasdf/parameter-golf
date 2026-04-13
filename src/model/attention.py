import torch
from .block import CastedLinear, Rotary, apply_rotary_emb 
from torch import Tensor, nn

import torch.nn.functional as F




class SlidingWindowCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _local_causal_mask(self, seqlen: int, device: torch.device) -> Tensor:
        row = torch.arange(seqlen, device=device)[:, None]
        col = torch.arange(seqlen, device=device)[None, :]
        return (col <= row) & (col >= row - self.window_size + 1)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn_mask = self._local_causal_mask(seqlen, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=False,
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class SlidingWindowGQACausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _local_causal_mask(self, seqlen: int, device: torch.device) -> Tensor:
        row = torch.arange(seqlen, device=device)[:, None]
        col = torch.arange(seqlen, device=device)[None, :]
        return (col <= row) & (col >= row - self.window_size + 1)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn_mask = self._local_causal_mask(seqlen, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)




def _local_causal_mask(seqlen: int, window_size: int, device: torch.device) -> Tensor:
    row = torch.arange(seqlen, device=device)[:, None]
    col = torch.arange(seqlen, device=device)[None, :]
    return (col <= row) & (col >= row - window_size + 1)


def _apply_xsa_projection(y: Tensor, v: Tensor, eps: float = 1e-6) -> Tensor:
    # y, v: (B, H, T, Dh)
    # XSA paper pseudocode:
    # Vn = normalize(V, dim=-1)
    # Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
    v_norm = F.normalize(v, dim=-1, eps=eps)
    return y - (y * v_norm).sum(dim=-1, keepdim=True) * v_norm


class XSACausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int | None = None,  # ignored, kept for constructor compatibility
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_kv_heads != num_heads:
            raise ValueError("XSACausalSelfAttention expects num_kv_heads == num_heads")
        if window_size is not None:
            raise ValueError("XSACausalSelfAttention does not use window_size")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=False,
        )
        z = _apply_xsa_projection(y, v)

        z = z.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(z)


class XSAGQACausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int | None = None,  # ignored, kept for constructor compatibility
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if window_size is not None:
            raise ValueError("XSAGQACausalSelfAttention does not use window_size")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=True,
        )
        z = _apply_xsa_projection(y, v)

        z = z.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(z)


class XSASlidingWindowCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_kv_heads != num_heads:
            raise ValueError("XSASlidingWindowCausalSelfAttention expects num_kv_heads == num_heads")
        if window_size is None or window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn_mask = _local_causal_mask(seqlen, self.window_size, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=False,
        )
        z = _apply_xsa_projection(y, v)

        z = z.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(z)


class XSASlidingWindowGQACausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if window_size is None or window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn_mask = _local_causal_mask(seqlen, self.window_size, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
        )
        z = _apply_xsa_projection(y, v)

        z = z.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(z)