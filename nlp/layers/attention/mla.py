import math

import torch
from torch import nn
from torch.nn import Linear, RMSNorm


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Headed Attention Layer (MLA) as per DeepSeek V3

    :param dim: Dimensionality of the input features.
    :param n_heads: Number of attention heads.
    :param q_lora_rank: Rank for low-rank query projection.
    :param kv_lora_rank: Rank for low-rank key/value projection.
    :param qk_nope_head_dim: Dimensionality of non-positional query/key projections.
    :param qk_rope_head_dim: Dimensionality of rotary-positional query/key projections.
    :param v_head_dim: Dimensionality of value projections.
    :param max_seq_len: Maximum sequence length.
    :param original_seq_len: Original sequence length.
    :param rope_factor: Factor for rotary positional embeddings.
    :param mscale: Scaling factor for softmax.
    :param max_batch_size: Maximum batch size.
    :param block_size: Block size for attention computation.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_seq_len: int,
        original_seq_len: int,
        rope_factor: float = 1.0,
        mscale: float = 1.0,
        max_batch_size: int = 1,
        block_size: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_seq_len = max_seq_len
        self.rope_factor = rope_factor
        self.mscale = mscale
        self.max_batch_size = max_batch_size
        self.block_size = block_size

        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim**-0.5
        if self.max_seq_len > original_seq_len:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.register_buffer(
            "k_cache",
            torch.zeros(max_batch_size, max_seq_len, self.n_local_heads, self.qk_head_dim),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(max_batch_size, max_seq_len, self.n_local_heads, self.v_head_dim),
            persistent=False,
        )

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional embeddings to the input tensor.

        :param x: Input tensor with positional embeddings to be applied.
        :param freqs_cis: Precomputed complex exponential values for positional embeddings.
        """
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Headed Latent Attention Layer

        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :param start_pos: Starting position in the sequence for caching.
        :param freqs_cis: Precomputed complex exponential values for rotary embeddings.
        :param mask: Mask tensor to exclude certain positions from attention.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = self.apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        scores = (
            torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        )

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])

        x = self.wo(x.flatten(2))
        return x
