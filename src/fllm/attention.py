"""Grouped-query attention with RoPE and a persistent KV cache.

Matches the Qwen3 shape: num_heads queries, num_kv_heads K/V (broadcast at
attention time). RoPE is applied to Q and K at the absolute position implied
by the cache offset, not the local index. The cache is a fixed-size ring per
layer; decode appends one (k, v) row per step instead of recomputing prefix.

Reference path stays float; FPGA path will replace inner matmuls with INT4
matvec and BFP8 KV reads (see src/fllm/bfp.py, src/fllm/quant.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from fllm.config import FLLMConfig


def compressed_block_summary(kv: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool a KV tensor along seq_len into blocks of `block_size`.

    Shape: (B, H, T, D) -> (B, H, ceil(T/block_size), D).
    The FPGA implementation maintains these summaries incrementally: each new
    KV write updates the latest block's running mean; once full, a new block
    slot opens. This Python form is the bulk equivalent for training/eval.
    """
    batch, heads, seq_len, dim = kv.shape
    pad = (-seq_len) % block_size
    if pad:
        padding = torch.zeros(batch, heads, pad, dim, device=kv.device, dtype=kv.dtype)
        kv = torch.cat([kv, padding], dim=2)
    blocks = kv.shape[2] // block_size
    kv = kv.reshape(batch, heads, blocks, block_size, dim)
    return kv.mean(dim=3)


@dataclass
class KVCache:
    k: torch.Tensor      # (batch, num_kv_heads, max_len, head_dim)
    v: torch.Tensor      # (batch, num_kv_heads, max_len, head_dim)
    length: int          # number of valid positions written


def build_rope_table(head_dim: int, max_len: int, theta: float, device, dtype):
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, num_heads, seq, head_dim). cos/sin: (seq, head_dim/2).
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot_even = x1 * cos - x2 * sin
    rot_odd = x1 * sin + x2 * cos
    out = torch.stack([rot_even, rot_odd], dim=-1)
    return out.reshape(*x.shape)


class GQAttention(nn.Module):
    """Grouped-query attention with RoPE and optional persistent KV cache.

    When `cache=None`, runs prefill over the full input sequence (standard
    causal mask, optional local window). When `cache` is provided, expects
    `x` of shape (batch, 1, hidden) and appends one KV row per call.
    """

    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        n_kv = config.kv_heads()
        if config.num_heads % n_kv != 0:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = n_kv
        self.head_dim = config.hidden_size // config.num_heads
        self.local_window = config.local_window
        self.use_rope = config.use_rope
        self.rope_theta = config.rope_theta
        self.context_length = config.context_length
        self.use_compressed_global = config.use_compressed_global
        self.compressed_block_size = config.compressed_block_size
        self.compressed_top_k = config.compressed_top_k

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _rope(self, max_len: int, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].size(0) < max_len
            or self._rope_cache[0].device != device
            or self._rope_cache[0].dtype != dtype
        ):
            self._rope_cache = build_rope_table(
                self.head_dim, max_len, self.rope_theta, device, dtype
            )
        return self._rope_cache

    def new_cache(self, batch: int, device, dtype) -> KVCache:
        k = torch.zeros(batch, self.num_kv_heads, self.context_length, self.head_dim, device=device, dtype=dtype)
        v = torch.zeros_like(k)
        return KVCache(k=k, v=v, length=0)

    def forward(self, x: torch.Tensor, *, cache: KVCache | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            start = cache.length if cache is not None else 0
            cos, sin = self._rope(start + seq_len, x.device, x.dtype)
            cos_slice = cos[start : start + seq_len]
            sin_slice = sin[start : start + seq_len]
            q = apply_rope(q, cos_slice, sin_slice)
            k = apply_rope(k, cos_slice, sin_slice)

        if cache is not None:
            end = cache.length + seq_len
            if end > cache.k.size(2):
                raise ValueError("KV cache overflow; raise context_length")
            cache.k[:, :, cache.length : end] = k
            cache.v[:, :, cache.length : end] = v
            cache.length = end
            k_full = cache.k[:, :, :end]
            v_full = cache.v[:, :, :end]
        else:
            k_full = k
            v_full = v

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_pos_start = cache.length - seq_len if cache is not None else 0
        q_positions = torch.arange(q_pos_start, q_pos_start + seq_len, device=x.device)
        k_positions = torch.arange(k_full.size(-2), device=x.device)
        causal = q_positions[:, None] >= k_positions[None, :]
        local_mask = (q_positions[:, None] - k_positions[None, :]) < self.local_window
        attn_mask = causal & local_mask

        if self.use_compressed_global and k_full.size(-2) > self.local_window:
            global_mask = self._compressed_global_mask(
                q, k_full, q_positions, k_positions, causal
            )
            attn_mask = attn_mask | global_mask

        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        out = probs @ v_full
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.o_proj(out))

    def _compressed_global_mask(
        self,
        q: torch.Tensor,
        k_full: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        causal: torch.Tensor,
    ) -> torch.Tensor:
        """Pick top-k compressed blocks of past context to admit into attention.

        Mean-pools K along blocks of `compressed_block_size`, scores each block
        against each query, then selects `compressed_top_k` blocks per query
        and adds those tokens to the mask. Blocks fully inside the local
        window are excluded (already admitted). Block selection is causal.
        """
        bs = self.compressed_block_size
        block_summaries = compressed_block_summary(k_full, bs)  # (B,H,Nb,D)
        block_scores = (q @ block_summaries.transpose(-2, -1)) / math.sqrt(self.head_dim)

        seq_len = q.size(-2)
        num_blocks = block_summaries.size(-2)
        block_positions = torch.arange(num_blocks, device=q.device) * bs
        block_causal = q_positions[:, None] >= block_positions[None, :]

        outside_local = (q_positions[:, None] - block_positions[None, :]) >= self.local_window
        block_eligible = block_causal & outside_local
        block_scores = block_scores.masked_fill(
            ~block_eligible.unsqueeze(0).unsqueeze(0), torch.finfo(block_scores.dtype).min
        )

        k_eff = min(self.compressed_top_k, num_blocks)
        _, top_blocks = torch.topk(block_scores, k=k_eff, dim=-1)  # (B,H,seq,k)

        # Expand selected blocks back to token positions
        offsets = torch.arange(bs, device=q.device)
        selected_tokens = (top_blocks.unsqueeze(-1) * bs + offsets).reshape(
            *top_blocks.shape[:-1], k_eff * bs
        )
        full_mask = torch.zeros(
            *q.shape[:-1], k_full.size(-2), dtype=torch.bool, device=q.device,
        )
        valid = selected_tokens < k_full.size(-2)
        safe_idx = selected_tokens.clamp(max=k_full.size(-2) - 1)
        full_mask.scatter_(-1, safe_idx, valid)
        return full_mask & causal.unsqueeze(0).unsqueeze(0)
