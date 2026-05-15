"""Minimal PyTorch reference model.

This is not the final FPGA architecture. It is the first executable reference used
to freeze tensor shapes and build correctness tests before hardware work.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fllm.attention import GQAttention, KVCache
from fllm.config import FLLMConfig
from fllm.moe import MoEMLP


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.local_window = config.local_window
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        positions = torch.arange(seq_len, device=x.device)
        causal = positions[:, None] >= positions[None, :]
        local = positions[:, None] - positions[None, :] < self.local_window
        mask = causal & local
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.resid_dropout(self.out(out))


class MLP(nn.Module):
    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        inner = config.hidden_size * config.mlp_ratio
        self.up = nn.Linear(config.hidden_size, inner, bias=False)
        self.gate = nn.Linear(config.hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class FLLMBlock(nn.Module):
    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = GQAttention(config) if config.use_gqa else LocalCausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = MoEMLP(config) if config.use_moe else MLP(config)
        self.use_gqa = config.use_gqa

    def forward(self, x: torch.Tensor, *, cache: KVCache | None = None) -> torch.Tensor:
        if self.use_gqa:
            x = x + self.attn(self.attn_norm(x), cache=cache)
        else:
            x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FLLMForCausalLM(nn.Module):
    def __init__(self, config: FLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.blocks = nn.ModuleList(FLLMBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if caches is None and seq_len > self.config.context_length:
            raise ValueError("sequence length exceeds configured context_length")

        x = self.token_embedding(input_ids)
        if not self.config.use_rope:
            offset = caches[0].length if caches is not None and len(caches) > 0 else 0
            positions = torch.arange(offset, offset + seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        for idx, block in enumerate(self.blocks):
            cache = caches[idx] if caches is not None else None
            x = block(x, cache=cache) if self.config.use_gqa else block(x)
        return self.lm_head(self.final_norm(x))

    def new_caches(self, batch: int, device, dtype) -> list[KVCache]:
        return [block.attn.new_cache(batch, device, dtype) for block in self.blocks]

    def collect_aux_loss(self) -> torch.Tensor:
        """Sum of last MoE aux losses across blocks. Zero if no MoE."""
        total = torch.zeros((), device=self.token_embedding.weight.device)
        if not self.config.use_moe:
            return total
        from fllm.moe import MoEMLP
        for block in self.blocks:
            if isinstance(block.mlp, MoEMLP):
                total = total + block.mlp.last_aux_loss()
        return total

    def set_vocab_cache(self, cache_token_ids: torch.Tensor) -> None:
        """Replace lm_head with a VocabCacheLMHead for FPGA-fast inference."""
        from fllm.vocab_cache import VocabCacheLMHead
        if self.config.tie_embeddings:
            raise ValueError("vocab cache not supported with tied embeddings")
        cached = VocabCacheLMHead(self.lm_head, cache_token_ids)
        self.lm_head = cached
        self.config = self.config  # mutable swap

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        if self.config.use_gqa:
            caches = self.new_caches(input_ids.size(0), input_ids.device, self.token_embedding.weight.dtype)
            _ = self(input_ids, caches=caches)
            for _ in range(max_new_tokens):
                logits = self(input_ids[:, -1:], caches=caches)[:, -1, :]
                logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
                next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                    break
            return input_ids

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length :]
            logits = self(context)[:, -1, :]
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_idx in range(input_ids.size(0)):
        seen = torch.unique(input_ids[batch_idx])
        selected = adjusted[batch_idx, seen]
        adjusted[batch_idx, seen] = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
    return adjusted
