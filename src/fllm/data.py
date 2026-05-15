"""Data loading helpers for tiny language-model experiments."""

from __future__ import annotations

from pathlib import Path

import torch

from fllm.tokenizer import TextTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def encode_text(text: str, tokenizer: TextTokenizer) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return torch.tensor(token_ids, dtype=torch.long)


def split_tokens(tokens: torch.Tensor, train_fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = max(1, int(tokens.numel() * train_fraction))
    split = min(split, tokens.numel() - 1)
    return tokens[:split], tokens[split:]


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= seq_len + 1:
        raise ValueError("token stream is too short for requested seq_len")

    starts = torch.randint(0, tokens.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([tokens[start : start + seq_len] for start in starts])
    y = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)
