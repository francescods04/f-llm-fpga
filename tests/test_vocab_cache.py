"""Tests for the vocabulary-cache LM head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM
from fllm.vocab_cache import VocabCacheLMHead, build_vocab_cache_from_corpus


def test_vocab_cache_output_matches_full_head():
    """With random weights, cache head + fallback must equal full head for cache tokens."""
    cfg = FLLMConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4, use_gqa=True, tie_embeddings=False)
    model = FLLMForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 4))

    with torch.no_grad():
        full_logits = model(x)

    # Build cache with top-8 tokens
    cache_ids = torch.arange(8)
    model.set_vocab_cache(cache_ids)

    with torch.no_grad():
        cache_logits = model(x)

    # Cached positions must match exactly
    assert torch.allclose(cache_logits[..., cache_ids], full_logits[..., cache_ids], atol=1e-5)
    # Uncached positions come from fallback head -> should also match exactly
    # because we scatter cache logits back into the full tensor.
    assert torch.allclose(cache_logits, full_logits, atol=1e-5)


def test_cache_hit_rate_computation():
    cfg = FLLMConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4, tie_embeddings=False)
    model = FLLMForCausalLM(cfg).eval()
    cache_ids = torch.tensor([0, 2, 4, 6, 8])
    model.set_vocab_cache(cache_ids)

    targets = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    rate = model.lm_head.cache_hit_rate(targets)
    assert 0.0 < rate <= 1.0


def test_build_cache_from_corpus():
    cfg = FLLMConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4, tie_embeddings=False)
    model = FLLMForCausalLM(cfg).eval()
    corpus = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    cache_head = build_vocab_cache_from_corpus(model.lm_head, corpus, cache_size=3)
    assert cache_head.cache_size == 3
    # Most frequent are 3, 1, 0
    assert cache_head.cache_token_ids[0].item() == 3
    assert cache_head.cache_token_ids[1].item() == 1
    assert cache_head.cache_token_ids[2].item() == 0


if __name__ == "__main__":
    test_vocab_cache_output_matches_full_head()
    test_cache_hit_rate_computation()
    test_build_cache_from_corpus()
    print("All vocab-cache tests passed.")
