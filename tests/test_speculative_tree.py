#!/usr/bin/env python3
"""Tests for tree speculation and draft cascade (Novelty N3)."""

import pytest
import torch

from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM
from fllm.speculative import (
    DraftModel,
    speculative_generate_simple,
    tree_speculative_generate,
    DraftCascade,
)


def make_models():
    cfg = FLLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        moe_inner=128,
        num_experts=4,
        top_k=2,
        context_length=128,
    )
    target = FLLMForCausalLM(cfg)
    draft = DraftModel(cfg, num_layers=1)
    return target, draft, cfg


def test_tree_spec_runs():
    target, draft, cfg = make_models()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = tree_speculative_generate(
        target, draft, input_ids, max_new_tokens=8, gamma=2, tree_depth=2
    )
    assert out.shape[0] == 1
    assert out.shape[1] >= 5


def test_draft_cascade_stub():
    target, draft, cfg = make_models()
    cascade = DraftCascade([draft])
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = cascade.generate(target, input_ids, max_new_tokens=4)
    assert out.shape[0] == 1


def test_tree_vs_simple_equivalent_when_depth_one():
    target, draft, cfg = make_models()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out_simple = speculative_generate_simple(
        target, draft, input_ids, max_new_tokens=8, gamma=2
    )
    out_tree = tree_speculative_generate(
        target, draft, input_ids, max_new_tokens=8, gamma=2, tree_depth=1
    )
    # tree_depth=1 should be equivalent to linear draft
    assert torch.equal(out_simple, out_tree)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
