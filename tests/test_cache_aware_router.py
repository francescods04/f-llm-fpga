#!/usr/bin/env python3
"""Tests for cache-aware MoE router (Novelty N1)."""

import pytest
import torch

from fllm.cache_aware_router import CacheAwareConfig, CacheAwareRouter, pick_resident_set


def test_router_basic():
    cfg = CacheAwareConfig(num_experts=8, top_k=2, alpha=2.0)
    router = CacheAwareRouter(hidden_size=64, cfg=cfg)
    hidden = torch.randn(2, 4, 64)
    weights, indices = router(hidden)
    assert weights.shape == (8, 2)
    assert indices.shape == (8, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_resident_bias():
    cfg = CacheAwareConfig(num_experts=8, top_k=2, alpha=10.0)
    router = CacheAwareRouter(hidden_size=64, cfg=cfg)
    hidden = torch.randn(2, 4, 64)
    resident = torch.zeros(8, dtype=torch.bool)
    resident[0] = True
    resident[3] = True
    weights, indices = router(hidden, resident_mask=resident)
    # With huge alpha, top-2 should almost always include resident experts
    all_indices = indices.flatten().tolist()
    assert 0 in all_indices or 3 in all_indices


def test_loss_computation():
    cfg = CacheAwareConfig(num_experts=8, top_k=2)
    router = CacheAwareRouter(hidden_size=64, cfg=cfg)
    hidden = torch.randn(2, 4, 64)
    resident = torch.zeros(8, dtype=torch.bool)
    resident[0] = True
    weights, indices = router(hidden, resident_mask=resident)
    loss = router.compute_loss(hidden, resident, indices)
    assert "total" in loss
    assert "hit_rate" in loss
    assert 0 <= loss["hit_rate"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
