#!/usr/bin/env python3
"""Tests for sparsity retrain hook and cold-expert identification."""

import pytest
import torch

from fllm.sparsity import (
    NMSparsity,
    apply_nm_mask,
    apply_nm_to_model,
    identify_cold_experts,
)


def test_identify_cold_experts():
    hist = torch.arange(10, dtype=torch.float)
    mask = identify_cold_experts(hist, cold_fraction=0.3)
    assert mask.sum() == 3
    assert not mask[9]
    assert mask[0]


def test_apply_nm_mask_preserves_topn():
    w = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    pattern = NMSparsity(n=2, m=4)
    m = apply_nm_mask(w, pattern)
    # Top-2 magnitudes are 0.9 and 0.8
    assert m[0, 1] != 0
    assert m[0, 3] != 0
    assert m[0, 0] == 0
    assert m[0, 2] == 0


def test_apply_nm_to_model_counts():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )
    count = apply_nm_to_model(model, NMSparsity(n=2, m=4))
    assert count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
