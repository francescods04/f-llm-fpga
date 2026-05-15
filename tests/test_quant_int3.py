#!/usr/bin/env python3
"""Tests for INT3 weight quantization (Gate G4)."""

import pytest
import torch

from fllm.quant import fake_quant_weight_int3, QuantConfig, QuantLinear


def test_int3_range():
    w = torch.randn(16, 32)
    wq = fake_quant_weight_int3(w, group_size=-1)
    # Quantized values must be multiples of the scale and within [-4,3]
    scale = w.abs().amax(dim=-1, keepdim=True) / 3.0
    diff = (wq / scale).round()
    assert diff.min() >= -4
    assert diff.max() <= 3


def test_int3_group64():
    w = torch.randn(8, 128)
    wq = fake_quant_weight_int3(w, group_size=64)
    assert wq.shape == w.shape


def test_quant_linear_int3():
    base = torch.nn.Linear(32, 16, bias=False)
    qcfg = QuantConfig(weight_bits=4, use_int3=True, weight_group_size=64)
    ql = QuantLinear(base, qcfg)
    x = torch.randn(2, 32)
    out = ql(x)
    assert out.shape == (2, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
