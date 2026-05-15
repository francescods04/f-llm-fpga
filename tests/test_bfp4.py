#!/usr/bin/env python3
"""Tests for BFP4 KV cache (Gate G7)."""

import pytest
import torch

from fllm.bfp import BFPConfig, bfp_quantize, bfp_pack, bfp_unpack, bytes_per_value


def test_bfp4_quantize():
    cfg = BFPConfig(mantissa_bits=4, block_size=32)
    x = torch.randn(2, 128)
    y = bfp_quantize(x, cfg)
    assert y.shape == x.shape
    # Relative error should be modest for 4-bit mantissa
    rel_err = (x - y).norm() / x.norm()
    assert rel_err < 0.15


def test_bfp4_pack_roundtrip():
    cfg = BFPConfig(mantissa_bits=4, block_size=32)
    x = torch.randn(2, 128)
    packed = bfp_pack(x, cfg)
    x2 = bfp_unpack(packed)
    # Unpack is exact because it re-uses the same scale
    assert torch.allclose(x, x2, atol=1e-3)


def test_bfp4_bytes():
    cfg = BFPConfig(mantissa_bits=4, block_size=32)
    bpv = bytes_per_value(cfg)
    # 4 bits mantissa = 0.5 bytes + 1 byte exp / 32 values = 0.53125
    assert abs(bpv - 0.53125) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
