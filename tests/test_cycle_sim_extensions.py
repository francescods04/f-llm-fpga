#!/usr/bin/env python3
"""Tests for cycle_sim extensions (cache hit, sparse skip, cross-FPGA, tree-spec)."""

import pytest

from fllm.cycle_sim import (
    TileSpec,
    ModelShape,
    matvec_cycles,
    moe_cycles,
    hbm_bytes_per_token,
    estimate_token,
)


def test_matvec_cycles_with_density():
    tile = TileSpec()
    c_dense = matvec_cycles(4096, 4096, tile, density=1.0)
    c_sparse = matvec_cycles(4096, 4096, tile, density=0.5)
    assert c_sparse < c_dense


def test_moe_cycles_sparse():
    tile = TileSpec()
    shape_dense = ModelShape(use_sparse_skip=False, sparse_density=1.0)
    shape_sparse = ModelShape(use_sparse_skip=True, sparse_density=0.5)
    c_dense = moe_cycles(shape_dense, tile)
    c_sparse = moe_cycles(shape_sparse, tile)
    assert c_sparse < c_dense


def test_cross_fpga_latency_increases_ms():
    tile = TileSpec()
    shape0 = ModelShape(cross_fpga_dispatch_us=0.0)
    shape50 = ModelShape(cross_fpga_dispatch_us=50.0)
    r0 = estimate_token(shape0, tile)
    r50 = estimate_token(shape50, tile)
    assert r50.ms_per_token > r0.ms_per_token


def test_speculative_factor_increases_tps():
    tile = TileSpec()
    shape1 = ModelShape(speculative_effective_factor=1.0)
    shape3 = ModelShape(speculative_effective_factor=3.0)
    r1 = estimate_token(shape1, tile)
    r3 = estimate_token(shape3, tile)
    assert r3.tokens_per_s > r1.tokens_per_s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
