#!/usr/bin/env python3
"""Tests for named operating presets."""

import pytest

from fllm.presets import PRESETS, preset_to_fpga_optimizations


def test_presets_exist():
    assert "conservative" in PRESETS
    assert "moderate" in PRESETS
    assert "aggressive" in PRESETS


def test_preset_fpga_kwargs():
    for name, preset in PRESETS.items():
        kwargs = preset_to_fpga_optimizations(preset)
        assert "fpga_opts" in kwargs
        opts = kwargs["fpga_opts"]
        assert 0.0 <= opts.expert_cache_hit_rate <= 1.0
        assert 0.0 < opts.kv_byte_factor <= 1.0
        assert opts.speculative_effective_factor >= 1.0


def test_moderate_has_cache_aware():
    p = PRESETS["moderate"]
    assert p.cache_aware is not None
    assert p.speculative_gamma == 4


def test_conservative_no_cache_aware():
    p = PRESETS["conservative"]
    assert p.cache_aware is None
    assert p.speculative_gamma == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
