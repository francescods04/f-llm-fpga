"""Stability and numerical sanity tests for FPGA-sim on Qwen3-A3B shape.

These tests do NOT require real Qwen weights. They exercise the exact tensor
shapes of Qwen3-A3B (4096 hidden, 32 heads, 8 KV heads, RoPE, MoE) but keep
the model small enough to run on a laptop CPU without swapping.

Strategy to avoid memory bloat:
  - Test a single FLLMBlock (not the full model) so embedding + LM head
    are excluded.
  - MoE test uses num_experts=4 (not 128) so expert memory stays under ~200M.
  - fpga_sim_mode runs with restore_weights=False to avoid a deep copy of
    the state dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.config import FLLMConfig
from fllm.fpga_sim import fpga_sim_mode
from fllm.model import FLLMBlock


def _dense_qwen_block_cfg() -> FLLMConfig:
    """Single dense block with Qwen tensor shapes."""
    return FLLMConfig(
        vocab_size=8192,  # dummy, not used for block-only test
        context_length=64,
        hidden_size=4096,
        num_layers=1,
        num_heads=32,
        num_kv_heads=8,
        local_window=64,
        mlp_ratio=4,
        use_moe=False,
        use_rope=True,
        rope_theta=10_000_000.0,
        use_gqa=True,
    )


def _moe_qwen_block_cfg() -> FLLMConfig:
    """Single MoE block with Qwen tensor shapes (few experts to keep RAM low)."""
    return FLLMConfig(
        vocab_size=8192,
        context_length=64,
        hidden_size=4096,
        num_layers=1,
        num_heads=32,
        num_kv_heads=8,
        local_window=64,
        mlp_ratio=4,
        use_moe=True,
        num_experts=4,      # reduced from 128 for test speed
        active_experts=2,
        moe_inner_size=1408,  # exact Qwen expert inner dim
        use_rope=True,
        rope_theta=10_000_000.0,
        use_gqa=True,
    )


def test_dense_block_fpga_sim_no_nan():
    cfg = _dense_qwen_block_cfg()
    block = FLLMBlock(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)

    with torch.no_grad():
        y_ref = block(x)

    sim_block = type(block)(cfg).eval()
    sim_block.load_state_dict(block.state_dict())

    with fpga_sim_mode(sim_block, restore_weights=False):
        with torch.no_grad():
            y_sim = sim_block(x)

    assert torch.isfinite(y_sim).all(), "dense block: FPGA-sim output contains NaN/Inf"
    cos = torch.nn.functional.cosine_similarity(
        y_ref.flatten(), y_sim.flatten(), dim=0
    ).item()
    print(f"dense_block cos_sim = {cos:.4f}")
    assert cos > 0.80, f"dense block cosine similarity too low: {cos}"


def test_moe_block_fpga_sim_no_nan():
    cfg = _moe_qwen_block_cfg()
    block = FLLMBlock(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)

    with torch.no_grad():
        y_ref = block(x)

    sim_block = type(block)(cfg).eval()
    sim_block.load_state_dict(block.state_dict())

    with fpga_sim_mode(sim_block, restore_weights=False):
        with torch.no_grad():
            y_sim = sim_block(x)

    assert torch.isfinite(y_sim).all(), "moe block: FPGA-sim output contains NaN/Inf"
    cos = torch.nn.functional.cosine_similarity(
        y_ref.flatten(), y_sim.flatten(), dim=0
    ).item()
    print(f"moe_block cos_sim = {cos:.4f}")
    assert cos > 0.75, f"moe block cosine similarity too low: {cos}"


def test_moe_block_routing_survives_quantization():
    """The router must still produce sensible expert assignments after INT4."""
    cfg = _moe_qwen_block_cfg()
    block = FLLMBlock(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)

    with torch.no_grad():
        _ = block(x)
        if cfg.use_moe and hasattr(block.mlp, "routing_stats"):
            ref_stats = block.mlp.routing_stats(x)

    sim_block = type(block)(cfg).eval()
    sim_block.load_state_dict(block.state_dict())

    with fpga_sim_mode(sim_block, restore_weights=False):
        with torch.no_grad():
            _ = sim_block(x)
            if cfg.use_moe and hasattr(sim_block.mlp, "routing_stats"):
                sim_stats = sim_block.mlp.routing_stats(x)

    # After quantization the expert loads should still sum to ~1 and entropy
    # should not collapse to zero (all tokens to one expert).
    assert abs(ref_stats["expert_load"].sum().item() - 1.0) < 1e-3
    assert abs(sim_stats["expert_load"].sum().item() - 1.0) < 1e-3
    assert sim_stats["router_entropy"] > 0.1, "router collapsed after quantization"


def test_stepwise_decode_with_cache_qwen_shape():
    """GQA + KV cache must stay numerically stable under fpga_sim."""
    cfg = _dense_qwen_block_cfg()
    block = FLLMBlock(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)

    cache = block.attn.new_cache(1, x.device, x.dtype)

    with torch.no_grad():
        y_ref_prefill = block(x, cache=cache)

    # One decode step
    x_step = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        y_ref_step = block(x_step, cache=cache)

    # Reset cache and run under fpga_sim
    cache_sim = block.attn.new_cache(1, x.device, x.dtype)
    sim_block = type(block)(cfg).eval()
    sim_block.load_state_dict(block.state_dict())

    with fpga_sim_mode(sim_block, restore_weights=False):
        with torch.no_grad():
            y_sim_prefill = sim_block(x, cache=cache_sim)
            y_sim_step = sim_block(x_step, cache=cache_sim)

    assert torch.isfinite(y_sim_step).all()
    # Step output should be close between BF16 and sim (looser bound because
    # quantization errors accumulate in the KV cache).
    cos = torch.nn.functional.cosine_similarity(
        y_ref_step.flatten(), y_sim_step.flatten(), dim=0
    ).item()
    print(f"decode_step cos_sim = {cos:.4f}")
    assert cos > 0.70, f"decode step cosine similarity too low: {cos}"


if __name__ == "__main__":
    test_dense_block_fpga_sim_no_nan()
    test_moe_block_fpga_sim_no_nan()
    test_moe_block_routing_survives_quantization()
    test_stepwise_decode_with_cache_qwen_shape()
    print("All Qwen-shape FPGA-sim stability tests passed.")
