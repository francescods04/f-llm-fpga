"""Tests for the Qwen3-A3B-shape integration: GQA+RoPE+KV cache, aux loss,
HF config import, FPGA-sim forward, cycle simulator."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.config import FLLMConfig
from fllm.cycle_sim import ModelShape, TileSpec, estimate_token
from fllm.fpga_sim import fpga_sim_mode
from fllm.hf_config import load_hf_config, write_qwen3_a3b_template
from fllm.model import FLLMForCausalLM


def _gqa_cfg(use_rope: bool = True) -> FLLMConfig:
    return FLLMConfig(
        vocab_size=256, context_length=32, hidden_size=64,
        num_layers=2, num_heads=8, num_kv_heads=2, local_window=32,
        use_gqa=True, use_rope=use_rope, rope_theta=1e4,
    )


def test_gqa_full_vs_cached_match():
    cfg = _gqa_cfg()
    m = FLLMForCausalLM(cfg).eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        y_full = m(x)
        caches = m.new_caches(1, x.device, m.token_embedding.weight.dtype)
        y_cached = m(x, caches=caches)
    assert torch.allclose(y_full, y_cached, atol=1e-4)
    assert caches[0].length == 8


def test_gqa_step_wise_decode():
    cfg = _gqa_cfg()
    m = FLLMForCausalLM(cfg).eval()
    x = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        full = m(torch.cat([x, torch.zeros(1, 1, dtype=torch.long)], dim=-1))
        caches = m.new_caches(1, x.device, m.token_embedding.weight.dtype)
        m(x, caches=caches)
        step = m(torch.zeros(1, 1, dtype=torch.long), caches=caches)
    assert torch.allclose(full[:, -1, :], step[:, -1, :], atol=1e-4)


def test_moe_aux_loss_nonzero():
    cfg = FLLMConfig(
        vocab_size=128, context_length=16, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, local_window=16,
        use_gqa=True, use_moe=True, num_experts=4, active_experts=2,
    )
    m = FLLMForCausalLM(cfg)
    x = torch.randint(0, 128, (2, 8))
    m(x)
    aux = m.collect_aux_loss()
    assert aux.numel() == 1
    assert float(aux) > 0
    # untrained router should be near uniform => aux near 1.0
    assert 0.5 < float(aux) < 4.0


def test_hf_config_loads_qwen3_template():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        write_qwen3_a3b_template(path)
        cfg, report = load_hf_config(path)
    assert cfg.hidden_size == 4096
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 8
    assert cfg.use_moe is True
    assert cfg.num_experts == 128
    assert cfg.active_experts == 8
    assert cfg.moe_inner_size == 1408
    assert cfg.use_rope is True
    assert cfg.use_gqa is True
    assert "hidden_size" in report.matched


def test_fpga_sim_forward_close_to_bf16():
    cfg = _gqa_cfg()
    m = FLLMForCausalLM(cfg).eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        y_bf16 = m(x)
    sim = copy.deepcopy(m)
    with fpga_sim_mode(sim):
        with torch.no_grad():
            y_sim = sim(x)
    cos = torch.nn.functional.cosine_similarity(y_bf16.flatten(), y_sim.flatten(), dim=0).item()
    assert cos > 0.95, f"FPGA-sim cos-sim too low: {cos}"


def test_cycle_sim_scales_with_tile_count():
    shape = ModelShape(avg_ctx=1024)
    small = estimate_token(shape, TileSpec(num_tiles=2, tile_rows=16))
    big = estimate_token(shape, TileSpec(num_tiles=16, tile_rows=32))
    assert big.tokens_per_s > small.tokens_per_s
    assert big.cycles_per_token < small.cycles_per_token


def test_cycle_sim_moe_dominates_at_a3b_shape():
    shape = ModelShape()
    r = estimate_token(shape, TileSpec())
    assert r.breakdown_per_block["moe"] > r.breakdown_per_block["qkv_proj"]
    assert r.breakdown_per_block["moe"] > r.breakdown_per_block["attention"]
