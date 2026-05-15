"""Tests for compressed global attention, N:M sparsity, and unified cycle sim."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.config import FLLMConfig
from fllm.cycle_sim import ModelShape, TileSpec, estimate_token
from fllm.model import FLLMForCausalLM
from fllm.sparsity import NMSparsity, apply_nm_mask, apply_nm_to_model


def test_compressed_global_attention_runs():
    cfg = FLLMConfig(
        vocab_size=64, context_length=64, hidden_size=32,
        num_layers=2, num_heads=4, num_kv_heads=2, local_window=8,
        use_gqa=True, use_rope=True,
        use_compressed_global=True, compressed_block_size=4, compressed_top_k=2,
    )
    m = FLLMForCausalLM(cfg).eval()
    x = torch.randint(0, 64, (1, 32))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 32, 64)
    assert torch.isfinite(y).all()


def test_compressed_global_admits_more_tokens_than_local():
    cfg_no = FLLMConfig(
        vocab_size=64, context_length=64, hidden_size=32,
        num_layers=1, num_heads=4, num_kv_heads=2, local_window=4,
        use_gqa=True,
    )
    cfg_yes = FLLMConfig(
        vocab_size=64, context_length=64, hidden_size=32,
        num_layers=1, num_heads=4, num_kv_heads=2, local_window=4,
        use_gqa=True,
        use_compressed_global=True, compressed_block_size=4, compressed_top_k=4,
    )
    torch.manual_seed(0)
    m_no = FLLMForCausalLM(cfg_no).eval()
    torch.manual_seed(0)
    m_yes = FLLMForCausalLM(cfg_yes).eval()
    x = torch.randint(0, 64, (1, 32))
    with torch.no_grad():
        y_no = m_no(x)
        y_yes = m_yes(x)
    # Different attention coverage -> outputs must differ
    assert (y_no - y_yes).abs().max().item() > 1e-3


def test_nm_sparsity_density_correct():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    masked = apply_nm_mask(w, NMSparsity(n=1, m=4))
    nonzeros = (masked != 0).sum().item()
    assert nonzeros == 8 * (32 // 4)  # exactly 1 nonzero per 4-group


def test_nm_sparsity_keeps_largest():
    torch.manual_seed(0)
    w = torch.tensor([[1.0, -3.0, 2.0, 0.5]])
    masked = apply_nm_mask(w, NMSparsity(n=1, m=4))
    assert masked[0, 1].item() == -3.0
    assert (masked[0, 0] == 0).item()
    assert (masked[0, 2] == 0).item()


def test_nm_sparsity_applied_to_model_count():
    cfg = FLLMConfig(
        vocab_size=64, context_length=16, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, local_window=16,
        use_gqa=True,
    )
    m = FLLMForCausalLM(cfg)
    count = apply_nm_to_model(m, NMSparsity(n=2, m=4))
    # Each GQA block has q/k/v/o = 4 linears; 2 blocks = 8
    assert count >= 8


def test_cycle_sim_reports_bottleneck():
    shape = ModelShape(avg_ctx=1024)
    r_compute_bound = estimate_token(shape, TileSpec(num_tiles=1, tile_rows=8))
    r_hbm_bound = estimate_token(shape, TileSpec(num_tiles=32, tile_rows=64))
    assert r_compute_bound.bottleneck == "compute"
    assert r_hbm_bound.bottleneck == "hbm"
    assert r_compute_bound.tokens_per_s <= r_hbm_bound.tokens_per_s


def test_cycle_sim_hbm_bytes_positive():
    r = estimate_token(ModelShape(), TileSpec())
    assert r.bytes_per_token > 0
    assert r.hbm_tokens_per_s > 0
