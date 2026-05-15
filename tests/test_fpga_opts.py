"""Quality + math sanity tests for FPGA-specific optimization simulations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.bfp import BFPConfig, bfp_pack, bfp_quantize, bfp_unpack, bytes_per_value
from fllm.cost_model import FPGAOptimizations, MODELS, compare
from fllm.lut_activations import LUTConfig, lut_quality, rsqrt_lut, silu_lut, softmax_shifted


def test_silu_lut_matches_torch_within_tol():
    torch.manual_seed(0)
    cfg = LUTConfig(num_entries=1024, input_min=-8.0, input_max=8.0)
    x = torch.randn(4096) * 2.0  # stay inside LUT support
    x = x.clamp(cfg.input_min + 1e-3, cfg.input_max - 1e-3)
    exact = F.silu(x)
    approx = silu_lut(x, cfg)
    err = (exact - approx).abs().max().item()
    assert err < 1e-2, f"SiLU LUT max err too high: {err}"


def test_silu_lut_quality_reports_reasonable():
    q = lut_quality(1024)
    assert q["max_abs_err"] < 5e-3
    assert q["bram_kb"] <= 8.0


def test_softmax_shifted_close_to_torch():
    torch.manual_seed(1)
    x = torch.randn(4, 64) * 4.0
    exact = F.softmax(x, dim=-1)
    approx = softmax_shifted(x, dim=-1)
    err = (exact - approx).abs().max().item()
    assert err < 5e-3, f"softmax LUT max err too high: {err}"
    assert torch.allclose(approx.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_rsqrt_lut_within_tol():
    x = torch.linspace(0.1, 100.0, 1000)
    exact = 1.0 / torch.sqrt(x)
    approx = rsqrt_lut(x)
    rel = ((exact - approx).abs() / exact).max().item()
    assert rel < 1e-2, f"rsqrt LUT rel err too high: {rel}"


def test_bfp_quantize_roundtrip_quality():
    torch.manual_seed(2)
    x = torch.randn(8, 128) * 5.0
    deq = bfp_quantize(x, BFPConfig(mantissa_bits=8, block_size=32))
    rel = ((x - deq).pow(2).mean() / x.pow(2).mean()).sqrt().item()
    assert rel < 0.05, f"BFP8 reconstruction NMSE too high: {rel}"


def test_bfp_pack_byte_cost_reasonable():
    bpv = bytes_per_value(BFPConfig(mantissa_bits=8, block_size=32))
    assert 0.99 < bpv < 1.10  # ~1.03 B/value, just over INT8


def test_bfp_pack_unpack_matches_quantize():
    torch.manual_seed(3)
    x = torch.randn(2, 4, 64)
    cfg = BFPConfig(mantissa_bits=8, block_size=16)
    rec = bfp_pack(x, cfg)
    rebuilt = bfp_unpack(rec)
    direct = bfp_quantize(x, cfg)
    assert torch.allclose(rebuilt, direct, atol=1e-5)


def test_fpga_tuned_beats_naive_in_cost_model():
    results = compare("qwen3.6-35b-a3b-int4", fpga_shard_devices=2)
    naive = [r for r in results if "naive port" in r.hardware][0]
    tuned = [r for r in results if "tuned" in r.hardware][0]
    assert tuned.realistic_tokens_per_s > naive.realistic_tokens_per_s
    assert tuned.joule_per_token < naive.joule_per_token
    assert tuned.bytes_per_token < naive.bytes_per_token


def test_fpga_opts_dataclass_defaults_sane():
    o = FPGAOptimizations()
    assert 0.0 <= o.expert_cache_hit_rate <= 1.0
    assert 0.0 <= o.kv_byte_factor <= 1.0
    assert 0.0 <= o.dataflow_overhead_savings <= 1.0
    assert 0.0 <= o.hbm_efficiency <= 1.0
