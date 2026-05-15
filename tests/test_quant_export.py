"""Round-trip + numeric sanity tests for quantization and INT4 export."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fllm.export import dequantize, export_linear_int4, load_linear_int4
from fllm.quant import QuantConfig, fake_quant_weight, quantize_model_
from fllm.config import FLLMConfig
from fllm.model import FLLMForCausalLM


def test_fake_quant_int4_in_range():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    wq = fake_quant_weight(w, bits=4)
    assert wq.shape == w.shape
    err = (w - wq).abs().mean().item()
    assert err < 0.2, f"INT4 fake-quant error too high: {err}"


def test_int4_export_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "linear.bin"
        export_linear_int4(w, path)
        rec = load_linear_int4(path)
        dq = dequantize(rec)
    assert dq.shape == w.shape
    err = (w - dq).abs().mean().item()
    assert err < 0.2, f"export round-trip error too high: {err}"
    assert rec["qweight"].min() >= -8 and rec["qweight"].max() <= 7


def test_quantize_model_runs():
    cfg = FLLMConfig(
        vocab_size=64, context_length=16, hidden_size=32,
        num_layers=2, num_heads=4, local_window=8,
    )
    model = FLLMForCausalLM(cfg)
    replaced = quantize_model_(model, QuantConfig(), skip=("lm_head",))
    assert replaced > 0
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = model(x)
    assert y.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()


def test_moe_block_forward():
    cfg = FLLMConfig(
        vocab_size=64, context_length=16, hidden_size=32,
        num_layers=2, num_heads=4, local_window=8,
        use_moe=True, num_experts=4, active_experts=2,
    )
    model = FLLMForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = model(x)
    assert y.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()
