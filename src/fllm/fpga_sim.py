"""Bit-faithful Python simulation of the FPGA decode datapath.

Wraps a trained FLLM model and reroutes its math through the FPGA primitives:

    Linear            -> QuantLinear (INT4 weights, INT8 activations)
    SiLU              -> LUT-based piecewise linear
    Softmax (attn)    -> shifted-softmax with LUT exp
    KV cache          -> BFP8 round-trip on every write/read
    RMSNorm rsqrt     -> log-domain LUT

Forward returns the same shape as the BF16 forward; quality delta is the
research signal. Use scripts/fpga_sim_eval.py to measure ppl vs BF16.

This module monkey-patches the running model in-place (and reverses on close).
It does *not* persist; reload the checkpoint to recover BF16 behavior.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from fllm.attention import GQAttention
from fllm.bfp import BFPConfig, bfp_quantize
from fllm.lut_activations import LUTConfig, lut_apply, rsqrt_lut, silu_lut, softmax_shifted, build_silu_lut
from fllm.model import MLP, RMSNorm
from fllm.moe import Expert
from fllm.quant import QuantConfig, quantize_model_


def _patched_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
    return x * rsqrt_lut(mean_sq) * self.weight


def _patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_out = silu_lut(self.gate(x), self._silu_cfg)
    return self.dropout(self.down(gate_out * self.up(x)))


def _patched_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_out = silu_lut(self.gate(x), self._silu_cfg)
    return self.down(gate_out * self.up(x))


def _patched_gqa_forward(self, x: torch.Tensor, *, cache=None) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    if self.use_rope:
        start = cache.length if cache is not None else 0
        cos, sin = self._rope(start + seq_len, x.device, x.dtype)
        cos_slice = cos[start : start + seq_len]
        sin_slice = sin[start : start + seq_len]
        from fllm.attention import apply_rope
        q = apply_rope(q, cos_slice, sin_slice)
        k = apply_rope(k, cos_slice, sin_slice)

    # BFP8 round-trip on the KV that gets written to HBM
    k_q = bfp_quantize(k, BFPConfig(mantissa_bits=8, block_size=32))
    v_q = bfp_quantize(v, BFPConfig(mantissa_bits=8, block_size=32))

    if cache is not None:
        end = cache.length + seq_len
        cache.k[:, :, cache.length : end] = k_q
        cache.v[:, :, cache.length : end] = v_q
        cache.length = end
        k_full = cache.k[:, :, :end]
        v_full = cache.v[:, :, :end]
    else:
        k_full = k_q
        v_full = v_q

    repeat = self.num_heads // self.num_kv_heads
    if repeat > 1:
        k_full = k_full.repeat_interleave(repeat, dim=1)
        v_full = v_full.repeat_interleave(repeat, dim=1)

    scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

    q_pos_start = cache.length - seq_len if cache is not None else 0
    q_positions = torch.arange(q_pos_start, q_pos_start + seq_len, device=x.device)
    k_positions = torch.arange(k_full.size(-2), device=x.device)
    causal = q_positions[:, None] >= k_positions[None, :]
    local = (q_positions[:, None] - k_positions[None, :]) < self.local_window
    mask = causal & local
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    probs = softmax_shifted(scores, dim=-1)
    out = probs @ v_full
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return self.o_proj(out)


@contextmanager
def fpga_sim_mode(
    model,
    *,
    quant: QuantConfig = QuantConfig(),
    silu_cfg: LUTConfig = LUTConfig(num_entries=1024),
    restore_weights: bool = True,
):
    """Apply FPGA-equivalent math to model in-place; restore on exit.

    Args:
        restore_weights: if False, skips the deep copy of model.state_dict()
            used to restore BF16 weights on exit.  Use False for large-model
            inference-only smoke tests where the model is discarded afterward.
    """
    state_backup = copy.deepcopy(model.state_dict()) if restore_weights else None
    rmsnorm_orig = RMSNorm.forward
    mlp_orig = MLP.forward
    expert_orig = Expert.forward
    gqa_orig = GQAttention.forward

    quantize_model_(model, quant, skip=("lm_head",))

    for module in model.modules():
        if isinstance(module, (MLP, Expert)):
            module._silu_cfg = silu_cfg

    RMSNorm.forward = _patched_rmsnorm_forward
    MLP.forward = _patched_mlp_forward
    Expert.forward = _patched_expert_forward
    GQAttention.forward = _patched_gqa_forward
    try:
        yield model
    finally:
        RMSNorm.forward = rmsnorm_orig
        MLP.forward = mlp_orig
        Expert.forward = expert_orig
        GQAttention.forward = gqa_orig
        if state_backup is not None:
            model.load_state_dict(state_backup, strict=False)
