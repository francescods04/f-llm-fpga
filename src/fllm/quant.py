"""Quantization simulation for FPGA-targeted Linear layers.

Goal: simulate per-channel symmetric INT4 weights and per-tensor symmetric INT8
activations with the exact rounding rules the FPGA matvec engine will use, so
quality can be measured before any RTL exists.

Design choices forced by hardware:

- symmetric quantization (no zero-point) -> single multiply on FPGA.
- per-channel scales on the weight output dim (group_size=-1 means per-row),
  optional finer grouping along input dim for ablations.
- "fake-quant": forward path mimics int math, gradient is straight-through.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QuantConfig:
    weight_bits: int = 4
    activation_bits: int = 8
    weight_group_size: int = -1  # -1 = per output channel only


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


def fake_quant_weight(weight: torch.Tensor, bits: int, group_size: int = -1) -> torch.Tensor:
    qmax = _qmax(bits)
    if group_size <= 0 or group_size >= weight.size(-1):
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = ste_round(weight / scale).clamp(-qmax - 1, qmax)
        return q * scale

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError("group_size must divide in_features")
    groups = in_features // group_size
    w = weight.reshape(out_features, groups, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = ste_round(w / scale).clamp(-qmax - 1, qmax)
    return (q * scale).reshape(out_features, in_features)


def fake_quant_activation(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = _qmax(bits)
    scale = x.detach().abs().amax().clamp_min(1e-8) / qmax
    q = ste_round(x / scale).clamp(-qmax - 1, qmax)
    return q * scale


class QuantLinear(nn.Module):
    """Drop-in fake-quant replacement for nn.Linear."""

    def __init__(self, base: nn.Linear, qcfg: QuantConfig) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("FPGA target uses bias-free Linear; rebuild without bias")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.qcfg = qcfg
        self.weight = nn.Parameter(base.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant_activation(x, self.qcfg.activation_bits)
        wq = fake_quant_weight(self.weight, self.qcfg.weight_bits, self.qcfg.weight_group_size)
        return torch.nn.functional.linear(xq, wq)


def quantize_model_(model: nn.Module, qcfg: QuantConfig, *, skip: tuple[str, ...] = ()) -> int:
    """Recursively replace nn.Linear modules in-place with QuantLinear. Returns count replaced."""
    replaced = 0
    for name, child in list(model.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear) and child.bias is None:
            setattr(model, name, QuantLinear(child, qcfg))
            replaced += 1
        else:
            replaced += quantize_model_(child, qcfg, skip=skip)
    return replaced
