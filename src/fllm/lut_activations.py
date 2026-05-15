"""Piecewise-linear LUT activations matching FPGA BRAM-LUT implementation.

The FPGA implementation will precompute a table of N entries on a fixed input
range, then use the input MSBs as table index and LSBs as the interpolation
weight. This file does the same in Python so we can:

1. measure the quality cost of LUT-vs-exact activations before any RTL exists;
2. ship the exact same table to the bitstream (deterministic at hex level).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LUTConfig:
    """LUT geometry. Matches the BRAM that will hold it on the FPGA."""
    num_entries: int = 1024
    input_min: float = -8.0
    input_max: float = 8.0


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def build_silu_lut(cfg: LUTConfig) -> torch.Tensor:
    grid = torch.linspace(cfg.input_min, cfg.input_max, cfg.num_entries + 1)
    return _silu(grid).to(torch.float32)


def lut_apply(x: torch.Tensor, table: torch.Tensor, cfg: LUTConfig) -> torch.Tensor:
    """Piecewise-linear interpolation. Inputs outside [min, max] are clamped."""
    span = cfg.input_max - cfg.input_min
    step = span / cfg.num_entries
    xc = x.clamp(cfg.input_min, cfg.input_max - step * 1e-3)
    idx_f = (xc - cfg.input_min) / step
    idx_lo = idx_f.floor().to(torch.long).clamp(0, cfg.num_entries - 1)
    frac = (idx_f - idx_lo.to(idx_f.dtype)).to(table.dtype)
    lo = table[idx_lo]
    hi = table[idx_lo + 1]
    return lo + (hi - lo) * frac


def silu_lut(x: torch.Tensor, cfg: LUTConfig = LUTConfig()) -> torch.Tensor:
    table = build_silu_lut(cfg).to(x.device)
    return lut_apply(x, table, cfg)


def softmax_shifted(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax variant matching the FPGA shift-add path.

    Replaces exp() with a base-2 LUT after subtracting max(x). Uses 256 entries
    over [-16, 0] (everything below is treated as 0). Maintains exact denominator
    so probabilities still sum to 1.
    """
    cfg = LUTConfig(num_entries=256, input_min=-16.0, input_max=0.0)
    table = torch.exp(torch.linspace(cfg.input_min, cfg.input_max, cfg.num_entries + 1)).to(x.dtype).to(x.device)
    x_max, _ = x.max(dim=dim, keepdim=True)
    shifted = (x - x_max).clamp_min(cfg.input_min)
    exps = lut_apply(shifted, table, cfg)
    return exps / exps.sum(dim=dim, keepdim=True).clamp_min(1e-9)


def rsqrt_lut(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rsqrt via log2-domain LUT, matching the FPGA RMSNorm path.

    Decomposes x = 2^e * m, m in [1, 2). Returns 2^(-e/2) * rsqrt(m). The
    rsqrt(m) table covers m in [1,2) with 512 entries (1 BRAM).
    """
    if (x <= 0).any():
        x = x.clamp_min(eps)
    e = torch.floor(torch.log2(x))
    m = x / (2.0 ** e)
    cfg = LUTConfig(num_entries=512, input_min=1.0, input_max=2.0)
    grid = torch.linspace(cfg.input_min, cfg.input_max, cfg.num_entries + 1)
    table = (1.0 / torch.sqrt(grid)).to(x.dtype).to(x.device)
    rsqrt_m = lut_apply(m, table, cfg)
    return rsqrt_m * (2.0 ** (-e / 2.0))


def lut_quality(num_entries: int = 1024) -> dict[str, float]:
    """Diagnostic helper: max + mean abs error of the SiLU LUT on a test grid."""
    cfg = LUTConfig(num_entries=num_entries)
    grid = torch.linspace(cfg.input_min, cfg.input_max, 10_000)
    exact = _silu(grid)
    approx = silu_lut(grid, cfg)
    err = (exact - approx).abs()
    return {
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "entries": num_entries,
        "bram_kb": num_entries * 4 / 1024,
    }
