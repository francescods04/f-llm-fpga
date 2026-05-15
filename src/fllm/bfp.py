"""Block-floating-point quantization for KV cache and activations.

A BFP block of width `block_size` shares one exponent (int8) across all lanes;
each lane stores a `mantissa_bits` signed integer mantissa. Effective bits/value
= `mantissa_bits + 8 / block_size`. For block_size=32, mantissa_bits=8 this is
~8.25 bits/value with FP16-class dynamic range.

The FPGA implementation packs (exp, mantissas) in HBM. On read: mantissa is
sign-extended and shifted by exp. On write: a block-max reduce picks the new
exponent.

This file is the bit-exact Python reference. Forward path quantizes; gradient
is straight-through.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BFPConfig:
    mantissa_bits: int = 8
    block_size: int = 32


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def bfp_quantize(x: torch.Tensor, cfg: BFPConfig = BFPConfig()) -> torch.Tensor:
    """Per-block-along-last-dim shared-exponent quantize-dequantize.

    The block dimension is the **last** axis of x. Last axis must be a multiple
    of cfg.block_size, or it is padded with zeros and un-padded after.
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    bs = cfg.block_size
    pad = (-last) % bs
    if pad:
        zeros = torch.zeros(*orig_shape[:-1], pad, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=-1)

    blocks = x.shape[-1] // bs
    leading = x.shape[:-1]
    xb = x.reshape(*leading, blocks, bs)

    block_max = xb.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    qmax = _qmax(cfg.mantissa_bits)
    exp_float = torch.ceil(torch.log2(block_max / qmax))
    scale = torch.pow(2.0, exp_float)

    mant_round = torch.round(xb / scale).clamp(-qmax - 1, qmax)
    deq = mant_round * scale
    deq = deq.reshape(*leading, blocks * bs)
    if pad:
        deq = deq[..., :last]
    return _ste(x[..., :last] if pad else x, deq)


def _ste(x: torch.Tensor, deq: torch.Tensor) -> torch.Tensor:
    return x + (deq - x).detach()


def bfp_pack(x: torch.Tensor, cfg: BFPConfig = BFPConfig()) -> dict:
    """Return packed components mantissa (int8) + exponent (int8) per block.

    This is what the FPGA writes to HBM. Useful for measuring real byte cost.
    """
    last = x.shape[-1]
    bs = cfg.block_size
    pad = (-last) % bs
    if pad:
        zeros = torch.zeros(*x.shape[:-1], pad, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=-1)
    blocks = x.shape[-1] // bs
    leading = x.shape[:-1]
    xb = x.reshape(*leading, blocks, bs)
    block_max = xb.abs().amax(dim=-1).clamp_min(1e-30)
    qmax = _qmax(cfg.mantissa_bits)
    exp = torch.ceil(torch.log2(block_max / qmax)).to(torch.int8)
    scale = torch.pow(2.0, exp.to(torch.float32)).unsqueeze(-1)
    mant = torch.round(xb / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return {
        "mantissa": mant,
        "exponent": exp,
        "block_size": bs,
        "mantissa_bits": cfg.mantissa_bits,
        "original_last_dim": last,
    }


def bfp_unpack(record: dict) -> torch.Tensor:
    mant = record["mantissa"].to(torch.float32)
    exp = record["exponent"].to(torch.float32).unsqueeze(-1)
    scale = torch.pow(2.0, exp)
    out = mant * scale
    last = record["original_last_dim"]
    out = out.reshape(*out.shape[:-2], -1)
    return out[..., :last]


def bytes_per_value(cfg: BFPConfig) -> float:
    return cfg.mantissa_bits / 8.0 + 1.0 / cfg.block_size
