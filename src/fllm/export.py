"""FPGA-friendly packed weight export.

Binary layout per Linear tensor:

    header (32 bytes):
        magic[4]      = b"FLLM"
        version[2]    = uint16 little-endian = 1
        bits[1]       = uint8 (4 supported)
        flags[1]      = uint8 (bit0 = per-channel scales)
        out_features  = uint32
        in_features   = uint32
        group_size    = uint32 (0 = per-row)
        reserved[12]
    packed_weight: int4 values packed 2 per byte, row-major.
                   row stride = ceil(in_features / 2) bytes.
    scales: float32 per channel (or per group), row-major.

The FPGA loader reads this directly into BRAM/HBM. Layout is deterministic
across runs to enable bit-exact regressions between Python reference and RTL
testbenches.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

from fllm.quant import _qmax

MAGIC = b"FLLM"
VERSION = 1


def _quantize_int4(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = _qmax(4)
    out_features, in_features = weight.shape
    if group_size <= 0 or group_size >= in_features:
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(weight / scale).clamp(-qmax - 1, qmax).to(torch.int8)
        return q, scale.squeeze(-1).to(torch.float32)
    if in_features % group_size != 0:
        raise ValueError("group_size must divide in_features")
    groups = in_features // group_size
    w = weight.reshape(out_features, groups, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(w / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q.reshape(out_features, in_features), scale.squeeze(-1).to(torch.float32)


def _pack_pairs(qweight: torch.Tensor) -> bytes:
    out_features, in_features = qweight.shape
    if in_features % 2 == 1:
        pad = torch.zeros(out_features, 1, dtype=torch.int8)
        qweight = torch.cat([qweight, pad], dim=1)
        in_features += 1
    low = (qweight[:, 0::2] & 0x0F).to(torch.uint8)
    high = ((qweight[:, 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    return packed.numpy().tobytes()


def export_linear_int4(weight: torch.Tensor, path: str | Path, *, group_size: int = -1) -> Path:
    if weight.dim() != 2:
        raise ValueError("expected 2D weight")
    qw, scales = _quantize_int4(weight.detach().to(torch.float32), group_size)
    out_features, in_features = qw.shape
    flags = 0x01
    gsz = 0 if group_size <= 0 else group_size
    header = struct.pack(
        "<4sHBBIII12x",
        MAGIC,
        VERSION,
        4,
        flags,
        out_features,
        in_features,
        gsz,
    )
    body = _pack_pairs(qw)
    scales_bytes = scales.contiguous().numpy().tobytes()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + body + scales_bytes)
    return out_path


def load_linear_int4(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    header = raw[:32]
    magic, version, bits, flags, out_f, in_f, gsz = struct.unpack("<4sHBBIII12x", header)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if bits != 4:
        raise ValueError(f"unsupported bits {bits}")
    row_bytes = (in_f + 1) // 2
    body_size = out_f * row_bytes
    body = raw[32 : 32 + body_size]
    scales_raw = raw[32 + body_size :]
    packed = torch.frombuffer(bytearray(body), dtype=torch.uint8).reshape(out_f, row_bytes)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    interleaved = torch.empty(out_f, row_bytes * 2, dtype=torch.int8)
    interleaved[:, 0::2] = low.to(torch.int8)
    interleaved[:, 1::2] = high.to(torch.int8)
    interleaved = interleaved[:, :in_f]
    signed = torch.where(interleaved >= 8, interleaved - 16, interleaved)
    scales = torch.frombuffer(bytearray(scales_raw), dtype=torch.float32).clone()
    if gsz == 0:
        scales = scales.reshape(out_f, 1)
    else:
        scales = scales.reshape(out_f, in_f // gsz)
    return {
        "qweight": signed,
        "scales": scales,
        "out_features": out_f,
        "in_features": in_f,
        "group_size": gsz,
        "flags": flags,
    }


def dequantize(record: dict) -> torch.Tensor:
    qw = record["qweight"].to(torch.float32)
    scales = record["scales"]
    if record["group_size"] == 0:
        return qw * scales
    out_f = record["out_features"]
    in_f = record["in_features"]
    gsz = record["group_size"]
    groups = in_f // gsz
    return (qw.reshape(out_f, groups, gsz) * scales.unsqueeze(-1)).reshape(out_f, in_f)
