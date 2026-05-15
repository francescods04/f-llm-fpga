"""Generate binary input + golden output files for FPGA kernel testbenches.

Pairs with:
  fpga/silu_lut_tb.cpp
  fpga/rmsnorm_engine_tb.cpp
  fpga/softmax_engine_tb.cpp

Usage:
  PYTHONPATH=src python3 scripts/fpga_kernel_golden.py --kernel all --out-dir /tmp/fllm_golden
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from fllm.lut_activations import LUTConfig, silu_lut, softmax_shifted


def write_floats(path: Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.float32
    path.write_bytes(arr.tobytes())


def read_floats(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype=np.float32)


def gen_silu(out_dir: Path, n: int = 4096) -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    cfg = LUTConfig(num_entries=1024, input_min=-8.0, input_max=8.0)
    y = silu_lut(torch.from_numpy(x), cfg).numpy()
    write_floats(out_dir / "silu_in.bin", x)
    write_floats(out_dir / "silu_golden.bin", y)
    print(f"silu: wrote {n} floats to silu_in.bin / silu_golden.bin")


def gen_rmsnorm(out_dir: Path, hidden: int = 512) -> None:
    rng = np.random.default_rng(1337)
    x = rng.normal(0.0, 1.0, size=hidden).astype(np.float32)
    weight = (1.0 + 0.1 * rng.normal(0.0, 1.0, size=hidden)).astype(np.float32)
    mean_sq = np.mean(x ** 2)
    scale = 1.0 / np.sqrt(mean_sq + 1e-6)
    y = x * scale * weight
    write_floats(out_dir / "rmsnorm_in.bin", x)
    write_floats(out_dir / "rmsnorm_weight.bin", weight)
    write_floats(out_dir / "rmsnorm_golden.bin", y)
    print(f"rmsnorm: wrote {hidden} floats to rmsnorm_in/weight/golden.bin")


def gen_softmax(out_dir: Path, seq: int = 128) -> None:
    rng = np.random.default_rng(2025)
    x = rng.normal(0.0, 2.0, size=seq).astype(np.float32)
    y = softmax_shifted(torch.from_numpy(x), dim=-1).numpy()
    write_floats(out_dir / "softmax_in.bin", x)
    write_floats(out_dir / "softmax_golden.bin", y)
    print(f"softmax: wrote {seq} floats to softmax_in/golden.bin")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", choices=["all", "silu", "rmsnorm", "softmax"], default="all")
    p.add_argument("--out-dir", default="/tmp/fllm_golden")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.kernel in ("all", "silu"):
        gen_silu(out_dir)
    if args.kernel in ("all", "rmsnorm"):
        gen_rmsnorm(out_dir)
    if args.kernel in ("all", "softmax"):
        gen_softmax(out_dir)
