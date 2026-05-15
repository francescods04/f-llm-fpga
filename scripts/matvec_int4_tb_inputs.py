"""Emit packed-INT4 weight + INT8 activation + golden output for the C++ testbench.

Pairs with fpga/matvec_int4_tb.cpp. Round-trip path:

    1. Sample random weight + activation in float.
    2. Quantize weight via src/fllm/export.py (deterministic FLLM v1 binary).
    3. Compute the dequant-and-matvec golden in float32 (same arithmetic the
       C++ testbench will emulate).
    4. Write activation as float32 binary (same length as in_features).
    5. Write golden as float32 binary.

Caller then:

    g++ -O2 -std=c++17 fpga/matvec_int4_tb.cpp -o /tmp/matvec_int4_tb
    /tmp/matvec_int4_tb /tmp/w.bin /tmp/x.bin --golden /tmp/y.bin

and expects max_abs_err close to zero (only float rounding noise).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fllm.export import dequantize, export_linear_int4, load_linear_int4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="/tmp")
    p.add_argument("--out-features", type=int, default=128)
    p.add_argument("--in-features", type=int, default=256)
    p.add_argument("--group-size", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = out_dir / "matvec_w.bin"
    activation_path = out_dir / "matvec_x.bin"
    golden_path = out_dir / "matvec_y.bin"

    weight = torch.randn(args.out_features, args.in_features)
    export_linear_int4(weight, weight_path, group_size=args.group_size)
    record = load_linear_int4(weight_path)
    deq = dequantize(record)

    x = torch.randn(args.in_features)
    y = deq @ x

    activation_path.write_bytes(x.contiguous().numpy().astype("float32").tobytes())
    golden_path.write_bytes(y.contiguous().numpy().astype("float32").tobytes())

    print(f"weight     -> {weight_path}")
    print(f"activation -> {activation_path}")
    print(f"golden     -> {golden_path}")
    print(f"shape: out={args.out_features} in={args.in_features} group={args.group_size}")
    print(f"first 8 golden: {[float(v) for v in y[:8].tolist()]}")


if __name__ == "__main__":
    main()
