"""Generate dummy Qwen-shaped weights and export them as FLLM binaries.

This is useful for validating the export/import pipeline and FPGA kernel
integration without downloading a real checkpoint.  It creates random
weights with the exact shapes of Qwen3-A3B linear layers, fake-quantizes
them to INT4, and writes FLLM v1 binaries plus a manifest.

Usage:
  PYTHONPATH=src python3 scripts/prepare_dummy_weights.py \
      --out-dir checkpoints/dummy-fpga
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fllm.export import export_linear_int4, load_linear_int4, dequantize
from fllm.quant import QuantConfig, fake_quant_weight


# Qwen3-A3B linear layer shapes (approximate, from config)
# hidden=4096, heads=32, kv_heads=4, head_dim=128, mlp_inner=14336
# num_layers=48, vocab=152064
_QWEN_SHAPES = {
    "qkv": (4096, 4096 * 3),      # hidden -> 3*hidden
    "o_proj": (4096, 4096),        # hidden -> hidden
    "up": (4096, 14336),           # hidden -> mlp_inner
    "gate": (4096, 14336),         # hidden -> mlp_inner
    "down": (14336, 4096),         # mlp_inner -> hidden
}


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def make_dummy_weight(name: str, out_dir: Path, qcfg: QuantConfig) -> dict:
    if name not in _QWEN_SHAPES:
        raise ValueError(f"Unknown dummy weight name: {name}")
    out_f, in_f = _QWEN_SHAPES[name]
    w = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1

    wq = fake_quant_weight(w, qcfg.weight_bits, qcfg.weight_group_size)
    mse = _mse(w, wq)

    out_path = out_dir / f"dummy_{name}.fllm"
    export_linear_int4(w, out_path, group_size=qcfg.weight_group_size)

    # Round-trip
    record = load_linear_int4(out_path)
    w_rebuilt = dequantize(record)
    mse_export = _mse(w, w_rebuilt)

    return {
        "name": f"dummy.{name}",
        "shape": [out_f, in_f],
        "mse_quant": mse,
        "mse_export": mse_export,
        "path": str(out_path.name),
        "bytes": out_path.stat().st_size,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="checkpoints/dummy-fpga")
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=-1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qcfg = QuantConfig(weight_bits=args.weight_bits, weight_group_size=args.group_size)
    print(f"Generating dummy weights -> {out_dir}")
    print(f"Quant config: {qcfg}")

    manifest = []
    total_params = 0
    total_bytes = 0
    max_mse = 0.0

    for name in _QWEN_SHAPES:
        print(f"  {name} ...")
        info = make_dummy_weight(name, out_dir, qcfg)
        manifest.append(info)
        total_params += math.prod(info["shape"])
        total_bytes += info["bytes"]
        max_mse = max(max_mse, info["mse_export"])

    # Also generate a scaled-down block set (representing one layer repeated)
    layer_manifest = []
    for layer_idx in range(48):
        for name in _QWEN_SHAPES:
            layer_manifest.append({
                "layer": layer_idx,
                "tensor": f"layers.{layer_idx}.{name}",
                "ref": f"dummy_{name}.fllm",
            })

    summary = {
        "model_id": "dummy_qwen3_35b_a3b",
        "quant_config": {
            "weight_bits": args.weight_bits,
            "group_size": args.group_size,
        },
        "total_params": total_params,
        "total_bytes": total_bytes,
        "max_mse": max_mse,
        "unique_tensors": manifest,
        "layers": layer_manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")
    print(f"Total params: {total_params:,}")
    print(f"Total bytes: {total_bytes / 1e6:.2f} MB")
    print(f"Max MSE: {max_mse:.6f}")


if __name__ == "__main__":
    main()
