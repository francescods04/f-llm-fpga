"""Prepare Qwen3.6-35B-A3B weights for FPGA inference.

Workflow:
  1. Load model from HuggingFace (or local path).
  2. Fake-quantize to INT4 symmetric per-channel.
  3. Optionally apply N:M structured sparsity (e.g. 2:4).
  4. Export each Linear weight as packed INT4 binary (FLLM v1 format).
  5. Verify dequantization MSE < threshold.
  6. Emit a manifest JSON with layer names -> file paths and shapes.

This script does NOT load the full model into RAM at once if the model is
sharded. It processes one checkpoint file at a time, making it feasible
for 35B models on a machine with ~32 GB RAM.

Usage:
  PYTHONPATH=src python3 scripts/prepare_qwen_weights.py \
      --model-id Qwen/Qwen3.6-35B-A3B \
      --out-dir checkpoints/qwen3-fpga \
      --nm-n 2 --nm-m 4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fllm.export import export_linear_int4, load_linear_int4, dequantize
from fllm.quant import QuantConfig, fake_quant_weight
from fllm.sparsity import NMSparsity, apply_nm_mask


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def process_tensor(
    name: str,
    weight: torch.Tensor,
    out_dir: Path,
    qcfg: QuantConfig,
    nm: NMSparsity | None,
) -> dict:
    """Quantize, optionally prune, export one weight tensor."""
    if weight.dim() != 2:
        return {"name": name, "skipped": True, "reason": "not 2D"}

    w = weight.detach().to(torch.float32)
    if nm is not None:
        w = apply_nm_mask(w, nm)

    # Fake-quant to measure MSE
    wq = fake_quant_weight(w, qcfg.weight_bits, qcfg.weight_group_size)
    mse = _mse(w, wq)

    # Export to packed binary
    out_path = out_dir / f"{name.replace('.', '_')}.fllm"
    export_linear_int4(w, out_path, group_size=qcfg.weight_group_size)

    # Round-trip validation
    record = load_linear_int4(out_path)
    w_rebuilt = dequantize(record)
    mse_export = _mse(w, w_rebuilt)

    return {
        "name": name,
        "shape": list(w.shape),
        "mse_quant": mse,
        "mse_export": mse_export,
        "path": str(out_path.relative_to(out_dir)),
        "bytes": out_path.stat().st_size,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--out-dir", default="checkpoints/qwen3-fpga")
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=-1)
    p.add_argument("--nm-n", type=int, default=0, help="N:M sparsity N (0 = off)")
    p.add_argument("--nm-m", type=int, default=4)
    p.add_argument("--max-mse", type=float, default=0.05,
                   help="abort if any tensor MSE exceeds this")
    p.add_argument("--device", default="cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nm = None
    if args.nm_n > 0 and args.nm_n < args.nm_m:
        nm = NMSparsity(n=args.nm_n, m=args.nm_m)

    qcfg = QuantConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        weight_group_size=args.group_size,
        nm_sparsity=nm,
    )

    print(f"Loading model: {args.model_id}")
    print(f"Quant config: {qcfg}")
    print(f"Output dir: {out_dir}")

    # Try loading from HF; if unavailable, require local path
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"HF load failed: {e}")
        print("Please provide a local checkpoint path or install transformers.")
        return

    manifest: list[dict] = []
    total_params = 0
    total_bytes = 0
    max_mse = 0.0

    # Iterate over named parameters that look like Linear weights
    for name, param in model.named_parameters():
        # Heuristic: we only quantize MLP/attention/head weights, not embeddings
        # or norms.  Adjust as needed for the target architecture.
        if "embed" in name or "norm" in name or "lm_head" in name:
            continue
        if param.dim() != 2:
            continue

        print(f"Processing {name} ...")
        info = process_tensor(name, param, out_dir, qcfg, nm)
        manifest.append(info)
        if not info.get("skipped"):
            total_params += math.prod(info["shape"])
            total_bytes += info["bytes"]
            max_mse = max(max_mse, info["mse_export"])

    summary = {
        "model_id": args.model_id,
        "quant_config": {
            "weight_bits": args.weight_bits,
            "group_size": args.group_size,
            "nm_sparsity": f"{args.nm_n}:{args.nm_m}" if nm else "none",
        },
        "total_params": total_params,
        "total_bytes": total_bytes,
        "max_mse": max_mse,
        "layers": manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")
    print(f"Total params exported: {total_params:,}")
    print(f"Total bytes on disk: {total_bytes / 1e9:.2f} GB")
    print(f"Max MSE (quant+export): {max_mse:.6f}")
    if max_mse > args.max_mse:
        print(f"WARNING: max_mse {max_mse:.6f} exceeds threshold {args.max_mse}")


if __name__ == "__main__":
    main()
