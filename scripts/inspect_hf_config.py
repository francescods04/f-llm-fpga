"""Load an HF config.json into FLLMConfig and report shape parity.

Usage:
    python3 scripts/inspect_hf_config.py --config datasets/qwen3-a3b/config.json
    python3 scripts/inspect_hf_config.py --emit-template datasets/qwen3-a3b/config.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from fllm.hf_config import load_hf_config, write_qwen3_a3b_template
from fllm.model import FLLMForCausalLM, count_parameters


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="datasets/qwen3-a3b/config.json")
    p.add_argument("--emit-template", default=None,
                   help="write an assumed Qwen3-A3B config template to this path")
    p.add_argument("--instantiate", action="store_true",
                   help="actually build the FLLMForCausalLM at the target shape (slow + memory heavy)")
    p.add_argument("--dummy-forward", action="store_true",
                   help="run a 1-token dummy forward to confirm the model runs end-to-end")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.emit_template:
        out = write_qwen3_a3b_template(args.emit_template)
        print(f"wrote template -> {out}")
        return

    cfg, report = load_hf_config(args.config)
    print(f"source: {report.source}")
    print("matched:", report.matched)
    print("inferred:", report.inferred)
    if report.warnings:
        print("warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    print()
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")

    if not args.instantiate:
        return

    print("\ninstantiating model (this allocates the full parameter count)...")
    model = FLLMForCausalLM(cfg)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,} (~{n_params/1e9:.2f} B)")

    if args.dummy_forward:
        x = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            y = model(x)
        print(f"forward ok: output shape {tuple(y.shape)}")


if __name__ == "__main__":
    main()
