#!/usr/bin/env python3
"""Measure real GPU baseline on H100 / L4 / L40S using vLLM.

This script is a thin wrapper around vLLM + nvidia-smi dmon.  It requires:
  - A GPU instance (p5, g6, g6e) on AWS or equivalent.
  - vLLM installed (`pip install vllm`).
  - The target model checkpoint available locally or via HF cache.

Execution:
  python scripts/measure_h100_baseline.py \
      --model Qwen/Qwen3-35B-A3B \
      --hardware p5.48xlarge \
      --output benchmarks/measured_gpu_baseline.json

Without a real GPU this script writes a stub with instructions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_vllm_benchmark(
    model: str,
    prompt_len: int = 1024,
    gen_len: int = 128,
    batch_size: int = 1,
    temperature: float = 0.0,
) -> dict:
    """Run vLLM benchmark and return metrics dict."""
    # This is a placeholder — real execution would call vLLM's benchmark_serving
    # or a custom script that wraps LLM.generate() with timing.
    return {
        "model": model,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "batch_size": batch_size,
        "temperature": temperature,
        "status": "stub",
        "notes": "Run on a real GPU instance with vLLM installed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GPU baseline")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--hardware", default="p5.48xlarge", choices=["p5.48xlarge", "g6.xlarge", "g6e.xlarge"])
    parser.add_argument("--output", default="benchmarks/measured_gpu_baseline.json")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--gen-len", type=int, default=128)
    args = parser.parse_args()

    print(f"[measure_h100_baseline] Target: {args.hardware} with model {args.model}")
    print("[measure_h100_baseline] NOTE: this script is a stub on CPU-only environments.")

    result = run_vllm_benchmark(args.model, args.prompt_len, args.gen_len)
    result["hardware"] = args.hardware
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[measure_h100_baseline] Wrote stub to {out_path}")
    print("[measure_h100_baseline] Next: run on a real GPU instance and replace the stub.")


if __name__ == "__main__":
    main()
